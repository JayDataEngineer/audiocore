// gguf_reader.cpp — GGUF weight loader implementation.
//
// Adapted from stable-diffusion.cpp (MIT, Copyright (c) 2023 leejet):
//   https://github.com/leejet/stable-diffusion.cpp/blob/master/src/model_io/gguf_io.cpp
//
// The fallback manual reader (GGUFReader class in gguf_reader_ext.h) handles
// edge cases that stock gguf_init_from_file() can't: split shards, alignment
// quirks, quantization type remapping for tensors stored with non-standard
// dtypes (e.g. f8_e4m3, i64). Both code paths produce the same
// vector<TensorStorage> output; model code can't tell them apart.
//
// License: MIT inherited from the source.

#include "audiocore/framework/io/gguf_reader.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "audiocore/framework/io/gguf_reader_ext.h"

namespace audiocore {

namespace {

void set_error(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
}

}  // namespace

// RAII wrapper around mmap(). Forward-declared in the header (without sys/mman.h)
// and defined here so the rest of the project doesn't need to see POSIX types.
// Windows support goes through a future MapViewOfFile branch — marked TODO.
struct MMapHandle {
    void*  base = MAP_FAILED;
    size_t size = 0;
    int    fd   = -1;
    ~MMapHandle() {
        if (base != MAP_FAILED) munmap(static_cast<char*>(base), size);
        if (fd   != -1)         close(fd);
    }
};

GgufReader::GgufReader() = default;   // out-of-line so member cleanup sees full MMapHandle
GgufReader::~GgufReader() = default;

bool GgufReader::is_gguf_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    char magic[4];
    f.read(magic, sizeof(magic));
    if (!f) return false;
    for (uint32_t i = 0; i < sizeof(magic); i++) {
        if (magic[i] != GGUF_MAGIC[i]) return false;
    }
    return true;
}

bool GgufReader::load(const std::string& path, std::string* error) {
    path_ = path;
    tensors_.clear();

    // ── Path 1: try upstream ggml GGUF parser ─────────────────────────────
    // no_alloc=true so meta_ctx_ holds tensor structs only; the actual weight
    // bytes stay in our mmap below (zero-copy, see tensor_data_ptr()).
    ctx_ = gguf_init_from_file(path.c_str(), {.no_alloc = true, .ctx = &meta_ctx_});
    if (ctx_) {
        // mmap the whole file once. tensor_data_ptr() returns
        // mmap_base_ + t.offset, so family code gets zero-copy access to
        // multi-GB weights without doubling RSS.
        std::unique_ptr<MMapHandle> mm(new MMapHandle);
        mm->fd = ::open(path.c_str(), O_RDONLY);
        if (mm->fd < 0) {
            set_error(error, "open() failed for '" + path + "'");
            return false;
        }
        struct stat st {};
        if (::fstat(mm->fd, &st) != 0) {
            set_error(error, "fstat() failed for '" + path + "'");
            return false;
        }
        mm->size = static_cast<size_t>(st.st_size);
        mm->base = ::mmap(nullptr, mm->size, PROT_READ, MAP_PRIVATE, mm->fd, 0);
        if (mm->base == MAP_FAILED) {
            set_error(error, "mmap() failed for '" + path + "'");
            return false;
        }
        // Keep the mmap alive for our lifetime. Released in destructor via
        // the unique_ptr below.
        mmap_owner_ = std::move(mm);
        mmap_base_  = static_cast<const char*>(mmap_owner_->base);

        const int n_tensors = static_cast<int>(gguf_get_n_tensors(ctx_));
        const size_t data_offset = gguf_get_data_offset(ctx_);
        tensors_.reserve(n_tensors);
        for (int i = 0; i < n_tensors; i++) {
            std::string name = gguf_get_tensor_name(ctx_, i);
            ggml_tensor* meta = ggml_get_tensor(meta_ctx_, name.c_str());
            if (!meta) {
                set_error(error, "tensor metadata missing for '" + name + "'");
                return false;
            }
            const size_t offset = data_offset + gguf_get_tensor_offset(ctx_, i);
            TensorStorage ts(name, meta->type, meta->ne, ggml_n_dims(meta),
                             /*file_index=*/0, offset);
            if (ggml_nbytes(meta) != ts.nbytes()) {
                set_error(error, "size mismatch for tensor '" + name + "'");
                return false;
            }
            tensors_.push_back(std::move(ts));
        }
        return true;
    }

    // ── Path 2: fallback manual reader ────────────────────────────────────
    // For files the stock parser rejects. Same TensorStorage output shape.
    // No mmap here — materialize() will read on demand. (Rare path; the cost
    // of always-mmap'ing files that fall through to Path 2 isn't worth it.)
    GGUFReaderExt reader;
    if (!reader.load(path)) {
        set_error(error, "failed to open '" + path + "' with GGUFReaderExt");
        return false;
    }
    const size_t data_offset = reader.data_offset();
    tensors_.reserve(reader.tensors().size());
    for (const auto& info : reader.tensors()) {
        TensorStorage ts(info.name, info.type, info.shape.data(),
                         static_cast<int>(info.shape.size()),
                         /*file_index=*/0, data_offset + info.offset);
        tensors_.push_back(std::move(ts));
    }
    // Cache for materialize() — the fallback reader has no live ggml_context
    // backing it, so we keep its parsed state alive for our lifetime.
    fallback_tensors_ = std::move(tensors_);
    tensors_ = fallback_tensors_;  // keep find()/tensors() consistent
    return true;
}

bool GgufReader::materialize(const TensorStorage& t, void* dst,
                             std::string* error) const {
    // Fast path: alias the mmap.
    if (mmap_base_) {
        if (t.offset + t.nbytes_to_read() > mmap_owner_->size) {
            set_error(error, "tensor '" + t.name + "' runs past EOF");
            return false;
        }
        std::memcpy(dst, mmap_base_ + t.offset, t.nbytes_to_read());
        return true;
    }
    // Fallback path (Path 2 above): open, seek, read.
    std::ifstream f(path_, std::ios::binary);
    if (!f.is_open()) {
        set_error(error, "could not reopen '" + path_ + "'");
        return false;
    }
    f.seekg(static_cast<std::streamoff>(t.offset), std::ios::beg);
    const int64_t to_read = t.nbytes_to_read();
    f.read(static_cast<char*>(dst), to_read);
    if (!f) {
        set_error(error, "short read on tensor '" + t.name + "'");
        return false;
    }
    return true;
}

// ---- Typed KV getters ----------------------------------------------------
// Thin wrappers over gguf_get_val_* so family code doesn't have to know the
// enum-as-int dance. All return false on missing key or type mismatch.

bool GgufReader::get_kv_i32(const char* key, int32_t* out) const {
    if (!ctx_) return false;
    const int64_t k = gguf_find_key(ctx_, key);
    if (k < 0) return false;
    if (gguf_get_kv_type(ctx_, k) != GGUF_TYPE_INT32) return false;
    *out = gguf_get_val_i32(ctx_, k);
    return true;
}

bool GgufReader::get_kv_i64(const char* key, int64_t* out) const {
    if (!ctx_) return false;
    const int64_t k = gguf_find_key(ctx_, key);
    if (k < 0) return false;
    if (gguf_get_kv_type(ctx_, k) != GGUF_TYPE_INT64) return false;
    *out = gguf_get_val_i64(ctx_, k);
    return true;
}

bool GgufReader::get_kv_u32(const char* key, uint32_t* out) const {
    if (!ctx_) return false;
    const int64_t k = gguf_find_key(ctx_, key);
    if (k < 0) return false;
    if (gguf_get_kv_type(ctx_, k) != GGUF_TYPE_UINT32) return false;
    *out = gguf_get_val_u32(ctx_, k);
    return true;
}

bool GgufReader::get_kv_bool(const char* key, bool* out) const {
    if (!ctx_) return false;
    const int64_t k = gguf_find_key(ctx_, key);
    if (k < 0) return false;
    if (gguf_get_kv_type(ctx_, k) != GGUF_TYPE_BOOL) return false;
    *out = gguf_get_val_bool(ctx_, k);
    return true;
}

bool GgufReader::get_kv_f32(const char* key, float* out) const {
    if (!ctx_) return false;
    const int64_t k = gguf_find_key(ctx_, key);
    if (k < 0) return false;
    if (gguf_get_kv_type(ctx_, k) != GGUF_TYPE_FLOAT32) return false;
    *out = gguf_get_val_f32(ctx_, k);
    return true;
}

bool GgufReader::get_kv_str(const char* key, std::string* out) const {
    if (!ctx_) return false;
    const int64_t k = gguf_find_key(ctx_, key);
    if (k < 0) return false;
    if (gguf_get_kv_type(ctx_, k) != GGUF_TYPE_STRING) return false;
    *out = gguf_get_val_str(ctx_, k);
    return true;
}

const void* GgufReader::tensor_data_ptr(const TensorStorage& t) const {
    if (mmap_base_) {
        // Bounds check; on overflow return null so callers can fall back to
        // materialize() rather than reading garbage.
        if (t.offset + t.nbytes_to_read() > mmap_owner_->size) return nullptr;
        return mmap_base_ + t.offset;
    }
    // Path 2 has no mmap; caller must use materialize() instead.
    return nullptr;
}

bool write_gguf_file(const std::string& path,
                     const std::vector<TensorWriteInfo>& tensors,
                     std::string* error) {
    gguf_context* ctx = gguf_init_empty();
    if (!ctx) {
        set_error(error, "gguf_init_empty failed");
        return false;
    }
    for (const TensorWriteInfo& info : tensors) {
        if (!info.tensor) {
            set_error(error, "null tensor cannot be written to GGUF");
            gguf_free(ctx);
            return false;
        }
        gguf_add_tensor(ctx, info.tensor);
    }
    const bool ok = gguf_write_to_file(ctx, path.c_str(), /*parallel=*/false);
    if (!ok) set_error(error, "failed to write GGUF '" + path + "'");
    gguf_free(ctx);
    return ok;
}

}  // namespace audiocore
