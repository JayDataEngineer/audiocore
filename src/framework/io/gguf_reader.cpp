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
#include <fstream>
#include <string>
#include <vector>

#include "audiocore/framework/io/gguf_reader_ext.h"

namespace audiocore {

namespace {

void set_error(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
}

}  // namespace

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
    // Handles well-formed files written by gguf_init_empty()/gguf_write_to_file(),
    // including the standard quantization types (Q4_0 … Q8_0, F16, BF16, …).
    ctx_ = gguf_init_from_file(path.c_str(), {.no_alloc = true, .ctx = &meta_ctx_});
    if (ctx_) {
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
    // Single-file GGUF: open, seek, read. Multi-file (sharded) variants
    // would key off t.storage_key instead. MOSS/ACE GGUFs are single-file
    // today; sharding is a Phase 2 concern.
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
    // f8/f64/i64 unpack → f16/f32. Same conversion rules as sd.cpp.
    // (Defer to a shared convert pass; see materialize_conversions.cpp.)
    return true;
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
