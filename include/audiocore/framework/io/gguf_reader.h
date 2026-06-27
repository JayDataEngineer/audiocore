// gguf_reader.h — GGUF weight loader.
//
// Adapted from stable-diffusion.cpp (MIT, Copyright (c) 2023 leejet):
//   https://github.com/leejet/stable-diffusion.cpp/blob/master/src/model_io/gguf_io.{h,cpp}
//   https://github.com/leejet/stable-diffusion.cpp/blob/master/src/model_io/gguf_reader_ext.h
//
// Two-stage strategy, same as sd.cpp:
//   1. Try upstream gguf_init_from_file() — handles 99% of well-formed files.
//   2. Fall back to a manual GGUFReader for edge cases the stock loader
//      can't handle (split shards, unusual alignment, quantization type
//      remapping for non-standard encodings).
//
// Writing (used by the converter / quantizer) uses gguf_init_empty() +
// gguf_add_tensor() + gguf_write_to_file() — same as sd.cpp's
// write_gguf_file().

#ifndef AUDIOCORE_FRAMEWORK_IO_GGUF_READER_H
#define AUDIOCORE_FRAMEWORK_IO_GGUF_READER_H

#include "gguf.h"
#include "weight_loader.h"

namespace audiocore {

// Forward decl — full definition is in gguf_reader.cpp (keeps sys/mman.h
// out of this header).
struct MMapHandle;

class GgufReader : public WeightLoader {
public:
    // Default ctor / dtor declared here, defined out-of-line in
    // gguf_reader.cpp. They must NOT be inline so the implicit cleanup of
    // the unique_ptr<MMapHandle> member happens in a TU that sees the full
    // MMapHandle definition (otherwise callers that just forward-declare
    // GgufReader hit "incomplete type" errors).
    GgufReader();
    ~GgufReader() override;
    bool load(const std::string& path, std::string* error = nullptr) override;
    bool materialize(const TensorStorage& t, void* dst,
                     std::string* error = nullptr) const override;

    // Direct file detection (used by make_weight_loader for dispatch).
    static bool is_gguf_file(const std::string& path);

    // Raw metadata key/value access (model code uses this for things like
    // the MOSS audio codec config stored in GGUF general metadata).
    const gguf_context* ctx() const { return ctx_; }

    // ---- Typed KV getters ------------------------------------------------
    // All return false if the key is absent OR the stored type doesn't match.
    // Family code uses these instead of raw gguf_get_val_* calls so that the
    // "what type did OpenMOSS ship this key as" question lives in one place.
    bool get_kv_i32(const char* key, int32_t* out) const;
    bool get_kv_i64(const char* key, int64_t* out) const;
    bool get_kv_u32(const char* key, uint32_t* out) const;
    bool get_kv_bool(const char* key, bool* out) const;
    bool get_kv_str(const char* key, std::string* out) const;

    // ---- Zero-copy tensor data access -----------------------------------
    // Returns a pointer into the mmap'd GGUF region for the tensor described
    // by `t`. Valid for the lifetime of this GgufReader. Returns nullptr if
    // the tensor isn't backed by a mmap'd region (caller must materialize()
    // into its own buffer in that case).
    //
    // This is the path family code uses for hot tensors — embedding tables,
    // LM heads, codec weights — where copying on every load is wasteful.
    const void* tensor_data_ptr(const TensorStorage& t) const;

    // Underlying ggml_context that owns the tensor structs (their ->data
    // pointers alias the mmap). Exposed so family code can use
    // ggml_get_tensor() / ggml_n_elements() without re-implementing lookup.
    ggml_context* meta_ctx() const { return meta_ctx_; }

private:
    std::string path_;
    gguf_context* ctx_        = nullptr;
    ggml_context* meta_ctx_   = nullptr;
    // Owned by GgufReader when the fallback manual parser is used.
    std::vector<TensorStorage> fallback_tensors_;
    // POSIX mmap of the GGUF file, kept alive so tensor_data_ptr() can return
    // pointers into it. Type is forward-declared below; full definition lives
    // in gguf_reader.cpp so this header doesn't need sys/mman.h.
    std::unique_ptr<MMapHandle> mmap_owner_;
    const char*                 mmap_base_ = nullptr;
};

// Write tensors to a GGUF file. Used by tools/convert_safetensors_to_gguf.py
// (via a small pybind shim) and tools/quantize.
bool write_gguf_file(const std::string& path,
                     const std::vector<TensorWriteInfo>& tensors,
                     std::string* error = nullptr);

}  // namespace audiocore

#endif  // AUDIOCORE_FRAMEWORK_IO_GGUF_READER_H
