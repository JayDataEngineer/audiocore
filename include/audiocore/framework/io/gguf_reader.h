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

class GgufReader : public WeightLoader {
public:
    bool load(const std::string& path, std::string* error = nullptr) override;
    bool materialize(const TensorStorage& t, void* dst,
                     std::string* error = nullptr) const override;

    // Direct file detection (used by make_weight_loader for dispatch).
    static bool is_gguf_file(const std::string& path);

    // Raw metadata key/value access (model code uses this for things like
    // the MOSS audio codec config stored in GGUF general metadata).
    const gguf_context* ctx() const { return ctx_; }

private:
    std::string path_;
    gguf_context* ctx_        = nullptr;
    ggml_context* meta_ctx_   = nullptr;
    // Owned by GgufReader when the fallback manual parser is used.
    std::vector<TensorStorage> fallback_tensors_;
};

// Write tensors to a GGUF file. Used by tools/convert_safetensors_to_gguf.py
// (via a small pybind shim) and tools/quantize.
bool write_gguf_file(const std::string& path,
                     const std::vector<TensorWriteInfo>& tensors,
                     std::string* error = nullptr);

}  // namespace audiocore

#endif  // AUDIOCORE_FRAMEWORK_IO_GGUF_READER_H
