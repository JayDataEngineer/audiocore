// tensor_storage.h — format-neutral weight descriptor.
//
// Adapted from stable-diffusion.cpp (MIT, Copyright (c) 2023 leejet):
//   https://github.com/leejet/stable-diffusion.cpp/blob/master/src/model_io/tensor_storage.h
//
// The TensorStorage struct is the seam between "weight format readers" and
// "model code". Every reader (GGUF, safetensors, ONNX, …) produces a
// vector<TensorStorage>; model code asks for tensors by name and the
// framework materializes them lazily into whichever backend is active
// (ggml CUDA/CPU today, ONNX Runtime later).
//
// Keeping this neutral is what lets us add new weight formats without
// touching model code, and new execution backends without touching readers.

#ifndef AUDIOCORE_FRAMEWORK_IO_TENSOR_STORAGE_H
#define AUDIOCORE_FRAMEWORK_IO_TENSOR_STORAGE_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "ggml.h"

namespace audiocore {

constexpr int kMaxDims = 5;

struct TensorStorage {
    std::string name;
    ggml_type type          = GGML_TYPE_F32;
    ggml_type expected_type = GGML_TYPE_COUNT;  // set by model code if it wants
                                                // to assert a specific dtype
    bool is_f8_e4m3 = false;
    bool is_f8_e5m2 = false;
    bool is_f64     = false;
    bool is_i64     = false;

    int64_t ne[kMaxDims] = {1, 1, 1, 1, 1};
    int n_dims           = 0;

    // Where the bytes actually live. Readers fill these in so the framework
    // can mmap/fread tensors on demand rather than holding the whole model
    // in RAM.
    std::string storage_key;   // e.g. "file:0" / "zip:1:2" / "onnx:weight"
    size_t file_index = 0;     // which file in a multi-file model dir
    int index_in_zip  = -1;    // >=0 means stored inside a zip archive
    uint64_t offset   = 0;     // byte offset within the storage_key

    TensorStorage() = default;

    TensorStorage(std::string name_, ggml_type type_, const int64_t* ne_,
                  int n_dims_, size_t file_index_, size_t offset_ = 0)
        : name(std::move(name_)),
          type(type_),
          n_dims(n_dims_),
          file_index(file_index_),
          offset(offset_) {
        for (int i = 0; i < n_dims_; i++) {
            ne[i] = ne_[i];
        }
    }

    int64_t nelements() const {
        int64_t n = 1;
        for (int i = 0; i < kMaxDims; i++) n *= ne[i];
        return n;
    }

    int64_t nbytes() const {
        return nelements() * ggml_type_size(type) / ggml_blck_size(type);
    }

    // f8/f64/i64 are stored on disk in a packed form; readers convert on load.
    int64_t nbytes_to_read() const {
        if (is_f8_e4m3 || is_f8_e5m2) return nbytes() / 2;
        if (is_f64 || is_i64)         return nbytes() * 2;
        return nbytes();
    }

    int64_t element_size() const {
        return ggml_type_size(type) / ggml_blck_size(type);
    }
};

// Used by GGUF *writing* (model export / quantization tools). Same field
// shape as TensorStorage but carries the live ggml_tensor pointer.
struct TensorWriteInfo {
    ggml_tensor* tensor = nullptr;
    std::string name;
};

}  // namespace audiocore

#endif  // AUDIOCORE_FRAMEWORK_IO_TENSOR_STORAGE_H
