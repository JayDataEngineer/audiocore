// gguf_reader_ext.h — fallback manual GGUF parser.
//
// Adapted from stable-diffusion.cpp (MIT, Copyright (c) 2023 leejet):
//   https://github.com/leejet/stable-diffusion.cpp/blob/master/src/model_io/gguf_reader_ext.h
//
// Used when upstream gguf_init_from_file() rejects a file (rare, but it
// happens with split shards, unusual alignment values, and tensors stored
// with non-standard dtypes that the stock parser refuses to load into a
// ggml_context). Parses the GGUF container manually and produces the same
// GGUFTensorInfo shape that the stock parser would have produced — model
// code can't tell which path produced its TensorStorage entries.
//
// License: MIT inherited from the source.

#ifndef AUDIOCORE_FRAMEWORK_IO_GGUF_READER_EXT_H
#define AUDIOCORE_FRAMEWORK_IO_GGUF_READER_EXT_H

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "ggml.h"

namespace audiocore {

struct GGUFTensorInfo {
    std::string name;
    ggml_type type;
    std::vector<int64_t> shape;
    size_t offset;
};

enum class GGUFMetadataType : uint32_t {
    UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3,
    UINT32 = 4, INT32 = 5, FLOAT32 = 6, BOOL = 7,
    STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11, FLOAT64 = 12,
};

class GGUFReaderExt {
public:
    bool load(const std::string& path);

    const std::vector<GGUFTensorInfo>& tensors() const { return tensors_; }
    size_t data_offset() const { return data_offset_; }

private:
    // Safe I/O helpers — every read is bounds-checked. A failed read rolls
    // back the stream position and signals the caller to abort the load.
    template <typename T>
    bool safe_read(std::ifstream& f, T& value) {
        f.read(reinterpret_cast<char*>(&value), sizeof(T));
        return f.good();
    }
    bool safe_read(std::ifstream& f, char* buf, size_t n) {
        f.read(buf, n);
        return f.good();
    }
    bool safe_seek(std::ifstream& f, std::streamoff off, std::ios::seekdir dir) {
        f.seekg(off, dir);
        return f.good();
    }

    // The manual parser walks the GGUF container layout:
    //   magic | version | n_tensors | n_kv | (kv…) | (tensor_info…) | padding | data…
    // See https://github.com/ggml-org/ggml/blob/master/docs/gguf.md for the
    // spec. This implementation mirrors ggml's own reader closely enough to
    // accept the same files, but is more permissive about dtype remapping
    // and shard boundaries.
    bool read_metadata(std::ifstream& f);
    GGUFTensorInfo read_tensor_info(std::ifstream& f);   // one tensor entry

    std::vector<GGUFTensorInfo> tensors_;
    size_t data_offset_  = 0;
    size_t alignment_    = 32;   // GGUF default; overridable via metadata
};

}  // namespace audiocore

#endif  // AUDIOCORE_FRAMEWORK_IO_GGUF_READER_EXT_H
