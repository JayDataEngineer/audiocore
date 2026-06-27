// gguf_reader_ext.cpp — fallback manual GGUF parser.
//
// Adapted verbatim from stable-diffusion.cpp (MIT, Copyright (c) 2023 leejet):
//   https://github.com/leejet/stable-diffusion.cpp/blob/master/src/model_io/gguf_reader_ext.h
//
// Used when upstream gguf_init_from_file() rejects a file. Parses the GGUF
// container layout (magic | version | n_tensors | n_kv | (kv…) | (info…) |
// padding | data…) manually, producing the same GGUFTensorInfo shape the
// stock parser would have. Model code can't tell which path produced its
// TensorStorage entries.
//
// License: MIT inherited from the source.

#include "audiocore/framework/io/gguf_reader_ext.h"

#include <cstring>
#include "ggml.h"

// Minimal LOG_* shims so we don't pull in sd.cpp's logging header.
//audiocore uses fprintf(stderr) for now; wire to a real logger later.
#include <cstdio>
#define LOG_DEBUG(...) do {} while (0)
#define LOG_ERROR(...) fprintf(stderr, "[gguf_reader_ext] " __VA_ARGS__); fprintf(stderr, "\n")
#define LOG_INFO(...)  fprintf(stderr, "[gguf_reader_ext] " __VA_ARGS__); fprintf(stderr, "\n")

namespace audiocore {

bool GGUFReaderExt::read_metadata(std::ifstream& fin) {
    uint64_t key_len = 0;
    if (!safe_read(fin, key_len)) return false;
    if (key_len > 4096) return false;

    std::string key(key_len, '\0');
    if (!safe_read(fin, key.data(), key_len)) return false;

    uint32_t type = 0;
    if (!safe_read(fin, type)) return false;

    if (key == "general.alignment") {
        uint32_t align_val = 0;
        if (!safe_read(fin, align_val)) return false;
        if (align_val != 0 && (align_val & (align_val - 1)) == 0) {
            alignment_ = align_val;
        }
        return true;
    }

    switch (static_cast<GGUFMetadataType>(type)) {
        case GGUFMetadataType::UINT8:
        case GGUFMetadataType::INT8:
        case GGUFMetadataType::BOOL:
            return safe_seek(fin, 1, std::ios::cur);
        case GGUFMetadataType::UINT16:
        case GGUFMetadataType::INT16:
            return safe_seek(fin, 2, std::ios::cur);
        case GGUFMetadataType::UINT32:
        case GGUFMetadataType::INT32:
        case GGUFMetadataType::FLOAT32:
            return safe_seek(fin, 4, std::ios::cur);
        case GGUFMetadataType::UINT64:
        case GGUFMetadataType::INT64:
        case GGUFMetadataType::FLOAT64:
            return safe_seek(fin, 8, std::ios::cur);
        case GGUFMetadataType::STRING: {
            uint64_t len = 0;
            if (!safe_read(fin, len)) return false;
            return safe_seek(fin, len, std::ios::cur);
        }
        case GGUFMetadataType::ARRAY: {
            uint32_t elem_type = 0;
            uint64_t len       = 0;
            if (!safe_read(fin, elem_type)) return false;
            if (!safe_read(fin, len))       return false;
            for (uint64_t i = 0; i < len; i++) {
                if (!read_metadata(fin)) return false;
            }
            return true;
        }
        default:
            LOG_ERROR("unknown metadata type=%u", type);
            return false;
    }
}

GGUFTensorInfo GGUFReaderExt::read_tensor_info(std::ifstream& fin) {
    GGUFTensorInfo info;
    uint64_t name_len;
    if (!safe_read(fin, name_len))
        throw std::runtime_error("read tensor name length failed");
    info.name.resize(name_len);
    if (!safe_read(fin, info.name.data(), name_len))
        throw std::runtime_error("read tensor name failed");

    uint32_t n_dims;
    if (!safe_read(fin, n_dims))
        throw std::runtime_error("read tensor dims failed");
    info.shape.resize(n_dims);
    for (uint32_t i = 0; i < n_dims; i++) {
        if (!safe_read(fin, info.shape[i]))
            throw std::runtime_error("read tensor shape failed");
    }
    if (n_dims > GGML_MAX_DIMS) {
        for (uint32_t i = GGML_MAX_DIMS; i < n_dims; i++) {
            info.shape[GGML_MAX_DIMS - 1] *= info.shape[i];
        }
        info.shape.resize(GGML_MAX_DIMS);
    }
    uint32_t type;
    if (!safe_read(fin, type))
        throw std::runtime_error("read tensor type failed");
    info.type = static_cast<ggml_type>(type);
    if (!safe_read(fin, info.offset))
        throw std::runtime_error("read tensor offset failed");
    return info;
}

bool GGUFReaderExt::load(const std::string& file_path) {
    std::ifstream fin(file_path, std::ios::binary);
    if (!fin) {
        LOG_ERROR("failed to open '%s'", file_path.c_str());
        return false;
    }
    char magic[4];
    if (!safe_read(fin, magic, 4) || strncmp(magic, "GGUF", 4) != 0) {
        LOG_ERROR("not a valid GGUF file");
        return false;
    }
    uint32_t version;
    if (!safe_read(fin, version)) return false;
    uint64_t tensor_count, metadata_kv_count;
    if (!safe_read(fin, tensor_count))      return false;
    if (!safe_read(fin, metadata_kv_count)) return false;

    for (uint64_t i = 0; i < metadata_kv_count; i++) {
        if (!read_metadata(fin)) {
            LOG_ERROR("read metadata failed");
            return false;
        }
    }
    tensors_.clear();
    try {
        for (uint64_t i = 0; i < tensor_count; i++) {
            tensors_.push_back(read_tensor_info(fin));
        }
    } catch (const std::runtime_error& e) {
        LOG_ERROR("%s", e.what());
        return false;
    }
    data_offset_ = static_cast<size_t>(fin.tellg());
    if ((data_offset_ % alignment_) != 0) {
        data_offset_ = ((data_offset_ + alignment_ - 1) / alignment_) * alignment_;
    }
    return true;
}

}  // namespace audiocore
