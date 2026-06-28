// loader.cpp — Kokoro weight loading + family registration.
//
// Loads:
//   1. kokoro-v1.0.onnx      — ONNX model (via ONNX Runtime)
//   2. voices-v1.0.bin        — voice style embeddings (.npz archive)
//
// The model path in server.json can be either:
//   - A directory containing both files (auto-discovered)
//   - The exact .onnx file path (voices_path from extras)
//
// The voicepack is a .npz (zip archive) of stored (uncompressed) .npy files,
// one per voice. Each .npy is a float32 array of shape (510, 1, 256).

#include "audiocore/models/kokoro/family.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

// ONNX Runtime C API — we use the C API for portability.
#include <onnxruntime_c_api.h>

#include "audiocore/framework/runtime/registry.h"

namespace audiocore::kokoro {

// ===========================================================================
// Minimal .npz / .npy parser (stored entries only)
// ===========================================================================
// The voicepack is a .npz file: a zip archive where each member is a .npy
// file containing a float32 ndarray. Entries are stored (uncompressed).
//
// We parse the zip central directory to find member offsets, then for each
// member we skip the .npy header to find the raw float32 data.

#pragma pack(push, 1)
struct ZipEOCD {
    uint32_t signature;       // 0x06054b50
    uint16_t disk_num;
    uint16_t cd_disk;
    uint16_t cd_entries_disk;
    uint16_t cd_entries_total;
    uint32_t cd_size;
    uint32_t cd_offset;
    uint16_t comment_len;
};

struct ZipCDEntry {
    uint32_t signature;       // 0x02014b50
    uint16_t maker_ver;
    uint16_t needed_ver;
    uint16_t flags;
    uint16_t compression;     // 0 = stored, 8 = deflated
    uint16_t mod_time;
    uint16_t mod_date;
    uint32_t crc32;
    uint32_t compressed_size;
    uint32_t uncompressed_size;
    uint16_t name_len;
    uint16_t extra_len;
    uint16_t comment_len;
    uint16_t disk_start;
    uint16_t internal_attrs;
    uint32_t external_attrs;
    uint32_t local_offset;
};

struct ZipLocalHeader {
    uint32_t signature;       // 0x04034b50
    uint16_t ver;
    uint16_t flags;
    uint16_t compression;
    uint16_t mod_time;
    uint16_t mod_date;
    uint32_t crc32;
    uint32_t compressed_size;
    uint32_t uncompressed_size;
    uint16_t name_len;
    uint16_t extra_len;
};
#pragma pack(pop)

// Read bytes from stream at a given offset.
template <typename T>
static bool read_at(std::ifstream& f, uint64_t offset, T& out) {
    f.seekg(static_cast<std::streamoff>(offset));
    f.read(reinterpret_cast<char*>(&out), sizeof(T));
    return f.good();
}

// Parse a .npy header embedded in a data buffer and find the float32 data.
// Returns pointer to the float32 data and sets data_bytes to the number
// of float32 elements.
static const float* parse_npy_header_inplace(const uint8_t* buffer,
                                              size_t buffer_size,
                                              int64_t& n_rows,
                                              int64_t& n_cols,
                                              int64_t& n_elems) {
    // Magic: \x93NUMPY
    if (buffer_size < 80 || std::memcmp(buffer, "\x93NUMPY", 6) != 0) {
        return nullptr;
    }

    const uint8_t ver_major = buffer[6];
    const uint8_t ver_minor = buffer[7];

    // Read header length.
    uint64_t header_len = 0;
    size_t hdr_pos = 8;
    if (ver_major == 1) {
        uint16_t hlen = 0;
        std::memcpy(&hlen, buffer + hdr_pos, 2);
        header_len = hlen;
        hdr_pos += 2;
    } else if (ver_major == 2 || ver_major == 3) {
        uint32_t hlen = 0;
        std::memcpy(&hlen, buffer + hdr_pos, 4);
        header_len = hlen;
        hdr_pos += 4;
    } else {
        return nullptr;
    }

    // Extract header string.
    if (hdr_pos + header_len + 1 > buffer_size) return nullptr;
    std::string header(reinterpret_cast<const char*>(buffer + hdr_pos),
                       static_cast<size_t>(header_len));

    const uint64_t data_offset = hdr_pos + header_len;

    // Parse "shape": (...) from header.
    auto shape_pos = header.find("shape");
    if (shape_pos == std::string::npos) return nullptr;
    auto paren_open = header.find('(', shape_pos);
    auto paren_close = header.find(')', paren_open);
    if (paren_open == std::string::npos || paren_close == std::string::npos)
        return nullptr;

    std::string shape_str = header.substr(paren_open + 1,
                                           paren_close - paren_open - 1);
    // Remove spaces.
    shape_str.erase(std::remove_if(shape_str.begin(), shape_str.end(),
                                   [](char c) { return c == ' ' || c == '\t'; }),
                    shape_str.end());

    // Parse dimensions.
    std::vector<int64_t> dims;
    size_t start = 0;
    while (start < shape_str.size()) {
        auto comma = shape_str.find(',', start);
        std::string tok;
        if (comma == std::string::npos) {
            tok = shape_str.substr(start);
            start = shape_str.size();
        } else {
            tok = shape_str.substr(start, comma - start);
            start = comma + 1;
        }
        if (!tok.empty()) {
            dims.push_back(std::stoll(tok));
        }
    }

    if (dims.empty()) return nullptr;

    // Verify it's float32.
    bool is_f32 = false;
    auto descr_pos = header.find("descr");
    if (descr_pos != std::string::npos) {
        auto col = header.find(':', descr_pos);
        auto q1 = header.find('\'', col);
        auto q2 = header.find('\'', q1 + 1);
        if (q1 != std::string::npos && q2 != std::string::npos) {
            std::string descr = header.substr(q1 + 1, q2 - q1 - 1);
            is_f32 = (descr == "<f4" || descr == "|f4" || descr == "<f8" || descr == "|f8");
        }
    }
    if (!is_f32) return nullptr;

    // Calculate total elements from dims.
    n_elems = 1;
    for (auto d : dims) n_elems *= d;

    // Check data fits.
    const int64_t elem_size = 4; // float32
    // Also check if it might be float64
    if (dims.size() >= 1) {
        // Check descr more carefully
        auto descr_pos2 = header.find("descr");
        if (descr_pos2 != std::string::npos) {
            auto col2 = header.find(':', descr_pos2);
            auto q1_2 = header.find('\'', col2);
            auto q2_2 = header.find('\'', q1_2 + 1);
            if (q1_2 != std::string::npos && q2_2 != std::string::npos) {
                std::string descr2 = header.substr(q1_2 + 1, q2_2 - q1_2 - 1);
                if (descr2 == "<f8" || descr2 == "|f8") {
                    // float64 — not expected but handle gracefully
                    return nullptr;
                }
            }
        }
    }

    if (data_offset + static_cast<uint64_t>(n_elems) * 4 > buffer_size)
        return nullptr;

    // Determine shape.
    if (dims.size() == 1) {
        n_rows = dims[0];
        n_cols = 1;
    } else if (dims.size() == 2) {
        n_rows = dims[0];
        n_cols = dims[1];
    } else if (dims.size() == 3) {
        // shape (510, 1, 256) — squeeze middle dim
        n_rows = dims[0];
        n_cols = dims[2];
    } else {
        return nullptr;
    }

    return reinterpret_cast<const float*>(buffer + data_offset);
}

// Load the .npz voicepack into voices_ map.
bool KokoroSession::load_voices(const std::string& voices_path,
                                 std::string* error) {
    std::ifstream f(voices_path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        if (error) *error = "cannot open voices file: " + voices_path;
        return false;
    }

    const size_t file_size = static_cast<size_t>(f.tellg());
    if (file_size < sizeof(ZipEOCD)) {
        if (error) *error = "voices file too small: " + voices_path;
        return false;
    }

    // Read entire file into memory (it's ~28MB — fine to keep resident).
    std::vector<uint8_t> file_data(file_size);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(file_data.data()),
           static_cast<std::streamsize>(file_size));
    if (!f) {
        if (error) *error = "failed to read voices file";
        return false;
    }
    f.close();

    // --- Find End of Central Directory ---
    // Scan backward from end of file for the EOCD signature (0x06054b50).
    const uint32_t kEOCDSig = 0x06054b50;
    int64_t eocd_pos = static_cast<int64_t>(file_size) - sizeof(ZipEOCD);
    while (eocd_pos >= 0) {
        uint32_t sig = 0;
        std::memcpy(&sig, file_data.data() + eocd_pos, 4);
        if (sig == kEOCDSig) break;
        eocd_pos--;
    }
    if (eocd_pos < 0) {
        if (error) *error = "voices file: EOCD not found (not a zip?)";
        return false;
    }

    ZipEOCD eocd;
    std::memcpy(&eocd, file_data.data() + eocd_pos, sizeof(eocd));

    if (eocd.cd_entries_total == 0) {
        if (error) *error = "voices file: empty central directory";
        return false;
    }

    // --- Scan Central Directory ---
    const uint8_t* cd_base = file_data.data() + eocd.cd_offset;
    for (uint16_t i = 0; i < eocd.cd_entries_total; ++i) {
        if (cd_base + sizeof(ZipCDEntry) > file_data.data() + file_size) {
            if (error) *error = "voices file: truncated central directory";
            return false;
        }

        ZipCDEntry entry;
        std::memcpy(&entry, cd_base, sizeof(entry));
        cd_base += sizeof(entry);

        if (entry.signature != 0x02014b50) {
            if (error) *error = "voices file: bad central directory entry";
            return false;
        }

        // Read filename.
        std::string name(reinterpret_cast<const char*>(cd_base),
                         entry.name_len);
        cd_base += entry.name_len + entry.extra_len + entry.comment_len;

        // Only process .npy files.
        if (name.size() < 4 || name.substr(name.size() - 4) != ".npy") continue;

        if (entry.compression != 0) {  // stored only
            if (error) *error = "voice '" + name + "': compressed entries not supported";
            return false;
        }

        // --- Read local file header + data ---
        if (entry.local_offset + sizeof(ZipLocalHeader) > file_size) {
            if (error) *error = "voice '" + name + "': bad local offset";
            return false;
        }

        ZipLocalHeader local;
        std::memcpy(&local, file_data.data() + entry.local_offset, sizeof(local));
        if (local.signature != 0x04034b50) {
            if (error) *error = "voice '" + name + "': bad local header";
            return false;
        }

        const uint64_t data_start = static_cast<uint64_t>(entry.local_offset) +
                                     sizeof(ZipLocalHeader) +
                                     local.name_len + local.extra_len;

        if (data_start + entry.uncompressed_size > file_size) {
            if (error) *error = "voice '" + name + "': data exceeds file";
            return false;
        }

        // Parse the .npy file (header + float32 data).
        const auto* npy_start = file_data.data() + data_start;
        const size_t npy_size = static_cast<size_t>(entry.uncompressed_size);

        int64_t n_rows = 0, n_cols = 0, n_elems = 0;
        const float* float_data = parse_npy_header_inplace(
            npy_start, npy_size, n_rows, n_cols, n_elems);

        if (!float_data) {
            if (error) *error = "voice '" + name + "': failed to parse npy";
            return false;
        }

        // Voice name without .npy extension.
        std::string voice_name = name.substr(0, name.size() - 4);

        // Store the voice data. Shape is (510, 1, 256) → (510, 256).
        VoiceData vd;
        vd.n_styles = static_cast<int32_t>(n_rows);
        vd.style_dim = static_cast<int32_t>(n_cols);
        vd.data.assign(float_data, float_data + n_elems);

        voices_[voice_name] = std::move(vd);

        if (i == 0) {
            std::fprintf(stderr, "kokoro: loaded voice '%s' (%d x %d, %zu elems)\n",
                         voice_name.c_str(), vd.n_styles, vd.style_dim,
                         n_elems);
        }
    }

    if (voices_.empty()) {
        if (error) *error = "voices file: no .npy entries found";
        return false;
    }

    std::fprintf(stderr, "kokoro: loaded %zu voices from %s\n",
                 voices_.size(), voices_path.c_str());
    return true;
}

// ===========================================================================
// ONNX model loading
// ===========================================================================
// We use the ONNX Runtime C API directly. The C++ API is also available
// (see codec.cpp), but the C API gives us the same functionality with
// fewer build quirks across ONNX Runtime versions.
//
// This mirrors the pattern in codec.cpp where we keep Ort types pimpl'd.

bool KokoroSession::load_onnx(const std::string& onnx_path, bool use_gpu,
                               std::string* error) {
    const OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!ort) {
        if (error) *error = "ONNX Runtime API not available";
        return false;
    }

    // Create environment.
    OrtEnv* env = nullptr;
    if (ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "kokoro", &env) != nullptr) {
        if (error) *error = "Failed to create ONNX Runtime environment";
        return false;
    }
    ort_env_ = env;

    // Create session options.
    OrtSessionOptions* opts = nullptr;
    if (ort->CreateSessionOptions(&opts) != nullptr) {
        if (error) *error = "Failed to create session options";
        return false;
    }
    ort->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL);
    ort->SetIntraOpNumThreads(opts, 4);

    // GPU support: the CUDA execution provider can be configured via
    // the ONNX Runtime C++ API when available (see codec.cpp for the
    // C++ pattern). The C API version for CUDA provider options varies
    // across ONNX Runtime releases, so we skip it here and default to CPU.
    (void)use_gpu;

    // Load the model.
    OrtSession* session = nullptr;
    auto result = ort->CreateSession(env, onnx_path.c_str(), opts, &session);
    if (result != nullptr) {
        if (error) {
            const char* msg = ort->GetErrorMessage(result);
            *error = std::string("Failed to load ONNX model: ") + (msg ? msg : "unknown");
            ort->ReleaseStatus(result);
        }
        ort->ReleaseSessionOptions(opts);
        return false;
    }

    ort_session_ = session;
    ort->ReleaseSessionOptions(opts);

    // Create CPU memory info for input tensors.
    OrtMemoryInfo* mem_info = nullptr;
    ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info);
    ort_mem_info_ = mem_info;

    onnx_loaded_ = true;

    // Log model inputs.
    {
        size_t n_inputs = 0;
        ort->SessionGetInputCount(session, &n_inputs);
        std::fprintf(stderr, "kokoro: ONNX model loaded (%zu inputs)\n", n_inputs);
        for (size_t i = 0; i < n_inputs; ++i) {
            OrtAllocator* alloc;
            ort->GetAllocatorWithDefaultOptions(&alloc);
            char* name = nullptr;
            ort->SessionGetInputName(session, i, alloc, &name);
            std::fprintf(stderr, "  input[%zu]: %s\n", i, name ? name : "?");
            if (name) ort->AllocatorFree(alloc, name);
        }
    }

    return true;
}

// ===========================================================================
// Main load()
// ===========================================================================
bool KokoroSession::load(const std::string& model_path,
                          const LoadOptions& opts,
                          const BackendConfig& backend_cfg,
                          std::string* error) {
    // Determine model path and voices path.
    std::string onnx_path = model_path;
    std::string voices_path;

    // Check if model_path is a directory or a file.
    {
        std::ifstream f(model_path + "/kokoro-v1.0.onnx");
        if (f.good()) {
            // Directory with standard filenames.
            onnx_path = model_path + "/kokoro-v1.0.onnx";
            voices_path = model_path + "/voices-v1.0.bin";
        } else {
            // It's a file — look for voices in extras or alongside.
            auto it = opts.extras.find("voices_path");
            if (it != opts.extras.end()) {
                voices_path = it->second;
            } else {
                // Try alongside the model file.
                auto slash = model_path.rfind('/');
                std::string dir = (slash == std::string::npos)
                    ? "." : model_path.substr(0, slash);
                voices_path = dir + "/voices-v1.0.bin";
            }
        }
    }

    cfg_.model_path = onnx_path;
    cfg_.voices_path = voices_path;

    // Default voice from extras.
    auto it_voice = opts.extras.find("voice");
    if (it_voice != opts.extras.end()) {
        cfg_.default_voice = it_voice->second;
    }

    // Load voices.
    if (!load_voices(voices_path, error)) return false;

    // Load ONNX model. GPU if the backend config says CUDA.
    bool use_gpu = (backend_cfg.kind == BackendKind::ggml_cuda);
    if (!load_onnx(onnx_path, use_gpu, error)) return false;

    loaded_ = true;
    return true;
}

// ===========================================================================
// Factory + family registration
// ===========================================================================
namespace {
std::unique_ptr<Session> make_kokoro_session() {
    return std::unique_ptr<Session>(new KokoroSession());
}
}  // namespace

AUDIOCORE_REGISTER_FAMILY(kokoro, make_kokoro_session)
AUDIOCORE_EXTERN_C_GUARD(kokoro, make_kokoro_session)

}  // namespace audiocore::kokoro
