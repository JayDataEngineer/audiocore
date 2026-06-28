// synthetic_gguf.h — write tiny GGUF files for unit tests.
//
// Tests need known-shape tensors + KV metadata without pulling multi-GB real
// weights. This helper builds a ggml_context, fills tensors with a deterministic
// pattern, writes them to a temp file via write_gguf_file, and returns the
// path. The reader tests round-trip via GgufReader::load.

#ifndef AUDIOCORE_TESTS_SYNTHETIC_GGUF_H
#define AUDIOCORE_TESTS_SYNTHETIC_GGUF_H

#include <cstdint>
#include <string>
#include <vector>

namespace audiocore::test {

// One named tensor with deterministic content. dtype is GGML_TYPE_F32 unless
// overridden; the writer fills element [i] with (start_value + i * step).
struct SyntheticTensorSpec {
    std::string name;
    std::vector<int64_t> ne;        // ne[0] is innermost (row length)
    int         dtype    = 0;       // ggml_type (GGML_TYPE_F32 = 0)
    float       start    = 0.0f;
    float       step     = 1.0f;
};

// One KV metadata entry. Stored as the given gguf_type.
// For strings, set value_str. For numerics, value_i is cast to the right
// width on write.
struct SyntheticKV {
    std::string key;
    // GGUF_TYPE_I32 = 5, _F32 = 6, _BOOL = 7, _STRING = 8, _I64 = 11.
    int         type      = 5;
    int64_t     value_i   = 0;
    std::string value_str;
};

struct SyntheticGguf {
    std::vector<SyntheticTensorSpec> tensors;
    std::vector<SyntheticKV>         kvs;
};

// Write `spec` to a fresh file under /tmp and return the path. The caller
// owns cleanup (just unlink the path when done).
std::string write_synthetic_gguf(const std::string& tag,
                                 const SyntheticGguf& spec);

// ---- Convenience builders ------------------------------------------------
// Avoid C++17's nested-designated-init limitations in test bodies.

inline SyntheticTensorSpec tspec(std::string name,
                                 std::vector<int64_t> ne,
                                 float start = 0.0f, float step = 1.0f) {
    SyntheticTensorSpec s;
    s.name  = std::move(name);
    s.ne    = std::move(ne);
    s.dtype = 0;       // GGML_TYPE_F32
    s.start = start;
    s.step  = step;
    return s;
}

inline SyntheticKV kv_i32(std::string k, int64_t v) {
    return SyntheticKV{std::move(k), /*type=*/5 /*I32*/, v, ""};
}
inline SyntheticKV kv_str(std::string k, std::string v) {
    return SyntheticKV{std::move(k), /*type=*/8 /*STRING*/, 0, std::move(v)};
}
inline SyntheticKV kv_bool(std::string k, bool v) {
    return SyntheticKV{std::move(k), /*type=*/7 /*BOOL*/, v ? 1 : 0, ""};
}

}  // namespace audiocore::test

#endif  // AUDIOCORE_TESTS_SYNTHETIC_GGUF_H
