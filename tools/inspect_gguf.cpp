// inspect_gguf.cpp — dump tensor names, shapes, and sizes from a GGUF file.
//
// Usage: inspect_gguf <path.gguf>

#include <cstdio>
#include <cstring>
#include <string>
#include "audiocore/framework/io/gguf_reader.h"

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <gguf> [--kv]\n", argv[0]); return 1; }
    bool show_kv = (argc >= 3 && std::strcmp(argv[2], "--kv") == 0);
    audiocore::GgufReader reader;
    std::string err;
    if (!reader.load(argv[1], &err)) {
        std::fprintf(stderr, "load: %s\n", err.c_str());
        return 1;
    }
    if (show_kv) {
        const gguf_context* gc = reader.ctx();
        const int n_kv = gguf_get_n_kv(gc);
        for (int i = 0; i < n_kv; i++) {
            const char* key = gguf_get_key(gc, i);
            enum gguf_type t = gguf_get_kv_type(gc, i);
            const char* tn = gguf_type_name(t);
            std::fprintf(stderr, "kv[%d] %s = (%s)", i, key ? key : "?", tn ? tn : "?");
            switch (t) {
                case GGUF_TYPE_STRING: {
                    std::string s;
                    if (reader.get_kv_str(key, &s)) std::fprintf(stderr, " \"%s\"", s.c_str());
                    break;
                }
                case GGUF_TYPE_UINT32: {
                    uint32_t v = 0;
                    if (reader.get_kv_u32(key, &v)) std::fprintf(stderr, " %u", v);
                    break;
                }
                case GGUF_TYPE_INT32: {
                    int32_t v = 0;
                    if (reader.get_kv_i32(key, &v)) std::fprintf(stderr, " %d", v);
                    break;
                }
                case GGUF_TYPE_INT64: {
                    int64_t v = 0;
                    if (reader.get_kv_i64(key, &v)) std::fprintf(stderr, " %lld", (long long)v);
                    break;
                }
                case GGUF_TYPE_FLOAT32: {
                    float v = 0;
                    if (reader.get_kv_f32(key, &v)) std::fprintf(stderr, " %g", v);
                    break;
                }
                case GGUF_TYPE_BOOL: {
                    bool v = false;
                    if (reader.get_kv_bool(key, &v)) std::fprintf(stderr, " %s", v ? "true" : "false");
                    break;
                }
                default: break;
            }
            std::fprintf(stderr, "\n");
        }
        return 0;
    }
    for (const auto& t : reader.tensors()) {
        std::fprintf(stderr, "%s: type=%s ndim=%d ne=[", t.name.c_str(),
                     ggml_type_name(t.type), t.n_dims);
        for (int i = 0; i < t.n_dims; i++) {
            if (i) std::fprintf(stderr, ",");
            std::fprintf(stderr, "%lld", (long long)t.ne[i]);
        }
        std::fprintf(stderr, "] nbytes=%lld\n", (long long)t.nbytes());
    }
    return 0;
}
