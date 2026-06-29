// inspect_gguf.cpp — dump tensor names, shapes, and sizes from a GGUF file.
//
// Usage: inspect_gguf <path.gguf>

#include <cstdio>
#include <string>
#include "audiocore/framework/io/gguf_reader.h"

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <gguf>\n", argv[0]); return 1; }
    audiocore::GgufReader reader;
    std::string err;
    if (!reader.load(argv[1], &err)) {
        std::fprintf(stderr, "load: %s\n", err.c_str());
        return 1;
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
