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
    auto names = reader.tensor_names();
    for (const auto& n : names) {
        int ndim = 0;
        auto dims = reader.tensor_shape(n, &ndim);
        size_t nbytes = 0;
        auto dtype = reader.tensor_info(n, &nbytes);
        // Element size
        size_t esize = 4; // assume f32 for codec embd
        std::fprintf(stderr, "%s: ndim=%d dims=[", n.c_str(), ndim);
        for (int i = 0; i < ndim; i++) {
            if (i) std::fprintf(stderr, ",");
            std::fprintf(stderr, "%zu", dims[i]);
        }
        std::fprintf(stderr, "] nbytes=%zu\n", nbytes);
    }
    return 0;
}
