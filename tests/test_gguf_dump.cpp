// Quick GGUF dump tool to debug tensor_data_ptr issue.
#include "audiocore/framework/io/gguf_reader.h"
#include <sys/stat.h>

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "Usage: %s <gguf_path>\n", argv[0]); return 1; }
    audiocore::GgufReader r;
    std::string err;
    if (!r.load(argv[1], &err)) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    struct stat st;
    stat(argv[1], &st);
    std::fprintf(stderr, "File size: %lld\n", (long long)st.st_size);
    std::fprintf(stderr, "Loaded %zu tensors\n", r.tensors().size());

    // count OK vs NULL, find first NULL
    size_t n_ok = 0, n_null = 0;
    uint64_t first_null_offset = 0;
    std::string first_null_name;
    for (const auto& t : r.tensors()) {
        const void* p = r.tensor_data_ptr(t);
        if (p) {
            ++n_ok;
        } else {
            ++n_null;
            if (first_null_name.empty()) {
                first_null_offset = t.offset;
                first_null_name = t.name;
            }
        }
    }
    std::fprintf(stderr, "OK: %zu  NULL: %zu\n", n_ok, n_null);

    // find the actual tensor
    for (const auto& t : r.tensors()) {
        if (t.name == first_null_name) {
            std::fprintf(stderr, "First NULL: name=%s ne=[%lld,%lld,%lld] type=%d offset=%llu nbytes=%lld nbytes_to_read=%lld\n",
                         t.name.c_str(),
                         (long long)t.ne[0], (long long)t.ne[1], (long long)t.ne[2],
                         (int)t.type,
                         (unsigned long long)t.offset,
                         (long long)t.nbytes(),
                         (long long)t.nbytes_to_read());
            std::fprintf(stderr, "  offset+to_read = %llu vs file_size = %lld\n",
                         (unsigned long long)(t.offset + t.nbytes_to_read()),
                         (long long)st.st_size);
            break;
        }
    }

    // check the last tensor
    if (!r.tensors().empty()) {
        const auto& last = r.tensors().back();
        const void* p = r.tensor_data_ptr(last);
        std::fprintf(stderr, "Last tensor: name=%s offset=%llu nbytes=%lld nbytes_to_read=%lld data_ptr=%s\n",
                     last.name.c_str(),
                     (unsigned long long)last.offset,
                     (long long)last.nbytes(),
                     (long long)last.nbytes_to_read(),
                     p ? "OK" : "NULL");
        std::fprintf(stderr, "  last offset+to_read = %llu vs file_size = %lld\n",
                     (unsigned long long)(last.offset + last.nbytes_to_read()),
                     (long long)st.st_size);
    }
    return 0;
}
