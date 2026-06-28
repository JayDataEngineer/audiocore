// synthetic_gguf.cpp — see synthetic_gguf.h.

#include "synthetic_gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>

#include "ggml.h"
#include "gguf.h"

namespace audiocore::test {

namespace {

// Build a fresh temp path under /tmp. Tag dedupes parallel runs.
std::string temp_path(const std::string& tag) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "/tmp/audiocore_test_%s_%ld.gguf",
                  tag.c_str(), static_cast<long>(::getpid()));
    return std::string(buf);
}

}  // namespace

std::string write_synthetic_gguf(const std::string& tag,
                                 const SyntheticGguf& spec) {
    // One ggml_context holds the tensor structs; the data buffers live in
    // a single std::vector<float> arena that backs every F32 tensor.
    size_t total_elems = 0;
    for (const auto& t : spec.tensors) {
        int64_t n = 1;
        for (int64_t d : t.ne) n *= d;
        total_elems += static_cast<size_t>(n);
    }
    std::vector<float> arena(total_elems);

    struct ggml_init_params gip {
        /*.mem_size   =*/ ggml_tensor_overhead() * (spec.tensors.size() + 4),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context* ctx = ggml_init(gip);
    if (!ctx) {
        std::fprintf(stderr, "write_synthetic_gguf: ggml_init failed\n");
        std::exit(2);
    }

    size_t offset = 0;
    std::vector<std::pair<ggml_tensor*, const SyntheticTensorSpec*>> built;
    built.reserve(spec.tensors.size());

    for (const auto& t : spec.tensors) {
        if (t.ne.empty() || t.ne.size() > 4) {
            std::fprintf(stderr, "write_synthetic_gguf: bad ne for %s\n",
                         t.name.c_str());
            std::exit(2);
        }
        int64_t ne4[4] = {1, 1, 1, 1};
        int64_t n = 1;
        for (size_t i = 0; i < t.ne.size(); ++i) {
            ne4[i] = t.ne[i];
            n *= t.ne[i];
        }
        ggml_tensor* gt = ggml_new_tensor(ctx,
                                          static_cast<ggml_type>(t.dtype),
                                          static_cast<int>(t.ne.size()),
                                          ne4);
        if (!gt) {
            std::fprintf(stderr, "write_synthetic_gguf: ggml_new_tensor failed\n");
            std::exit(2);
        }
        ggml_set_name(gt, t.name.c_str());
        // Fill the arena slice for this tensor with a deterministic pattern.
        float* dst = arena.data() + offset;
        for (int64_t i = 0; i < n; ++i) {
            dst[i] = t.start + static_cast<float>(i) * t.step;
        }
        // We need ->data to point at the slice before gguf_add_tensor, but
        // ggml has no_alloc=true so it's null. Borrow the arena pointer.
        gt->data = dst;
        offset += static_cast<size_t>(n);
        built.emplace_back(gt, &t);
    }

    // Build the GGUF context. KV first (gguf_set_*), then tensors.
    gguf_context* gctx = gguf_init_empty();
    if (!gctx) {
        std::fprintf(stderr, "write_synthetic_gguf: gguf_init_empty failed\n");
        std::exit(2);
    }
    for (const auto& kv : spec.kvs) {
        switch (static_cast<gguf_type>(kv.type)) {
            case GGUF_TYPE_INT32:
                gguf_set_val_i32(gctx, kv.key.c_str(),
                                 static_cast<int32_t>(kv.value_i));
                break;
            case GGUF_TYPE_INT64:
                gguf_set_val_i64(gctx, kv.key.c_str(), kv.value_i);
                break;
            case GGUF_TYPE_UINT32:
                gguf_set_val_u32(gctx, kv.key.c_str(),
                                 static_cast<uint32_t>(kv.value_i));
                break;
            case GGUF_TYPE_FLOAT32:
                gguf_set_val_f32(gctx, kv.key.c_str(),
                                 static_cast<float>(kv.value_i));
                break;
            case GGUF_TYPE_BOOL:
                gguf_set_val_bool(gctx, kv.key.c_str(), kv.value_i != 0);
                break;
            case GGUF_TYPE_STRING:
                gguf_set_val_str(gctx, kv.key.c_str(), kv.value_str.c_str());
                break;
            default:
                std::fprintf(stderr,
                             "write_synthetic_gguf: unsupported KV type %d\n",
                             kv.type);
                std::exit(2);
        }
    }
    for (auto [gt, _] : built) gguf_add_tensor(gctx, gt);

    const std::string path = temp_path(tag);
    if (!gguf_write_to_file(gctx, path.c_str(), /*only_meta=*/false)) {
        std::fprintf(stderr, "write_synthetic_gguf: gguf_write_to_file failed\n");
        std::exit(2);
    }

    gguf_free(gctx);
    ggml_free(ctx);
    return path;
}

}  // namespace audiocore::test
