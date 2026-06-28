// convert_acestep.cpp — rewrite an ACE-Step GGUF's tensor names from
// HuggingFace PyTorch naming to llama.cpp naming.
//
// ACE-Step ships its Qwen3-based LM and Text-Encoder GGUFs with HF names
// (model.embed_tokens.weight, model.layers.0.*, …). libllama refuses anything
// that isn't in its own naming, so audiocore's qwen3::Runner can't load them
// directly.
//
// This tool rewrites tensor names by copying every tensor + KV from the input
// GGUF to a fresh output GGUF, applying the rename table to the tensor names
// only. KV metadata is preserved verbatim. The DiT and VAE GGUFs are NOT
// Qwen3 transformers and DO NOT need this converter; the family code binds
// them by their native HF names directly.
//
// Usage:
//   convert_acestep <input.gguf> <output.gguf>            # apply rename
//   convert_acestep <input.gguf> --dry-run                # audit only
//
// Mapping reference: docs/GGUF_FORMAT.md → "ACE-Step" section. Names below
// are verified against ServeurpersoCom/acestep.cpp's qwen3-lm.h loader.

#include <cstdio>
#include <cstring>
#include <regex>
#include <string>
#include <vector>

#include "ggml.h"
#include "gguf.h"

#include "audiocore/framework/io/gguf_reader.h"

namespace {

struct Rule {
    std::regex pattern;
    std::string replacement;
};

// HF name → llama.cpp name. Patterns are tried in order; first match wins.
// The {i} layer index is captured and substituted verbatim.
const std::vector<Rule>& rules() {
    static const std::vector<Rule> R = {
        {std::regex("^model\\.embed_tokens\\.weight$"),               "token_embd.weight"},
        {std::regex("^model\\.norm\\.weight$"),                       "output_norm.weight"},
        {std::regex("^model\\.layers\\.(\\d+)\\.input_layernorm\\.weight$"),
            "blk.$1.attn_norm.weight"},
        {std::regex("^model\\.layers\\.(\\d+)\\.post_attention_layernorm\\.weight$"),
            "blk.$1.ffn_norm.weight"},
        {std::regex("^model\\.layers\\.(\\d+)\\.self_attn\\.q_proj\\.weight$"),
            "blk.$1.attn_q.weight"},
        {std::regex("^model\\.layers\\.(\\d+)\\.self_attn\\.k_proj\\.weight$"),
            "blk.$1.attn_k.weight"},
        {std::regex("^model\\.layers\\.(\\d+)\\.self_attn\\.v_proj\\.weight$"),
            "blk.$1.attn_v.weight"},
        {std::regex("^model\\.layers\\.(\\d+)\\.self_attn\\.o_proj\\.weight$"),
            "blk.$1.attn_output.weight"},
        {std::regex("^model\\.layers\\.(\\d+)\\.mlp\\.gate_proj\\.weight$"),
            "blk.$1.ffn_gate.weight"},
        {std::regex("^model\\.layers\\.(\\d+)\\.mlp\\.up_proj\\.weight$"),
            "blk.$1.ffn_up.weight"},
        {std::regex("^model\\.layers\\.(\\d+)\\.mlp\\.down_proj\\.weight$"),
            "blk.$1.ffn_down.weight"},
        // Text-encoder variant: same layers but no `model.` prefix.
        {std::regex("^embed_tokens\\.weight$"), "token_embd.weight"},
        {std::regex("^norm\\.weight$"),         "output_norm.weight"},
    };
    return R;
}

std::string rename(const std::string& name, bool* matched) {
    for (const auto& r : rules()) {
        if (std::regex_match(name, r.pattern)) {
            *matched = true;
            return std::regex_replace(name, r.pattern, r.replacement);
        }
    }
    *matched = false;
    return name;
}

int usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s <input.gguf> [<output.gguf> | --dry-run]\n"
        "\n"
        "  Rewrites ACE-Step HF tensor names to llama.cpp names so the\n"
        "  Qwen3 LM / Text-Encoder GGUFs load through qwen3::Runner.\n"
        "  Pass --dry-run to audit without writing.\n",
        argv0);
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) return usage(argv[0]);

    const std::string in_path  = argv[1];
    const std::string out_arg  = argv[2];
    const bool dry_run = (out_arg == "--dry-run");
    if (in_path == "--dry-run" || (!dry_run && out_arg.empty())) {
        return usage(argv[0]);
    }

    audiocore::GgufReader r;
    std::string err;
    if (!r.load(in_path, &err)) {
        std::fprintf(stderr, "load('%s') failed: %s\n", in_path.c_str(),
                     err.c_str());
        return 1;
    }

    // Build the rename plan and report.
    struct PlanItem {
        std::string old_name, new_name;
        bool        will_rename;
    };
    std::vector<PlanItem> plan;
    plan.reserve(r.tensors().size());
    int n_renamed = 0;
    for (const auto& t : r.tensors()) {
        bool matched = false;
        std::string new_name = rename(t.name, &matched);
        if (matched) ++n_renamed;
        plan.push_back({t.name, new_name, matched});
    }

    if (n_renamed == 0) {
        std::printf("%s: nothing to convert (already in llama.cpp layout, "
                    "or unrecognized names).\n", in_path.c_str());
        int shown = 0;
        for (const auto& p : plan) {
            if (!p.will_rename) {
                std::printf("  skipped: %s\n", p.old_name.c_str());
                if (++shown >= 5) break;
            }
        }
        return 0;
    }

    std::printf("%s: %d of %zu tensors will be renamed.\n",
                in_path.c_str(), n_renamed, plan.size());
    int shown = 0;
    for (const auto& p : plan) {
        if (!p.will_rename) continue;
        std::printf("  %s -> %s\n", p.old_name.c_str(), p.new_name.c_str());
        if (++shown >= 10) { std::printf("  …\n"); break; }
    }

    if (dry_run) {
        std::printf("dry-run: no file written.\n");
        return 0;
    }

    // Allocate one big arena that backs every output tensor's bytes. We
    // materialize each input tensor into the arena at its own offset, then
    // build a fresh ggml_context of tensor structs that point into the
    // arena and hand the lot to write_gguf_file.
    size_t total_bytes = 0;
    for (const auto& t : r.tensors()) total_bytes += t.nbytes();
    std::vector<char> arena(total_bytes);

    struct ggml_init_params gip {
        /*.mem_size   =*/ ggml_tensor_overhead() * (plan.size() + 4),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context* ctx = ggml_init(gip);
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed\n");
        return 1;
    }

    size_t offset = 0;
    std::vector<audiocore::TensorWriteInfo> writes;
    writes.reserve(plan.size());
    for (size_t i = 0; i < plan.size(); ++i) {
        const auto& p = plan[i];
        const auto& t = r.tensors()[i];
        char* dst = arena.data() + offset;
        if (!r.materialize(t, dst, &err)) {
            std::fprintf(stderr, "materialize('%s') failed: %s\n",
                         t.name.c_str(), err.c_str());
            ggml_free(ctx);
            return 1;
        }

        // ggml ne[] convention: ne[0] is innermost. TensorStorage is already
        // in this order.
        ggml_tensor* gt = ggml_new_tensor(ctx, t.type, t.n_dims, t.ne);
        if (!gt) {
            std::fprintf(stderr, "ggml_new_tensor failed for %s\n",
                         p.new_name.c_str());
            ggml_free(ctx);
            return 1;
        }
        // gguf_add_tensor reads the name from the ggml_tensor struct, so we
        // must set it here (not just on the TensorWriteInfo side).
        ggml_set_name(gt, p.new_name.c_str());
        gt->data = dst;
        offset  += t.nbytes();
        writes.push_back({gt, p.new_name});
    }

    // Carry over the KV metadata verbatim. We need to inspect the source
    // gguf_context to enumerate keys/types; the GgufReader exposes ctx().
    const gguf_context* src = r.ctx();
    if (src) {
        // Use a fresh empty gguf_context as the target, copy KVs via gguf_set_kv.
        gguf_context* dst = gguf_init_empty();
        if (dst) {
            gguf_set_kv(dst, src);
            // Add renamed tensors in source order.
            for (const auto& w : writes) gguf_add_tensor(dst, w.tensor);
            if (!gguf_write_to_file(dst, out_arg.c_str(), /*only_meta=*/false)) {
                std::fprintf(stderr, "gguf_write_to_file('%s') failed\n",
                             out_arg.c_str());
                gguf_free(dst);
                ggml_free(ctx);
                return 1;
            }
            gguf_free(dst);
        }
    } else {
        // No source KV (Path 2 reader); fall back to write_gguf_file, which
        // omits KV — caller will need to add metadata manually.
        std::fprintf(stderr,
            "warning: source GGUF read via Path 2; KV metadata not copied. "
            "Use the Path 1 reader or add KV by hand.\n");
        if (!audiocore::write_gguf_file(out_arg, writes, &err)) {
            std::fprintf(stderr, "write_gguf_file failed: %s\n", err.c_str());
            ggml_free(ctx);
            return 1;
        }
    }

    ggml_free(ctx);
    std::printf("wrote %s (%zu tensors, %zu bytes payload).\n",
                out_arg.c_str(), plan.size(), total_bytes);
    return 0;
}
