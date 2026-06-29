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
        // QK-norm (both LM and TE variants).
        {std::regex("^model\\.layers\\.(\\d+)\\.self_attn\\.k_norm\\.weight$"),
            "blk.$1.attn_k_norm.weight"},
        {std::regex("^model\\.layers\\.(\\d+)\\.self_attn\\.q_norm\\.weight$"),
            "blk.$1.attn_q_norm.weight"},

        // Text-encoder variant: same layers but no `model.` prefix.
        {std::regex("^embed_tokens\\.weight$"), "token_embd.weight"},
        {std::regex("^norm\\.weight$"),         "output_norm.weight"},
        {std::regex("^layers\\.(\\d+)\\.input_layernorm\\.weight$"),
            "blk.$1.attn_norm.weight"},
        {std::regex("^layers\\.(\\d+)\\.post_attention_layernorm\\.weight$"),
            "blk.$1.ffn_norm.weight"},
        {std::regex("^layers\\.(\\d+)\\.self_attn\\.q_proj\\.weight$"),
            "blk.$1.attn_q.weight"},
        {std::regex("^layers\\.(\\d+)\\.self_attn\\.k_proj\\.weight$"),
            "blk.$1.attn_k.weight"},
        {std::regex("^layers\\.(\\d+)\\.self_attn\\.v_proj\\.weight$"),
            "blk.$1.attn_v.weight"},
        {std::regex("^layers\\.(\\d+)\\.self_attn\\.o_proj\\.weight$"),
            "blk.$1.attn_output.weight"},
        {std::regex("^layers\\.(\\d+)\\.mlp\\.gate_proj\\.weight$"),
            "blk.$1.ffn_gate.weight"},
        {std::regex("^layers\\.(\\d+)\\.mlp\\.up_proj\\.weight$"),
            "blk.$1.ffn_up.weight"},
        {std::regex("^layers\\.(\\d+)\\.mlp\\.down_proj\\.weight$"),
            "blk.$1.ffn_down.weight"},
        {std::regex("^layers\\.(\\d+)\\.self_attn\\.k_norm\\.weight$"),
            "blk.$1.attn_k_norm.weight"},
        {std::regex("^layers\\.(\\d+)\\.self_attn\\.q_norm\\.weight$"),
            "blk.$1.attn_q_norm.weight"},
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
        if (dry_run) {
            int shown = 0;
            for (const auto& p : plan) {
                if (!p.will_rename) {
                    std::printf("  skipped: %s\n", p.old_name.c_str());
                    if (++shown >= 5) break;
                }
            }
            return 0;
        }
        // Still write with KV fix even when no tensors need renaming —
        // the upstream GGUFs ship general.file_type as str but llama.cpp
        // expects u32. Fall through to the write path below.
    } else {
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
            // Fix general.file_type: upstream ACE-Step GGUFs store it as str
            // ("Q8_0") but llama.cpp expects u32. Override with the correct
            // integer value (LLAMA_FTYPE_MOSTLY_Q8_0 = 7 for Q8_0 weights;
            // the converter preserves the quantization so this is safe).
            gguf_set_val_u32(dst, "general.file_type", 7);
            // Override architecture: ACE-Step Qwen3-based GGUFs use
            // "acestep-lm" (LM) and "acestep-text-enc" (TE) but llama.cpp
            // only recognizes "qwen3". Both are architecturally Qwen3
            // models (same tensor structure, config keys, etc.).
            gguf_set_val_str(dst, "general.architecture", "qwen3");
            // Remap arch-prefixed KV keys: the source uses acestep-lm.*,
            // acestep-text-enc.* or acestep.*, but llama.cpp reads qwen3.*
            // after the architecture change above.
            {
                const char* old_prefixes[] = {"acestep-lm.", "acestep-text-enc.", "acestep."};
                int nkvs = gguf_get_n_kv(src);
                for (int i = 0; i < nkvs; i++) {
                    const char* key = gguf_get_key(src, i);
                    for (const char* prefix : old_prefixes) {
                        size_t plen = std::strlen(prefix);
                        if (std::strncmp(key, prefix, plen) == 0) {
                            std::string new_key = std::string("qwen3.") + (key + plen);
                            int vt = gguf_get_kv_type(src, i);
                            switch (vt) {
                                case GGUF_TYPE_UINT32:
                                    gguf_set_val_u32(dst, new_key.c_str(),
                                        gguf_get_val_u32(src, i)); break;
                                case GGUF_TYPE_INT32:
                                    gguf_set_val_i32(dst, new_key.c_str(),
                                        gguf_get_val_i32(src, i)); break;
                                case GGUF_TYPE_FLOAT32:
                                    gguf_set_val_f32(dst, new_key.c_str(),
                                        gguf_get_val_f32(src, i)); break;
                                case GGUF_TYPE_BOOL:
                                    gguf_set_val_bool(dst, new_key.c_str(),
                                        gguf_get_val_bool(src, i)); break;
                                case GGUF_TYPE_STRING:
                                    gguf_set_val_str(dst, new_key.c_str(),
                                        gguf_get_val_str(src, i)); break;
                                default: break;
                            }
                            break;
                        }
                    }
                }
            }
            // Set attention.value_length explicitly if missing. Without it,
            // llama.cpp defaults to n_embd / n_head (64 for TE head_dim=128),
            // causing a shape mismatch on attn_v.weight. Match key_length.
            {
                int kl_idx = gguf_find_key(dst, "qwen3.attention.key_length");
                int vl_idx = gguf_find_key(dst, "qwen3.attention.value_length");
                if (kl_idx != -1 && vl_idx == -1) {
                    uint32_t kl = gguf_get_val_u32(dst, kl_idx);
                    std::fprintf(stderr, "  setting qwen3.attention.value_length = %u"
                                 " (was unset; default would be wrong for TE)\n", kl);
                    gguf_set_val_u32(dst, "qwen3.attention.value_length", kl);
                }
            }

            // Pad tokenizer arrays to match qwen3.vocab_size (ACE-Step LM
            // adds audio-code tokens beyond the BPE tokenizer; llama.cpp
            // uses len(tokenizer.ggml.tokens) as n_vocab for tensor dim
            // checks, so the array must be padded to avoid shape errors).
            {
                int vs_idx = gguf_find_key(dst, "qwen3.vocab_size");
                int tk_idx = gguf_find_key(dst, "tokenizer.ggml.tokens");
                if (vs_idx != -1 && tk_idx != -1) {
                    uint32_t target = gguf_get_val_u32(dst, vs_idx);
                    int cur = gguf_get_arr_n(dst, tk_idx);
                    if (cur > 0 && (uint32_t)cur < target) {
                        std::fprintf(stderr, "  padding tokenizer arrays %d → %u\n",
                                     cur, target);
                        // Read existing tokens into stable strings
                        std::vector<std::string> token_strs;
                        token_strs.reserve(target);
                        for (int i = 0; i < cur; i++)
                            token_strs.push_back(
                                gguf_get_arr_str(dst, tk_idx, i));
                        // Pad with unique placeholders (llama.cpp requires
                        // id_to_token.size() == token_to_id.size(), so each
                        // entry must be a distinct non-empty string).
                        for (uint32_t i = (uint32_t)cur; i < target; i++)
                            token_strs.push_back("<|extra_"
                                + std::to_string(i - (uint32_t)cur) + "|>");
                        std::vector<const char*> strs(target);
                        for (uint32_t i = 0; i < target; i++)
                            strs[i] = token_strs[i].c_str();
                        gguf_set_arr_str(dst, "tokenizer.ggml.tokens",
                                         strs.data(), (int)target);
                        // Pad scores if present
                        int sc_idx = gguf_find_key(dst, "tokenizer.ggml.scores");
                        if (sc_idx != -1) {
                            const void* raw = gguf_get_arr_data(dst, sc_idx);
                            int n_raw = gguf_get_arr_n(dst, sc_idx);
                            std::vector<float> scores(target, 0.0f);
                            std::memcpy(scores.data(), raw,
                                         (size_t)std::min(n_raw, (int)target) * sizeof(float));
                            gguf_set_arr_data(dst, "tokenizer.ggml.scores",
                                GGUF_TYPE_FLOAT32, scores.data(), (int)target);
                        }
                        // Pad token types if present
                        int tt_idx = gguf_find_key(dst, "tokenizer.ggml.token_type");
                        if (tt_idx != -1) {
                            const void* raw = gguf_get_arr_data(dst, tt_idx);
                            int n_raw = gguf_get_arr_n(dst, tt_idx);
                            std::vector<int32_t> types(target, 0);
                            std::memcpy(types.data(), raw,
                                         (size_t)std::min(n_raw, (int)target) * sizeof(int32_t));
                            gguf_set_arr_data(dst, "tokenizer.ggml.token_type",
                                GGUF_TYPE_INT32, types.data(), (int)target);
                        }
                    }
                }
            }
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
