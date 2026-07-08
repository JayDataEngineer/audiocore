// detokenizer_runner.cpp — FSQ codes → 25 Hz latent bridge for ACE-Step.
//
// See detokenizer_runner.h for the architecture overview. This file builds a
// single ggml graph per decode() call that references the model's detokenizer.*
// and tokenizer.quantizer.* tensors directly (no copy). The FSQ code → 6-D →
// 2048-D path runs on CPU; everything else runs in the graph.

#include "audiocore/models/ace_step/detokenizer_runner.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace audiocore::acestep {

// ═════════════════════════════════════════════════════════════════════════════
//  FSQ decode: single int code → 6-D vector
// ═════════════════════════════════════════════════════════════════════════════
//
// Matches vector_quantize_pytorch.FSQ._scale_and_shift_inverse with
// preserve_symmetry=True: each level index ci ∈ [0, L_i) maps to
//   value = 2·ci/(L_i − 1) − 1   ∈ [−1, 1]
// Levels are [8, 8, 8, 5, 5, 5] → 64 000 codes via mixed-radix.
static void fsq_decode_one(int32_t code, float out[6]) {
    static const int levels[6] = {8, 8, 8, 5, 5, 5};
    int tmp = code;
    for (int i = 0; i < 6; i++) {
        int ci = tmp % levels[i];
        tmp /= levels[i];
        out[i] = 2.0f * static_cast<float>(ci) / (levels[i] - 1) - 1.0f;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Tensor dequantization helper (BF16/F16/Q8_0 → F32)
// ═════════════════════════════════════════════════════════════════════════════
//
// Uses a small temporary ggml context to dequantize any tensor type to F32.
// For the small per-layer scale weights (RMSNorm γ, biases) we read directly
// from the GGUF; the detokenizer transformer weights themselves stay in their
// native Q8_0/BF16 format and are consumed natively by ggml_mul_mat.
static std::vector<float> dequant_to_f32(const ggml_tensor* t) {
    if (!t) return {};
    const size_t n = static_cast<size_t>(ggml_nelements(t));
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, n * sizeof(float));
        return out;
    }
    // Build a temp context just large enough for src view + dst + graph.
    const size_t src_bytes = ggml_row_size(t->type, t->ne[0]) *
                             static_cast<size_t>(ggml_nrows(t));
    const size_t dst_bytes = n * sizeof(float);
    const size_t overhead = (size_t)ggml_tensor_overhead() * 6 +
                            (size_t)ggml_graph_overhead() + 1024;
    const size_t mem_size = overhead + src_bytes + dst_bytes;
    std::vector<char> buf(mem_size);
    struct ggml_init_params p = { mem_size, buf.data(), /*no_alloc=*/false };
    ggml_context* ctx = ggml_init(p);
    if (!ctx) return out;
    // View the source tensor in our context (data pointer references the
    // external GGUF mmap; ggml_cpy will read from it).
    ggml_tensor* src = ggml_view_tensor(ctx, const_cast<ggml_tensor*>(t));
    src->data = const_cast<void*>(t->data);
    ggml_tensor* dst = ggml_new_tensor(ctx, GGML_TYPE_F32, ggml_n_dims(t), t->ne);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, src, dst));
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    std::memcpy(out.data(), dst->data, n * sizeof(float));
    ggml_free(ctx);
    return out;
}

// ═════════════════════════════════════════════════════════════════════════════
//  DetokenizerRunner
// ═════════════════════════════════════════════════════════════════════════════

DetokenizerRunner::DetokenizerRunner(ggml_context* ext_ctx)
    : ext_ctx_(ext_ctx) {}

DetokenizerRunner::~DetokenizerRunner() {
    if (detok_wctx_) ggml_free(detok_wctx_);
}

void DetokenizerRunner::build_weight_ctx() {
    if (detok_wctx_ready_) return;

    // First pass: calculate total memory needed for all detokenizer weights.
    size_t total_bytes = 0;
    int n_tensors = 0;
    for (ggml_tensor* t = ggml_get_first_tensor(ext_ctx_); t;
         t = ggml_get_next_tensor(ext_ctx_, t)) {
        const char* name = ggml_get_name(t);
        if (!name) continue;
        std::string nm(name);
        bool want = (nm.find("detokenizer.") == 0 ||
                     nm.find("tokenizer.quantizer.project_out") == 0);
        if (!want) continue;
        total_bytes += ggml_row_size(t->type, t->ne[0]) *
                       static_cast<size_t>(ggml_nrows(t));
        n_tensors++;
    }
    total_bytes += (size_t)ggml_tensor_overhead() * (n_tensors + 8) + 1024;
    fprintf(stderr, "[detok] building weight context: %d tensors, %zu bytes\n",
            n_tensors, total_bytes);

    // Allocate a persistent arena (intentionally leaked — detokenizer lives
    // for the entire session, and the arena must outlive any decode() call).
    char* arena = (char*)std::malloc(total_bytes);
    if (!arena) {
        fprintf(stderr, "[detok] FATAL: malloc(%zu) failed for weight arena\n",
                total_bytes);
        return;
    }
    struct ggml_init_params p = { total_bytes, arena, /*no_alloc=*/false };
    detok_wctx_ = ggml_init(p);
    if (!detok_wctx_) {
        std::free(arena);
        fprintf(stderr, "[detok] FATAL: ggml_init failed for weight context\n");
        return;
    }

    // Second pass: clone each tensor into the weight context (original type).
    int n_cloned = 0;
    for (ggml_tensor* t = ggml_get_first_tensor(ext_ctx_); t;
         t = ggml_get_next_tensor(ext_ctx_, t)) {
        const char* name = ggml_get_name(t);
        if (!name) continue;
        std::string nm(name);
        bool want = (nm.find("detokenizer.") == 0 ||
                     nm.find("tokenizer.quantizer.project_out") == 0);
        if (!want) continue;

        ggml_tensor* dst = ggml_new_tensor(detok_wctx_, t->type,
                                           ggml_n_dims(t), t->ne);
        size_t nbytes = ggml_row_size(t->type, t->ne[0]) *
                        static_cast<size_t>(ggml_nrows(t));
        std::memcpy(dst->data, t->data, nbytes);
        ggml_set_name(dst, name);
        n_cloned++;
    }

    detok_wctx_ready_ = true;
    fprintf(stderr, "[detok] weight context built: %d tensors, %zu bytes\n",
            n_cloned, total_bytes);
    fflush(stderr);
}

bool DetokenizerRunner::decode(const int32_t* codes, int32_t n_codes,
                                std::vector<float>* latents, std::string* error) {
    if (!ext_ctx_ || !codes || n_codes <= 0) {
        if (error) *error = "Detokenizer: invalid state";
        return false;
    }

    // ── Architecture constants (configuration_acestep_v15.py) ────────────
    constexpr int   P       = 5;        // pool_window_size (5 Hz → 25 Hz)
    constexpr int   H       = 2048;     // hidden_size
    constexpr int   nh      = 16;       // num_attention_heads
    constexpr int   nk      = 8;        // num_key_value_heads (GQA 2:1)
    constexpr int   hd      = 128;      // head_dim
    constexpr int   inter   = 6144;     // intermediate_size
    constexpr int   out_dim = 64;       // audio_acoustic_hidden_dim
    constexpr float eps     = 1e-6f;    // rms_norm_eps
    constexpr float theta   = 1000000.0f; // rope_theta
    const int N = n_codes;

    fprintf(stderr, "[detok] decode: N=%d codes, P=%d → %d latent frames\n",
            N, P, N * P);
    fflush(stderr);

    // Clone all detokenizer weights to a CPU ggml context BEFORE any CUDA
    // migration. After the first DiT forward, ext_ctx_ tensors migrate to
    // GPU and CPU-side reads crash. The weight context is built once (first
    // decode call, before any DiT forward) and reused.
    build_weight_ctx();
    if (!detok_wctx_ready_ || !detok_wctx_) {
        if (error) *error = "Detokenizer: failed to build CPU weight context";
        return false;
    }

    // ═══ Phase 1: FSQ code → 6-D → 204-D continuous (CPU) ═══
    // For each LM code: decode to 6-D via fsq_decode_one, then apply
    // tokenizer.quantizer.project_out (Linear 6 → 2048). This reconstructs
    // the `quantized` representation that the upstream detokenizer consumes.
    ggml_tensor* proj_w_t = ggml_get_tensor(detok_wctx_,
        "tokenizer.quantizer.project_out.weight");
    ggml_tensor* proj_b_t = ggml_get_tensor(detok_wctx_,
        "tokenizer.quantizer.project_out.bias");
    if (!proj_w_t) {
        if (error) *error = "Detokenizer: tokenizer.quantizer.project_out.weight not found";
        return false;
    }
    auto proj_w = dequant_to_f32(proj_w_t);
    auto proj_b = proj_b_t ? dequant_to_f32(proj_b_t) : std::vector<float>(H, 0.0f);

    std::vector<float> quantized(static_cast<size_t>(N) * H);
    for (int n = 0; n < N; n++) {
        float f6[6];
        fsq_decode_one(codes[n], f6);
        const size_t base = static_cast<size_t>(n) * H;
        for (int j = 0; j < H; j++) {
            float s = proj_b[static_cast<size_t>(j)];
            for (int k = 0; k < 6; k++) {
                s += f6[k] * proj_w[static_cast<size_t>(k) * H + j];
            }
            quantized[base + j] = s;
        }
    }

    // ═══ Phase 2: detokenizer transformer (single ggml graph) ═══
    //
    // Tensor shapes (ggml ne order, fastest-first):
    //   x (input)         : [H, N]
    //   after embed_tokens: [H, N]
    //   after expand+add  : [H, P, N]            (each token broadcast to P patches)
    //   per layer:
    //     h_norm          : [H, P, N]
    //     q, k, v         : [nh*hd|nk*hd, P, N]
    //     q/k after norm  : [hd, nh|nk, P, N]   (rms_norm over ne[0]=hd)
    //     q, k for RoPE   : [hd, nh|nk, P*N]    (3D as required by rope_ext)
    //     q, k, v for attn: [hd, P, nh|nk, N]   (4D for flash_attn_ext batched)
    //     attn out        : [hd, nh, P, N]      (permuted result)
    //     reshape for o   : [nh*hd, P, N]
    //     o_proj out      : [H, P, N]
    //   final:
    //     norm            : [H, P, N]
    //     proj_out        : [out_dim, P, N]     (then flatten to [out_dim, P*N])

    // Memory budget: graph nodes are small but activations can be large for
    // long generations. For N=600 (120s) we have N*P=3000 patches, peak
    // activation ~3000*2048*4 = 25 MB. 256 MB is plenty.
    const size_t buf_size = 512ULL * 1024 * 1024;
    std::vector<char> buf(buf_size);
    struct ggml_init_params gip = { buf_size, buf.data(), /*no_alloc=*/false };
    ggml_context* ctx = ggml_init(gip);
    if (!ctx) {
        if (error) *error = "Detokenizer: ggml_init failed";
        return false;
    }
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 8192, false);

    // All weight tensors come from detok_wctx_ (CPU-only clone), never from
    // ext_ctx_ (which migrates to GPU after the first DiT forward).
    // We use ggml_get_tensor on detok_wctx_ and reference the returned tensor
    // in our computation graph. The weight tensors live in detok_wctx_ which
    // is a separate CPU ggml context — it persists across decode() calls and
    // is never migrated to GPU.

    // ── Input: quantized [N, H] → ggml 2D ne=[H, N] ──────────────────────
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, N);
    std::memcpy(x->data, quantized.data(), quantized.size() * sizeof(float));

    // ── embed_tokens Linear (H → H) ──────────────────────────────────────
    {
        ggml_tensor* w = ggml_get_tensor(detok_wctx_, "detokenizer.embed_tokens.weight");
        ggml_tensor* b = ggml_get_tensor(detok_wctx_, "detokenizer.embed_tokens.bias");
        if (!w) {
            if (error) *error = "Detokenizer: detokenizer.embed_tokens.weight missing from cache";
            ggml_free(ctx); return false;
        }
        x = ggml_mul_mat(ctx, w, x);  // ne=[H, N]
        if (b) x = ggml_add(ctx, x, ggml_repeat(ctx,
            ggml_reshape_2d(ctx, b, H, 1), x));
    }

    // ── Expand to N×P patches + add special_tokens ───────────────────────
    // x: [H, N] → [H, 1, N] → broadcast to [H, P, N]
    x = ggml_reshape_3d(ctx, x, H, 1, N);
    {
        ggml_tensor* target = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H, P, N);
        x = ggml_repeat(ctx, x, target);  // [H, P, N]
    }
    // special_tokens in GGUF: ne=[H=2048, P=5]. ggml_repeat only supports
    // F32/F16/BF16/I32 source types; if the weight is Q8_0 we dequantize,
    // otherwise we reference it directly from detok_wctx_.
    {
        ggml_tensor* st_src = ggml_get_tensor(detok_wctx_,
            "detokenizer.special_tokens");
        if (!st_src) {
            if (error) *error = "Detokenizer: detokenizer.special_tokens missing";
            ggml_free(ctx); return false;
        }
        // If already F32, reference directly; otherwise dequantize into local.
        ggml_tensor* st;
        if (st_src->type == GGML_TYPE_F32 || st_src->type == GGML_TYPE_F16 ||
            st_src->type == GGML_TYPE_BF16) {
            st = st_src;
        } else {
            auto st_f32 = dequant_to_f32(st_src);
            st = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, P);
            std::memcpy(st->data, st_f32.data(), st_f32.size() * sizeof(float));
        }
        st = ggml_reshape_3d(ctx, st, H, P, 1);
        ggml_tensor* st_target = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H, P, N);
        st = ggml_repeat(ctx, st, st_target);  // [H, P, N]
        x = ggml_add(ctx, x, st);  // [H, P, N]
    }

    // ── 2 transformer layers ─────────────────────────────────────────────
    for (int layer = 0; layer < 2; layer++) {
        char prefix[64];
        std::snprintf(prefix, sizeof(prefix), "detokenizer.layers.%d", layer);

        auto W = [&](const char* suffix) -> ggml_tensor* {
            char nm[256];
            std::snprintf(nm, sizeof(nm), "%s.%s", prefix, suffix);
            return ggml_get_tensor(detok_wctx_, nm);
        };

        // ── Pre-norm (RMSNorm over ne[0]=H) ──────────────────────────────
        ggml_tensor* ln1_w = W("input_layernorm.weight");
        ggml_tensor* h_norm = ggml_rms_norm(ctx, x, eps);
        if (ln1_w) {
            h_norm = ggml_mul(ctx, h_norm, ggml_repeat(ctx,
                ggml_reshape_3d(ctx, ln1_w, H, 1, 1), h_norm));
        }

        // ── Attention projections ────────────────────────────────────────
        // Weights stored as ne=[in, out]. mul_mat contracts over ne[0]=in.
        // Result ne=[out, P, N].
        ggml_tensor* q = ggml_mul_mat(ctx, W("self_attn.q_proj.weight"), h_norm);  // [nh*hd, P, N]
        ggml_tensor* k = ggml_mul_mat(ctx, W("self_attn.k_proj.weight"), h_norm);  // [nk*hd, P, N]
        ggml_tensor* v = ggml_mul_mat(ctx, W("self_attn.v_proj.weight"), h_norm);  // [nk*hd, P, N]

        // ── QK-norm (RMSNorm per head, over head_dim) ────────────────────
        // Reshape [n*hd, P, N] → [hd, n, P, N] then rms_norm over ne[0]=hd.
        q = ggml_reshape_4d(ctx, q, hd, nh, P, N);  // [hd, nh, P, N]
        k = ggml_reshape_4d(ctx, k, hd, nk, P, N);  // [hd, nk, P, N]
        q = ggml_rms_norm(ctx, q, eps);
        k = ggml_rms_norm(ctx, k, eps);
        if (ggml_tensor* qn_w = W("self_attn.q_norm.weight")) {
            q = ggml_mul(ctx, q, ggml_repeat(ctx,
                ggml_reshape_4d(ctx, qn_w, hd, 1, 1, 1), q));
        }
        if (ggml_tensor* kn_w = W("self_attn.k_norm.weight")) {
            k = ggml_mul(ctx, k, ggml_repeat(ctx,
                ggml_reshape_4d(ctx, kn_w, hd, 1, 1, 1), k));
        }

        // ── RoPE (Neox-style mode=2) ─────────────────────────────────────
        // rope_ext requires 3D input [hd, n_heads, T]. Reshape [hd, n, P, N]
        // → [hd, n, P*N]. Positions are local within each P=5 group: [0..4]
        // repeated N times.
        q = ggml_reshape_3d(ctx, q, hd, nh, P * N);
        k = ggml_reshape_3d(ctx, k, hd, nk, P * N);
        {
            ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, P * N);
            int32_t* pd = static_cast<int32_t*>(pos->data);
            for (int n = 0; n < N; n++)
                for (int p = 0; p < P; p++)
                    pd[n * P + p] = p;
            q = ggml_rope_ext(ctx, q, pos, nullptr,
                              hd, 2, 0, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            k = ggml_rope_ext(ctx, k, pos, nullptr,
                              hd, 2, 0, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        }

        // ── flash_attn_ext (4D batched, batch=N over the ne3 dim) ────────
        // The op expects q,k,v as [hd, seq_len, n_heads, batch].
        // Reshape [hd, n_heads, P*N] → [hd, P, n_heads, N].
        q = ggml_reshape_4d(ctx, q, hd, P, nh, N);
        k = ggml_reshape_4d(ctx, k, hd, P, nk, N);
        v = ggml_reshape_4d(ctx, v, hd, P, nk, N);

        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(hd));
        // No mask → bidirectional attention within each [hd, P, *, N] group.
        // Result is permuted: [hd, n_heads, P, N] (matches our pre-RoPE shape).
        ggml_tensor* attn = ggml_flash_attn_ext(ctx, q, k, v, nullptr,
                                                 attn_scale, 0.0f, 0.0f);

        // ── o_proj: reshape [hd, nh, P, N] → [nh*hd, P, N], mul_mat ──────
        attn = ggml_reshape_3d(ctx, attn, nh * hd, P, N);  // [H, P, N]
        attn = ggml_mul_mat(ctx, W("self_attn.o_proj.weight"), attn);  // [H, P, N]

        // Residual add.
        x = ggml_add(ctx, x, attn);

        // ── Post-attn norm + SwiGLU MLP ──────────────────────────────────
        ggml_tensor* ln2_w = W("post_attention_layernorm.weight");
        ggml_tensor* h_norm2 = ggml_rms_norm(ctx, x, eps);
        if (ln2_w) {
            h_norm2 = ggml_mul(ctx, h_norm2, ggml_repeat(ctx,
                ggml_reshape_3d(ctx, ln2_w, H, 1, 1), h_norm2));
        }
        // gate = silu(W_gate @ h_norm2), up = W_up @ h_norm2
        // down = W_down @ (gate * up)
        ggml_tensor* gate = ggml_silu(ctx, ggml_mul_mat(ctx,
            W("mlp.gate_proj.weight"), h_norm2));  // [inter, P, N]
        ggml_tensor* up = ggml_mul_mat(ctx, W("mlp.up_proj.weight"), h_norm2);
        ggml_tensor* mlp = ggml_mul_mat(ctx, W("mlp.down_proj.weight"),
                                         ggml_mul(ctx, gate, up));  // [H, P, N]

        // Residual add.
        x = ggml_add(ctx, x, mlp);
    }

    // ── Final RMSNorm + proj_out (H → out_dim) ───────────────────────────
    {
        ggml_tensor* norm_w = ggml_get_tensor(detok_wctx_, "detokenizer.norm.weight");
        x = ggml_rms_norm(ctx, x, eps);
        if (norm_w) {
            x = ggml_mul(ctx, x, ggml_repeat(ctx,
                ggml_reshape_3d(ctx, norm_w, H, 1, 1), x));
        }
    }
    {
        ggml_tensor* po_w = ggml_get_tensor(detok_wctx_, "detokenizer.proj_out.weight");
        ggml_tensor* po_b = ggml_get_tensor(detok_wctx_, "detokenizer.proj_out.bias");
        if (!po_w) {
            if (error) *error = "Detokenizer: detokenizer.proj_out.weight missing from cache";
            ggml_free(ctx); return false;
        }
        x = ggml_mul_mat(ctx, po_w, x);  // [out_dim, P, N]
        if (po_b) {
            x = ggml_add(ctx, x, ggml_repeat(ctx,
                ggml_reshape_3d(ctx, po_b, out_dim, 1, 1), x));
        }
    }

    // ── Build & compute ──────────────────────────────────────────────────
    fprintf(stderr, "[detok] graph build + compute (N=%d, wctx=%p)...\n",
            N, (void*)detok_wctx_);
    fflush(stderr);
    ggml_build_forward_expand(gf, x);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    fprintf(stderr, "[detok] graph compute done\n");
    fflush(stderr);

    // ── Copy out: x is [out_dim=64, P=5, N] (ne order fastest-first).
    // We want time-major [T=N*P, out_dim] where T is the 25 Hz frame index.
    // ggml data layout: data[(n*P + p)*out_dim + c] for the [c, p, n] ne order.
    // That's already time-major in the order we want: (n*P + p) ∈ [0, N*P).
    latents->resize(static_cast<size_t>(N) * P * out_dim);
    std::memcpy(latents->data(), x->data, latents->size() * sizeof(float));

    // Quick stats for sanity.
    {
        double mn = 1e30, mx = -1e30, sum = 0;
        for (float v : *latents) {
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += v;
        }
        fprintf(stderr, "[detok] out: T=%d C=%d range=[%g, %g] mean=%g\n",
                N * P, out_dim, mn, mx, sum / latents->size());
    }

    ggml_free(ctx);
    return true;
}

}  // namespace audiocore::acestep
