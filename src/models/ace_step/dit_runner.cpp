// dit_runner.cpp — DiT graph builder + forward pass.
//
// Builds a ggml_cgraph for one DiT velocity-prediction forward pass, then
// computes it via ggml_graph_compute_with_ctx (CPU).  One temporary
// ggml_context per call (~512 MB, no_alloc=false).  All weight tensors
// live in ext_ctx_ (no_alloc=true, backed by GGUF mmap).
//
// The session handles proj_in / proj_out externally — this runner only
// covers the core transformer (hidden_size → hidden_size).
//
// Architecture (24 layers, each with):
//   1. RMS norm × learned weight + AdaLN modulation
//   2. QK-norm + GQA self-attention + RoPE
//   3. RMS norm × learned weight + AdaLN + QK-norm + cross-attention to TE cond
//   4. RMS norm × learned weight + AdaLN + SwiGLU MLP
//
// Tensor names match the GGUF from Serveurperso/ACE-Step-1.5-GGUF:
//   decoder.layers.{i}.self_attn.{q,k,v,o}_proj.weight  (separate Q/K/V)
//   decoder.layers.{i}.self_attn_{q,k}_norm.weight       (QK-norm)
//   decoder.layers.{i}.self_attn_norm.weight             (pre-SA RMS weight)
//   decoder.layers.{i}.cross_attn.{q,k,v,o}_proj.weight
//   decoder.layers.{i}.cross_attn_{q,k}_norm.weight
//   decoder.layers.{i}.cross_attn_norm.weight
//   decoder.layers.{i}.mlp.{gate,up,down}_proj.weight
//   decoder.layers.{i}.mlp_norm.weight
//   decoder.layers.{i}.scale_shift_table                 (bf16, [H, 6])
//   decoder.time_embed.linear_{1,2}.weight/bias
//   decoder.time_embed.time_proj.weight/bias
//   decoder.norm_out.weight                              (final RMS weight)
//   decoder.scale_shift_table                            (global, [H, 2])
//   decoder.condition_embedder.weight/bias
//   null_condition_emb
//
// CFG runs two separate graphs (cond / uncond) and blends outputs.

#include "audiocore/models/ace_step/dit_runner.h"

#include "ggml.h"
#include "ggml-cpu.h"   // ggml_new_f32, ggml_graph_compute_with_ctx

#include <cmath>
#include <cstdio>
#include <cstring>
#include <new>
#include <vector>

namespace audiocore::acestep {

// ── BF16 → F32 helpers ─────────────────────────────────────────────────────
static float bf16_to_f32(uint16_t bits) {
    // BF16 is F32 truncated to top 16 bits — just zero-extend.
    uint32_t f32_bits = static_cast<uint32_t>(bits) << 16;
    float f;
    memcpy(&f, &f32_bits, sizeof(f));
    return f;
}

static void bf16_buf_to_f32(const void* src, float* dst, int n) {
    const uint16_t* s = static_cast<const uint16_t*>(src);
    for (int i = 0; i < n; i++) dst[i] = bf16_to_f32(s[i]);
}

// ── AdaLN modulation ─────────────────────────────────────────────────────────
// x' = x * (1 + scale) + shift   where scale/shift are rows of time_mod.
// time_mod is [6, H] → ne0=H, ne1=6 in ggml layout.
static ggml_tensor* adaln(ggml_context* ctx, ggml_tensor* x,
                           ggml_tensor* time_mod,
                           int row_s, int row_h, int H) {
    auto vr = [&](int r) {
        return ggml_view_1d(ctx, time_mod, H,
                            static_cast<size_t>(r) * H * sizeof(float));
    };
    ggml_tensor* s = ggml_reshape_2d(ctx, vr(row_s), H, 1);
    ggml_tensor* h = ggml_reshape_2d(ctx, vr(row_h), H, 1);
    // x + x*s + h  ≡  x*(1+s) + h
    return ggml_add(ctx,
               ggml_add(ctx, x, ggml_mul(ctx, x, ggml_repeat(ctx, s, x))),
               ggml_repeat(ctx, h, x));
}

// ── Apply QK-norm (RMS norm per-head + learned scale) ───────────────────────
// Input:  [T, nh*hd]  in ggml ne: ne[0]=nh*hd, ne[1]=T
// Output: [T, nh*hd]  same shape
static ggml_tensor* apply_qk_norm(ggml_context* ctx, ggml_tensor* t,
                                    int hd, int nh,
                                    ggml_tensor* norm_w) {
    if (!norm_w) return t;
    // Compute T from total elements: input is [nh*hd, T]
    int64_t T = ggml_nelements(t) / (static_cast<int64_t>(hd) * nh);
    // Reshape to [hd, nh, T] for per-head norm
    ggml_tensor* t3d = ggml_reshape_3d(ctx, t, hd, nh, T);
    t3d = ggml_rms_norm(ctx, t3d, 1e-6f);
    // norm_w is [hd] → expand to [hd, nh, T]
    ggml_tensor* nw = ggml_reshape_3d(ctx, norm_w, hd, 1, 1);
    nw = ggml_repeat(ctx, nw, t3d);
    t3d = ggml_mul(ctx, t3d, nw);
    return ggml_reshape_2d(ctx, t3d, nh * hd, T);
}

// ── Self-attention (GQA + QK-norm + RoPE) ───────────────────────────────────
// Returns just the attention (or O-projected) output — NO residual added.
static ggml_tensor* self_attn(ggml_context* ctx, ggml_tensor* x,
                                int T, int H, int nh, int nk, int hd,
                                ggml_tensor* q_w, ggml_tensor* k_w,
                                ggml_tensor* v_w, ggml_tensor* o_w,
                                ggml_tensor* q_norm_w, ggml_tensor* k_norm_w,
                                float rope_theta) {
    if (!q_w || !k_w || !v_w) return nullptr;

    auto q = ggml_mul_mat(ctx, q_w, x);
    auto k = ggml_mul_mat(ctx, k_w, x);
    auto v = ggml_mul_mat(ctx, v_w, x);

    // QK-norm (per-head RMS norm before RoPE)
    q = apply_qk_norm(ctx, q, hd, nh, q_norm_w);
    k = apply_qk_norm(ctx, k, hd, nk, k_norm_w);

    // Reshape to 3D [hd, n_heads, T] before RoPE (ggml_rope_ext requires 3D)
    auto to_3d = [&](ggml_tensor* t, int n_h) {
        return ggml_reshape_3d(ctx, t, hd, n_h, T);
    };
    q = to_3d(q, nh);
    k = to_3d(k, nk);
    v = to_3d(v, nk);

    // Build position IDs [0, 1, 2, ..., T-1] for RoPE
    ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    {
        int32_t* pd = static_cast<int32_t*>(pos->data);
        for (int i = 0; i < T; i++) pd[i] = i;
    }

    // RoPE (Neox-style: mode=2) on 3D tensors
    auto rope = [&](ggml_tensor* t) {
        return ggml_rope_ext(ctx, t, pos, nullptr,
                              hd, 2, 0,
                              rope_theta, 1.0f, 0.0f, 1.0f,
                              0.0f, 0.0f);
    };
    q = rope(q);
    k = rope(k);

    // Permute for flash_attn_ext → [hd, T, n_heads]
    auto rsh = [&](ggml_tensor* t, int n_h) {
        return ggml_cont(ctx, ggml_permute(ctx, t, 0, 2, 1, 3));
    };
    q = rsh(q, nh);
    k = rsh(k, nk);
    v = rsh(v, nk);

    float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    auto a = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);

    // Back to [T, nh*hd]
    a = ggml_cont(ctx, ggml_permute(ctx, a, 0, 2, 1, 3));
    a = ggml_reshape_2d(ctx, a, nh * hd, T);
    if (o_w) a = ggml_mul_mat(ctx, o_w, a);
    return a;
}

// ── Cross-attention (QK-norm, no RoPE) ──────────────────────────────────────
// Returns just the attention output — NO residual added.
static ggml_tensor* cross_attn(ggml_context* ctx, ggml_tensor* x,
                                 ggml_tensor* cond, int T, int T_cond,
                                 int H, int nh, int nk, int hd,
                                 ggml_tensor* ca_q, ggml_tensor* ca_k,
                                 ggml_tensor* ca_v, ggml_tensor* ca_o,
                                 ggml_tensor* q_norm_w, ggml_tensor* k_norm_w) {
    if (!ca_q || !ca_k || !ca_v || !ca_o || !cond) return nullptr;

    auto rsh = [&](ggml_tensor* t, int n_h, int len) {
        return ggml_cont(ctx, ggml_permute(ctx,
            ggml_reshape_3d(ctx, t, hd, n_h, len), 0, 2, 1, 3));
    };

    auto q_lin = ggml_mul_mat(ctx, ca_q, x);
    auto k_lin = ggml_mul_mat(ctx, ca_k, cond);
    auto v_lin = ggml_mul_mat(ctx, ca_v, cond);

    // QK-norm on cross-attention Q and K
    q_lin = apply_qk_norm(ctx, q_lin, hd, nh, q_norm_w);
    k_lin = apply_qk_norm(ctx, k_lin, hd, nk, k_norm_w);

    auto q = rsh(q_lin, nh, T);
    auto k = rsh(k_lin, nk, T_cond);
    auto v = rsh(v_lin, nk, T_cond);

    float s = 1.0f / std::sqrt(static_cast<float>(hd));
    auto a = ggml_flash_attn_ext(ctx, q, k, v, nullptr, s, 0.0f, 0.0f);
    a = ggml_cont(ctx, ggml_permute(ctx, a, 0, 2, 1, 3));
    a = ggml_reshape_2d(ctx, a, nh * hd, T);
    a = ggml_mul_mat(ctx, ca_o, a);
    return a;
}

// ── SwiGLU MLP ──────────────────────────────────────────────────────────────
// Returns just the MLP output — NO residual added.
static ggml_tensor* swiglu_mlp(ggml_context* ctx, ggml_tensor* x,
                                ggml_tensor* gw, ggml_tensor* uw,
                                ggml_tensor* dw) {
    if (!gw || !uw || !dw) return nullptr;
    auto g = ggml_silu(ctx, ggml_mul_mat(ctx, gw, x));
    auto u = ggml_mul_mat(ctx, uw, x);
    return ggml_mul_mat(ctx, dw, ggml_mul(ctx, g, u));
}

// ══════════════════════════════════════════════════════════════════════════════
// DiTRunner
// ══════════════════════════════════════════════════════════════════════════════

DiTRunner::DiTRunner(ggml_context* ext_ctx, const DitConfig& cfg)
    : ext_ctx_(ext_ctx), cfg_(cfg) {
    // Pre-convert per-layer scale_shift_table (bf16 → f32) and store for
    // use during graph construction. This avoids bf16 handling in the
    // hot graph-build path.
    if (cfg_.n_layers > 0) {
        ss_table_f32_.resize(cfg_.n_layers);
        for (int i = 0; i < cfg_.n_layers; i++) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                          "decoder.layers.%d.scale_shift_table", i);
            ggml_tensor* sst = ggml_get_tensor(ext_ctx_, buf);
            if (sst && sst->type == GGML_TYPE_BF16) {
                int n = static_cast<int>(sst->ne[0] * sst->ne[1]);
                ss_table_f32_[i].resize(n);
                bf16_buf_to_f32(sst->data, ss_table_f32_[i].data(), n);
            }
        }
    }
    // Pre-convert global scale_shift_table (applied after final norm)
    {
        ggml_tensor* gsst = ggml_get_tensor(ext_ctx_, "decoder.scale_shift_table");
        if (gsst && gsst->type == GGML_TYPE_BF16) {
            int n = static_cast<int>(gsst->ne[0] * gsst->ne[1]);
            global_ss_f32_.resize(n);
            bf16_buf_to_f32(gsst->data, global_ss_f32_.data(), n);
        }
    }
}

DiTRunner::~DiTRunner() = default;

ggml_tensor* DiTRunner::weight(const char* name) const {
    return ggml_get_tensor(ext_ctx_, name);
}

// ── Timestep sinusoidal embedding ────────────────────────────────────────────
// Matches ggml_compute_forward_timestep_embedding_f32 (ops.cpp:8157):
//   freq = exp(-log(max_period) * j / half)
//   embed[j]      = cos(t * freq)   (first half = COS)
//   embed[j+half] = sin(t * freq)   (second half = SIN)
static void timestep_embed(float t, float* out, int dim) {
    const int half = dim / 2;
    const float log_period = std::log(10000.0f);
    for (int j = 0; j < half; j++) {
        float freq = std::exp(-log_period * static_cast<float>(j) /
                              static_cast<float>(half));
        float arg = t * freq;
        out[j]        = std::cos(arg);
        out[half + j] = std::sin(arg);
    }
}

// ── Single forward: build graph + compute ────────────────────────────────────
static bool run_one_forward(
    const DitConfig& cfg, ggml_context* ext_ctx,
    const float* x_t, int32_t T, int32_t H,
    const float* temb, int temb_dim,
    const float* cond_data, int ct_len, int cond_hidden,
    float* result,
    const std::vector<std::vector<float>>& ss_table_f32,
    const std::vector<float>& global_ss_f32,
    std::string* error)
{
    // Input NaN diagnostics
    {
        int nx = 0, nt = 0, nc = 0;
        for (int32_t i = 0; i < T * H; i++) if (std::isnan(x_t[i])) nx++;
        for (int i = 0; i < temb_dim; i++) if (std::isnan(temb[i])) nt++;
        if (cond_data && ct_len > 0) {
            for (int i = 0; i < ct_len * cond_hidden; i++) if (std::isnan(cond_data[i])) nc++;
        }
        fprintf(stderr, "[dit] input NaN: x=%d/%d temb=%d/%d cond=%d/%d\n",
                nx, T*H, nt, temb_dim, nc, cond_data ? ct_len*cond_hidden : 0);
    }
    const int32_t nh      = cfg.n_heads     > 0 ? cfg.n_heads     : 24;
    const int32_t nk      = cfg.n_kv_heads  > 0 ? cfg.n_kv_heads  : 8;
    const int32_t hd      = cfg.head_dim    > 0 ? cfg.head_dim    : 128;
    const int32_t n_layer = cfg.n_layers    > 0 ? cfg.n_layers    : 24;
    const float   eps     = cfg.rms_norm_eps > 0 ? cfg.rms_norm_eps : 1e-6f;
    const float   theta   = cfg.rope_theta  > 0 ? cfg.rope_theta  : 1000000.0f;
    const int     nthr    = 4;

    // Context memory: scales with T_latent. 6 GB handles T≤375 (15 s @ 25 Hz).
    // For longer sequences we scale linearly (graph is mostly matmuls with
    // T-dependent activations). Capped at 24 GB to avoid system OOM.
    size_t mem;
    if (T <= 375)       mem = 6144ULL * 1024 * 1024;       // 6 GB  (≤15 s)
    else if (T <= 750)  mem = 12288ULL * 1024 * 1024;      // 12 GB (≤30 s)
    else if (T <= 1125) mem = 18432ULL * 1024 * 1024;      // 18 GB (≤45 s)
    else                mem = 24576ULL * 1024 * 1024;      // 24 GB (>45 s)
    char* ctx_buf = new (std::nothrow) char[mem];
    if (!ctx_buf) { if (error) *error = "DiT OOM"; return false; }

    ggml_init_params p = { mem, ctx_buf, /*no_alloc=*/false };
    ggml_context* ctx = ggml_init(p);
    ggml_cgraph* gf  = ggml_new_graph_custom(ctx, 4096, false);
    if (!ctx || !gf) {
        delete[] ctx_buf; if (error) *error = "DiT ggml_init"; return false;
    }

    // ── Input x [T, H] ───────────────────────────────────────────────────
    ggml_tensor* cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
    memcpy(cur->data, x_t, static_cast<size_t>(T) * H * sizeof(float));

    // ── Time embedding → time_mod [H, 6] ─────────────────────────────────
    ggml_tensor* tp = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, temb_dim);
    memcpy(tp->data, temb, static_cast<size_t>(temb_dim) * sizeof(float));
    tp = ggml_reshape_2d(ctx, tp, temb_dim, 1);   // [temb_dim, 1]

    // MLP 1: linear_1 → SiLU → linear_2
    ggml_tensor* t1w = ggml_get_tensor(ext_ctx, "decoder.time_embed.linear_1.weight");
    ggml_tensor* t1b = ggml_get_tensor(ext_ctx, "decoder.time_embed.linear_1.bias");
    if (t1w) {
        tp = ggml_mul_mat(ctx, t1w, tp);
        if (t1b) tp = ggml_add(ctx, tp, ggml_repeat(ctx, t1b, tp));
    }
    tp = ggml_silu(ctx, tp);
    ggml_tensor* t2w = ggml_get_tensor(ext_ctx, "decoder.time_embed.linear_2.weight");
    ggml_tensor* t2b = ggml_get_tensor(ext_ctx, "decoder.time_embed.linear_2.bias");
    if (t2w) {
        tp = ggml_mul_mat(ctx, t2w, tp);
        if (t2b) tp = ggml_add(ctx, tp, ggml_repeat(ctx, t2b, tp));
    }

    // Time projection → [H, 6] modulation
    // HOT-Step dit-graph.h:144 applies SiLU before time_proj: h2 = silu(temb)
    ggml_tensor* time_mod = nullptr;
    ggml_tensor* tpw = ggml_get_tensor(ext_ctx, "decoder.time_embed.time_proj.weight");
    ggml_tensor* tpb = ggml_get_tensor(ext_ctx, "decoder.time_embed.time_proj.bias");
    if (tpw) {
        tp = ggml_silu(ctx, tp);  // SiLU before time_proj (HOT-Step dit-graph.h:144)
        auto pj = ggml_mul_mat(ctx, tpw, tp);
        if (tpb) pj = ggml_add(ctx, pj, ggml_repeat(ctx, tpb, pj));
        time_mod = ggml_reshape_2d(ctx, pj, H, 6);   // ne0=H, ne1=6
    } else {
        // No time_proj → repeat time-embedding as time_mod
        time_mod = ggml_repeat(ctx, tp,
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 6));
    }

    // ── Condition embedder ───────────────────────────────────────────────
    // Pipeline: TE hidden (1024-dim) → encoder.text_projector (1024→2048)
    //           → decoder.condition_embedder (2048→2048) → cross-attn
    ggml_tensor* cond_emb = nullptr;
    ggml_tensor* tp_w = ggml_get_tensor(ext_ctx, "encoder.text_projector.weight");
    ggml_tensor* cew = ggml_get_tensor(ext_ctx, "decoder.condition_embedder.weight");
    ggml_tensor* ceb = ggml_get_tensor(ext_ctx, "decoder.condition_embedder.bias");
    if (tp_w && cew && ceb && cond_data && ct_len > 0) {
        // (1) Load raw TE hidden states [te_hs, ct_len] (te_hs=1024)
        int64_t te_hs = (cond_hidden > 0) ? cond_hidden : 1024;
        ggml_tensor* ct = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, te_hs, ct_len);
        memcpy(ct->data, cond_data,
               static_cast<size_t>(ct_len) * static_cast<size_t>(te_hs) *
               sizeof(float));
        // (2) Project TE hidden (1024) → encoder_hidden (2048) via
        //     encoder.text_projector. Weight ne=[1024, 2048] → mul_mat
        //     maps inner 1024 → outer 2048.
        ggml_tensor* proj = ggml_mul_mat(ctx, tp_w, ct);
        // (3) Apply condition_embedder: encoder_hidden (2048) → H.
        //     The bias repeat target MUST be the mul_mat output (shape H),
        //     not `proj` (shape 2048) — those only coincide when H==2048
        //     (turbo). For xl-base H=2560, repeating ceb=[2560] against
        //     proj=[2048,N] is a shape error.
        ggml_tensor* ce_out = ggml_mul_mat(ctx, cew, proj);
        cond_emb = ggml_add(ctx, ce_out, ggml_repeat(ctx, ceb, ce_out));
    } else if (cew && ceb && (!cond_data || ct_len == 0)) {
        // No input condition — use learned null_condition_emb.
        // null_condition_emb is encoder_hidden-dim (2048); condition_embedder
        // maps it to H. Same bias-repeat fix as above.
        ggml_tensor* nce = ggml_get_tensor(ext_ctx, "null_condition_emb");
        if (nce) {
            ggml_tensor* ce_out = ggml_mul_mat(ctx, cew, nce);
            cond_emb = ggml_add(ctx, ce_out, ggml_repeat(ctx, ceb, ce_out));
        }
    }

    // ── DiT layers ───────────────────────────────────────────────────────
    for (int i = 0; i < n_layer && i < 48; i++) {
        char buf[128];

        // Build per-layer time_mod with scale_shift_table bias
        ggml_tensor* layer_time_mod = time_mod;
        if (i < (int)ss_table_f32.size() && !ss_table_f32[i].empty()) {
            ggml_tensor* sst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 6);
            memcpy(sst->data, ss_table_f32[i].data(),
                   static_cast<size_t>(H) * 6 * sizeof(float));
            layer_time_mod = ggml_add(ctx, time_mod, sst);
        }

        // ── Self-attention block ─────────────────────────────────────────
        {
            ggml_tensor* h = ggml_rms_norm(ctx, cur, eps);
            // Learned norm weight
            std::snprintf(buf, sizeof(buf),
                          "decoder.layers.%d.self_attn_norm.weight", i);
            ggml_tensor* san_w = ggml_get_tensor(ext_ctx, buf);
            if (san_w) {
                h = ggml_mul(ctx, h, ggml_repeat(ctx, san_w, h));
            }
            h = adaln(ctx, h, layer_time_mod, 0, 1, H);

            // Self-attention with separate Q/K/V + QK-norm
            std::snprintf(buf, sizeof(buf),
                          "decoder.layers.%d.self_attn.q_proj.weight", i);
            auto q_w = ggml_get_tensor(ext_ctx, buf);
            std::snprintf(buf, sizeof(buf),
                          "decoder.layers.%d.self_attn.k_proj.weight", i);
            auto k_w = ggml_get_tensor(ext_ctx, buf);
            std::snprintf(buf, sizeof(buf),
                          "decoder.layers.%d.self_attn.v_proj.weight", i);
            auto v_w = ggml_get_tensor(ext_ctx, buf);
            std::snprintf(buf, sizeof(buf),
                          "decoder.layers.%d.self_attn.o_proj.weight", i);
            auto o_w = ggml_get_tensor(ext_ctx, buf);

            // QK-norm weights
            std::snprintf(buf, sizeof(buf),
                          "decoder.layers.%d.self_attn.q_norm.weight", i);
            auto qn_w = ggml_get_tensor(ext_ctx, buf);
            std::snprintf(buf, sizeof(buf),
                          "decoder.layers.%d.self_attn.k_norm.weight", i);
            auto kn_w = ggml_get_tensor(ext_ctx, buf);

            h = self_attn(ctx, h, T, H, nh, nk, hd,
                          q_w, k_w, v_w, o_w,
                          qn_w, kn_w, theta);
            if (h) cur = ggml_add(ctx, cur, h);
        }

        // ── Cross-attention block ────────────────────────────────────────
        if (cond_emb) {
            ggml_tensor* h = ggml_rms_norm(ctx, cur, eps);
            // Learned norm weight
            std::snprintf(buf, sizeof(buf),
                          "decoder.layers.%d.cross_attn_norm.weight", i);
            ggml_tensor* can_w = ggml_get_tensor(ext_ctx, buf);
            if (can_w) {
                h = ggml_mul(ctx, h, ggml_repeat(ctx, can_w, h));
            }
            h = adaln(ctx, h, layer_time_mod, 2, 3, H);

            int c_len = static_cast<int>(cond_emb->ne[1]);
            std::snprintf(buf, sizeof(buf),
                          "decoder.layers.%d.cross_attn.q_proj.weight", i);
            auto ca_q = ggml_get_tensor(ext_ctx, buf);
            std::snprintf(buf, sizeof(buf),
                          "decoder.layers.%d.cross_attn.k_proj.weight", i);
            auto ca_k = ggml_get_tensor(ext_ctx, buf);
            std::snprintf(buf, sizeof(buf),
                          "decoder.layers.%d.cross_attn.v_proj.weight", i);
            auto ca_v = ggml_get_tensor(ext_ctx, buf);
            std::snprintf(buf, sizeof(buf),
                          "decoder.layers.%d.cross_attn.o_proj.weight", i);
            auto ca_o = ggml_get_tensor(ext_ctx, buf);

            // Cross-attn QK-norm weights
            std::snprintf(buf, sizeof(buf),
                          "decoder.layers.%d.cross_attn.q_norm.weight", i);
            auto ca_qn = ggml_get_tensor(ext_ctx, buf);
            std::snprintf(buf, sizeof(buf),
                          "decoder.layers.%d.cross_attn.k_norm.weight", i);
            auto ca_kn = ggml_get_tensor(ext_ctx, buf);

            h = cross_attn(ctx, h, cond_emb, T, c_len, H, nh, nk, hd,
                           ca_q, ca_k, ca_v, ca_o,
                           ca_qn, ca_kn);
            if (h) cur = ggml_add(ctx, cur, h);
        }

        // ── MLP block ────────────────────────────────────────────────────
        {
            ggml_tensor* h = ggml_rms_norm(ctx, cur, eps);
            // Learned norm weight
            std::snprintf(buf, sizeof(buf),
                          "decoder.layers.%d.mlp_norm.weight", i);
            ggml_tensor* mn_w = ggml_get_tensor(ext_ctx, buf);
            if (mn_w) {
                h = ggml_mul(ctx, h, ggml_repeat(ctx, mn_w, h));
            }
            h = adaln(ctx, h, layer_time_mod, 4, 5, H);

            std::snprintf(buf, sizeof(buf),
                          "decoder.layers.%d.mlp.gate_proj.weight", i);
            auto gw = ggml_get_tensor(ext_ctx, buf);
            std::snprintf(buf, sizeof(buf),
                          "decoder.layers.%d.mlp.up_proj.weight", i);
            auto uw = ggml_get_tensor(ext_ctx, buf);
            std::snprintf(buf, sizeof(buf),
                          "decoder.layers.%d.mlp.down_proj.weight", i);
            auto dw = ggml_get_tensor(ext_ctx, buf);

            h = swiglu_mlp(ctx, h, gw, uw, dw);
            if (h) cur = ggml_add(ctx, cur, h);
        }
    }

    // ── Final norm + scale/shift ─────────────────────────────────────────
    {
        // Learned output norm weight
        ggml_tensor* norm_out_w = ggml_get_tensor(ext_ctx, "decoder.norm_out.weight");
        ggml_tensor* n_out = ggml_rms_norm(ctx, cur, eps);
        if (norm_out_w) {
            n_out = ggml_mul(ctx, n_out, ggml_repeat(ctx, norm_out_w, n_out));
        }

        // Global scale/shift: decoder.scale_shift_table [H, 2] → ne0=H, ne1=2
        if (!global_ss_f32.empty()) {
            ggml_tensor* gss = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 2);
            memcpy(gss->data, global_ss_f32.data(),
                   static_cast<size_t>(H) * 2 * sizeof(float));
            auto os = ggml_reshape_2d(ctx,
                ggml_view_1d(ctx, gss, H, 0), H, 1);
            auto oh = ggml_reshape_2d(ctx,
                ggml_view_1d(ctx, gss, H,
                             static_cast<size_t>(H) * sizeof(float)),
                H, 1);
            n_out = ggml_add(ctx,
                ggml_mul(ctx, n_out, ggml_repeat(ctx, os, n_out)),
                ggml_repeat(ctx, oh, n_out));
        }
        cur = n_out;
    }

    // ── Compute ──────────────────────────────────────────────────────────
    ggml_build_forward_expand(gf, cur);
    int st = ggml_graph_compute_with_ctx(ctx, gf, nthr);
    if (st != 0) {
        if (error) *error = "DiT compute failed (status " +
                            std::to_string(st) + ")";
        ggml_free(ctx); delete[] ctx_buf; return false;
    }

    memcpy(result, cur->data,
           static_cast<size_t>(T) * H * sizeof(float));
    ggml_free(ctx);
    delete[] ctx_buf;
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
bool DiTRunner::forward(
    const float* x_t, float t,
    const float* cond, int32_t T_cond, int32_t cond_hidden,
    const float* cond_nc, int32_t T_cond_nc,
    float guidance_scale, int32_t n_patches,
    float* output, std::string* error)
{
    const int32_t H = cfg_.hidden_size;
    const int32_t T = n_patches;
    const int temb_dim = 256;

    // Pre-compute timestep embedding (shared by cond + uncond)
    std::vector<float> temb(static_cast<size_t>(temb_dim));
    // Scale timestep by 1000 (diffusion convention, matches HOT-Step
    // dit-graph.h:118: ggml_scale(ctx, t_scalar, 1000.0f))
    timestep_embed(t * 1000.0f, temb.data(), temb_dim);

    if (guidance_scale == 1.0f) {
        return run_one_forward(cfg_, ext_ctx_, x_t, T, H,
                                temb.data(), temb_dim,
                                cond, T_cond, cond_hidden,
                                output,
                                ss_table_f32_, global_ss_f32_, error);
    }

    // CFG: cond + uncond, then blend
    std::vector<float> c_out(static_cast<size_t>(T) * H);
    std::vector<float> u_out(static_cast<size_t>(T) * H);

    if (!run_one_forward(cfg_, ext_ctx_, x_t, T, H,
                          temb.data(), temb_dim,
                          cond, T_cond, cond_hidden,
                          c_out.data(),
                          ss_table_f32_, global_ss_f32_, error)) {
        if (error) *error = "DiT: cond forward failed";
        return false;
    }

    const float* uc = (cond_nc && T_cond_nc > 0) ? cond_nc : nullptr;
    int uc_len = (cond_nc && T_cond_nc > 0) ? T_cond_nc : 0;
    if (!run_one_forward(cfg_, ext_ctx_, x_t, T, H,
                          temb.data(), temb_dim,
                          uc, uc_len, cond_hidden,
                          u_out.data(),
                          ss_table_f32_, global_ss_f32_, error)) {
        if (error) *error = "DiT: uncond forward failed";
        return false;
    }

    int64_t n_el = static_cast<int64_t>(T) * H;
    for (int64_t i = 0; i < n_el; i++) {
        output[i] = u_out[static_cast<size_t>(i)] +
                    guidance_scale * (c_out[static_cast<size_t>(i)] -
                                      u_out[static_cast<size_t>(i)]);
    }
    return true;
}

}  // namespace audiocore::acestep
