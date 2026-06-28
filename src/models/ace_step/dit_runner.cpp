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
//   1. RMS norm + AdaLN modulation  (scale/shift from time_mod rows)
//   2. GQA self-attention + RoPE
//   3. RMS norm + AdaLN + cross-attention to TE condition
//   4. RMS norm + AdaLN + SwiGLU MLP
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

// ── Self-attention (GQA + RoPE) ──────────────────────────────────────────────
// Returns just the attention (or O-projected) output — NO residual added.
static ggml_tensor* self_attn(ggml_context* ctx, ggml_tensor* x,
                                int T, int H, int nh, int nk, int hd,
                                ggml_tensor* qkv,
                                ggml_tensor* q_w, ggml_tensor* k_w,
                                ggml_tensor* v_w, ggml_tensor* o_w,
                                float rope_theta) {
    ggml_tensor *q, *k, *v;

    if (qkv) {
        int qd = nh * hd, kd = nk * hd;
        int64_t row_b = qkv->ne[0];   // bytes per row
        ggml_tensor* f = ggml_mul_mat(ctx, qkv, x);
        q = ggml_view_2d(ctx, f, qd, T, f->nb[1], 0);
        k = ggml_view_2d(ctx, f, kd, T, f->nb[1],
                         static_cast<size_t>(qd) * sizeof(float));
        v = ggml_view_2d(ctx, f, kd, T, f->nb[1],
                         static_cast<size_t>(qd + kd) * sizeof(float));
    } else if (q_w && k_w && v_w) {
        q = ggml_mul_mat(ctx, q_w, x);
        k = ggml_mul_mat(ctx, k_w, x);
        v = ggml_mul_mat(ctx, v_w, x);
    } else {
        return nullptr;
    }

    // RoPE (Neox-style: mode=2)
    auto rope = [&](ggml_tensor* t) {
        return ggml_rope_ext(ctx, t, nullptr, nullptr,
                              hd, 2, 0,
                              rope_theta, 1.0f, 0.0f, 1.0f,
                              0.0f, 0.0f);
    };
    q = rope(q);
    k = rope(k);

    // Reshape for flash_attn_ext → [hd, n_heads, T]
    auto rsh = [&](ggml_tensor* t, int n_h, int len) {
        return ggml_cont(ctx, ggml_permute(ctx,
            ggml_reshape_3d(ctx, t, hd, n_h, len), 0, 2, 1, 3));
    };
    q = rsh(q, nh, T);
    k = rsh(k, nk, T);
    v = rsh(v, nk, T);

    float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    auto a = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);

    // Back to [T, nh*hd]
    a = ggml_cont(ctx, ggml_permute(ctx, a, 0, 2, 1, 3));
    a = ggml_reshape_2d(ctx, a, nh * hd, T);
    if (o_w) a = ggml_mul_mat(ctx, o_w, a);
    return a;
}

// ── Cross-attention (no RoPE) ────────────────────────────────────────────────
// Returns just the attention output — NO residual added.
static ggml_tensor* cross_attn(ggml_context* ctx, ggml_tensor* x,
                                 ggml_tensor* cond, int T, int T_cond,
                                 int H, int nh, int nk, int hd,
                                 ggml_tensor* ca_q, ggml_tensor* ca_k,
                                 ggml_tensor* ca_v, ggml_tensor* ca_o) {
    if (!ca_q || !ca_k || !ca_v || !ca_o || !cond) return nullptr;

    auto rsh = [&](ggml_tensor* t, int n_h, int len) {
        return ggml_cont(ctx, ggml_permute(ctx,
            ggml_reshape_3d(ctx, t, hd, n_h, len), 0, 2, 1, 3));
    };

    auto q = rsh(ggml_mul_mat(ctx, ca_q, x), nh, T);
    auto k = rsh(ggml_mul_mat(ctx, ca_k, cond), nk, T_cond);
    auto v = rsh(ggml_mul_mat(ctx, ca_v, cond), nk, T_cond);

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
    : ext_ctx_(ext_ctx), cfg_(cfg) {}

DiTRunner::~DiTRunner() = default;

ggml_tensor* DiTRunner::weight(const char* name) const {
    return ggml_get_tensor(ext_ctx_, name);
}

// ── Timestep sinusoidal embedding ────────────────────────────────────────────
static void timestep_embed(float t, float* out, int dim) {
    int half = dim / 2;
    for (int i = 0; i < half; i++) {
        float f = std::exp(-static_cast<float>(i) *
                            std::log(10000.0f) / (half - 1));
        out[i]        = std::sin(t * f);
        out[half + i] = std::cos(t * f);
    }
}

// ── Single forward: build graph + compute ────────────────────────────────────
static bool run_one_forward(
    const DitConfig& cfg, ggml_context* ext_ctx,
    const float* x_t, int32_t T, int32_t H,
    const float* temb, int temb_dim,
    const float* cond_data, int ct_len, int cond_hidden,
    float* result,
    std::string* error)
{
    const int32_t nh      = cfg.n_heads     > 0 ? cfg.n_heads     : 24;
    const int32_t nk      = cfg.n_kv_heads  > 0 ? cfg.n_kv_heads  : 8;
    const int32_t hd      = cfg.head_dim    > 0 ? cfg.head_dim    : 128;
    const int32_t n_layer = cfg.n_layers    > 0 ? cfg.n_layers    : 24;
    const float   eps     = cfg.rms_norm_eps > 0 ? cfg.rms_norm_eps : 1e-6f;
    const float   theta   = cfg.rope_theta  > 0 ? cfg.rope_theta  : 10000.0f;
    const int     nthr    = 4;

    // Context memory: 512 MB for T up to ~250 (10 s at 25 Hz).
    // For longer sequences, switch to the ggml-alloc + backend API pattern.
    const size_t mem = 512ULL * 1024 * 1024;
    char* ctx_buf = new (std::nothrow) char[mem];
    if (!ctx_buf) { if (error) *error = "DiT OOM"; return false; }

    ggml_init_params p = { mem, ctx_buf, /*no_alloc=*/false };
    ggml_context* ctx = ggml_init(p);
    ggml_cgraph* gf  = ggml_new_graph(ctx);
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

    ggml_tensor* t1w = ggml_get_tensor(ext_ctx, "decoder.time_embed.linear_1_w");
    ggml_tensor* t1b = ggml_get_tensor(ext_ctx, "decoder.time_embed.linear_1_b");
    if (t1w) {
        tp = ggml_mul_mat(ctx, t1w, tp);
        if (t1b) tp = ggml_add(ctx, tp, ggml_repeat(ctx, t1b, tp));
    }
    tp = ggml_silu(ctx, tp);
    ggml_tensor* t2w = ggml_get_tensor(ext_ctx, "decoder.time_embed.linear_2_w");
    ggml_tensor* t2b = ggml_get_tensor(ext_ctx, "decoder.time_embed.linear_2_b");
    if (t2w) {
        tp = ggml_mul_mat(ctx, t2w, tp);
        if (t2b) tp = ggml_add(ctx, tp, ggml_repeat(ctx, t2b, tp));
    }

    ggml_tensor* time_mod = nullptr;
    ggml_tensor* tpw = ggml_get_tensor(ext_ctx, "decoder.time_embed.time_proj_w");
    ggml_tensor* tpb = ggml_get_tensor(ext_ctx, "decoder.time_embed.time_proj_b");
    if (tpw) {
        auto pj = ggml_mul_mat(ctx, tpw, tp);
        if (tpb) pj = ggml_add(ctx, pj, ggml_repeat(ctx, tpb, pj));
        time_mod = ggml_reshape_2d(ctx, pj, H, 6);   // ne0=H, ne1=6
    } else {
        // No time_proj → repeat time-embedding as time_mod
        time_mod = ggml_repeat(ctx, tp,
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 6));
    }

    // ── Condition embedder ───────────────────────────────────────────────
    ggml_tensor* cond_emb = nullptr;
    ggml_tensor* cew = ggml_get_tensor(ext_ctx, "decoder.condition_embedder.weight");
    ggml_tensor* ceb = ggml_get_tensor(ext_ctx, "decoder.condition_embedder.bias");
    if (cew && ceb && cond_data && ct_len > 0) {
        int64_t ch = (cond_hidden > 0) ? cond_hidden : 1024;
        ggml_tensor* ct = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ch, ct_len);
        memcpy(ct->data, cond_data,
               static_cast<size_t>(ct_len) * static_cast<size_t>(ch) *
               sizeof(float));
        cond_emb = ggml_add(ctx, ggml_mul_mat(ctx, cew, ct),
                             ggml_repeat(ctx, ceb, ct));
    } else if (cew && ceb && (!cond_data || ct_len == 0)) {
        // No input condition — use learned null_condition_emb
        ggml_tensor* nce = ggml_get_tensor(ext_ctx, "null_condition_emb");
        if (nce) {
            cond_emb = ggml_add(ctx, ggml_mul_mat(ctx, cew, nce),
                                 ggml_repeat(ctx, ceb, nce));
        }
    }

    // ── DiT layers ───────────────────────────────────────────────────────
    for (int i = 0; i < n_layer && i < 48; i++) {
        char buf[128];
        auto bn = [&](const char* sfx) -> const char* {
            std::snprintf(buf, sizeof(buf), "decoder.block.%d.%s", i, sfx);
            return buf;
        };

        // Self-attention block
        {
            ggml_tensor* h = ggml_rms_norm(ctx, cur, eps);
            h = adaln(ctx, h, time_mod, 0, 1, H);
            h = self_attn(ctx, h, T, H, nh, nk, hd,
                           ggml_get_tensor(ext_ctx, bn("sa_qkv.weight")),
                           ggml_get_tensor(ext_ctx, bn("sa_q_proj.weight")),
                           ggml_get_tensor(ext_ctx, bn("sa_k_proj.weight")),
                           ggml_get_tensor(ext_ctx, bn("sa_v_proj.weight")),
                           ggml_get_tensor(ext_ctx, bn("sa_o_proj.weight")),
                           theta);
            if (h) cur = ggml_add(ctx, cur, h);
        }

        // Cross-attention block (if condition available)
        if (cond_emb) {
            ggml_tensor* h = ggml_rms_norm(ctx, cur, eps);
            h = adaln(ctx, h, time_mod, 2, 3, H);
            int c_len = static_cast<int>(cond_emb->ne[1]);
            h = cross_attn(ctx, h, cond_emb, T, c_len, H, nh, nk, hd,
                           ggml_get_tensor(ext_ctx, bn("ca_q_proj.weight")),
                           ggml_get_tensor(ext_ctx, bn("ca_k_proj.weight")),
                           ggml_get_tensor(ext_ctx, bn("ca_v_proj.weight")),
                           ggml_get_tensor(ext_ctx, bn("ca_o_proj.weight")));
            if (h) cur = ggml_add(ctx, cur, h);
        }

        // MLP block
        {
            ggml_tensor* h = ggml_rms_norm(ctx, cur, eps);
            h = adaln(ctx, h, time_mod, 4, 5, H);
            h = swiglu_mlp(ctx, h,
                           ggml_get_tensor(ext_ctx, bn("gate_proj.weight")),
                           ggml_get_tensor(ext_ctx, bn("up_proj.weight")),
                           ggml_get_tensor(ext_ctx, bn("down_proj.weight")));
            if (h) cur = ggml_add(ctx, cur, h);
        }
    }

    // ── Final norm + scale/shift ─────────────────────────────────────────
    {
        ggml_tensor* n_out = ggml_rms_norm(ctx, cur, eps);
        ggml_tensor* oss = ggml_get_tensor(ext_ctx, "decoder.out_scale_shift");
        if (oss) {
            auto os = ggml_reshape_2d(ctx,
                ggml_view_1d(ctx, oss, H, 0), H, 1);
            auto oh = ggml_reshape_2d(ctx,
                ggml_view_1d(ctx, oss, H,
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
    timestep_embed(t, temb.data(), temb_dim);

    if (guidance_scale == 1.0f) {
        return run_one_forward(cfg_, ext_ctx_, x_t, T, H,
                                temb.data(), temb_dim,
                                cond, T_cond, cond_hidden,
                                output, error);
    }

    // CFG: cond + uncond, then blend
    std::vector<float> c_out(static_cast<size_t>(T) * H);
    std::vector<float> u_out(static_cast<size_t>(T) * H);

    if (!run_one_forward(cfg_, ext_ctx_, x_t, T, H,
                          temb.data(), temb_dim,
                          cond, T_cond, cond_hidden,
                          c_out.data(), error)) {
        if (error) *error = "DiT: cond forward failed";
        return false;
    }

    const float* uc = (cond_nc && T_cond_nc > 0) ? cond_nc : nullptr;
    int uc_len = (cond_nc && T_cond_nc > 0) ? T_cond_nc : 0;
    if (!run_one_forward(cfg_, ext_ctx_, x_t, T, H,
                          temb.data(), temb_dim,
                          uc, uc_len, cond_hidden,
                          u_out.data(), error)) {
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
