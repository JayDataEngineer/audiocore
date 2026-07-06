// dit_runner.cpp — DiT graph builder + forward pass for MOSS-SoundEffect-v2.
//
// Architecture (WanAudioModel, 30 layers):
//   1. patch_embedding (Linear in_dim→dim, patch_size=1 → no-op reshape)
//   2. text_embedding  (Linear text_dim→dim, then SiLU, then Linear dim→dim)
//   3. time_embedding  (sinusoidal→Linear→SiLU→Linear→SiLU→time_proj→[dim*6])
//   4. 30× blocks, each with:
//      a. RMS norm1 + AdaLN → QK-norm → self-attention + RoPE
//      b. (no norm) → QK-norm → cross-attention + RoPE
//      c. RMS norm3 + AdaLN → GELU_tanh FFN
//   5. head norm + global modulation scale/shift + Linear(dim, out_dim)
//
// GGUF tensor names (all prefixed with "moss_sfx_v2."):
//   patch_embedding.{weight,bias}
//   text_embedding.{0,2}.{weight,bias}
//   time_embedding.{0,2}.{weight,bias}
//   time_projection.1.{weight,bias}
//   blocks.{i}.norm1.weight
//   blocks.{i}.norm3.weight     (HF norm2 → renamed)
//   blocks.{i}.self_attn.{q,k,v,o}.weight
//   blocks.{i}.self_attn.{norm_q,norm_k}.weight
//   blocks.{i}.cross_attn.{q,k,v,o}.weight
//   blocks.{i}.cross_attn.{norm_q,norm_k}.weight
//   blocks.{i}.ffn.{0,2}.weight
//   blocks.{i}.modulation
//   head.norm.weight
//   head.head.{weight,bias}
//   head.modulation

#include "audiocore/models/moss_sfx_v2/dit_runner.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

namespace audiocore::moss_sfx_v2 {

// ── BF16 → F32 helpers ─────────────────────────────────────────────────────
static float bf16_to_f32(uint16_t bits) {
    uint32_t f32_bits = static_cast<uint32_t>(bits) << 16;
    float f;
    memcpy(&f, &f32_bits, sizeof(f));
    return f;
}

static void bf16_buf_to_f32(const void* src, float* dst, int n) {
    const uint16_t* s = static_cast<const uint16_t*>(src);
    for (int i = 0; i < n; i++) dst[i] = bf16_to_f32(s[i]);
}

// ── Sinusoidal timestep embedding ─────────────────────────────────────────
static void timestep_sinusoidal(const float* t, float* out,
                                 int n_timesteps, int freq_dim) {
    int half = freq_dim / 2;
    float log_period = std::log(10000.0f);
    for (int b = 0; b < n_timesteps; b++) {
        float* row = out + static_cast<size_t>(b) * freq_dim;
        for (int j = 0; j < half; j++) {
            float freq = std::exp(-log_period * j / static_cast<float>(half));
            float arg = t[b] * 1000.0f * freq;
            row[j]          = std::cos(arg);
            row[half + j]   = std::sin(arg);
        }
    }
}

// ── GELU (tanh approximation) ────────────────────────────────────────────
static ggml_tensor* gelu_tanh(ggml_context* ctx, ggml_tensor* x) {
    const float s = std::sqrt(2.0f / (float)M_PI);
    const float c = 0.044715f;
    ggml_tensor* x3 = ggml_mul(ctx, x, ggml_mul(ctx, x, x));
    ggml_tensor* inner = ggml_scale(ctx,
        ggml_add(ctx, x, ggml_scale(ctx, x3, c)), s);
    ggml_tensor* th = ggml_tanh(ctx, inner);
    return ggml_mul(ctx,
        ggml_scale(ctx, x, 0.5f),
        ggml_add(ctx, th, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1)));
}

// ── FFN: Linear(dim, 2*ffn_dim) → GELU_tanh → Linear(2*ffn_dim, dim) ────
static ggml_tensor* ffn_gelu(ggml_context* ctx, ggml_tensor* x,
                               ggml_tensor* w0, ggml_tensor* w2) {
    if (!w0 || !w2) return nullptr;
    auto h = ggml_mul_mat(ctx, w0, x);
    h = gelu_tanh(ctx, h);
    return ggml_mul_mat(ctx, w2, h);
}

// ── QK-norm (RMS norm per-head + learned scale) ───────────────────────────
static ggml_tensor* apply_qk_norm(ggml_context* ctx, ggml_tensor* t,
                                    int hd, int nh, ggml_tensor* norm_w) {
    if (!norm_w) return t;
    int64_t T = ggml_nelements(t) / (static_cast<int64_t>(hd) * nh);
    ggml_tensor* t3d = ggml_reshape_3d(ctx, t, hd, nh, T);
    t3d = ggml_rms_norm(ctx, t3d, 1e-6f);
    ggml_tensor* nw = ggml_reshape_3d(ctx, norm_w, hd, nh, 1);
    nw = ggml_repeat(ctx, nw, t3d);
    t3d = ggml_mul(ctx, t3d, nw);
    return ggml_reshape_2d(ctx, t3d, nh * hd, T);
}

// ── Self-attention (QK-norm + RoPE) ──────────────────────────────────────
static ggml_tensor* self_attn(ggml_context* ctx, ggml_tensor* x,
                                int T, int H, int nh, int hd,
                                ggml_tensor* q_w, ggml_tensor* k_w,
                                ggml_tensor* v_w, ggml_tensor* o_w,
                                ggml_tensor* q_norm_w, ggml_tensor* k_norm_w) {
    if (!q_w || !k_w || !v_w) return nullptr;

    auto q = ggml_mul_mat(ctx, q_w, x);
    auto k = ggml_mul_mat(ctx, k_w, x);
    auto v = ggml_mul_mat(ctx, v_w, x);

    q = apply_qk_norm(ctx, q, hd, nh, q_norm_w);
    k = apply_qk_norm(ctx, k, hd, nh, k_norm_w);

    auto to_3d = [&](ggml_tensor* t, int n_h) {
        return ggml_reshape_3d(ctx, t, hd, n_h, T);
    };
    q = to_3d(q, nh);
    k = to_3d(k, nh);
    v = to_3d(v, nh);

    // RoPE
    ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    {
        int32_t* pd = static_cast<int32_t*>(pos->data);
        for (int i = 0; i < T; i++) pd[i] = i;
    }
    auto rope_fn = [&](ggml_tensor* t) {
        return ggml_rope_ext(ctx, t, pos, nullptr,
                              hd, 2, 0,
                              10000.0f, 1.0f, 0.0f, 1.0f,
                              0.0f, 0.0f);
    };
    q = rope_fn(q);
    k = rope_fn(k);

    auto rsh = [&](ggml_tensor* t, int n_h) {
        return ggml_cont(ctx, ggml_permute(ctx, t, 0, 2, 1, 3));
    };
    q = rsh(q, nh);
    k = rsh(k, nh);
    v = rsh(v, nh);

    float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    auto a = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(a, GGML_PREC_F32);

    // flash_attn_ext returns [hd, nh, T, 1] — reshape directly (NO permute).
    a = ggml_reshape_2d(ctx, a, nh * hd, T);
    if (o_w) a = ggml_mul_mat(ctx, o_w, a);
    return a;
}

// ── Cross-attention (QK-norm + RoPE on Q from x, K from context) ────────
static ggml_tensor* cross_attn(ggml_context* ctx, ggml_tensor* x,
                                 ggml_tensor* cond, int T, int T_cond,
                                 int H, int nh, int hd,
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

    q_lin = apply_qk_norm(ctx, q_lin, hd, nh, q_norm_w);
    k_lin = apply_qk_norm(ctx, k_lin, hd, nh, k_norm_w);

    // RoPE on Q (using x positions)
    {
        ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
        int32_t* pd = static_cast<int32_t*>(pos->data);
        for (int i = 0; i < T; i++) pd[i] = i;
        auto q_3d = ggml_reshape_3d(ctx, q_lin, hd, nh, T);
        q_lin = ggml_reshape_2d(ctx,
            ggml_rope_ext(ctx, q_3d, pos, nullptr,
                           hd, 2, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f),
            nh * hd, T);
    }
    // RoPE on K (using context positions truncated)
    if (T_cond > 0) {
        ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_cond);
        int32_t* pd = static_cast<int32_t*>(pos->data);
        for (int i = 0; i < T_cond && i < T; i++) pd[i] = i;
        auto k_3d = ggml_reshape_3d(ctx, k_lin, hd, nh, T_cond);
        k_lin = ggml_reshape_2d(ctx,
            ggml_rope_ext(ctx, k_3d, pos, nullptr,
                           hd, 2, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f),
            nh * hd, T_cond);
    }

    auto q = rsh(q_lin, nh, T);
    auto k = rsh(k_lin, nh, T_cond);
    auto v = rsh(v_lin, nh, T_cond);

    float s = 1.0f / std::sqrt(static_cast<float>(hd));
    auto a = ggml_flash_attn_ext(ctx, q, k, v, nullptr, s, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(a, GGML_PREC_F32);
    // flash_attn_ext returns [hd, nh, T, 1] — reshape directly (NO permute).
    a = ggml_reshape_2d(ctx, a, nh * hd, T);
    a = ggml_mul_mat(ctx, ca_o, a);
    return a;
}

// ══════════════════════════════════════════════════════════════════════════════
// DiTRunner — constructor, destructor, member helpers
// ══════════════════════════════════════════════════════════════════════════════

DiTRunner::DiTRunner(ggml_context* ext_ctx, const DitConfig& cfg)
    : ext_ctx_(ext_ctx), cfg_(cfg) {
    if (cfg_.n_layers > 0) {
        modulation_f32_.resize(cfg_.n_layers);
        for (int i = 0; i < cfg_.n_layers; i++) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                          "moss_sfx_v2.blocks.%d.modulation", i);
            ggml_tensor* mod = ggml_get_tensor(ext_ctx_, buf);
            if (mod && mod->type == GGML_TYPE_BF16) {
                int n = static_cast<int>(ggml_nelements(mod));
                modulation_f32_[i].resize(n);
                bf16_buf_to_f32(mod->data, modulation_f32_[i].data(), n);
            }
        }
    }
    {
        ggml_tensor* hmod = ggml_get_tensor(ext_ctx_,
            "moss_sfx_v2.head.modulation");
        if (hmod && hmod->type == GGML_TYPE_BF16) {
            int n = static_cast<int>(ggml_nelements(hmod));
            head_modulation_f32_.resize(n);
            bf16_buf_to_f32(hmod->data, head_modulation_f32_.data(), n);
        }
    }
}

DiTRunner::~DiTRunner() {
    if (cached_ctx_) {
        ggml_free(cached_ctx_);
        cached_ctx_ = nullptr;
    }
    free(cached_buf_);
    cached_buf_ = nullptr;
    cached_mem_ = 0;
}

// ══════════════════════════════════════════════════════════════════════════════
// Static graph-builder + compute function (avoids member-access issues)
// ══════════════════════════════════════════════════════════════════════════════

static bool run_one_forward(
    const DitConfig& cfg, ggml_context* ext_ctx,
    const float* x_t, int32_t T_latent, int32_t H,
    const float* temb, int temb_dim,
    const float* cond_data, int ct_len, int cond_hidden,
    float* result,
    const std::vector<std::vector<float>>& modulation_f32,
    const std::vector<float>& head_modulation_f32,
    std::string* error,
    // Cached ggml context (reused to avoid 12GB alloc/free per forward)
    size_t& cached_mem, char*& cached_buf, ggml_context*& cached_ctx)
{
    const int32_t nh     = cfg.n_heads;
    const int32_t hd     = cfg.head_dim;
    const int32_t n_lyr  = cfg.n_layers;
    const float   eps    = cfg.eps;
    const int32_t ffn_d  = cfg.ffn_dim;
    const int     nthr   = 4;

    auto weight = [&](const char* name) -> ggml_tensor* {
        return ggml_get_tensor(ext_ctx, name);
    };

    // Context memory — reuse cached context when possible
    size_t mem;
    if (T_latent <= 750)       mem = 12288ULL * 1024 * 1024;
    else if (T_latent <= 1500) mem = 16384ULL * 1024 * 1024;
    else                       mem = 24576ULL * 1024 * 1024;

    ggml_context* ctx;
    if (cached_ctx && cached_mem >= mem) {
        // Reuse existing context
        ggml_reset(cached_ctx);
        ctx = cached_ctx;
    } else {
        // Allocate new context (first call or T_latent grew)
        if (cached_ctx) { ggml_free(cached_ctx); cached_ctx = nullptr; }
        free(cached_buf); cached_buf = nullptr;

        char* ctx_buf = (char*)aligned_alloc(64, mem);
        if (!ctx_buf) { if (error) *error = "DiT OOM"; return false; }
        cached_buf = ctx_buf;
        cached_mem = mem;

        ggml_init_params p = { mem, cached_buf, /*no_alloc=*/false };
        cached_ctx = ggml_init(p);
        if (!cached_ctx) {
            free(cached_buf); cached_buf = nullptr; cached_mem = 0;
            if (error) *error = "DiT ggml_init"; return false;
        }
        ctx = cached_ctx;
    }
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 4096, false);
    if (!gf) { if (error) *error = "DiT ggml_new_graph"; return false; }

    // ── 1. Load timestep embedding ─────────────────────────────────────
    ggml_tensor* te = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, temb_dim, 1);
    memcpy(te->data, temb, static_cast<size_t>(temb_dim) * sizeof(float));

    // time_embedding.0: Linear(freq_dim, dim)
    {
        ggml_tensor* w = weight("moss_sfx_v2.time_embedding.0.weight");
        ggml_tensor* b = weight("moss_sfx_v2.time_embedding.0.bias");
        if (w) {
            te = ggml_mul_mat(ctx, w, te);
            if (b) te = ggml_add(ctx, te, ggml_repeat(ctx, b, te));
        }
    }
    te = ggml_silu(ctx, te);

    // time_embedding.2: Linear(dim, dim)
    {
        ggml_tensor* w = weight("moss_sfx_v2.time_embedding.2.weight");
        ggml_tensor* b = weight("moss_sfx_v2.time_embedding.2.bias");
        if (w) {
            te = ggml_mul_mat(ctx, w, te);
            if (b) te = ggml_add(ctx, te, ggml_repeat(ctx, b, te));
        }
    }
    te = ggml_silu(ctx, te);

    // time_projection.1: Linear(dim, dim*6)
    ggml_tensor* t_mod = nullptr;
    {
        ggml_tensor* w = weight("moss_sfx_v2.time_projection.1.weight");
        ggml_tensor* b = weight("moss_sfx_v2.time_projection.1.bias");
        if (w) {
            t_mod = ggml_mul_mat(ctx, w, te);
            if (b) t_mod = ggml_add(ctx, t_mod, ggml_repeat(ctx, b, t_mod));
        }
    }
    // t_mod is [dim*6, 1]

    // ── 2. Text embedding ──────────────────────────────────────────────
    ggml_tensor* ctx_emb = nullptr;
    if (cond_data && ct_len > 0) {
        ggml_tensor* ct = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                               cond_hidden, ct_len);
        memcpy(ct->data, cond_data,
               static_cast<size_t>(ct_len) * cond_hidden * sizeof(float));

        ggml_tensor* w = weight("moss_sfx_v2.text_embedding.0.weight");
        ggml_tensor* b = weight("moss_sfx_v2.text_embedding.0.bias");
        if (w) {
            ct = ggml_mul_mat(ctx, w, ct);
            if (b) ct = ggml_add(ctx, ct, ggml_repeat(ctx, b, ct));
        }
        ct = ggml_silu(ctx, ct);

        w = weight("moss_sfx_v2.text_embedding.2.weight");
        b = weight("moss_sfx_v2.text_embedding.2.bias");
        if (w) {
            ct = ggml_mul_mat(ctx, w, ct);
            if (b) ct = ggml_add(ctx, ct, ggml_repeat(ctx, b, ct));
        }
        ctx_emb = ct;
    }

    // ── 3. Patch embedding: Linear(in_dim, dim) ────────────────────────
    ggml_tensor* cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                            cfg.in_dim, T_latent);
    memcpy(cur->data, x_t,
           static_cast<size_t>(T_latent) * cfg.in_dim * sizeof(float));
    {
        ggml_tensor* w = weight("moss_sfx_v2.patch_embedding.weight");
        ggml_tensor* b = weight("moss_sfx_v2.patch_embedding.bias");
        if (w) {
            // GGUF stores patch_embedding as 3D [kernel=1, in, out] (converter
            // reverses the safetensors shape). For ggml_mul_mat we need 2D
            // [in, out], so reshape away the singleton kernel dim.
            if (ggml_n_dims(w) > 2) {
                w = ggml_reshape_2d(ctx, w, cfg.in_dim, cfg.dim);
            }
            if (w->ne[0] != cfg.in_dim) {
                w = ggml_cont(ctx, ggml_transpose(ctx, w));
            }
            cur = ggml_mul_mat(ctx, w, cur);
            if (b) cur = ggml_add(ctx, cur, ggml_repeat(ctx, b, cur));
        }
    }

    // ── 4. DiT layers ─────────────────────────────────────────────────
    for (int i = 0; i < n_lyr; i++) {
        char buf[128];

        // Add block modulation bias to t_mod
        ggml_tensor* layer_mod = t_mod;
        if (i < (int)modulation_f32.size() && !modulation_f32[i].empty()) {
            ggml_tensor* mod_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,
                                                         H * 6);
            memcpy(mod_bias->data, modulation_f32[i].data(),
                   static_cast<size_t>(H) * 6 * sizeof(float));
            mod_bias = ggml_reshape_2d(ctx, mod_bias, H * 6, 1);
            layer_mod = ggml_add(ctx, t_mod, mod_bias);
        }

        // Split [H*6, 1] → 6 × [H, 1]
        auto mod_chunk = [&](int idx) -> ggml_tensor* {
            return ggml_reshape_2d(ctx,
                ggml_view_1d(ctx, layer_mod, H,
                              static_cast<size_t>(idx) * H * sizeof(float)),
                H, 1);
        };
        ggml_tensor* shift1 = mod_chunk(0);
        ggml_tensor* scale1 = mod_chunk(1);
        ggml_tensor* gate1  = mod_chunk(2);
        ggml_tensor* shift2 = mod_chunk(3);
        ggml_tensor* scale2 = mod_chunk(4);
        ggml_tensor* gate2  = mod_chunk(5);

        // ── 4a. Self-attention block ─────────────────────────────────
        {
            ggml_tensor* h = ggml_rms_norm(ctx, cur, eps);
            std::snprintf(buf, sizeof(buf),
                          "moss_sfx_v2.blocks.%d.norm1.weight", i);
            ggml_tensor* nw = ggml_get_tensor(ext_ctx, buf);
            if (nw) h = ggml_mul(ctx, h, ggml_repeat(ctx, nw, h));

            // AdaLN: h * (1 + scale) + shift
            h = ggml_add(ctx,
                ggml_mul(ctx, h, ggml_add(ctx,
                    ggml_repeat(ctx, scale1, h),
                    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1))),
                ggml_repeat(ctx, shift1, h));

            std::snprintf(buf, sizeof(buf),
                          "moss_sfx_v2.blocks.%d.self_attn.q.weight", i);
            auto q_w = ggml_get_tensor(ext_ctx, buf);
            std::snprintf(buf, sizeof(buf),
                          "moss_sfx_v2.blocks.%d.self_attn.k.weight", i);
            auto k_w = ggml_get_tensor(ext_ctx, buf);
            std::snprintf(buf, sizeof(buf),
                          "moss_sfx_v2.blocks.%d.self_attn.v.weight", i);
            auto v_w = ggml_get_tensor(ext_ctx, buf);
            std::snprintf(buf, sizeof(buf),
                          "moss_sfx_v2.blocks.%d.self_attn.o.weight", i);
            auto o_w = ggml_get_tensor(ext_ctx, buf);
            std::snprintf(buf, sizeof(buf),
                          "moss_sfx_v2.blocks.%d.self_attn.norm_q.weight", i);
            auto qn_w = ggml_get_tensor(ext_ctx, buf);
            std::snprintf(buf, sizeof(buf),
                          "moss_sfx_v2.blocks.%d.self_attn.norm_k.weight", i);
            auto kn_w = ggml_get_tensor(ext_ctx, buf);

            auto sa = self_attn(ctx, h, T_latent, H, nh, hd,
                                q_w, k_w, v_w, o_w, qn_w, kn_w);
            if (sa) {
                cur = ggml_add(ctx, cur,
                    ggml_mul(ctx, sa, ggml_repeat(ctx, gate1, sa)));
            }
        }

        // ── 4b. Cross-attention block ───────────────────────────────
        if (ctx_emb) {
            int c_len = static_cast<int>(ctx_emb->ne[1]);
            std::snprintf(buf, sizeof(buf),
                          "moss_sfx_v2.blocks.%d.cross_attn.q.weight", i);
            auto ca_q = ggml_get_tensor(ext_ctx, buf);
            std::snprintf(buf, sizeof(buf),
                          "moss_sfx_v2.blocks.%d.cross_attn.k.weight", i);
            auto ca_k = ggml_get_tensor(ext_ctx, buf);
            std::snprintf(buf, sizeof(buf),
                          "moss_sfx_v2.blocks.%d.cross_attn.v.weight", i);
            auto ca_v = ggml_get_tensor(ext_ctx, buf);
            std::snprintf(buf, sizeof(buf),
                          "moss_sfx_v2.blocks.%d.cross_attn.o.weight", i);
            auto ca_o = ggml_get_tensor(ext_ctx, buf);
            std::snprintf(buf, sizeof(buf),
                          "moss_sfx_v2.blocks.%d.cross_attn.norm_q.weight", i);
            auto ca_qn = ggml_get_tensor(ext_ctx, buf);
            std::snprintf(buf, sizeof(buf),
                          "moss_sfx_v2.blocks.%d.cross_attn.norm_k.weight", i);
            auto ca_kn = ggml_get_tensor(ext_ctx, buf);

            auto ca = cross_attn(ctx, cur, ctx_emb,
                                  T_latent, c_len, H, nh, hd,
                                  ca_q, ca_k, ca_v, ca_o,
                                  ca_qn, ca_kn);
            if (ca) {
                cur = ggml_add(ctx, cur,
                    ggml_mul(ctx, ca, ggml_repeat(ctx, gate1, ca)));
            }
        }

        // ── 4c. FFN block ───────────────────────────────────────────
        {
            ggml_tensor* h = ggml_rms_norm(ctx, cur, eps);
            std::snprintf(buf, sizeof(buf),
                          "moss_sfx_v2.blocks.%d.norm3.weight", i);
            ggml_tensor* nw = ggml_get_tensor(ext_ctx, buf);
            if (nw) h = ggml_mul(ctx, h, ggml_repeat(ctx, nw, h));

            h = ggml_add(ctx,
                ggml_mul(ctx, h, ggml_add(ctx,
                    ggml_repeat(ctx, scale2, h),
                    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1))),
                ggml_repeat(ctx, shift2, h));

            std::snprintf(buf, sizeof(buf),
                          "moss_sfx_v2.blocks.%d.ffn.0.weight", i);
            auto ffn_w0 = ggml_get_tensor(ext_ctx, buf);
            std::snprintf(buf, sizeof(buf),
                          "moss_sfx_v2.blocks.%d.ffn.2.weight", i);
            auto ffn_w2 = ggml_get_tensor(ext_ctx, buf);

            auto ffn_out = ffn_gelu(ctx, h, ffn_w0, ffn_w2);
            if (ffn_out) {
                cur = ggml_add(ctx, cur,
                    ggml_mul(ctx, ffn_out, ggml_repeat(ctx, gate2, ffn_out)));
            }
        }
    }

    // ── 5. Head ──────────────────────────────────────────────────────────
    {
        ggml_tensor* n_out = ggml_rms_norm(ctx, cur, eps);
        ggml_tensor* nw = weight("moss_sfx_v2.head.norm.weight");
        if (nw) n_out = ggml_mul(ctx, n_out, ggml_repeat(ctx, nw, n_out));

        if (!head_modulation_f32.empty()) {
            ggml_tensor* hmod = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 2);
            memcpy(hmod->data, head_modulation_f32.data(),
                   static_cast<size_t>(H) * 2 * sizeof(float));
            auto os = ggml_reshape_2d(ctx,
                ggml_view_1d(ctx, hmod, H, 0), H, 1);
            auto oh = ggml_reshape_2d(ctx,
                ggml_view_1d(ctx, hmod, H,
                              static_cast<size_t>(H) * sizeof(float)),
                H, 1);
            n_out = ggml_add(ctx,
                ggml_mul(ctx, n_out, ggml_repeat(ctx, os, n_out)),
                ggml_repeat(ctx, oh, n_out));
        }

        ggml_tensor* hw = weight("moss_sfx_v2.head.head.weight");
        ggml_tensor* hb = weight("moss_sfx_v2.head.head.bias");
        if (hw) {
            n_out = ggml_mul_mat(ctx, hw, n_out);
            if (hb) n_out = ggml_add(ctx, n_out, ggml_repeat(ctx, hb, n_out));
        }
        cur = n_out;
    }

    // ── 6. Compute ──────────────────────────────────────────────────────
    ggml_build_forward_expand(gf, cur);
    int st = ggml_graph_compute_with_ctx(ctx, gf, nthr);
    if (st != 0) {
        if (error) *error = "DiT compute failed (status " +
                            std::to_string(st) + ")";
        ggml_reset(ctx); return false;
    }

    int64_t n_out = static_cast<int64_t>(T_latent) * cfg.out_dim;
    memcpy(result, cur->data, static_cast<size_t>(n_out) * sizeof(float));
    ggml_reset(ctx);
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
// Forward (public API)
// ══════════════════════════════════════════════════════════════════════════════

bool DiTRunner::forward(const float* x_t, const float* t,
                         const float* context, int32_t T_text,
                         int32_t B, int32_t T_latent,
                         float* output, std::string* error) {
    if (B != 1) {
        if (error) *error = "DiTRunner only supports B=1 for now";
        return false;
    }
    const int32_t H       = cfg_.dim;
    const int32_t freq_d  = cfg_.freq_dim;

    std::vector<float> temb(static_cast<size_t>(freq_d));
    timestep_sinusoidal(t, temb.data(), 1, freq_d);

    return run_one_forward(cfg_, ext_ctx_, x_t, T_latent, H,
                            temb.data(), freq_d,
                            context, T_text, cfg_.text_dim,
                            output,
                            modulation_f32_, head_modulation_f32_, error,
                            cached_mem_, cached_buf_, cached_ctx_);
}

bool DiTRunner::forward_cfg(const float* x_t, const float* t,
                             const float* context_cond, int32_t T_cond,
                             const float* context_uncond, int32_t T_uncond,
                             float guidance_scale,
                             int32_t B, int32_t T_latent,
                             float* output, std::string* error) {
    if (B != 1) {
        if (error) *error = "CFG only supports B=1 for now";
        return false;
    }
    if (guidance_scale == 1.0f) {
        return forward(x_t, t, context_cond, T_cond, B, T_latent,
                        output, error);
    }

    int64_t n_el = static_cast<int64_t>(T_latent) * cfg_.out_dim;
    std::vector<float> c_out(static_cast<size_t>(n_el));
    std::vector<float> u_out(static_cast<size_t>(n_el));

    if (!forward(x_t, t, context_cond, T_cond, B, T_latent,
                  c_out.data(), error)) {
        if (error) *error = "DiT: cond forward failed";
        return false;
    }
    if (!forward(x_t, t, context_uncond, T_uncond, B, T_latent,
                  u_out.data(), error)) {
        if (error) *error = "DiT: uncond forward failed";
        return false;
    }

    for (int64_t i = 0; i < n_el; i++) {
        output[i] = u_out[static_cast<size_t>(i)] +
                    guidance_scale * (c_out[static_cast<size_t>(i)] -
                                      u_out[static_cast<size_t>(i)]);
    }
    return true;
}

}  // namespace audiocore::moss_sfx_v2
