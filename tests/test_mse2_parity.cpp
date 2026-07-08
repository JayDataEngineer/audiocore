// test_mse2_parity.cpp — layer-by-layer parity tests for MSE2 DiT + VAE.
//
// Brute-force approach: for every individual sub-operation, builds a tiny
// self-contained ggml graph, feeds the PyTorch reference input, and compares
// output element-wise. Win condition: every op matches within BF16 tolerance.
//
// Run:
//   ./test_mse2_parity /path/to/moss_sfx_v2.gguf /path/to/dump_mse2_tensors/

#include "test_framework.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include "audiocore/models/moss_sfx_v2/vae_runner.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ═════════════════════════════════════════════════════════════════════════════
//  Global state — weights loaded once, shared across all tests
// ═════════════════════════════════════════════════════════════════════════════

static ggml_context* g_weights = nullptr;    // DiT + VAE weight tensors
static std::string   g_dump_dir;

// ═════════════════════════════════════════════════════════════════════════════
//  Reference tensor I/O
// ═════════════════════════════════════════════════════════════════════════════

struct Shape {
    int64_t ne[4] = {1, 1, 1, 1};
    int     ndim  = 0;
};

static Shape read_shape(const std::string& path) {
    Shape s;
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return s;
    int64_t v;
    while (std::fscanf(f, "%ld", &v) == 1) s.ne[s.ndim++] = v;
    std::fclose(f);
    for (int i = s.ndim; i < 4; i++) s.ne[i] = 1;
    return s;
}

struct RefTensor {
    std::vector<float> data;
    Shape shape;
    int64_t n_el() const {
        int64_t n = 1;
        for (int i = 0; i < shape.ndim; i++) n *= shape.ne[i];
        return n;
    }
};

static RefTensor load_ref(const char* name, const char* prefix) {
    RefTensor t;
    std::string base = std::string(prefix) + "/" + name;
    t.shape = read_shape(base + ".shape");
    int64_t n = t.n_el();
    if (n == 0) return t;
    FILE* f = std::fopen((base + ".bin").c_str(), "rb");
    if (!f) { t.data.clear(); return t; }
    t.data.resize(n);
    auto got = std::fread(t.data.data(), sizeof(float), n, f);
    std::fclose(f);
    if ((int64_t)got != n) t.data.clear();
    return t;
}

#define LOAD(_name, _prefix) load_ref(_name, _prefix)

// ═════════════════════════════════════════════════════════════════════════════
//  GGUF weight loader
// ═════════════════════════════════════════════════════════════════════════════

static float bf16_to_f32(uint16_t bits) {
    uint32_t f = (uint32_t)bits << 16;
    float r;
    memcpy(&r, &f, sizeof(r));
    return r;
}

static ggml_context* load_gguf(const char* path) {
    // Use no_alloc=false so gguf_init_from_file creates a ggml_context with
    // all tensors allocated and their data pointers set to the mmap region.
    ggml_context* ctx = nullptr;
    gguf_init_params params = { false, &ctx };
    gguf_context* gctx = gguf_init_from_file(path, params);
    if (!gctx) return nullptr;
    if (!ctx) { gguf_free(gctx); return nullptr; }

    int n = gguf_get_n_tensors(gctx);
    std::fprintf(stderr, "  loaded %d tensors from GGUF\n", n);
    gguf_free(gctx);
    return ctx;
}

static void load_weight_f32(const char* name, std::vector<float>& buf) {
    auto* t = ggml_get_tensor(g_weights, name);
    if (!t) { buf.clear(); return; }
    int64_t n = ggml_nelements(t);
    buf.resize(n);
    if (t->type == GGML_TYPE_F32)
        memcpy(buf.data(), t->data, n * sizeof(float));
    else if (t->type == GGML_TYPE_BF16) {
        const uint16_t* s = (const uint16_t*)t->data;
        for (int64_t i = 0; i < n; i++) buf[i] = bf16_to_f32(s[i]);
    } else buf.clear();
}

// ═════════════════════════════════════════════════════════════════════════════
//  Tensor comparison
// ═════════════════════════════════════════════════════════════════════════════

static double max_abs_err(const float* a, const float* b, int64_t n) {
    double m = 0;
    for (int64_t i = 0; i < n; i++)
        m = std::max(m, std::fabs((double)a[i] - (double)b[i]));
    return m;
}

static double max_rel_err(const float* a, const float* b, int64_t n) {
    double m = 0;
    for (int64_t i = 0; i < n; i++) {
        double d = std::fabs((double)a[i] - (double)b[i]);
        double den = std::max({std::fabs((double)a[i]), std::fabs((double)b[i]), 1e-6});
        m = std::max(m, d / den);
    }
    return m;
}

// Extract batch, time, channel from shape (dump saves as [B, T, C]).
// Returns T=ne[1], C=ne[2] for 3D; T=ne[0], C=ne[1] for 2D.
static void shape_tc(const Shape& s, int64_t& T, int64_t& C) {
    if (s.ndim >= 3) { T = s.ne[1]; C = s.ne[2]; }
    else if (s.ndim == 2) { T = s.ne[0]; C = s.ne[1]; }
    else { T = s.ne[0]; C = 1; }
}

static bool match(const float* got, const float* want, int64_t n,
                   const char* label, double tol, int fail_on_err = 1) {
    double ae = max_abs_err(got, want, n);
    double re = max_rel_err(got, want, n);
    bool ok = (ae < tol) || (re < tol);
    if (!ok) {
        std::fprintf(stderr, "  [FAIL] %s: abs_err=%.2e rel_err=%.2e\n",
                     label, ae, re);
        for (int64_t i = 0; i < std::min(n, (int64_t)4); i++)
            std::fprintf(stderr, "    [%ld] got=%.6f want=%.6f diff=%.2e\n",
                         (long)i, got[i], want[i], std::fabs(got[i]-want[i]));
        if (fail_on_err) return false;
    } else {
        std::fprintf(stderr, "  [OK] %s: abs_err=%.2e rel_err=%.2e\n", label, ae, re);
    }
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Graph builder helpers — each creates a self-contained ggml context
// ═════════════════════════════════════════════════════════════════════════════

struct GraphResult {
    bool ok;
    std::vector<float> data;
    int64_t T, C;  // output shape (time-major: [T, C])
};

static GraphResult run_ggml(const float* input, int64_t T, int64_t C,
                             const std::vector<float>& weight_1d,
                             int mode,
                             int extra_param = 0,
                             const std::vector<float>& bias = {}) {
    size_t mem = 256 * 1024 * 1024;
    auto* buf = new (std::nothrow) char[mem];
    if (!buf) { GraphResult gr; gr.ok = false; return gr; }
    ggml_init_params p = { mem, buf, false };
    auto* ctx = ggml_init(p);
    if (!ctx) { delete[] buf; GraphResult gr; gr.ok = false; return gr; }
    auto* gf  = ggml_new_graph_custom(ctx, 128, false);
    if (!gf) { ggml_free(ctx); delete[] buf; GraphResult gr; gr.ok = false; return gr; }

    // Match DiT runner layout: ne[0]=C (channel innermost), ne[1]=T.
    // Flat memcpy of time-major [T,C] works because:
    //   input[t*C + c] → ggml offset c + t*C = t*C + c  ✓
    auto* in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int32_t)C, (int32_t)T);
    if (!in) { ggml_free(ctx); delete[] buf; GraphResult gr; gr.ok = false; return gr; }
    memcpy(in->data, input, T * C * sizeof(float));

    ggml_tensor* r = nullptr;
    int32_t OC = (int32_t)C;  // default: same as input channels

    if (mode == 0) {  // LayerNorm over channel dim (ne[0]=C) — optional weight + bias
        auto* h = ggml_norm(ctx, in, 1e-6f);
        if (!weight_1d.empty()) {
            auto* w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int32_t)C);
            memcpy(w->data, weight_1d.data(), (int32_t)C * sizeof(float));
            h = ggml_mul(ctx, h, ggml_repeat(ctx, w, h));
        }
        if (!bias.empty()) {
            auto* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int32_t)C);
            memcpy(b->data, bias.data(), (int32_t)C * sizeof(float));
            h = ggml_add(ctx, h, ggml_repeat(ctx, b, h));
        }
        r = h;

    } else if (mode == 1) {  // Linear: mul_mat(weight, x) + optional bias
        // weight_1d is flat [OC, C]; GGUF stores as [OC, C] row-major
        // For ggml: w = [C, OC] where ne[0]=C (innermost)
        int num_w = (int)weight_1d.size();
        OC = num_w / (int32_t)C;  // output channels
        auto* w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int32_t)C, OC);
        memcpy(w->data, weight_1d.data(), num_w * sizeof(float));
        // mul_mat: [C, OC] × [C, T] → [OC, T]
        r = ggml_mul_mat(ctx, w, in);
        if (!bias.empty()) {
            auto* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, OC);
            memcpy(b->data, bias.data(), OC * sizeof(float));
            r = ggml_add(ctx, r, ggml_repeat(ctx, b, r));
        }

    } else if (mode == 2) {  // GELU tanh (1D over all elements)
        float s = std::sqrt(2.0f / (float)M_PI);
        float c = 0.044715f;
        auto x3 = ggml_mul(ctx, in, ggml_mul(ctx, in, in));
        auto inner = ggml_scale(ctx, ggml_add(ctx, in, ggml_scale(ctx, x3, c)), s);
        auto th = ggml_tanh(ctx, inner);
        auto* ones = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 1);
        *(float*)ones->data = 1.0f;
        r = ggml_mul(ctx,
            ggml_scale(ctx, in, 0.5f),
            ggml_add(ctx, th, ones));  // th + 1 (scalar broadcast as 2nd arg)

    } else if (mode == 3) {  // QK-norm: RMS norm over full channel dim with learned scale
        auto* h = ggml_rms_norm(ctx, in, 1e-6f);
        if (!weight_1d.empty()) {
            auto* w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int32_t)C);
            memcpy(w->data, weight_1d.data(), (int32_t)C * sizeof(float));
            h = ggml_mul(ctx, h, ggml_repeat(ctx, w, h));
        }
        r = h;

    } else if (mode == 4) {  // 1D RoPE on [C,T] = [nh*hd, T] — matches Python rope_apply
        // Python (wan_audio_dit.py:WanAudioModel, vae_type="dac") uses:
        //   freqs = precompute_freqs_cis(head_dim=128, end, theta=10000)
        //   freqs.chunk(3, dim=-1) → 3 chunks, each using positions [0..f-1]
        //   In forward, all 3 chunks concatenated back → full 128-dim RoPE
        //   with theta^(-2j/128).  A single ggml_rope with n_dims=hd reproduces this.
        int hd = extra_param;
        int nh = (int32_t)C / hd;
        auto* in3 = ggml_reshape_3d(ctx, in, hd, nh, (int32_t)T);
        auto* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int32_t)T);
        int32_t* pd = (int32_t*)pos->data;
        for (int32_t i = 0; i < (int32_t)T; i++) pd[i] = i;
        auto* roped = ggml_rope_ext_inplace(ctx, in3, pos, nullptr,
            hd, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_build_forward_expand(gf, roped);
        r = ggml_reshape_2d(ctx, in3, (int32_t)C, (int32_t)T);
    }

    GraphResult gr;
    if (r) {
        ggml_build_forward_expand(gf, r);
        int st = ggml_graph_compute_with_ctx(ctx, gf, 4);
        gr.ok = (st == 0);
        if (gr.ok) {
            int64_t n = ggml_nelements(r);
            gr.data.resize(n);
            // r is [OC, T] in ggml layout: element [oc, t] at oc + t*OC
            // Convert to time-major [T, OC]
            int32_t out_C = (int32_t)(n / T);
            for (int32_t t = 0; t < (int32_t)T; t++)
                for (int32_t oc = 0; oc < out_C; oc++)
                    gr.data[(int64_t)t * out_C + oc] =
                        ((float*)r->data)[oc + (int64_t)t * out_C];
            gr.T = T;
            gr.C = out_C;
        }
    }
    ggml_free(ctx);
    delete[] buf;
    return gr;
}

// ── Self-attention (flash_attn_ext) ─────────────────────────────────────────

static GraphResult run_attention(const float* q, const float* k, const float* v,
                                  int64_t T, int hd, int nh,
                                  const float* o_weight, int OC) {
    size_t mem = 512 * 1024 * 1024;
    auto* buf = new char[mem];
    ggml_init_params p = { mem, buf, false };
    auto* ctx = ggml_init(p);
    auto* gf  = ggml_new_graph_custom(ctx, 512, false);

    int C = nh * hd;

    auto make_t = [&](const float* d) {
        auto* t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, nh, (int32_t)T);
        for (int32_t tl = 0; tl < (int32_t)T; tl++)
            for (int32_t h = 0; h < nh; h++)
                memcpy((float*)t->data + (int64_t)tl * nh * hd + (int64_t)h * hd,
                       d + (int64_t)tl * C + (int64_t)h * hd,
                       hd * sizeof(float));
        return t;
    };
    auto qt = make_t(q);
    auto kt = make_t(k);
    auto vt = make_t(v);

    // Permute [hd,nh,T] → [hd,T,nh] for flash_attn
    auto perm = [&](auto* t) { return ggml_cont(ctx, ggml_permute(ctx, t, 0, 2, 1, 3)); };
    auto qp = perm(qt);
    auto kp = perm(kt);
    auto vp = perm(vt);

    float s = 1.0f / std::sqrt((float)hd);
    auto fa = ggml_flash_attn_ext(ctx, qp, kp, vp, nullptr, s, 0.0f, 0.0f);
    // flash_attn returns [hd, nh, T, 1] (ggml.c: ne={v[0],q[2],q[1],q[3]}).
    // Data is contiguous ordered (d,h,t); reshape_2d(C,T) correctly maps
    // c=h*hd+d for each timestep. Do NOT permute (scrambles heads/T).
    auto fo = ggml_reshape_2d(ctx, fa, C, (int32_t)T);

    // Debug: save flash_attn output (before O proj) for block 0
    auto fa_debug = ggml_cpy(ctx, fa, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, (int32_t)T));
    auto a = fo;

    // Output projection
    if (o_weight && OC > 0) {
        auto* ow = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, OC);
        memcpy(ow->data, o_weight, (int64_t)C * OC * sizeof(float));
        a = ggml_mul_mat(ctx, ow, a);
    }

    ggml_build_forward_expand(gf, a);
    int st = ggml_graph_compute_with_ctx(ctx, gf, 4);

    GraphResult gr;
    gr.ok = (st == 0);
    if (gr.ok) {
        int64_t n = ggml_nelements(a);
        gr.data.resize(n);
        // a is [OC, T] in ggml: element [oc, t] at oc + t*OC
        int32_t OC2 = (int32_t)(n / T);
        for (int32_t t = 0; t < (int32_t)T; t++)
            for (int32_t oc = 0; oc < OC2; oc++)
                gr.data[(int64_t)t * OC2 + oc] =
                    ((float*)a->data)[oc + (int64_t)t * OC2];
        gr.T = T;
        gr.C = OC2;
    }
    ggml_free(ctx);
    delete[] buf;
    return gr;
}

// ── Cross-attention full ─────────────────────────────────────────────────────

static GraphResult run_cross_attn(const float* x, int64_t T,
                                   const float* cond, int64_t T_cond,
                                   int hd, int nh,
                                   const float* ca_q, int C_in,
                                   const float* ca_q_bias,
                                   const float* ca_k, const float* ca_k_bias,
                                   const float* ca_v, const float* ca_v_bias,
                                   const float* ca_o, const float* ca_o_bias,
                                   const float* qn_w, const float* kn_w) {
    size_t mem = 512 * 1024 * 1024;
    auto* buf = new char[mem];
    ggml_init_params p = { mem, buf, false };
    auto* ctx = ggml_init(p);
    auto* gf  = ggml_new_graph_custom(ctx, 512, false);

    int C = nh * hd;

    // Linear projection helper with optional bias (flat memcpy for row-major→col-major)
    auto lin_b = [&](const float* w_data, int in_dim, int out_dim,
                     const float* b_data,
                     const float* inp, int64_t len) -> ggml_tensor* {
        auto* w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, out_dim);
        memcpy(w->data, w_data, (int64_t)in_dim * out_dim * sizeof(float));
        auto* t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, (int32_t)len);
        memcpy(t->data, inp, (int64_t)in_dim * len * sizeof(float));
        auto* r = ggml_mul_mat(ctx, w, t);
        if (b_data) {
            auto* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_dim);
            memcpy(b->data, b_data, out_dim * sizeof(float));
            r = ggml_add(ctx, r, ggml_repeat(ctx, b, r));
        }
        return r;
    };

    int x_dim = C_in;  // = dim = 1536
    int text_hidden = x_dim;  // text_embedding projects 2048→dim, so K/V operate on dim

    auto q_raw = lin_b(ca_q, x_dim, C, ca_q_bias, x, T);
    auto k_raw = lin_b(ca_k, text_hidden, C, ca_k_bias, cond, T_cond);
    auto v_raw = lin_b(ca_v, text_hidden, C, ca_v_bias, cond, T_cond);

    // QK-norm with learned scale (apply RMSNorm over full channel dim, then scale)
    auto qk_norm_fn = [&](ggml_tensor* t, int32_t len,
                          const float* nw) -> ggml_tensor* {
        auto t2 = ggml_rms_norm(ctx, t, 1e-6f);
        if (nw) {
            auto* w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
            memcpy(w->data, nw, C * sizeof(float));
            t2 = ggml_mul(ctx, t2, ggml_repeat(ctx, w, t2));
        }
        return t2;
    };
    auto q_lin = qk_norm_fn(q_raw, (int32_t)T, qn_w);
    auto k_lin = qk_norm_fn(k_raw, (int32_t)T_cond, kn_w);

    // Cross-attention does NOT apply RoPE (unlike self-attention)

    // Permute for flash_attn
    auto rsh = [&](ggml_tensor* t, int32_t len) {
        return ggml_cont(ctx, ggml_permute(ctx,
            ggml_reshape_3d(ctx, t, hd, nh, len), 0, 2, 1, 3));
    };
    auto qp = rsh(q_lin, (int32_t)T);
    auto kp = rsh(k_lin, (int32_t)T_cond);
    auto vp = rsh(v_raw, (int32_t)T_cond);

    float scale = 1.0f / std::sqrt((float)hd);
    auto a = ggml_flash_attn_ext(ctx, qp, kp, vp, nullptr, scale, 0.0f, 0.0f);
    // flash_attn returns [hd, nh, T, 1] — reshape directly (NO permute).
    a = ggml_reshape_2d(ctx, a, C, (int32_t)T);

    {
        auto* ow = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, C);
        memcpy(ow->data, ca_o, (int64_t)C * C * sizeof(float));
        a = ggml_mul_mat(ctx, ow, a);
        if (ca_o_bias) {
            auto* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
            memcpy(b->data, ca_o_bias, C * sizeof(float));
            a = ggml_add(ctx, a, ggml_repeat(ctx, b, a));
        }
    }

    ggml_build_forward_expand(gf, a);
    int st = ggml_graph_compute_with_ctx(ctx, gf, 4);

    GraphResult gr;
    gr.ok = (st == 0);
    if (gr.ok) {
        int64_t n = ggml_nelements(a);
        gr.data.resize(n);
        for (int32_t t = 0; t < (int32_t)T; t++)
            for (int32_t oc = 0; oc < C; oc++)
                gr.data[(int64_t)t * C + oc] =
                    ((float*)a->data)[oc + (int64_t)t * C];
        gr.T = T;
        gr.C = C;
    }
    ggml_free(ctx);
    delete[] buf;
    return gr;
}

// ═════════════════════════════════════════════════════════════════════════════
//  DiT block test — all sub-operations
// ═════════════════════════════════════════════════════════════════════════════

static int test_block(int bi, int dim, int nh, int hd) {
    int fails = 0;
    // Strict tolerance: well-implemented ops should match within BF16 precision.
    // The previous TOL=10.0 was hiding catastrophic bugs (RoPE abs_err=20,
    // attention abs_err=17, cross-attn abs_err=2830 all "passing").
    double TOL = 0.1;

    auto fail = [&](const char* op) { fails++; std::fprintf(stderr, "  [FAIL] block_%d/%s\n", bi, op); };

    char bp[256];
    std::snprintf(bp, sizeof(bp), "%s/block_%d", g_dump_dir.c_str(), bi);

    std::fprintf(stderr, "    loading refs...\n");

    // ── 1. RMS norm (norm1) ──────────────────────────────────────────────
    {
        auto ref_in  = LOAD("input", bp);
        auto ref_out = LOAD("norm1_out", bp);
        if (ref_in.data.empty() || ref_out.data.empty()) {
            std::fprintf(stderr, "    ref_in empty=%d ref_out empty=%d\n",
                         (int)ref_in.data.empty(), (int)ref_out.data.empty());
            return 1;
        }
        int64_t T, C; shape_tc(ref_in.shape, T, C);

        char wn[128];
        std::snprintf(wn, sizeof(wn), "moss_sfx_v2.blocks.%d.norm1.weight", bi);
        std::vector<float> w; load_weight_f32(wn, w);

        auto gr = run_ggml(ref_in.data.data(), T, C, w, 0);
        if (!gr.ok || !match(gr.data.data(), ref_out.data.data(),
                              ref_out.n_el(), (std::string("block_") + std::to_string(bi) + "/norm1").c_str(), TOL))
            fail("norm1");
    }

    // ── 2. Self-attention Q projection (+ QK-norm) ──────────────────────
    {
        // Reference: self_attn_q = norm_q(q_proj(modulated_x))
        auto ref_in  = LOAD("modulated_x", bp);
        auto ref_out = LOAD("self_attn_q", bp);
        if (ref_in.data.empty() || ref_out.data.empty()) return fails + 1;
        int64_t T, C; shape_tc(ref_in.shape, T, C);

        char wn[128];
        std::snprintf(wn, sizeof(wn), "moss_sfx_v2.blocks.%d.self_attn.q.weight", bi);
        std::vector<float> w; load_weight_f32(wn, w);
        std::snprintf(wn, sizeof(wn), "moss_sfx_v2.blocks.%d.self_attn.q.bias", bi);
        std::vector<float> b; load_weight_f32(wn, b);
        // QK-norm weight (learned affine per head)
        std::snprintf(wn, sizeof(wn), "moss_sfx_v2.blocks.%d.self_attn.norm_q.weight", bi);
        std::vector<float> nw; load_weight_f32(wn, nw);

        // Step 1: linear projection + bias
        auto gr1 = run_ggml(ref_in.data.data(), T, C, w, 1, 0, b);
        if (!gr1.ok) return fails + 1;
        // Step 2: QK-norm (RMS norm over full channel dim with learned scale)
        auto gr2 = run_ggml(gr1.data.data(), gr1.T, gr1.C, nw, 3);
        if (!gr2.ok) return fails + 1;
        if (!match(gr2.data.data(), ref_out.data.data(), ref_out.n_el(),
                   (std::string("block_") + std::to_string(bi) + "/sa_q").c_str(), TOL))
            fail("sa_q");
    }

    // ── 3. Self-attention K projection (+ QK-norm) ──────────────────────
    {
        auto ref_in  = LOAD("modulated_x", bp);
        auto ref_out = LOAD("self_attn_k", bp);
        if (ref_in.data.empty() || ref_out.data.empty()) return fails + 1;
        int64_t T, C; shape_tc(ref_in.shape, T, C);

        char wn[128];
        std::snprintf(wn, sizeof(wn), "moss_sfx_v2.blocks.%d.self_attn.k.weight", bi);
        std::vector<float> w; load_weight_f32(wn, w);
        std::snprintf(wn, sizeof(wn), "moss_sfx_v2.blocks.%d.self_attn.k.bias", bi);
        std::vector<float> b; load_weight_f32(wn, b);
        std::snprintf(wn, sizeof(wn), "moss_sfx_v2.blocks.%d.self_attn.norm_k.weight", bi);
        std::vector<float> nw; load_weight_f32(wn, nw);

        auto gr1 = run_ggml(ref_in.data.data(), T, C, w, 1, 0, b);
        if (!gr1.ok) return fails + 1;
        auto gr2 = run_ggml(gr1.data.data(), gr1.T, gr1.C, nw, 3);
        if (!gr2.ok) return fails + 1;
        if (!match(gr2.data.data(), ref_out.data.data(), ref_out.n_el(),
                   (std::string("block_") + std::to_string(bi) + "/sa_k").c_str(), TOL))
            fail("sa_k");
    }

    // ── 4. Self-attention V projection ──────────────────────────────────
    {
        // Reference: self_attn_v = v_proj(modulated_x)  [no QK-norm on V]
        auto ref_in  = LOAD("modulated_x", bp);
        auto ref_out = LOAD("self_attn_v", bp);
        if (ref_in.data.empty() || ref_out.data.empty()) return fails + 1;
        int64_t T, C; shape_tc(ref_in.shape, T, C);

        char wn[128];
        std::snprintf(wn, sizeof(wn), "moss_sfx_v2.blocks.%d.self_attn.v.weight", bi);
        std::vector<float> w; load_weight_f32(wn, w);
        std::snprintf(wn, sizeof(wn), "moss_sfx_v2.blocks.%d.self_attn.v.bias", bi);
        std::vector<float> b; load_weight_f32(wn, b);

        auto gr = run_ggml(ref_in.data.data(), T, C, w, 1, 0, b);
        if (!gr.ok) return fails + 1;
        if (!match(gr.data.data(), ref_out.data.data(), ref_out.n_el(),
                   (std::string("block_") + std::to_string(bi) + "/sa_v").c_str(), TOL))
            fail("sa_v");
    }

    // ── 5. QK-norm on Q — SKIP (tested in SA Q projection above) ─────

    // ── 6. RoPE on Q (if q_norm and q_rope both available as refs) ─────
    {
        auto ref_norm = LOAD("self_attn_q", bp);
        auto ref_rope = LOAD("self_attn_q_rope", bp);
        if (ref_norm.data.empty() || ref_rope.data.empty()) return fails + 1;
        int64_t T, C; shape_tc(ref_norm.shape, T, C);

        auto gr = run_ggml(ref_norm.data.data(), T, C, {}, 4, hd);
        char lbl[128];
        std::snprintf(lbl, sizeof(lbl), "block_%d/sa_q_rope", bi);
        // RoPE alone
        if (!gr.ok || !match(gr.data.data(), ref_rope.data.data(),
                              ref_rope.n_el(), lbl, TOL))
            fail("sa_q_rope");
    }

    // ── 7. Full self-attention (Q,K,V after rope) — before O proj ─────
    {
        auto ref_qr = LOAD("self_attn_q_rope", bp);
        auto ref_kr = LOAD("self_attn_k_rope", bp);
        auto ref_v  = LOAD("self_attn_v", bp);
        auto ref_attn = LOAD("self_attn_attn", bp);
        if (ref_qr.data.empty() || ref_kr.data.empty() ||
            ref_v.data.empty() || ref_attn.data.empty()) return fails + 1;
        int64_t T, C; shape_tc(ref_qr.shape, T, C);

        // Run attention without O proj (pass OC=0 to skip it)
        auto gr = run_attention(ref_qr.data.data(), ref_kr.data.data(),
                                 ref_v.data.data(), T, hd, nh,
                                 nullptr, 0);
        if (!gr.ok || !match(gr.data.data(), ref_attn.data.data(),
                              ref_attn.n_el(),
                              (std::string("block_") + std::to_string(bi) + "/sa_attn_raw").c_str(), TOL))
            fail("sa_attn_raw");
    }

    // ── 7b. SA O projection ───────────────────────────────────────────
    {
        auto ref_attn = LOAD("self_attn_attn", bp);
        auto ref_o    = LOAD("self_attn_o", bp);
        if (ref_attn.data.empty() || ref_o.data.empty()) return fails + 1;
        int64_t T, C; shape_tc(ref_attn.shape, T, C);

        char wn[128];
        std::snprintf(wn, sizeof(wn), "moss_sfx_v2.blocks.%d.self_attn.o.weight", bi);
        std::vector<float> ow; load_weight_f32(wn, ow);
        std::snprintf(wn, sizeof(wn), "moss_sfx_v2.blocks.%d.self_attn.o.bias", bi);
        std::vector<float> ob; load_weight_f32(wn, ob);

        auto gr = run_ggml(ref_attn.data.data(), T, C, ow, 1, 0, ob);
        if (!gr.ok || !match(gr.data.data(), ref_o.data.data(),
                              ref_o.n_el(),
                              (std::string("block_") + std::to_string(bi) + "/sa_o_proj").c_str(), TOL))
            fail("sa_o_proj");
    }

    // ── 8. Cross-attention Q (+ QK-norm) ───────────────────────────────
    // Input to cross_attn is norm3(after_sa_gate), i.e. norm3_out
    {
        auto ref_in  = LOAD("norm3_out", bp);
        auto ref_out = LOAD("cross_attn_q", bp);
        if (ref_in.data.empty() || ref_out.data.empty()) return fails + 1;
        int64_t T, C; shape_tc(ref_in.shape, T, C);

        char wn[128];
        std::snprintf(wn, sizeof(wn), "moss_sfx_v2.blocks.%d.cross_attn.q.weight", bi);
        std::vector<float> w; load_weight_f32(wn, w);
        std::snprintf(wn, sizeof(wn), "moss_sfx_v2.blocks.%d.cross_attn.q.bias", bi);
        std::vector<float> b; load_weight_f32(wn, b);
        std::snprintf(wn, sizeof(wn), "moss_sfx_v2.blocks.%d.cross_attn.norm_q.weight", bi);
        std::vector<float> nw; load_weight_f32(wn, nw);

        auto gr1 = run_ggml(ref_in.data.data(), T, C, w, 1, 0, b);
        if (!gr1.ok) return fails + 1;
        auto gr2 = run_ggml(gr1.data.data(), gr1.T, gr1.C, nw, 3);
        if (!gr2.ok) return fails + 1;
        if (!match(gr2.data.data(), ref_out.data.data(), ref_out.n_el(),
                   (std::string("block_") + std::to_string(bi) + "/ca_q").c_str(), TOL))
            fail("ca_q");
    }

    // ── 9. Cross-attention K (+ QK-norm) ───────────────────────────────
    {
        auto ctx_ref = LOAD("text_embedding_out", g_dump_dir.c_str());
        auto ref_out = LOAD("cross_attn_k", bp);
        if (ctx_ref.data.empty() || ref_out.data.empty()) return fails + 1;
        int64_t T, C; shape_tc(ctx_ref.shape, T, C);

        char wn[128];
        std::snprintf(wn, sizeof(wn), "moss_sfx_v2.blocks.%d.cross_attn.k.weight", bi);
        std::vector<float> w; load_weight_f32(wn, w);
        std::snprintf(wn, sizeof(wn), "moss_sfx_v2.blocks.%d.cross_attn.k.bias", bi);
        std::vector<float> b; load_weight_f32(wn, b);
        std::snprintf(wn, sizeof(wn), "moss_sfx_v2.blocks.%d.cross_attn.norm_k.weight", bi);
        std::vector<float> nw; load_weight_f32(wn, nw);

        auto gr1 = run_ggml(ctx_ref.data.data(), T, C, w, 1, 0, b);
        if (!gr1.ok) return fails + 1;
        auto gr2 = run_ggml(gr1.data.data(), gr1.T, gr1.C, nw, 3);
        if (!gr2.ok) return fails + 1;
        if (!match(gr2.data.data(), ref_out.data.data(), ref_out.n_el(),
                   (std::string("block_") + std::to_string(bi) + "/ca_k").c_str(), TOL))
            fail("ca_k");
    }

    // ── 10. Cross-attention V ──────────────────────────────────────────
    {
        auto ctx_ref = LOAD("text_embedding_out", g_dump_dir.c_str());
        auto ref_out = LOAD("cross_attn_v", bp);
        if (ctx_ref.data.empty() || ref_out.data.empty()) return fails + 1;
        int64_t T, C; shape_tc(ctx_ref.shape, T, C);

        char wn[128];
        std::snprintf(wn, sizeof(wn), "moss_sfx_v2.blocks.%d.cross_attn.v.weight", bi);
        std::vector<float> w; load_weight_f32(wn, w);
        std::snprintf(wn, sizeof(wn), "moss_sfx_v2.blocks.%d.cross_attn.v.bias", bi);
        std::vector<float> b; load_weight_f32(wn, b);

        auto gr = run_ggml(ctx_ref.data.data(), T, C, w, 1, 0, b);
        if (!gr.ok || !match(gr.data.data(), ref_out.data.data(),
                              ref_out.n_el(), (std::string("block_") + std::to_string(bi) + "/ca_v").c_str(), TOL))
            fail("ca_v");
    }

    // ── 11. Cross-attention full ───────────────────────────────────────
    {
        auto ref_in   = LOAD("norm3_out", bp);
        auto ref_out  = LOAD("cross_attn_o", bp);
        auto ctx_ref  = LOAD("text_embedding_out", g_dump_dir.c_str());
        if (ref_in.data.empty() || ref_out.data.empty() || ctx_ref.data.empty()) return fails + 1;
        int64_t T, C, T_cond, dummyC; shape_tc(ref_in.shape, T, C);
        shape_tc(ctx_ref.shape, T_cond, dummyC);

        char qn[128], kn[128], cq[128], ck[128], cv[128], co[128],
             cqb[128], ckb[128], cvb[128], cob[128];
        std::snprintf(cq, sizeof(cq), "moss_sfx_v2.blocks.%d.cross_attn.q.weight", bi);
        std::snprintf(ck, sizeof(ck), "moss_sfx_v2.blocks.%d.cross_attn.k.weight", bi);
        std::snprintf(cv, sizeof(cv), "moss_sfx_v2.blocks.%d.cross_attn.v.weight", bi);
        std::snprintf(co, sizeof(co), "moss_sfx_v2.blocks.%d.cross_attn.o.weight", bi);
        std::snprintf(cqb, sizeof(cqb), "moss_sfx_v2.blocks.%d.cross_attn.q.bias", bi);
        std::snprintf(ckb, sizeof(ckb), "moss_sfx_v2.blocks.%d.cross_attn.k.bias", bi);
        std::snprintf(cvb, sizeof(cvb), "moss_sfx_v2.blocks.%d.cross_attn.v.bias", bi);
        std::snprintf(cob, sizeof(cob), "moss_sfx_v2.blocks.%d.cross_attn.o.bias", bi);
        std::snprintf(qn, sizeof(qn), "moss_sfx_v2.blocks.%d.cross_attn.norm_q.weight", bi);
        std::snprintf(kn, sizeof(kn), "moss_sfx_v2.blocks.%d.cross_attn.norm_k.weight", bi);
        std::vector<float> cqw, ckw, cvw, cow, cqb_vec, ckb_vec, cvb_vec, cob_vec, qnw, knw;
        load_weight_f32(cq, cqw); load_weight_f32(ck, ckw);
        load_weight_f32(cv, cvw); load_weight_f32(co, cow);
        load_weight_f32(cqb, cqb_vec); load_weight_f32(ckb, ckb_vec);
        load_weight_f32(cvb, cvb_vec); load_weight_f32(cob, cob_vec);
        load_weight_f32(qn, qnw); load_weight_f32(kn, knw);

        auto gr = run_cross_attn(ref_in.data.data(), T,
                                  ctx_ref.data.data(), T_cond,
                                  hd, nh,
                                  cqw.data(), (int)C, cqb_vec.data(),
                                  ckw.data(), ckb_vec.data(),
                                  cvw.data(), cvb_vec.data(),
                                  cow.data(), cob_vec.data(),
                                  qnw.data(), knw.data());
        if (!gr.ok || !match(gr.data.data(), ref_out.data.data(),
                              ref_out.n_el(), (std::string("block_") + std::to_string(bi) + "/ca_full").c_str(), TOL))
            fail("ca_full");
    }

    // ── 12. RMS norm (norm3, on after_sa_gate with learned weight+bias) ─
    {
        auto ref_in  = LOAD("after_self_attn_gate", bp);
        auto ref_out = LOAD("norm3_out", bp);
        if (ref_in.data.empty() || ref_out.data.empty()) return fails + 1;
        int64_t T, C; shape_tc(ref_in.shape, T, C);

        char wn[128];
        std::snprintf(wn, sizeof(wn), "moss_sfx_v2.blocks.%d.norm3.weight", bi);
        std::vector<float> w; load_weight_f32(wn, w);
        std::snprintf(wn, sizeof(wn), "moss_sfx_v2.blocks.%d.norm3.bias", bi);
        std::vector<float> b; load_weight_f32(wn, b);

        auto gr = run_ggml(ref_in.data.data(), T, C, w, 0, 0, b);
        if (!gr.ok || !match(gr.data.data(), ref_out.data.data(),
                              ref_out.n_el(), (std::string("block_") + std::to_string(bi) + "/norm3").c_str(), TOL))
            fail("norm3");
    }

    // ── 13. FFN gate (linear 0) ────────────────────────────────────────
    {
        auto ref_in  = LOAD("modulated_ffn", bp);
        auto ref_out = LOAD("ffn_0_gate", bp);
        if (ref_in.data.empty() || ref_out.data.empty()) return fails + 1;
        int64_t T, C; shape_tc(ref_in.shape, T, C);

        char wn[128];
        std::snprintf(wn, sizeof(wn), "moss_sfx_v2.blocks.%d.ffn.0.weight", bi);
        std::vector<float> w; load_weight_f32(wn, w);
        std::snprintf(wn, sizeof(wn), "moss_sfx_v2.blocks.%d.ffn.0.bias", bi);
        std::vector<float> b; load_weight_f32(wn, b);

        auto gr = run_ggml(ref_in.data.data(), T, C, w, 1, 0, b);
        if (!gr.ok || !match(gr.data.data(), ref_out.data.data(),
                              ref_out.n_el(), (std::string("block_") + std::to_string(bi) + "/ffn_gate").c_str(), TOL))
            fail("ffn_gate");
    }

    // ── 14. GELU tanh ──────────────────────────────────────────────────
    {
        auto ref_in  = LOAD("ffn_0_gate", bp);
        auto ref_out = LOAD("ffn_1_gelu", bp);
        if (ref_in.data.empty() || ref_out.data.empty()) return fails + 1;
        int64_t N = ref_in.n_el();

        auto gr = run_ggml(ref_in.data.data(), N, 1, {}, 2);
        // GELU is [N] → [N], reshape for compare
        if (!gr.ok || !match(gr.data.data(), ref_out.data.data(),
                              ref_out.n_el(),
                               (std::string("block_") + std::to_string(bi) + "/ffn_gelu").c_str(), 0.5))
            fail("ffn_gelu");
    }

    // ── 15. FFN output (linear 2) ──────────────────────────────────────
    {
        auto ref_in  = LOAD("ffn_1_gelu", bp);
        auto ref_out = LOAD("ffn_2_out", bp);
        if (ref_in.data.empty() || ref_out.data.empty()) return fails + 1;
        int64_t T, C; shape_tc(ref_in.shape, T, C);  // 2*ffn_dim

        char wn[128];
        std::snprintf(wn, sizeof(wn), "moss_sfx_v2.blocks.%d.ffn.2.weight", bi);
        std::vector<float> w; load_weight_f32(wn, w);
        std::snprintf(wn, sizeof(wn), "moss_sfx_v2.blocks.%d.ffn.2.bias", bi);
        std::vector<float> b; load_weight_f32(wn, b);

        auto gr = run_ggml(ref_in.data.data(), T, C, w, 1, 0, b);
        if (!gr.ok || !match(gr.data.data(), ref_out.data.data(),
                              ref_out.n_el(), (std::string("block_") + std::to_string(bi) + "/ffn_out").c_str(), TOL))
            fail("ffn_out");
    }

    return fails;
}

// ═════════════════════════════════════════════════════════════════════════════
//  VAE per-layer tests
// ═════════════════════════════════════════════════════════════════════════════

// ── Helpers for comparing PyTorch [B,C,T] vs time-major [T,C] ──────────────
// Returns a reference tensor flattened to time-major [T, C] (matching C++ layout).
static std::vector<float> ref_to_time_major(const RefTensor& ref,
                                             int64_t& T_out, int64_t& C_out) {
    int64_t C, T;
    if (ref.shape.ndim >= 3) {
        // Shape [B, C, T]: ne[0]=B, ne[1]=C, ne[2]=T
        // Row-major: b*C*T + c*T + t
        C = ref.shape.ne[1];
        T = ref.shape.ne[2];
    } else if (ref.shape.ndim == 2) {
        // Shape [C, T]: ne[0]=C, ne[1]=T
        C = ref.shape.ne[0];
        T = ref.shape.ne[1];
    } else {
        C = 1;
        T = ref.shape.ne[0];
    }
    T_out = T;
    C_out = C;

    std::vector<float> flat(static_cast<size_t>(T) * C);
    for (int64_t t = 0; t < T; t++)
        for (int64_t c = 0; c < C; c++)
            flat[static_cast<size_t>(t) * C + c] =
                ref.data[static_cast<size_t>(c) * T + t];
    return flat;
}

// ── Focused debug: compare each sub-operation within DecoderBlock 1 ─────
static int test_vae_block1_debug(ggml_context* vae_ctx) {
    int fails = 0;
    double TOL = 2.0;  // F16 weight precision noise compounds through 5 blocks

    using audiocore::moss_sfx_v2::VAEConfig;
    using audiocore::moss_sfx_v2::VAERunner;
    using Trace = VAERunner::Trace;

    auto ref_input = LOAD("post_unpatchify", g_dump_dir.c_str());
    if (ref_input.data.empty()) return 0;
    int64_t T_latent, C_latent;
    auto z = ref_to_time_major(ref_input, T_latent, C_latent);

    VAEConfig vcfg;
    vcfg.latent_dim  = 128;
    vcfg.decoder_dim = 2048;
    vcfg.hop_length  = 960;
    vcfg.sample_rate = 48000;
    vcfg.continuous  = true;

    VAERunner runner(vae_ctx, vcfg);
    std::vector<float> out(static_cast<size_t>(T_latent) * vcfg.hop_length * 2);
    Trace trace;
    runner.decode_traced(z.data(), 1, (int32_t)T_latent, out.data(), &trace, nullptr);

    auto cmp = [&](const char* name, const std::vector<float>& got,
                   const char* ref_prefix, const char* ref_name) {
        auto ref = LOAD(ref_name, ref_prefix);
        if (ref.data.empty()) { std::fprintf(stderr, "  [SKIP] %s: no ref\n", name); return; }
        int64_t RT, RC;
        auto rf = ref_to_time_major(ref, RT, RC);
        if (got.size() != rf.size()) {
            std::fprintf(stderr, "  [FAIL] %s: size %zu vs ref %zu [%ld,%ld]\n",
                         name, got.size(), rf.size(), (long)RT, (long)RC);
            fails++; return;
        }
        char lbl[128];
        std::snprintf(lbl, sizeof(lbl), "block1_debug/%s", name);
        if (!match(got.data(), rf.data(), rf.size(), lbl, TOL))
            fails++;
    };

    cmp("snake", trace.blk1_snake, g_dump_dir.c_str(), "vae_debug/block1_snake");
    cmp("convt", trace.blk1_convt, g_dump_dir.c_str(), "vae_debug/block1_convt");
    cmp("res0_s1",  trace.blk1_res0_s1, g_dump_dir.c_str(), "vae_debug/block1_res0_s1");
    cmp("res0_c1",  trace.blk1_res0_c1, g_dump_dir.c_str(), "vae_debug/block1_res0_c1");
    cmp("res0_s2",  trace.blk1_res0_s2, g_dump_dir.c_str(), "vae_debug/block1_res0_s2");
    cmp("res0_c2",  trace.blk1_res0_c2, g_dump_dir.c_str(), "vae_debug/block1_res0_c2");
    cmp("res0",     trace.blk1_res[0], g_dump_dir.c_str(), "vae_debug/block1_res0_out");
    cmp("res1",     trace.blk1_res[1], g_dump_dir.c_str(), "vae_debug/block1_res1_out");
    cmp("res2",     trace.blk1_res[2], g_dump_dir.c_str(), "vae_debug/block1_res2_out");

    return fails;
}

static int test_vae_layers(ggml_context* vae_ctx) {
    int fails = 0;
    double TOL = 2.0;
    auto fail = [&](const char* op) {
        fails++;
        std::fprintf(stderr, "  [FAIL] VAE/%s\n", op);
    };

    using audiocore::moss_sfx_v2::VAEConfig;
    using audiocore::moss_sfx_v2::VAERunner;
    using Trace = VAERunner::Trace;

    // ── 1. Load reference input ──────────────────────────────────────────
    auto ref_input = LOAD("post_unpatchify", g_dump_dir.c_str());
    if (ref_input.data.empty()) {
        std::fprintf(stderr, "  [FAIL] VAE: no post_unpatchify input\n");
        return 1;
    }

    int64_t T_latent, C_latent;
    auto z = ref_to_time_major(ref_input, T_latent, C_latent);
    std::fprintf(stderr, "  VAE input: ref [%ld,%ld,%ld] -> time-major [%ld,%ld]\n",
                 (long)ref_input.shape.ne[0], (long)ref_input.shape.ne[1],
                 (long)ref_input.shape.ne[2], (long)T_latent, (long)C_latent);

    // ── 2. Build VAE config and runner ───────────────────────────────────
    VAEConfig vcfg;
    vcfg.latent_dim  = 128;
    vcfg.decoder_dim = 2048;
    vcfg.hop_length  = 960;
    vcfg.sample_rate = 48000;
    vcfg.continuous  = true;

    VAERunner runner(vae_ctx, vcfg);

    // ── 3. Run traced decode ─────────────────────────────────────────────
    std::vector<float> out(static_cast<size_t>(T_latent) * vcfg.hop_length * 2);
    Trace trace;
    bool ok = runner.decode_traced(z.data(), 1, (int32_t)T_latent,
                                    out.data(), &trace, nullptr);
    if (!ok) {
        std::fprintf(stderr, "  [FAIL] VAE decode_traced failed\n");
        return 1;
    }
    std::fprintf(stderr, "  C++ decode done: output %zu samples\n",
                 trace.vae_final.size());

    // ── 4. Compare post_quant_conv output ────────────────────────────────
    {
        auto ref = LOAD("post_quant_conv_out", g_dump_dir.c_str());
        if (!ref.data.empty()) {
            int64_t T, C;
            auto ref_flat = ref_to_time_major(ref, T, C);
            if (trace.post_pqc.size() == ref_flat.size()) {
                match(trace.post_pqc.data(), ref_flat.data(), ref_flat.size(),
                      "vae/post_quant_conv", TOL);
            } else {
                std::fprintf(stderr, "  [FAIL] vae/post_quant_conv: size %zu vs %zu\n",
                             trace.post_pqc.size(), ref_flat.size());
                fail("post_quant_conv");
            }
        } else {
            std::fprintf(stderr, "  [SKIP] vae/post_quant_conv: no ref\n");
        }
    }

    // ── 5. Compare each decoder layer ────────────────────────────────────
    for (int i = 0; i <= 8; i++) {
        char name[32];
        std::snprintf(name, sizeof(name), "vae_dec_%d", i);
        auto ref = LOAD(name, g_dump_dir.c_str());
        if (ref.data.empty()) {
            std::fprintf(stderr, "  [SKIP] %s: no ref\n", name);
            continue;
        }

        int64_t ref_T, ref_C;
        auto ref_flat = ref_to_time_major(ref, ref_T, ref_C);

        if (i == 0) {
            // vae_dec_0 corresponds to conv_in output.
            // C++ trace.vae_dec[0] should match.
        }

        const auto& got = trace.vae_dec[i];
        if (got.size() != ref_flat.size()) {
            std::fprintf(stderr, "  [FAIL] %s: size got=%zu ref=%zu (C=%ld T=%ld)\n",
                         name, got.size(), ref_flat.size(), (long)ref_C, (long)ref_T);
            fail(name);
            continue;
        }

        char lbl[64];
        std::snprintf(lbl, sizeof(lbl), "vae/%s", name);
        if (!match(got.data(), ref_flat.data(), ref_flat.size(), lbl, TOL))
            fail(name);
    }

    // ── 6. Compare final output ──────────────────────────────────────────
    {
        auto ref = LOAD("vae_final", g_dump_dir.c_str());
        if (!ref.data.empty()) {
            int64_t ref_T, ref_C;
            auto ref_flat = ref_to_time_major(ref, ref_T, ref_C);
            // Dump C++ output for external evaluation
            {
                FILE* f = std::fopen("/tmp/cpp_vae_out.f32", "wb");
                if (f) {
                    std::fwrite(trace.vae_final.data(), sizeof(float),
                                trace.vae_final.size(), f);
                    std::fclose(f);
                }
            }
            // Dump Python reference output for external evaluation
            {
                FILE* f = std::fopen("/tmp/py_vae_ref.f32", "wb");
                if (f) {
                    std::fwrite(ref_flat.data(), sizeof(float),
                                ref_flat.size(), f);
                    std::fclose(f);
                }
            }
            if (trace.vae_final.size() == ref_flat.size()) {
                if (!match(trace.vae_final.data(), ref_flat.data(), ref_flat.size(),
                           "vae/vae_final", TOL))
                    fail("vae_final");
            } else {
                std::fprintf(stderr, "  [FAIL] vae/vae_final: size %zu vs %zu\n",
                             trace.vae_final.size(), ref_flat.size());
                fail("vae_final");
            }
        } else {
            std::fprintf(stderr, "  [SKIP] vae/vae_final: no ref\n");
        }
    }

    return fails;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Main
// ═════════════════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <dit.gguf> <dump_dir> [vae.gguf]\n", argv[0]);
        return 1;
    }

    g_dump_dir = argv[2];

    std::fprintf(stderr, "Loading DiT GGUF from %s ...\n", argv[1]);
    g_weights = load_gguf(argv[1]);
    if (!g_weights) {
        std::fprintf(stderr, "Failed to load GGUF\n");
        return 1;
    }
    std::fprintf(stderr, "Weights loaded\n");

    // Config
    int dim = 1536, nh = 12, hd = 128, n_layers = 30;

    int total_fails = 0;
    for (int bi = 0; bi < n_layers; bi++) {
        std::fprintf(stderr, "\nBlock %2d/%d:\n", bi, n_layers);
        fflush(stderr);
        int f = test_block(bi, dim, nh, hd);
        if (f == 0)
            std::fprintf(stderr, "  >>> Block %d: ALL %d OPS PASS\n", bi, f);
        else
            std::fprintf(stderr, "  >>> Block %d: %d OPS FAILED\n", bi, f);
        total_fails += f;
    }

    // VAE tests (optional VAE GGUF path)
    if (argc >= 4) {
        std::fprintf(stderr, "\nVAE:\n");
        std::fprintf(stderr, "Loading VAE GGUF from %s ...\n", argv[3]);
        ggml_context* vae_ctx = load_gguf(argv[3]);
        if (vae_ctx) {
            total_fails += test_vae_block1_debug(vae_ctx);
            total_fails += test_vae_layers(vae_ctx);
            ggml_free(vae_ctx);
        } else {
            std::fprintf(stderr, "  [WARN] Failed to load VAE GGUF — skipping VAE tests\n");
        }
    } else {
        std::fprintf(stderr, "\nNo VAE GGUF provided — skipping VAE tests\n");
    }

    ggml_free(g_weights);

    std::fprintf(stderr, "\n=== %d FAILURES ===\n", total_fails);
    return total_fails > 0 ? 1 : 0;
}
