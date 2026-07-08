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
// NOTE: vanilla CFG formula = u + g*(c-u). Upstream uses normalized_guidance
// with pred_cond base + orthogonal update + norm_threshold=2.5. The vanilla
// formula over-amplifies when diff is large (FIXME: upgrade to match upstream).

#include "audiocore/models/ace_step/dit_runner.h"
#include "audiocore/framework/ggml/backend_helper.h"

#include "ggml.h"
#include "ggml-alloc.h"  // ggml_gallocr — lifecycle-based graph allocation
#include "ggml-cpu.h"   // ggml_new_f32 (used by scheduler fallback paths)

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
// Input:  [nh*hd, T]  in ggml ne: ne[0]=nh*hd, ne[1]=T
// Output: [nh*hd, T]  same shape
//
// Matches upstream's dit-graph.h pattern exactly:
//   t = reshape_4d(t, hd, nh, T, 1)     # [hd, nh, T, 1]
//   t = rms_norm(t, eps)                # normalizes over ne[0]=hd per (nh, T)
//   t = mul(t, norm_w)                  # norm_w is [hd], broadcast via ggml_mul
//   return reshape_2d(t, nh*hd, T)
static ggml_tensor* apply_qk_norm(ggml_context* ctx, ggml_tensor* t,
                                    int hd, int nh,
                                    ggml_tensor* norm_w) {
    if (!norm_w) return t;
    // Compute T from total elements: input is [nh*hd, T]
    int64_t T = ggml_nelements(t) / (static_cast<int64_t>(hd) * nh);
    // Reshape to 4D [hd, nh, T, 1] for per-head norm. Using 4D matches
    // upstream's pattern and avoids any 3D rms_norm edge cases.
    ggml_tensor* t4d = ggml_reshape_4d(ctx, t, hd, nh, T, 1);
    t4d = ggml_rms_norm(ctx, t4d, 1e-6f);
    // norm_w is [hd] (1D) — ggml_mul broadcasts it across the other dims.
    // Upstream does this directly without ggml_repeat.
    t4d = ggml_mul(ctx, t4d, norm_w);
    return ggml_reshape_2d(ctx, t4d, nh * hd, T);
}

// Helper: mark a tensor for dumping. Wraps in ggml_cont() to force a unique
// data buffer (without this, the ggml scheduler can reuse the buffer for later
// tensors, causing the dump to read stale/overwritten data). Then names and
// flags it as an output so dump_named() can find it after sched_graph_compute.
static ggml_tensor* dump_mark(ggml_context* ctx, ggml_tensor* t,
                                const char* name) {
    t = ggml_cont(ctx, t);
    ggml_set_name(t, name);
    ggml_set_output(t);
    return t;
}

// ── Input upload record (tensor + cpu data to copy into it after alloc) ────
struct DiTInputUpload { ggml_tensor* t; const void* data; size_t nbytes; };

// ── Self-attention (GQA + QK-norm + RoPE) ───────────────────────────────────
// Returns just the attention (or O-projected) output — NO residual added.
// When `dump_layer0` is true, names intermediate tensors for layer 0 to enable
// layer-by-layer comparison with upstream (matches upstream's layer0_* dumps).
static ggml_tensor* self_attn(ggml_context* ctx, ggml_tensor* x,
                                int T, int H, int nh, int nk, int hd,
                                ggml_tensor* q_w, ggml_tensor* k_w,
                                ggml_tensor* v_w, ggml_tensor* o_w,
                                ggml_tensor* q_norm_w, ggml_tensor* k_norm_w,
                                float rope_theta,
                                std::vector<DiTInputUpload>& inputs,
                                bool dump_layer0 = false) {
    if (!q_w || !k_w || !v_w) return nullptr;

    auto q = ggml_mul_mat(ctx, q_w, x);
    auto k = ggml_mul_mat(ctx, k_w, x);
    auto v = ggml_mul_mat(ctx, v_w, x);

    // Debug: dump raw Q/K/V (post mul_mat, pre-QK-norm) for layer 0.
    if (dump_layer0) {
        q = dump_mark(ctx, q, "layer0_q_raw_after_proj");
        k = dump_mark(ctx, k, "layer0_k_raw_after_proj");
    }

    // QK-norm (per-head RMS norm before RoPE)
    q = apply_qk_norm(ctx, q, hd, nh, q_norm_w);
    k = apply_qk_norm(ctx, k, hd, nk, k_norm_w);

    // Debug dumps: Q/K AFTER QK-norm but BEFORE RoPE.
    // At RoPE position 0 the rotation is identity, so post-rope values at
    // position 0 must equal these. Comparing against upstream isolates whether
    // divergence is in (q_proj + qk_norm) vs in RoPE itself.
    if (dump_layer0) {
        q = dump_mark(ctx, q, "layer0_q_after_qknorm");
        k = dump_mark(ctx, k, "layer0_k_after_qknorm");
        v = dump_mark(ctx, v, "layer0_v_raw");
    }

    // Reshape to 3D [hd, n_heads, T] before RoPE (ggml_rope_ext requires 3D)
    auto to_3d = [&](ggml_tensor* t, int n_h) {
        return ggml_reshape_3d(ctx, t, hd, n_h, T);
    };
    q = to_3d(q, nh);
    k = to_3d(k, nk);
    v = to_3d(v, nk);

    // Build position IDs [0, 1, 2, ..., T-1] for RoPE — uploaded as input.
    ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_input(pos);
    {
        static thread_local std::vector<int32_t> pos_buf;
        pos_buf.resize(static_cast<size_t>(T));
        for (int i = 0; i < T; i++) pos_buf[i] = i;
        inputs.push_back({pos, pos_buf.data(),
                          static_cast<size_t>(T) * sizeof(int32_t)});
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

    // Upstream names the post-rope Q and K for layer 0. These are 3D [hd, nh, T]
    // tensors; upstream dumps the first sample slice (= hd*nh elements).
    if (dump_layer0) {
        q = dump_mark(ctx, q, "layer0_q_after_rope");
        k = dump_mark(ctx, k, "layer0_k_after_rope");
    }

    // Permute for flash_attn_ext → [hd, T, n_heads]
    auto rsh = [&](ggml_tensor* t, int n_h) {
        return ggml_cont(ctx, ggml_permute(ctx, t, 0, 2, 1, 3));
    };
    q = rsh(q, nh);
    k = rsh(k, nk);
    v = rsh(v, nk);

    float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    auto a = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(a, GGML_PREC_F32);

    // flash_attn_ext returns [hd, nh, T, 1] — reshape directly to [nh*hd, T].
    // Do NOT permute first: that would scramble head/position mapping (the
    // innermost hd varies faster than nh, which is exactly what reshape_2d
    // expects to combine into nh*hd along ne[0]). Matches upstream's
    // ggml_reshape_3d(attn, Nh*D, S, N).
    a = ggml_reshape_2d(ctx, a, nh * hd, T);

    // Upstream names the raw attention output (pre-o_proj) for layer 0.
    if (dump_layer0) {
        a = dump_mark(ctx, a, "layer0_attn_out");
    }

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
    ggml_flash_attn_ext_set_prec(a, GGML_PREC_F32);

    // flash_attn_ext returns [hd, nh, T, 1] — reshape directly to [nh*hd, T].
    // NO permute (see self_attn comment): permute scrambles the head/T mapping.
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

DiTRunner::~DiTRunner() {
    if (sched_) {
        ggml_backend_sched_free(sched_);
        sched_ = nullptr;
    }
    backend_pair_.reset();
}

ggml_tensor* DiTRunner::weight(const char* name) const {
    return ggml_get_tensor(ext_ctx_, name);
}

bool DiTRunner::ensure_backend() {
    if (backend_ready_) return cuda_backend_ != nullptr;

    namespace bu = audiocore::ggml_utils;
    backend_pair_ = std::make_unique<bu::BackendPair>(bu::backend_init("DiT"));
    if (!backend_pair_ || !backend_pair_->backend) {
        return false;
    }
    cuda_backend_ = backend_pair_->backend;
    backend_ready_ = true;

    // Migrate weight tensors from GGUF mmap → CUDA buffer.
    // We bypass the scheduler (it forces INPUT-flagged tensors to CPU) and
    // use ggml_gallocr for lifecycle-based allocation directly on CUDA.
    if (backend_pair_->has_gpu) {
        ggml_backend_buffer_t buf =
            bu::migrate_ctx_to_backend(ext_ctx_, cuda_backend_, "DiT");
    }
    return true;
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

// (struct DiTInputUpload defined earlier, before self_attn)

// ── Timbre encoder: refer_audio [T_refer, 64] → [1, 2048] CLS-pooled token ──
// 4-layer pre-LN transformer with GQA + QK-norm + RoPE.
// Matches encoder.timbre_encoder in ace_step/modeling_ace_step.py:325.
// Bidirectional attention (NOT causal); layer_types alternate
// [sliding, full, sliding, full]. Upstream cond-enc.h:266 applies a
// sliding-window mask (|i-j| <= 128) on EVEN layers and full attention
// on ODD layers. For text2music the caller passes T_refer=1 (a single
// silence frame per upstream ops_encode_timbre), so the sliding mask
// degenerates to a single unmasked token and attention is identity.
static ggml_tensor* timbre_encode(ggml_context* ctx,
                                    ggml_context* ext_ctx,
                                    const float* refer_audio,
                                    int T_refer, int timbre_dim,
                                    int H, int nh, int nk, int hd,
                                    float rope_theta,
                                    std::vector<DiTInputUpload>& inputs) {
    constexpr int kNumTimbreLayers = 4;

    // embed_tokens: Linear(timbre_dim=64 → H=2048) with bias.
    // Validated at load time — no null check needed.
    auto et_w = ggml_get_tensor(ext_ctx, "encoder.timbre_encoder.embed_tokens.weight");
    auto et_b = ggml_get_tensor(ext_ctx, "encoder.timbre_encoder.embed_tokens.bias");

    // Load refer_audio [timbre_dim, T_refer]
    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, timbre_dim, T_refer);
    ggml_set_name(input, "timbre_in");
    ggml_set_input(input);
    inputs.push_back({input, refer_audio,
                      static_cast<size_t>(timbre_dim) * T_refer * sizeof(float)});

    // Project to H: weight [H, 64] @ x [64, T] → [H, T]
    ggml_tensor* hidden = ggml_mul_mat(ctx, et_w, input);
    if (et_b) hidden = ggml_add(ctx, hidden, ggml_repeat(ctx, et_b, hidden));

    // Position IDs [0..T_refer-1] — built on CPU then uploaded as input.
    ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_refer);
    ggml_set_name(pos, "timbre_pos");
    ggml_set_input(pos);
    {
        // Build position data on CPU stack then queue for upload.
        static thread_local std::vector<int32_t> pos_buf;
        pos_buf.resize(static_cast<size_t>(T_refer));
        for (int i = 0; i < T_refer; i++) pos_buf[i] = i;
        inputs.push_back({pos, pos_buf.data(),
                          static_cast<size_t>(T_refer) * sizeof(int32_t)});
    }

    // Sliding-window attention mask (matches upstream cond-enc.h:342-352).
    // Bidirectional window of size 128: position i may attend to j when |i-j| <= 128.
    // Applied to EVEN layers (i % 2 == 0); ODD layers use full attention.
    // Trains-distribution requirement: the timbre encoder was trained with
    // this mask. Using full attention for all layers produces OOD behaviour
    // and steers the DiT toward industrial/machinery output.
    constexpr int kSlideWindow = 128;
    ggml_tensor* slide_mask = nullptr;
    static thread_local std::vector<ggml_fp16_t> mask_buf;
    if (T_refer > kSlideWindow) {
        slide_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, T_refer, T_refer);
        ggml_set_name(slide_mask, "timbre_slide_mask");
        ggml_set_input(slide_mask);
        const ggml_fp16_t kZero = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t kNegInf = ggml_fp32_to_fp16(-INFINITY);
        mask_buf.assign(static_cast<size_t>(T_refer) * T_refer, kZero);
        for (int i = 0; i < T_refer; i++) {
            for (int j = 0; j < T_refer; j++) {
                int d = i - j; if (d < 0) d = -d;
                mask_buf[static_cast<size_t>(i) * T_refer + j] =
                    (d <= kSlideWindow) ? kZero : kNegInf;
            }
        }
        inputs.push_back({slide_mask, mask_buf.data(),
                          static_cast<size_t>(T_refer) * T_refer * sizeof(ggml_fp16_t)});
    }

    // 4 transformer layers
    for (int i = 0; i < kNumTimbreLayers; i++) {
        char buf[160];

        // ── pre-LN self-attention ──
        std::snprintf(buf, sizeof(buf),
                      "encoder.timbre_encoder.layers.%d.input_layernorm.weight", i);
        ggml_tensor* ln_w = ggml_get_tensor(ext_ctx, buf);
        ggml_tensor* h = ggml_rms_norm(ctx, hidden, 1e-6f);
        h = ggml_mul(ctx, h, ggml_repeat(ctx, ln_w, h));

        // Self-attn projections (GQA: q has nh heads, kv has nk heads)
        std::snprintf(buf, sizeof(buf),
                      "encoder.timbre_encoder.layers.%d.self_attn.q_proj.weight", i);
        auto q_w = ggml_get_tensor(ext_ctx, buf);
        std::snprintf(buf, sizeof(buf),
                      "encoder.timbre_encoder.layers.%d.self_attn.k_proj.weight", i);
        auto k_w = ggml_get_tensor(ext_ctx, buf);
        std::snprintf(buf, sizeof(buf),
                      "encoder.timbre_encoder.layers.%d.self_attn.v_proj.weight", i);
        auto v_w = ggml_get_tensor(ext_ctx, buf);
        std::snprintf(buf, sizeof(buf),
                      "encoder.timbre_encoder.layers.%d.self_attn.o_proj.weight", i);
        auto o_w = ggml_get_tensor(ext_ctx, buf);
        std::snprintf(buf, sizeof(buf),
                      "encoder.timbre_encoder.layers.%d.self_attn.q_norm.weight", i);
        auto qn_w = ggml_get_tensor(ext_ctx, buf);
        std::snprintf(buf, sizeof(buf),
                      "encoder.timbre_encoder.layers.%d.self_attn.k_norm.weight", i);
        auto kn_w = ggml_get_tensor(ext_ctx, buf);

        // All timbre encoder weights validated at load time — no null checks.
        auto q = ggml_mul_mat(ctx, q_w, h);
        auto k = ggml_mul_mat(ctx, k_w, h);
        auto v = ggml_mul_mat(ctx, v_w, h);

        q = apply_qk_norm(ctx, q, hd, nh, qn_w);
        k = apply_qk_norm(ctx, k, hd, nk, kn_w);

        // Reshape 3D for RoPE: [hd, n_heads, T]
        q = ggml_reshape_3d(ctx, q, hd, nh, T_refer);
        k = ggml_reshape_3d(ctx, k, hd, nk, T_refer);
        v = ggml_reshape_3d(ctx, v, hd, nk, T_refer);

        // RoPE on Q and K
        auto rope = [&](ggml_tensor* t) {
            return ggml_rope_ext(ctx, t, pos, nullptr,
                                  hd, 2, 0,
                                  rope_theta, 1.0f, 0.0f, 1.0f,
                                  0.0f, 0.0f);
        };
        q = rope(q);
        k = rope(k);

        // Permute → [hd, T, n_heads] for flash_attn_ext
        auto rsh = [&](ggml_tensor* t, int n_h) {
            return ggml_cont(ctx, ggml_permute(ctx, t, 0, 2, 1, 3));
        };
        q = rsh(q, nh);
        k = rsh(k, nk);
        v = rsh(v, nk);

        float scale = 1.0f / std::sqrt(static_cast<float>(hd));
        // Even layers use sliding window mask; odd layers use full attention.
        // Matches upstream cond-enc.h:266 layer_mask = (i % 2 == 0) ? slide : NULL.
        ggml_tensor* layer_mask = (slide_mask && (i % 2 == 0)) ? slide_mask : nullptr;
        auto a = ggml_flash_attn_ext(ctx, q, k, v, layer_mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(a, GGML_PREC_F32);
        // flash_attn_ext returns [hd, nh, T, 1] — reshape directly (NO permute).
        a = ggml_reshape_2d(ctx, a, nh * hd, T_refer);
        a = ggml_mul_mat(ctx, o_w, a);
        hidden = ggml_add(ctx, hidden, a);   // residual

        // ── post-LN SwiGLU MLP ──
        std::snprintf(buf, sizeof(buf),
                      "encoder.timbre_encoder.layers.%d.post_attention_layernorm.weight", i);
        ggml_tensor* post_ln_w = ggml_get_tensor(ext_ctx, buf);
        h = ggml_rms_norm(ctx, hidden, 1e-6f);
        h = ggml_mul(ctx, h, ggml_repeat(ctx, post_ln_w, h));

        std::snprintf(buf, sizeof(buf),
                      "encoder.timbre_encoder.layers.%d.mlp.gate_proj.weight", i);
        auto gw = ggml_get_tensor(ext_ctx, buf);
        std::snprintf(buf, sizeof(buf),
                      "encoder.timbre_encoder.layers.%d.mlp.up_proj.weight", i);
        auto uw = ggml_get_tensor(ext_ctx, buf);
        std::snprintf(buf, sizeof(buf),
                      "encoder.timbre_encoder.layers.%d.mlp.down_proj.weight", i);
        auto dw = ggml_get_tensor(ext_ctx, buf);
        auto mlp_out = swiglu_mlp(ctx, h, gw, uw, dw);
        hidden = ggml_add(ctx, hidden, mlp_out);
    }

    // Final RMSNorm (validated at load time).
    auto final_ln_w = ggml_get_tensor(ext_ctx, "encoder.timbre_encoder.norm.weight");
    hidden = ggml_rms_norm(ctx, hidden, 1e-6f);
    hidden = ggml_mul(ctx, hidden, ggml_repeat(ctx, final_ln_w, hidden));

    // CLS pooling: take first token → [H, 1]
    // hidden is [H, T_refer] → view 1d first column
    ggml_tensor* cls = ggml_view_1d(ctx, hidden, H, 0);   // offset 0, length H
    cls = ggml_reshape_2d(ctx, cls, H, 1);                // [H, 1]

    // Prepend the learned special_token ([H]) — upstream prepends it as the
    // first sequence position before pooling takes [:, 0, :]. Effectively, the
    // special_token IS the CLS representation. We add it as an additional
    // conditioning token (more conservative — keeps the pooled representation
    // distinct from the learned special token).
    // NOTE: upstream timbre_encoder returns (B, 1, H) — just the pooled CLS.
    // We return [H, 1] (single token). The special_token is not added here.
    return cls;
}

// ── Lyric encoder ────────────────────────────────────────────────────────────
// 8-layer bidirectional transformer that processes raw TE token embeddings
// for the lyric conditioning path. Matches encoder.lyric_encoder in the DiT
// GGUF. Input: [1024, S_lyric] (raw TE embeddings via embed_lookup). Output:
// [2048, S_lyric].
//
// Architecture (cond-enc.h:188-215):
//   1. Linear embed: [1024, S] → [2048, S] (embed_tokens.weight + bias)
//   2. 8 transformer layers (alternating sliding-window/full attention)
//   3. Final RMS norm
//
// The lyric encoder ALWAYS runs, even for text2music without lyrics —
// upstream encodes "# Languages\nunknown\n\n# Lyric\n<|endoftext|>" as the
// default lyric string, producing ~11 tokens of conditioning. Without this,
// the cross-attention has 2 fewer tokens (wrong sequence length), corrupting
// the K/V projections and causing the "frantic clicking" output.
static ggml_tensor* lyric_encode(ggml_context* ctx,
                                   ggml_context* ext_ctx,
                                   const float* lyric_data,
                                   int S_lyric, int te_hidden,
                                   int H, int nh, int nk, int hd,
                                   float rope_theta,
                                   std::vector<DiTInputUpload>& inputs) {
    constexpr int kNumLyricLayers = 8;

    auto el_w = ggml_get_tensor(ext_ctx, "encoder.lyric_encoder.embed_tokens.weight");
    auto el_b = ggml_get_tensor(ext_ctx, "encoder.lyric_encoder.embed_tokens.bias");

    // Load lyric_data [te_hidden, S_lyric]
    ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, te_hidden, S_lyric);
    ggml_set_name(input, "lyric_in");
    ggml_set_input(input);
    inputs.push_back({input, lyric_data,
                      static_cast<size_t>(te_hidden) * S_lyric * sizeof(float)});

    // Project to H: weight [H, te_hidden] @ x [te_hidden, S] → [H, S]
    ggml_tensor* hidden = ggml_mul_mat(ctx, el_w, input);
    if (el_b) hidden = ggml_add(ctx, hidden, ggml_repeat(ctx, el_b, hidden));

    // Position IDs [0..S_lyric-1]
    ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S_lyric);
    ggml_set_name(pos, "lyric_pos");
    ggml_set_input(pos);
    {
        static thread_local std::vector<int32_t> pos_buf;
        pos_buf.resize(static_cast<size_t>(S_lyric));
        for (int i = 0; i < S_lyric; i++) pos_buf[i] = i;
        inputs.push_back({pos, pos_buf.data(),
                          static_cast<size_t>(S_lyric) * sizeof(int32_t)});
    }

    // Sliding-window attention mask (bidirectional, |i-j| <= 128).
    // Applied to EVEN layers; ODD layers use full attention.
    constexpr int kSlideWindow = 128;
    ggml_tensor* slide_mask = nullptr;
    static thread_local std::vector<ggml_fp16_t> mask_buf;
    if (S_lyric > 1) {
        slide_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, S_lyric, S_lyric);
        ggml_set_name(slide_mask, "lyric_slide_mask");
        ggml_set_input(slide_mask);
        const ggml_fp16_t kZero = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t kNegInf = ggml_fp32_to_fp16(-INFINITY);
        mask_buf.assign(static_cast<size_t>(S_lyric) * S_lyric, kZero);
        for (int i = 0; i < S_lyric; i++) {
            for (int j = 0; j < S_lyric; j++) {
                int d = i - j; if (d < 0) d = -d;
                mask_buf[static_cast<size_t>(i) * S_lyric + j] =
                    (d <= kSlideWindow) ? kZero : kNegInf;
            }
        }
        inputs.push_back({slide_mask, mask_buf.data(),
                          static_cast<size_t>(S_lyric) * S_lyric * sizeof(ggml_fp16_t)});
    }

    // 8 transformer layers
    for (int i = 0; i < kNumLyricLayers; i++) {
        char buf[160];

        // ── pre-LN self-attention ──
        std::snprintf(buf, sizeof(buf),
                      "encoder.lyric_encoder.layers.%d.input_layernorm.weight", i);
        ggml_tensor* ln_w = ggml_get_tensor(ext_ctx, buf);
        ggml_tensor* h = ggml_rms_norm(ctx, hidden, 1e-6f);
        h = ggml_mul(ctx, h, ggml_repeat(ctx, ln_w, h));

        std::snprintf(buf, sizeof(buf),
                      "encoder.lyric_encoder.layers.%d.self_attn.q_proj.weight", i);
        auto q_w = ggml_get_tensor(ext_ctx, buf);
        std::snprintf(buf, sizeof(buf),
                      "encoder.lyric_encoder.layers.%d.self_attn.k_proj.weight", i);
        auto k_w = ggml_get_tensor(ext_ctx, buf);
        std::snprintf(buf, sizeof(buf),
                      "encoder.lyric_encoder.layers.%d.self_attn.v_proj.weight", i);
        auto v_w = ggml_get_tensor(ext_ctx, buf);
        std::snprintf(buf, sizeof(buf),
                      "encoder.lyric_encoder.layers.%d.self_attn.o_proj.weight", i);
        auto o_w = ggml_get_tensor(ext_ctx, buf);
        std::snprintf(buf, sizeof(buf),
                      "encoder.lyric_encoder.layers.%d.self_attn.q_norm.weight", i);
        auto qn_w = ggml_get_tensor(ext_ctx, buf);
        std::snprintf(buf, sizeof(buf),
                      "encoder.lyric_encoder.layers.%d.self_attn.k_norm.weight", i);
        auto kn_w = ggml_get_tensor(ext_ctx, buf);

        auto q = ggml_mul_mat(ctx, q_w, h);
        auto k = ggml_mul_mat(ctx, k_w, h);
        auto v = ggml_mul_mat(ctx, v_w, h);

        q = apply_qk_norm(ctx, q, hd, nh, qn_w);
        k = apply_qk_norm(ctx, k, hd, nk, kn_w);

        q = ggml_reshape_3d(ctx, q, hd, nh, S_lyric);
        k = ggml_reshape_3d(ctx, k, hd, nk, S_lyric);
        v = ggml_reshape_3d(ctx, v, hd, nk, S_lyric);

        auto rope = [&](ggml_tensor* t) {
            return ggml_rope_ext(ctx, t, pos, nullptr,
                                  hd, 2, 0,
                                  rope_theta, 1.0f, 0.0f, 1.0f,
                                  0.0f, 0.0f);
        };
        q = rope(q);
        k = rope(k);

        auto rsh = [&](ggml_tensor* t, int n_h) {
            return ggml_cont(ctx, ggml_permute(ctx, t, 0, 2, 1, 3));
        };
        q = rsh(q, nh);
        k = rsh(k, nk);
        v = rsh(v, nk);

        float scale = 1.0f / std::sqrt(static_cast<float>(hd));
        ggml_tensor* layer_mask = (slide_mask && (i % 2 == 0)) ? slide_mask : nullptr;
        auto a = ggml_flash_attn_ext(ctx, q, k, v, layer_mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(a, GGML_PREC_F32);
        a = ggml_reshape_2d(ctx, a, nh * hd, S_lyric);
        a = ggml_mul_mat(ctx, o_w, a);
        hidden = ggml_add(ctx, hidden, a);   // residual

        // ── post-LN SwiGLU MLP ──
        std::snprintf(buf, sizeof(buf),
                      "encoder.lyric_encoder.layers.%d.post_attention_layernorm.weight", i);
        ggml_tensor* post_ln_w = ggml_get_tensor(ext_ctx, buf);
        h = ggml_rms_norm(ctx, hidden, 1e-6f);
        h = ggml_mul(ctx, h, ggml_repeat(ctx, post_ln_w, h));

        std::snprintf(buf, sizeof(buf),
                      "encoder.lyric_encoder.layers.%d.mlp.gate_proj.weight", i);
        auto gw = ggml_get_tensor(ext_ctx, buf);
        std::snprintf(buf, sizeof(buf),
                      "encoder.lyric_encoder.layers.%d.mlp.up_proj.weight", i);
        auto uw = ggml_get_tensor(ext_ctx, buf);
        std::snprintf(buf, sizeof(buf),
                      "encoder.lyric_encoder.layers.%d.mlp.down_proj.weight", i);
        auto dw = ggml_get_tensor(ext_ctx, buf);
        auto mlp_out = swiglu_mlp(ctx, h, gw, uw, dw);
        hidden = ggml_add(ctx, hidden, mlp_out);
    }

    // Final RMSNorm
    auto final_ln_w = ggml_get_tensor(ext_ctx, "encoder.lyric_encoder.norm.weight");
    hidden = ggml_rms_norm(ctx, hidden, 1e-6f);
    hidden = ggml_mul(ctx, hidden, ggml_repeat(ctx, final_ln_w, hidden));

    // Return [H, S_lyric] — NO pooling, all tokens go into the conditioning.
    return hidden;
}

// ── Single forward: build graph + compute ────────────────────────────────────
static bool run_one_forward(
    const DitConfig& cfg, ggml_context* ext_ctx,
    ggml_backend_t cuda_backend,
    const float* x_t, int32_t T, int32_t H,
    const float* temb, int temb_dim,
    const float* cond_data, int ct_len, int cond_hidden,
    const float* lyric_data, int lyric_len, int lyric_hidden,
    const float* refer_audio, int T_refer,
    float* result,
    const std::vector<std::vector<float>>& ss_table_f32,
    const std::vector<float>& global_ss_f32,
    std::string* error)
{
    // Input NaN check + RMS diagnostics
    double inp_rms = 0.0;
    {
        int nx = 0, nt = 0, nc = 0;
        double sq = 0.0;
        for (int32_t i = 0; i < T * H; i++) {
            if (std::isnan(x_t[i])) nx++;
            else sq += static_cast<double>(x_t[i]) * x_t[i];
        }
        inp_rms = std::sqrt(sq / std::max(1, T * H));
        for (int i = 0; i < temb_dim; i++) if (std::isnan(temb[i])) nt++;
        if (cond_data && ct_len > 0) {
            for (int i = 0; i < ct_len * cond_hidden; i++) if (std::isnan(cond_data[i])) nc++;
        }
        if (nx || nt || nc) {
            fprintf(stderr, "[dit] WARNING input NaN: x=%d/%d temb=%d/%d cond=%d/%d\n",
                    nx, T*H, nt, temb_dim, nc, cond_data ? ct_len*cond_hidden : 0);
        }
        if (std::getenv("ACE_STEP_DIT_DEBUG")) {
            fprintf(stderr, "[dit] input RMS=%.4f T=%d H=%d\n", inp_rms, T, H);
        }
    }
    const int32_t nh      = cfg.n_heads     > 0 ? cfg.n_heads     : 24;
    const int32_t nk      = cfg.n_kv_heads  > 0 ? cfg.n_kv_heads  : 8;
    const int32_t hd      = cfg.head_dim    > 0 ? cfg.head_dim    : 128;
    const int32_t n_layer = cfg.n_layers    > 0 ? cfg.n_layers    : 24;
    const float   eps     = cfg.rms_norm_eps > 0 ? cfg.rms_norm_eps : 1e-6f;
    const float   theta   = cfg.rope_theta  > 0 ? cfg.rope_theta  : 1000000.0f;
    if (std::getenv("ACE_STEP_DIT_DEBUG")) {
        fprintf(stderr, "[dit] rope_theta=%g (cfg=%g)\n", theta, cfg.rope_theta);
    }

    // ── Graph context (no_alloc=true: gallocr allocates intermediates) ──
    // We use ggml_gallocr for lifecycle-based allocation directly on CUDA.
    // This avoids the scheduler's CPU fallback (INPUT-flagged → CPU) while
    // keeping memory-efficient allocation (tensors with non-overlapping
    // lifetimes share the same memory).
    size_t mem = ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(8192, false);
    char* ctx_buf = new (std::nothrow) char[mem];
    if (!ctx_buf) { if (error) *error = "DiT OOM (metadata)"; return false; }

    ggml_init_params p = { mem, ctx_buf, /*no_alloc=*/true };
    ggml_context* ctx = ggml_init(p);
    ggml_cgraph* gf  = ggml_new_graph_custom(ctx, 8192, false);
    if (!ctx || !gf) {
        delete[] ctx_buf; if (error) *error = "DiT ggml_init"; return false;
    }

    // ── Input x [T, H] ───────────────────────────────────────────────────
    // Mark as scheduler input; data is set after sched_alloc_graph via
    // ggml_backend_tensor_set (see "Compute" block below).
    // We queue all input uploads in `inputs` and flush them after alloc.
    std::vector<DiTInputUpload> inputs;

    ggml_tensor* cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
    ggml_set_name(cur, "x_t");
    ggml_set_input(cur);
    inputs.push_back({cur, x_t, static_cast<size_t>(T) * H * sizeof(float)});

    // ── Dual-timestep embedding (mean-flow two-branch) ───────────────────
    // Upstream AceStepTransformer1DModel.forward (ace_step_transformer.py:560):
    //   temb_t, timestep_proj_t = self.time_embed(timestep)
    //   temb_r, timestep_proj_r = self.time_embed_r(timestep - timestep_r)
    //   temb = temb_t + temb_r
    //   timestep_proj = timestep_proj_t + timestep_proj_r
    // For standard inference timestep_r == timestep, so the r-branch evaluates
    // at t=0 (constant). Skipping it leaves both the per-layer AdaLN modulation
    // AND the global scale_shift_table modulation off by a constant, producing
    // audio that decodes to static. We compute both branches and sum.
    auto build_time_mlp = [&](const char* prefix, const float* sin_buf,
                              int sin_dim) -> ggml_tensor* {
        ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, sin_dim);
        // Name input sinusoid tensors so we can verify they're uploaded
        // correctly (debug aid for layer-by-layer comparison).
        if (std::string(prefix) == "decoder.time_embed") {
            ggml_set_name(t, "sinusoid_t_input");
        } else if (std::string(prefix) == "decoder.time_embed_r") {
            ggml_set_name(t, "sinusoid_r_input");
        }
        ggml_set_input(t);
        // Call set_output AFTER set_input — ggml_set_input resets flags.
        if (std::getenv("ACE_STEP_DUMP_DIR")) ggml_set_output(t);
        inputs.push_back({t, sin_buf, static_cast<size_t>(sin_dim) * sizeof(float)});
        t = ggml_reshape_2d(ctx, t, sin_dim, 1);

        char buf[160];
        std::snprintf(buf, sizeof(buf), "%s.linear_1.weight", prefix);
        ggml_tensor* l1w = ggml_get_tensor(ext_ctx, buf);
        std::snprintf(buf, sizeof(buf), "%s.linear_1.bias", prefix);
        ggml_tensor* l1b = ggml_get_tensor(ext_ctx, buf);
        std::snprintf(buf, sizeof(buf), "%s.linear_2.weight", prefix);
        ggml_tensor* l2w = ggml_get_tensor(ext_ctx, buf);
        std::snprintf(buf, sizeof(buf), "%s.linear_2.bias", prefix);
        ggml_tensor* l2b = ggml_get_tensor(ext_ctx, buf);
        t = ggml_mul_mat(ctx, l1w, t);
        if (l1b) t = ggml_add(ctx, t, ggml_repeat(ctx, l1b, t));
        // Dump lin1 output (before silu) for upstream comparison.
        // Upstream names these temb_lin1_t / temb_lin1_r.
        if (std::string(prefix) == "decoder.time_embed") {
            ggml_set_name(t, "temb_lin1_t");
            if (std::getenv("ACE_STEP_DUMP_DIR")) ggml_set_output(t);
        } else if (std::string(prefix) == "decoder.time_embed_r") {
            ggml_set_name(t, "temb_lin1_r");
            if (std::getenv("ACE_STEP_DUMP_DIR")) ggml_set_output(t);
        }
        t = ggml_silu(ctx, t);
        t = ggml_mul_mat(ctx, l2w, t);
        if (l2b) t = ggml_add(ctx, t, ggml_repeat(ctx, l2b, t));
        return t;   // [H, 1] — this is temb (linear_2 output)
    };

    // t-branch sinusoid (caller-supplied, t * 1000 already applied).
    ggml_tensor* temb_t = build_time_mlp("decoder.time_embed",
                                          temb, temb_dim);
    ggml_set_name(temb_t, "temb_t");
    if (std::getenv("ACE_STEP_DUMP_DIR")) ggml_set_output(temb_t);
    // r-branch sinusoid: t-r=0 → cos(0)=1 (first half), sin(0)=0 (second half).
    std::vector<float> r_sin(temb_dim, 0.0f);
    for (int j = 0; j < temb_dim / 2; j++) r_sin[j] = 1.0f;
    ggml_tensor* temb_r = build_time_mlp("decoder.time_embed_r",
                                          r_sin.data(), temb_dim);
    ggml_set_name(temb_r, "temb_r");
    if (std::getenv("ACE_STEP_DUMP_DIR")) ggml_set_output(temb_r);

    // temb_combined = temb_t + temb_r — used for the FINAL norm_out modulation.
    ggml_tensor* temb_combined = ggml_add(ctx, temb_t, temb_r);
    ggml_set_name(temb_combined, "temb");
    if (std::getenv("ACE_STEP_DUMP_DIR")) ggml_set_output(temb_combined);

    // Per-layer timestep_proj = time_proj(silu(temb_t)) + time_proj_r(silu(temb_r)).
    // Each time_proj maps [H, 1] → [H*6, 1], then reshape to [H, 6].
    auto proj_time = [&](const char* prefix, ggml_tensor* temb_branch) {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "%s.time_proj.weight", prefix);
        ggml_tensor* pw = ggml_get_tensor(ext_ctx, buf);
        std::snprintf(buf, sizeof(buf), "%s.time_proj.bias", prefix);
        ggml_tensor* pb = ggml_get_tensor(ext_ctx, buf);
        ggml_tensor* h = ggml_silu(ctx, temb_branch);
        h = ggml_mul_mat(ctx, pw, h);
        if (pb) h = ggml_add(ctx, h, ggml_repeat(ctx, pb, h));
        return h;   // [H*6, 1]
    };
    auto proj_t = proj_time("decoder.time_embed",     temb_t);
    auto proj_r = proj_time("decoder.time_embed_r",   temb_r);
    auto pj = ggml_add(ctx, proj_t, proj_r);
    ggml_tensor* time_mod = ggml_reshape_2d(ctx, pj, H, 6);   // ne0=H, ne1=6
    // Wrap in cont for a unique buffer — without this, the scheduler can reuse
    // time_mod's buffer for temb (they have overlapping lifetimes in the
    // scheduler's view), and the tproj dump reads temb's data instead.
    if (std::getenv("ACE_STEP_DUMP_DIR")) {
        time_mod = dump_mark(ctx, time_mod, "tproj");
    } else {
        ggml_set_name(time_mod, "tproj");
    }

    // ── Condition embedder ───────────────────────────────────────────────
    // Full upstream pipeline (modeling_ace_step.py:830):
    //   1. text_projector(TE_text_hs)         : [T_text, 1024] → [T_text, 2048]
    //   2. timbre_encoder(silence_latent[:750]) : [750, 64] → [1, 2048] (CLS pool)
    //   3. lyric_encoder(TE_lyric_hs)          : [T_lyric, 1024] → [T_lyric, 2048]
    //      (ALWAYS runs — upstream encodes "# Languages\nunknown..." by default
    //       when no lyrics are provided, producing ~11 conditioning tokens.
    //       Caller passes lyric_data for cond branch, null for CFG uncond.)
    //   4. pack [lyric | timbre | text]  (cond-enc.h:354 order)
    //   5. condition_embedder(packed)          : [T_packed, 2048] → [T_packed, H]
    //
    // Without timbre, the DiT produces white noise (verified: the silence_latent
    // is the "no music" reference that anchors the timbre encoder — passing
    // literal zeros is OOD and produces drone-like output).
    // Conditioning weights — validated at load time (loader.cpp must_have).
    ggml_tensor* tp_w = ggml_get_tensor(ext_ctx, "encoder.text_projector.weight");
    ggml_tensor* cew  = ggml_get_tensor(ext_ctx, "decoder.condition_embedder.weight");
    ggml_tensor* ceb  = ggml_get_tensor(ext_ctx, "decoder.condition_embedder.bias");

    // (A) Timbre token from silence_latent (single CLS-pooled 2048-dim token).
    //     Upstream ALWAYS supplies refer_audio (silence_latent slice for
    //     text2music). If the caller didn't, that's a bug — fail loudly.
    if (!refer_audio || T_refer <= 0) {
        if (error) *error = "DiT: refer_audio (silence_latent) required but not provided";
        return false;
    }
    const int timbre_dim = 64;   // timbre_hidden_dim
    ggml_tensor* timbre_tok = timbre_encode(ctx, ext_ctx, refer_audio, T_refer,
                                            timbre_dim, H,
                                            nh, nk, hd, theta, inputs);
    if (!timbre_tok) {
        if (error) *error = "DiT: timbre_encode returned null (missing tensor?)";
        return false;
    }

    // (B) Lyric encoder: raw TE embeddings [1024, S_lyric] → [2048, S_lyric].
    //     ALWAYS runs, even for text2music without lyrics — upstream encodes
    //     "# Languages\nunknown\n\n# Lyric\n<|endoftext|>" by default,
    //     producing ~11 conditioning tokens. Without the lyric path, the
    //     cross-attention sequence length is wrong and K/V projections diverge.
    //     CFG uncond branch passes lyric_data=null; lyrics are dropped to 0 tokens.
    ggml_tensor* lyric_proj = nullptr;
    if (lyric_data && lyric_len > 0) {
        int64_t lh = (lyric_hidden > 0) ? lyric_hidden : 1024;
        lyric_proj = lyric_encode(ctx, ext_ctx, lyric_data, lyric_len,
                                  static_cast<int>(lh),
                                  H, nh, nk, hd, theta, inputs);
        if (!lyric_proj) {
            if (error) *error = "DiT: lyric_encode returned null (missing tensor?)";
            return false;
        }
    }

    // (C) Text projection: TE_text_hs [1024, T_text] → [2048, T_text].
    //     CFG uncond branch passes cond_data=null; in that case null_condition_emb
    //     is already at encoder_hidden dim (2048), so use it directly as text_proj
    //     without going through text_projector (which maps 1024→2048 — wrong dim).
    ggml_tensor* text_proj;
    if (cond_data && ct_len > 0) {
        int64_t te_hs = (cond_hidden > 0) ? cond_hidden : 1024;
        ggml_tensor* ct = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, te_hs, ct_len);
        ggml_set_input(ct);
        inputs.push_back({ct, cond_data,
                          static_cast<size_t>(ct_len) * static_cast<size_t>(te_hs) *
                          sizeof(float)});
        text_proj = ggml_mul_mat(ctx, tp_w, ct);   // [2048, T_text]
    } else {
        // CFG uncond: null_condition_emb [2048] is already in encoder_hidden space.
        // Reshape to [2048, 1] for packing — no text_projector needed.
        ggml_tensor* nce = ggml_get_tensor(ext_ctx, "null_condition_emb");
        if (!nce) {
            if (error) *error = "DiT: null_condition_emb missing (CFG uncond)";
            return false;
        }
        text_proj = ggml_reshape_2d(ctx, nce, 2048, 1);  // [2048, 1]
    }

    // (D) Pack [lyric | timbre | text] → encoder_hidden_states [2048, T_packed]
    //     Upstream cond-enc.h:354 pack order: lyric, timbre[0:1], text_proj.
    //     For CFG uncond (no lyrics), pack [timbre | text] — lyrics dropped.
    //     For text2music (lyrics present), pack [lyric | timbre | text].
    ggml_tensor* packed;
    if (lyric_proj) {
        // [lyric | timbre | text]
        ggml_tensor* lt = ggml_concat(ctx, lyric_proj, timbre_tok, 1);
        packed = ggml_concat(ctx, lt, text_proj, 1);
    } else {
        // [timbre | text]
        packed = ggml_concat(ctx, timbre_tok, text_proj, 1);
    }

    // (D) condition_embedder: [2048, T_packed] → [H_dit, T_packed]
    ggml_tensor* ce_out = ggml_mul_mat(ctx, cew, packed);
    ggml_tensor* cond_emb = ggml_add(ctx, ce_out, ggml_repeat(ctx, ceb, ce_out));
    ggml_set_name(cond_emb, "enc_after_cond_emb");
    if (std::getenv("ACE_STEP_DUMP_DIR")) ggml_set_output(cond_emb);

    // ── Diagnostic: input cond stats (raw, before projection) ──────────────
    // NOTE: The intermediate-tensor stats_of diagnostics that were here used
    // ggml_graph_compute_with_ctx, which is incompatible with the scheduler
    // (no_alloc=true ctx). They were one-shot debug helpers; the final
    // output RMS below still runs.
    if (std::getenv("ACE_STEP_DIT_DEBUG") && cond_data && ct_len > 0) {
        int64_t te_hs = (cond_hidden > 0) ? cond_hidden : 1024;
        const float* cdv = cond_data;
        double sq = 0.0, sum = 0.0; float mn = 1e30f, mx = -1e30f;
        int64_t n = static_cast<int64_t>(ct_len) * te_hs;
        for (int64_t i = 0; i < n; i++) {
            float v = cdv[i]; sq += (double)v*v; sum += v;
            if (v < mn) mn = v; if (v > mx) mx = v;
        }
        fprintf(stderr, "[dit] te_cond: T=%d H=%ld mean=%.4f RMS=%.4f range=[%.4f,%.4f]\n",
                ct_len, (long)te_hs, sum/n, std::sqrt(sq/n), mn, mx);
    }

    // ── DiT layers ───────────────────────────────────────────────────────
    for (int i = 0; i < n_layer && i < 48; i++) {
        char buf[128];

        // Build per-layer time_mod with scale_shift_table bias
        ggml_tensor* layer_time_mod = time_mod;
        if (i < (int)ss_table_f32.size() && !ss_table_f32[i].empty()) {
            ggml_tensor* sst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 6);
            ggml_set_input(sst);
            inputs.push_back({sst, ss_table_f32[i].data(),
                              static_cast<size_t>(H) * 6 * sizeof(float)});
            layer_time_mod = ggml_add(ctx, time_mod, sst);
        }

        // Dump the per-layer time_mod (tptr + scale_shift_table) for layer 0
        // to compare against upstream's effective modulation values.
        if (i == 0 && std::getenv("ACE_STEP_DUMP_DIR")) {
            layer_time_mod = dump_mark(ctx, layer_time_mod, "layer0_time_mod");
        }

        // NOTE: A previous version of this block tried to log cur's RMS here,
        // but `cur->data` is unmaterialized until ggml_graph_compute below —
        // reading it produced misleading zeros. Post-compute diagnostics live
        // after the graph compute at the bottom of this function.

        // Set when this is layer 0 and the dump-dir env var is set. The named
        // intermediates below match upstream's layer0_* dumps in dit-graph.h.
        const bool dump_l0 = (i == 0) && std::getenv("ACE_STEP_DUMP_DIR");

        // ── Self-attention block ─────────────────────────────────────────
        // Upstream (modeling_acestep_v15_turbo.py:494-514):
        //   6 chunks = (shift_msa, scale_msa, gate_msa, c_shift, c_scale, c_gate)
        //   norm_h = norm(x) * (1 + scale_msa) + shift_msa   # scale=chunk[1], shift=chunk[0]
        //   attn_out = self_attn(norm_h)
        //   x = x + attn_out * gate_msa                       # gate=chunk[2]
        {
            ggml_tensor* h = ggml_rms_norm(ctx, cur, eps);
            // Learned norm weight (validated at load time)
            std::snprintf(buf, sizeof(buf),
                          "decoder.layers.%d.self_attn_norm.weight", i);
            ggml_tensor* san_w = ggml_get_tensor(ext_ctx, buf);
            h = ggml_mul(ctx, h, ggml_repeat(ctx, san_w, h));
            // AdaLN modulation: h * (1 + scale[row1]) + shift[row0]
            h = adaln(ctx, h, layer_time_mod, 1, 0, H);  // row_s=1 (scale), row_h=0 (shift)

            // Upstream: layer0_sa_input is norm_sa after adaln modulation.
            if (dump_l0) {
                h = dump_mark(ctx, h, "layer0_sa_input");
            }

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
                          qn_w, kn_w, theta, inputs, /*dump_layer0=*/dump_l0);

            // Upstream: layer0_sa_output is the post-o_proj self-attn output.
            if (dump_l0) {
                h = dump_mark(ctx, h, "layer0_sa_output");
            }

            // GATED residual: cur += h * gate[row2]
            auto gate = ggml_reshape_2d(ctx,
                ggml_view_1d(ctx, layer_time_mod, H,
                             static_cast<size_t>(2) * H * sizeof(float)),
                H, 1);
            cur = ggml_add(ctx, cur,
                            ggml_mul(ctx, h, ggml_repeat(ctx, gate, h)));

            // Upstream: layer0_after_self_attn = hidden after gated residual.
            if (dump_l0) {
                cur = dump_mark(ctx, cur, "layer0_after_self_attn");
            }
        }

        // ── Cross-attention block (NO modulation, plain residual) ───────
        // Upstream (modeling_acestep_v15_turbo.py:517-529):
        //   norm_h = cross_attn_norm(x)   # NO scale/shift modulation
        //   attn_out = cross_attn(norm_h, cond)
        //   x = x + attn_out              # plain residual, NO gate
        {
            ggml_tensor* h = ggml_rms_norm(ctx, cur, eps);
            // Learned norm weight (validated)
            std::snprintf(buf, sizeof(buf),
                          "decoder.layers.%d.cross_attn_norm.weight", i);
            ggml_tensor* can_w = ggml_get_tensor(ext_ctx, buf);
            h = ggml_mul(ctx, h, ggml_repeat(ctx, can_w, h));
            // NO adaln modulation for cross-attention

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
            cur = ggml_add(ctx, cur, h);   // plain residual

            // Upstream: layer0_after_cross_attn = hidden after cross-attn add.
            if (dump_l0) {
                cur = dump_mark(ctx, cur, "layer0_after_cross_attn");
            }
        }

        // ── MLP block ────────────────────────────────────────────────────
        // Upstream: norm_h = norm(x) * (1 + c_scale[4]) + c_shift[3]; x += mlp * c_gate[5]
        {
            ggml_tensor* h = ggml_rms_norm(ctx, cur, eps);
            // Learned norm weight (validated)
            std::snprintf(buf, sizeof(buf),
                          "decoder.layers.%d.mlp_norm.weight", i);
            ggml_tensor* mn_w = ggml_get_tensor(ext_ctx, buf);
            h = ggml_mul(ctx, h, ggml_repeat(ctx, mn_w, h));
            // AdaLN modulation: c_scale=row4, c_shift=row3
            h = adaln(ctx, h, layer_time_mod, 4, 3, H);  // row_s=4 (scale), row_h=3 (shift)

            std::snprintf(buf, sizeof(buf),
                          "decoder.layers.%d.mlp.gate_proj.weight", i);
            auto gw = ggml_get_tensor(ext_ctx, buf);
            std::snprintf(buf, sizeof(buf),
                          "decoder.layers.%d.mlp.up_proj.weight", i);
            auto uw = ggml_get_tensor(ext_ctx, buf);
            std::snprintf(buf, sizeof(buf),
                          "decoder.layers.%d.mlp.down_proj.weight", i);
            auto dw = ggml_get_tensor(ext_ctx, buf);

            if (dump_l0) {
                h = dump_mark(ctx, h, "layer0_mlp_input");
            }
            h = swiglu_mlp(ctx, h, gw, uw, dw);
            if (dump_l0) {
                h = dump_mark(ctx, h, "layer0_mlp_output");
            }
            // GATED residual: cur += h * c_gate[row5]
            auto c_gate = ggml_reshape_2d(ctx,
                ggml_view_1d(ctx, layer_time_mod, H,
                             static_cast<size_t>(5) * H * sizeof(float)),
                H, 1);
            cur = ggml_add(ctx, cur,
                            ggml_mul(ctx, h, ggml_repeat(ctx, c_gate, h)));
        }

        // NOTE: A previous version logged cur->data RMS here, but `cur` is
        // unmaterialized until the graph compute at the bottom of this
        // function. Reading the data here produced misleading zeros.

        // Name hidden state after specific layers for upstream comparison.
        // Upstream dumps hidden_after_layer{0,6,12,18,23}. We name the
        // corresponding tensors so we can fetch them via ggml_graph_get_tensor
        // and write matching dump files after compute.
        // CRITICAL: wrap in ggml_cont (via dump_mark) so the scheduler does NOT
        // reuse the tensor's memory for later layers (without this, all
        // hidden_after_layerN tensors can end up sharing the same data pointer
        // because their lifetimes don't overlap).
        if (i == 0 || i == 6 || i == 12 || i == 18 || i == n_layer - 1) {
            char nm[64];
            std::snprintf(nm, sizeof(nm), "hidden_after_layer%d", i);
            if (std::getenv("ACE_STEP_DUMP_DIR")) {
                cur = dump_mark(ctx, cur, nm);
            }
        }
    }

    // ── Final norm + scale/shift ─────────────────────────────────────────
    // Upstream (ace_step_transformer.py:616-619):
    //   shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
    //   out = norm_out(x) * (1 + scale[row1]) + shift[row0]
    // The temb addition is critical — without it the final norm_out modulation
    // is constant across all timesteps, so the velocity field doesn't actually
    // depend on `t` at the output projection. Combined with the dual-timestep
    // embedding above, temb_combined = temb_t + temb_r matches upstream exactly.
    {
        // Learned output norm weight (validated at load time)
        ggml_tensor* norm_out_w = ggml_get_tensor(ext_ctx, "decoder.norm_out.weight");
        ggml_tensor* n_out = ggml_rms_norm(ctx, cur, eps);
        n_out = ggml_mul(ctx, n_out, ggml_repeat(ctx, norm_out_w, n_out));

        // Global scale/shift: decoder.scale_shift_table [H, 2] → ne0=H, ne1=2
        // We need (scale_shift_table + temb) broadcast across the 2 rows.
        if (!global_ss_f32.empty()) {
            ggml_tensor* gss = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 2);
            ggml_set_input(gss);
            inputs.push_back({gss, global_ss_f32.data(),
                              static_cast<size_t>(H) * 2 * sizeof(float)});
            // temb_combined is [H, 1] — broadcast to [H, 2] via ggml_repeat.
            auto temb_bc = ggml_repeat(ctx, temb_combined, gss);
            auto gss_temb = ggml_add(ctx, gss, temb_bc);
            // shift = row 0, scale = row 1 of (scale_shift_table + temb)
            auto shift = ggml_reshape_2d(ctx,
                ggml_view_1d(ctx, gss_temb, H, 0), H, 1);
            auto scale = ggml_reshape_2d(ctx,
                ggml_view_1d(ctx, gss_temb, H,
                             static_cast<size_t>(H) * sizeof(float)),
                H, 1);
            // (1 + scale) * n_out + shift = n_out + n_out*scale + shift
            // (rewritten to avoid ggml_new_f32 which asserts on no_alloc ctx)
            auto n_scaled = ggml_mul(ctx, n_out, ggml_repeat(ctx, scale, n_out));
            n_out = ggml_add(ctx, ggml_add(ctx, n_out, n_scaled),
                              ggml_repeat(ctx, shift, n_out));
        }
        cur = n_out;
    }

    // ── Compute ──────────────────────────────────────────────────────────
    // Allocate graph tensors on CUDA via gallocr (lifecycle-based sharing),
    // upload inputs, compute, download output. This bypasses the scheduler's
    // CPU fallback while keeping memory-efficient allocation.
    ggml_set_name(cur, "dit_out");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    static bool kDebugSched = (std::getenv("ACE_STEP_SCHED_DEBUG") != nullptr);
    if (kDebugSched) {
        fprintf(stderr, "[dit] graph: %d nodes, %d inputs queued\n",
                ggml_graph_n_nodes(gf), (int)inputs.size());
    }

    // gallocr does lifecycle analysis: tensors with non-overlapping lifetimes
    // share memory, keeping peak VRAM low (unlike alloc_ctx_tensors which
    // allocates all simultaneously). Weight tensors in ext_ctx_ are skipped
    // (they already have buffers from migrate_ctx_to_backend).
    ggml_gallocr_t galloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(cuda_backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        if (error) *error = "DiT: gallocr_alloc_graph failed";
        ggml_gallocr_free(galloc);
        ggml_free(ctx); delete[] ctx_buf; return false;
    }

    // Upload input tensor data (x_t, sinusoids, cond, ss_tables).
    for (const auto& in : inputs) {
        ggml_backend_tensor_set(in.t, in.data, 0, in.nbytes);
    }

    ggml_backend_graph_compute(cuda_backend, gf);

    // Dump intermediate tensors by name (matches upstream's dump_named scheme)
    // when ACE_STEP_DUMP_DIR is set. MUST run before gallocr_free() — the
    // free frees the backend's intermediate buffers, leaving tensor data
    // pointers dangling.
    // Only dump on the FIRST call (step 0 of the diffusion loop) — upstream
    // does the same, and subsequent steps would overwrite the dumps.
    static thread_local int s_dump_call_count = 0;
    const char* dir_env = std::getenv("ACE_STEP_DUMP_DIR");
    const bool do_dump = (s_dump_call_count++ == 0) && dir_env;
    if (do_dump) {
        auto dump_named = [&](const char* name) {
            ggml_tensor* t = ggml_graph_get_tensor(gf, name);
            if (!t) return;
            int ndims = ggml_n_dims(t);
            // Cap at 3 written dims (we never need dim[3] since N=1 here).
            if (ndims > 3) ndims = 3;
            int64_t shape[3] = {1, 1, 1};
            int64_t total = 1;
            for (int i = 0; i < ndims; i++) {
                shape[i] = t->ne[i];
                total *= t->ne[i];
            }
            std::vector<float> buf(static_cast<size_t>(total));
            ggml_backend_tensor_get(t, buf.data(), 0, total * sizeof(float));
            fprintf(stderr, "[dump] %-32s data=%p ne=[%lld,%lld,%lld,%lld] writing=%lld floats\n",
                    name, t->data, (long long)t->ne[0], (long long)t->ne[1],
                    (long long)(ggml_n_dims(t) > 2 ? t->ne[2] : 0),
                    (long long)(ggml_n_dims(t) > 3 ? t->ne[3] : 0),
                    (long long)total);
            std::string path = std::string(dir_env) + "/" + name + ".bin";
            if (FILE* f = std::fopen(path.c_str(), "wb")) {
                int32_t ndims_w = static_cast<int32_t>(ndims);
                std::fwrite(&ndims_w, sizeof(int32_t), 1, f);
                for (int i = 0; i < ndims; i++) {
                    int32_t d = static_cast<int32_t>(shape[i]);
                    std::fwrite(&d, sizeof(int32_t), 1, f);
                }
                std::fwrite(buf.data(), sizeof(float),
                            static_cast<size_t>(total), f);
                std::fclose(f);
            }
        };
        dump_named("temb");
        dump_named("temb_t");
        dump_named("temb_r");
        dump_named("temb_lin1_t");
        dump_named("temb_lin1_r");
        dump_named("sinusoid_t_input");
        dump_named("sinusoid_r_input");
        dump_named("tproj");
        dump_named("layer0_time_mod");
        dump_named("enc_after_cond_emb");
        // Layer-0 sub-step dumps (matches upstream's layer0_* dump scheme).
        // These pinpoint which operation inside layer 0 first diverges.
        dump_named("layer0_sa_input");
        dump_named("layer0_q_raw_after_proj");
        dump_named("layer0_k_raw_after_proj");
        dump_named("layer0_q_after_qknorm");
        dump_named("layer0_k_after_qknorm");
        dump_named("layer0_v_raw");
        dump_named("layer0_q_after_rope");
        dump_named("layer0_k_after_rope");
        dump_named("layer0_attn_out");
        dump_named("layer0_sa_output");
        dump_named("layer0_after_self_attn");
        dump_named("layer0_after_cross_attn");
        dump_named("layer0_mlp_input");
        dump_named("layer0_mlp_output");
        char nm[64];
        for (int idx : {0, 6, 12, 18}) {
            std::snprintf(nm, sizeof(nm), "hidden_after_layer%d", idx);
            dump_named(nm);
        }
        std::snprintf(nm, sizeof(nm), "hidden_after_layer%d", n_layer - 1);
        dump_named(nm);
    }

    // Download output to caller's buffer.
    ggml_backend_tensor_get(cur, result, 0,
                            static_cast<size_t>(T) * H * sizeof(float));

    // Free the per-step gallocr buffer (intermediate tensors). Weight tensors
    // in ext_ctx_ are on a separate buffer and are NOT freed here.
    ggml_gallocr_free(galloc);

    // Output RMS diagnostic (gated — scans T*H floats per step)
    if (std::getenv("ACE_STEP_DIT_DEBUG")) {
        double sq = 0.0;
        for (int64_t ii = 0; ii < static_cast<int64_t>(T) * H; ii++)
            sq += (double)result[ii] * result[ii];
        double out_rms = std::sqrt(sq / (static_cast<double>(T) * H));
        fprintf(stderr, "[dit] output RMS=%.4f (in was %.4f, ratio=%.2f)\n",
                out_rms, inp_rms, out_rms / (inp_rms + 1e-10));
    }

    ggml_free(ctx);
    delete[] ctx_buf;
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
bool DiTRunner::forward(
    const float* x_t, float t,
    const float* cond, int32_t T_cond, int32_t cond_hidden,
    const float* cond_nc, int32_t T_cond_nc,
    const float* lyric, int32_t T_lyric, int32_t lyric_hidden,
    const float* lyric_nc, int32_t T_lyric_nc,
    const float* refer_audio, int32_t T_refer,
    float guidance_scale, int32_t n_patches,
    float* output, std::string* error)
{
    const int32_t H = cfg_.hidden_size;
    const int32_t T = n_patches;
    const int temb_dim = 256;

    // Initialize backend + scheduler on first call (GPU if available,
    // CPU fallback otherwise). Migrates ext_ctx_ weights to GPU buffer.
    if (!ensure_backend()) {
        if (error) *error = "DiT: backend initialization failed";
        return false;
    }

    // Pre-compute timestep embedding (shared by cond + uncond)
    std::vector<float> temb(static_cast<size_t>(temb_dim));
    // Scale timestep by 1000 (matches TimestepEmbedding.scale=1000 default
    // in vendor/acestep/models/turbo/modeling_acestep_v15_turbo.py:215 and
    // HOT-Step dit-graph.h:87 ggml_scale(ctx, t_scalar, 1000.0f)).
    // The diffusers pipeline_op.py wrapper passes raw [0,1] timesteps, but
    // the model's own TimestepEmbedding class applies scale=1000 internally
    // (line 247: t = t * self.scale) before computing the sinusoid.
    timestep_embed(t * 1000.0f, temb.data(), temb_dim);

    // Debug: dump the CPU-side temb buffer (before any graph upload) so we
    // can verify the sinusoid is computed correctly. This is independent of
    // any scheduler memory reuse issues.
    if (const char* dir_env = std::getenv("ACE_STEP_DUMP_DIR")) {
        std::string path = std::string(dir_env) + "/cpu_sinusoid_t.bin";
        if (FILE* f = std::fopen(path.c_str(), "wb")) {
            int32_t ndims = 1, d0 = temb_dim;
            std::fwrite(&ndims, sizeof(int32_t), 1, f);
            std::fwrite(&d0, sizeof(int32_t), 1, f);
            std::fwrite(temb.data(), sizeof(float), temb_dim, f);
            std::fclose(f);
        }
        fprintf(stderr, "[forward] t=%f, temb[0]=%.4f temb[1]=%.4f temb[128]=%.4f\n",
                t, temb[0], temb[1], temb[128]);
    }

    if (guidance_scale == 1.0f) {
        return run_one_forward(cfg_, ext_ctx_, cuda_backend_, x_t, T, H,
                                temb.data(), temb_dim,
                                cond, T_cond, cond_hidden,
                                lyric, T_lyric, lyric_hidden,
                                refer_audio, T_refer,
                                output,
                                ss_table_f32_, global_ss_f32_, error);
    }

    // CFG: cond + uncond, then blend
    std::vector<float> c_out(static_cast<size_t>(T) * H);
    std::vector<float> u_out(static_cast<size_t>(T) * H);

    if (!run_one_forward(cfg_, ext_ctx_, cuda_backend_, x_t, T, H,
                          temb.data(), temb_dim,
                          cond, T_cond, cond_hidden,
                          lyric, T_lyric, lyric_hidden,
                          refer_audio, T_refer,
                          c_out.data(),
                          ss_table_f32_, global_ss_f32_, error)) {
        if (error) *error = "DiT: cond forward failed";
        return false;
    }

    const float* uc       = (cond_nc && T_cond_nc > 0) ? cond_nc : nullptr;
    int          uc_len   = (cond_nc && T_cond_nc > 0) ? T_cond_nc : 0;
    const float* ul       = (lyric_nc && T_lyric_nc > 0) ? lyric_nc : nullptr;
    int          ul_len   = (lyric_nc && T_lyric_nc > 0) ? T_lyric_nc : 0;
    if (!run_one_forward(cfg_, ext_ctx_, cuda_backend_, x_t, T, H,
                          temb.data(), temb_dim,
                          uc, uc_len, cond_hidden,
                          ul, ul_len, lyric_hidden,
                          refer_audio, T_refer,
                          u_out.data(),
                          ss_table_f32_, global_ss_f32_, error)) {
        if (error) *error = "DiT: uncond forward failed";
        return false;
    }

    int64_t n_el = static_cast<int64_t>(T) * H;
    // ── CFG diagnostics (gated — scans 3×T*H floats per step) ──────────
    if (std::getenv("ACE_STEP_DIT_DEBUG")) {
        double c_rms = 0.0, u_rms = 0.0, diff_rms = 0.0;
        for (int64_t i = 0; i < n_el; i++) {
            c_rms += (double)c_out[i] * c_out[i];
            u_rms += (double)u_out[i] * u_out[i];
            double d = (double)c_out[i] - (double)u_out[i];
            diff_rms += d * d;
        }
        c_rms = std::sqrt(c_rms / n_el);
        u_rms = std::sqrt(u_rms / n_el);
        diff_rms = std::sqrt(diff_rms / n_el);
        fprintf(stderr, "[dit] CFG: c_rms=%.4f u_rms=%.4f diff_rms=%.4f g=%.1f\n",
                c_rms, u_rms, diff_rms, guidance_scale);
    }

    // ── Adaptive Projected Guidance (APG) ──────────────────────────────
    // Matches diffusers.guiders.adaptive_projected_guidance.normalized_guidance
    // called from pipeline_ace_step.py:1173 with:
    //   guidance_scale_arg = guidance_scale - 1.0
    //   eta                = 0.0   (drop parallel component)
    //   norm_threshold     = 2.5   (per-token L2 cap before projection)
    //   use_original_formulation = True (base = pred_cond, not pred_uncond)
    //   norm_dim           = (1,)  (per-token norm along hidden dim)
    //
    // Algorithm (per token t, along hidden dim H):
    //   1. diff = pred_cond - pred_uncond                    // [H]
    //   2. if ||diff|| > norm_threshold: diff *= norm_threshold / ||diff||
    //   3. v1 = pred_cond / ||pred_cond||                    // unit vector
    //      diff_parallel   = (diff · v1) * v1
    //      diff_orthogonal = diff - diff_parallel
    //      update          = diff_orthogonal + eta * diff_parallel
    //   4. output = pred_cond + guidance_scale_arg * update
    //
    // With eta=0 the parallel component (which would amplify the cond
    // magnitude) is dropped, leaving only the orthogonal steering signal.
    // This is the OPPOSITE of vanilla CFG which goes
    //   u + gs*(c-u) = gs*c + (1-gs)*u  — that blows up magnitude when gs>1.
    const float eta = 0.0f;
    const float norm_threshold = 2.5f;
    const float gs_arg = std::max(0.0f, guidance_scale - 1.0f);

    double post_rms = 0.0;
    int64_t n_clamped = 0;
    for (int64_t t = 0; t < T; t++) {
        const float* c_row = &c_out[static_cast<size_t>(t) * H];
        const float* u_row = &u_out[static_cast<size_t>(t) * H];
        float* out_row = &output[static_cast<size_t>(t) * H];

        // Step 1+2: compute diff with norm cap.
        double diff_sq = 0.0;
        for (int64_t h = 0; h < H; h++) {
            double d = (double)c_row[h] - (double)u_row[h];
            diff_sq += d * d;
        }
        float diff_scale = 1.0f;
        float diff_norm = static_cast<float>(std::sqrt(diff_sq));
        if (diff_norm > norm_threshold && diff_norm > 1e-12f) {
            diff_scale = norm_threshold / diff_norm;
            n_clamped++;
        }

        // Step 3: project diff onto pred_cond direction.
        //   v1 = pred_cond / ||pred_cond||  (per-token unit vector)
        //   parallel = (diff · v1) * v1
        //   ortho    = diff - parallel
        //   update   = ortho + eta * parallel
        double cond_sq = 0.0;
        for (int64_t h = 0; h < H; h++)
            cond_sq += (double)c_row[h] * (double)c_row[h];
        float cond_norm = static_cast<float>(std::sqrt(cond_sq));
        // Handle degenerate ||pred_cond||≈0 — fall back to raw diff as update.
        if (cond_norm < 1e-12f) {
            // output = pred_cond + gs_arg * diff  (no projection possible)
            for (int64_t h = 0; h < H; h++) {
                float d = (c_row[h] - u_row[h]) * diff_scale;
                float v = c_row[h] + gs_arg * d;
                out_row[h] = v;
                post_rms += (double)v * v;
            }
            continue;
        }
        float inv_cn = 1.0f / cond_norm;
        // diff · v1 = sum(diff[h] * c_row[h]) / ||c||
        double dot_dc = 0.0;
        for (int64_t h = 0; h < H; h++) {
            float d = (c_row[h] - u_row[h]) * diff_scale;
            dot_dc += (double)d * (double)c_row[h];
        }
        float parallel_factor = static_cast<float>(dot_dc) * inv_cn;  // scalar = (diff·v1)
        // For each h:
        //   diff_parallel[h] = parallel_factor * c_row[h] / cond_norm  (= parallel_factor * v1[h])
        //   diff_ortho[h]    = diff[h] - diff_parallel[h]
        //   update[h]        = diff_ortho[h] + eta * diff_parallel[h]
        //                   = diff[h] - (1 - eta) * parallel_factor * c_row[h] / cond_norm
        //   output[h]        = c_row[h] + gs_arg * update[h]
        float one_minus_eta = 1.0f - eta;
        float proj_coef = one_minus_eta * parallel_factor * inv_cn;  // multiplies c_row[h]
        for (int64_t h = 0; h < H; h++) {
            float d = (c_row[h] - u_row[h]) * diff_scale;
            float update_h = d - proj_coef * c_row[h];
            float v = c_row[h] + gs_arg * update_h;
            out_row[h] = v;
            post_rms += (double)v * v;
        }
    }
    post_rms = std::sqrt(post_rms / static_cast<double>(n_el));
    fprintf(stderr,
            "[dit] APG: gs_arg=%.2f eta=%.2f threshold=%.1f "
            "tokens_clamped=%lld/%lld post_rms=%.4f\n",
            gs_arg, eta, norm_threshold,
            (long long)n_clamped, (long long)T, post_rms);
    return true;
}

}  // namespace audiocore::acestep
