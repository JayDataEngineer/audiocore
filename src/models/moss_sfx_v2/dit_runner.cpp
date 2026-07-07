// dit_runner.cpp — DiT graph builder + forward pass for MOSS-SoundEffect-v2.
//
// Architecture (WanAudioModel, 30 layers):
//   1. patch_embedding (Linear in_dim→dim, patch_size=1 → no-op reshape)
//   2. text_embedding  (Linear text_dim→dim, then GELU(tanh), then Linear dim→dim)
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
#include "ggml-alloc.h"  // ggml_gallocr — lifecycle-based graph allocation

#include <cmath>
#include <cstdio>
#include <cstring>
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

// ── Zero-context fallback buffer (shared across all callers) ───────────
static std::vector<float>& get_zero_ctx() {
    static std::vector<float> z;
    return z;
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

// ── 3D RoPE in-place (Python precompute_freqs_cis_3d matching) ──────────
//    Splits head_dim=128 into 3 groups (44+42+42) with dim=44/42/42
//    frequency schedules. Applies ggml_rope_ext_inplace on each group's
//    view and expands the nodes into the graph for correct CUDA ordering.
static void apply_rope_3d(ggml_context* ctx, ggml_tensor* t,
                           ggml_tensor* pos, int hd, int nh, int T,
                           ggml_cgraph* gf) {
    if (!t) return;
    auto rope = [&](ggml_tensor* v, int nd) {
        return ggml_rope_ext_inplace(ctx, v, pos, nullptr,
            nd, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    };
    auto mkview = [&](int off, int nd) {
        return ggml_view_3d(ctx, t, nd, nh, T,
            static_cast<int64_t>(hd) * 4,
            static_cast<int64_t>(hd) * nh * 4,
            static_cast<size_t>(off) * 4);
    };
    auto r0 = rope(mkview(0, 44), 44);
    auto r1 = rope(mkview(44, 42), 42);
    auto r2 = rope(mkview(86, 42), 42);
    if (gf) {
        ggml_build_forward_expand(gf, r0);
        ggml_build_forward_expand(gf, r1);
        ggml_build_forward_expand(gf, r2);
    }
}

// ── Self-attention (QK-norm + 3D RoPE) ───────────────────────────────────
static ggml_tensor* self_attn(ggml_context* ctx, ggml_tensor* x,
                                int T, int H, int nh, int hd,
                                ggml_tensor* q_w, ggml_tensor* k_w,
                                ggml_tensor* v_w, ggml_tensor* o_w,
                                ggml_tensor* q_norm_w, ggml_tensor* k_norm_w,
                                ggml_tensor* pos, ggml_cgraph* gf) {
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

    apply_rope_3d(ctx, q, pos, hd, nh, T, gf);
    apply_rope_3d(ctx, k, pos, hd, nh, T, gf);

    auto rsh = [&](ggml_tensor* t, int n_h) {
        return ggml_cont(ctx, ggml_permute(ctx, t, 0, 2, 1, 3));
    };
    q = rsh(q, nh);
    k = rsh(k, nh);
    v = rsh(v, nh);

    float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    auto a = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(a, GGML_PREC_DEFAULT);

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
                                 ggml_tensor* q_norm_w, ggml_tensor* k_norm_w,
                                 ggml_tensor* pos_q, ggml_tensor* pos_k,
                                 ggml_cgraph* gf) {
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

    // 3D RoPE on Q (using pos_q)
    {
        auto q_3d = ggml_reshape_3d(ctx, q_lin, hd, nh, T);
        apply_rope_3d(ctx, q_3d, pos_q, hd, nh, T, gf);
        q_lin = ggml_reshape_2d(ctx, q_3d, nh * hd, T);
    }
    // 3D RoPE on K (using pos_k)
    if (T_cond > 0) {
        auto k_3d = ggml_reshape_3d(ctx, k_lin, hd, nh, T_cond);
        apply_rope_3d(ctx, k_3d, pos_k, hd, nh, T_cond, gf);
        k_lin = ggml_reshape_2d(ctx, k_3d, nh * hd, T_cond);
    }

    auto q = rsh(q_lin, nh, T);
    auto k = rsh(k_lin, nh, T_cond);
    auto v = rsh(v_lin, nh, T_cond);

    float s = 1.0f / std::sqrt(static_cast<float>(hd));
    auto a = ggml_flash_attn_ext(ctx, q, k, v, nullptr, s, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(a, GGML_PREC_DEFAULT);
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
    release_cached();
    if (sched_) {
        ggml_backend_sched_free(sched_);
        sched_ = nullptr;
    }
    bp_.reset();
}

// ══════════════════════════════════════════════════════════════════════════════
// CPU weight helpers — read time_embedding weights before GPU migration
// ══════════════════════════════════════════════════════════════════════════════

void DiTRunner::load_cpu_weights() {
    auto read_f32 = [&](const char* name) -> std::vector<float> {
        ggml_tensor* t = ggml_get_tensor(ext_ctx_, name);
        if (!t) return {};
        std::vector<float> v(ggml_nelements(t));
        if (t->type == GGML_TYPE_BF16) {
            bf16_buf_to_f32(t->data, v.data(), (int)v.size());
        } else {
            memcpy(v.data(), t->data, ggml_nbytes(t));
        }
        return v;
    };
    cpu_te0_w_ = read_f32("moss_sfx_v2.time_embedding.0.weight");
    cpu_te0_b_ = read_f32("moss_sfx_v2.time_embedding.0.bias");
    cpu_te2_w_ = read_f32("moss_sfx_v2.time_embedding.2.weight");
    cpu_te2_b_ = read_f32("moss_sfx_v2.time_embedding.2.bias");
    cpu_tp_w_  = read_f32("moss_sfx_v2.time_projection.1.weight");
    cpu_tp_b_  = read_f32("moss_sfx_v2.time_projection.1.bias");
}

void DiTRunner::compute_modulation(const float* temb, int temb_dim,
                                    int H, float* mod_buf) const {
    // time_embedding.0: Linear(freq_dim, dim) → SiLU
    std::vector<float> buf0(static_cast<size_t>(H));
    if (cpu_te0_w_.empty()) {
        for (int j = 0; j < H; j++) buf0[(size_t)j] = 0.0f;
    } else {
        for (int j = 0; j < H; j++) {
            float sum = 0.0f;
            int col = size_t(j) * temb_dim;
            for (int i = 0; i < temb_dim; i++)
                sum += cpu_te0_w_[(size_t)col + i] * temb[i];
            if (!cpu_te0_b_.empty()) sum += cpu_te0_b_[(size_t)j];
            buf0[(size_t)j] = sum / (1.0f + std::exp(-sum));
        }
    }

    // time_embedding.2: Linear(dim, dim) → SiLU
    std::vector<float> buf1(static_cast<size_t>(H));
    if (cpu_te2_w_.empty()) {
        for (int j = 0; j < H; j++) buf1[(size_t)j] = 0.0f;
    } else {
        for (int j = 0; j < H; j++) {
            float sum = 0.0f;
            for (int i = 0; i < H; i++)
                sum += cpu_te2_w_[(size_t)j * H + i] * buf0[(size_t)i];
            if (!cpu_te2_b_.empty()) sum += cpu_te2_b_[(size_t)j];
            buf1[(size_t)j] = sum / (1.0f + std::exp(-sum));
        }
    }

    // time_projection.1: Linear(dim, dim*6)
    if (cpu_tp_w_.empty()) {
        for (int j = 0; j < H * 6; j++) mod_buf[j] = 0.0f;
    } else {
        for (int j = 0; j < H * 6; j++) {
            float sum = 0.0f;
            for (int i = 0; i < H; i++)
                sum += cpu_tp_w_[(size_t)j * H + i] * buf1[(size_t)i];
            if (!cpu_tp_b_.empty()) sum += cpu_tp_b_[(size_t)j];
            mod_buf[j] = sum;
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Lazy backend init — called once on first forward
// ══════════════════════════════════════════════════════════════════════════════

bool DiTRunner::ensure_backend(std::string* error) {
    if (backend_initialized_) return true;

    // Pre-read time_embedding weights before GPU migration
    load_cpu_weights();

    namespace bu = audiocore::ggml_utils;
    bp_ = std::make_unique<bu::BackendPair>(bu::backend_init("MSE2-DiT"));
    if (!bp_ || !bp_->backend) {
        if (error) *error = "no GPU backend available";
        return false;
    }

    cuda_backend_ = bp_->backend;

    // Migrate weight tensors in ext_ctx_ to GPU.
    // We bypass the scheduler (it forces INPUT-flagged tensors to CPU) and
    // use ggml_gallocr for lifecycle-based allocation directly on CUDA.
    if (bp_->has_gpu) {
        migrated_buf_ = bu::migrate_ctx_to_backend(ext_ctx_, cuda_backend_,
                                                     "MSE2-DiT");
        if (!migrated_buf_) {
            if (error) *error = "failed to migrate DiT weights to GPU";
            return false;
        }
    }

    backend_initialized_ = true;
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
// Graph-builder + compute, with cached temp context for reuse
// ══════════════════════════════════════════════════════════════════════════════

void DiTRunner::update_mod_tensors(const float* mod_buf, int H, int n_lyr) {
    for (int i = 0; i < n_lyr; i++) {
        float chunks[6][1536];
        for (int j = 0; j < H * 6; j++) {
            float val = mod_buf[j];
            if (i < (int)modulation_f32_.size() && !modulation_f32_[i].empty())
                val += modulation_f32_[i][j];
            int c = j / H;
            chunks[c][j - c * H] = val;
        }
        for (int c = 0; c < 6; c++) {
            size_t idx = static_cast<size_t>(i) * 6 + c;
            if (idx < cg_.mod_tensors.size() && cg_.mod_tensors[idx]) {
                ggml_backend_tensor_set(cg_.mod_tensors[idx], chunks[c], 0,
                                        static_cast<size_t>(H) * sizeof(float));
            }
        }
    }
}

void DiTRunner::release_cached() {
    if (gallocr_) {
        ggml_gallocr_free(gallocr_);
        gallocr_ = nullptr;
    }
    cg_ = CachedGraph{};
    pos_scratch_.clear();
    if (cached_ctx_) {
        ggml_free(cached_ctx_);
        cached_ctx_ = nullptr;
    }
    delete[] cached_ctx_buf_;
    cached_ctx_buf_ = nullptr;
    graph_built_ = false;
}

bool DiTRunner::run_one_forward(
    const float* x_t, int32_t T_latent, int32_t H,
    const float* mod_buf,
    const float* cond_data, int ct_len, int cond_hidden,
    float* result, std::string* error,
    bool rebuild)
{
    const auto& cfg = cfg_;
    ggml_context* ext_ctx = ext_ctx_;
    const int32_t nh     = cfg.n_heads;
    const int32_t hd     = cfg.head_dim;
    const int32_t n_lyr  = cfg.n_layers;
    const float   eps    = cfg.eps;
    const auto&   cuda_backend = cuda_backend_;

    auto weight = [&](const char* name) -> ggml_tensor* {
        return ggml_get_tensor(ext_ctx, name);
    };

    int32_t ct_effective = (cond_data && ct_len > 0) ? ct_len : 1;

    // ── Rebuild graph from scratch ───────────────────────────────────────
    if (rebuild) {
        // Allocate cached context if needed
        if (!cached_ctx_buf_) {
            size_t mem = ggml_tensor_overhead() * 16384 +
                         ggml_graph_overhead_custom(16384, false);
            cached_ctx_buf_ = new (std::nothrow) char[mem];
            if (!cached_ctx_buf_) { if (error) *error = "DiT ctx OOM"; return false; }
            ggml_init_params p = { mem, cached_ctx_buf_, true };
            cached_ctx_ = ggml_init(p);
            if (!cached_ctx_) { delete[] cached_ctx_buf_; cached_ctx_buf_ = nullptr; if (error) *error = "DiT init"; return false; }
        }

        ggml_reset(cached_ctx_);
        ggml_context* ctx = cached_ctx_;

        // Clear cached tensor struct (NOT release_cached — that frees ctx)
        cg_ = CachedGraph{};
        pos_scratch_.clear();
        graph_built_ = false;

        cg_.gf = ggml_new_graph_custom(ctx, 8192, false);
        if (!cg_.gf) { if (error) *error = "DiT graph init"; return false; }

        // ── 1. Text embedding (Linear→GELU(tanh)→Linear) ─────────────
        const float* ctx_src = cond_data;
        if (!cond_data || ct_len <= 0) {
            auto& zc = get_zero_ctx();
            size_t n = static_cast<size_t>(cond_hidden);
            if (zc.size() < n) zc.resize(n);
            ctx_src = zc.data();
        }

        cg_.ctx_emb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                          cond_hidden, ct_effective);
        ggml_set_input(cg_.ctx_emb);

        auto ctx_emb = cg_.ctx_emb;
        {
            auto w = weight("moss_sfx_v2.text_embedding.0.weight");
            auto b = weight("moss_sfx_v2.text_embedding.0.bias");
            if (w) {
                ctx_emb = ggml_mul_mat(ctx, w, ctx_emb);
                if (b) ctx_emb = ggml_add(ctx, ctx_emb, ggml_repeat(ctx, b, ctx_emb));
            }
            ctx_emb = ggml_gelu(ctx, ctx_emb);

            w = weight("moss_sfx_v2.text_embedding.2.weight");
            b = weight("moss_sfx_v2.text_embedding.2.bias");
            if (w) {
                ctx_emb = ggml_mul_mat(ctx, w, ctx_emb);
                if (b) ctx_emb = ggml_add(ctx, ctx_emb, ggml_repeat(ctx, b, ctx_emb));
            }
        }

        // ── 2. Patch embedding ────────────────────────────────────────
        cg_.x_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                      cfg.in_dim, T_latent);
        ggml_set_input(cg_.x_t);

        ggml_tensor* cur = cg_.x_t;
        {
            auto w = weight("moss_sfx_v2.patch_embedding.weight");
            auto b = weight("moss_sfx_v2.patch_embedding.bias");
            if (w) {
                if (ggml_n_dims(w) > 2)
                    w = ggml_reshape_2d(ctx, w, cfg.in_dim, cfg.dim);
                if (w->ne[0] != cfg.in_dim)
                    w = ggml_cont(ctx, ggml_transpose(ctx, w));
                cur = ggml_mul_mat(ctx, w, cur);
                if (b) cur = ggml_add(ctx, cur, ggml_repeat(ctx, b, cur));
            }
        }

        // ── 3. RoPE position arrays ───────────────────────────────────
        cg_.pos_q = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_latent);
        ggml_set_input(cg_.pos_q);

        cg_.pos_k = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ct_effective);
        ggml_set_input(cg_.pos_k);

        // ── 4. DiT layers ─────────────────────────────────────────────
        cg_.mod_tensors.resize(static_cast<size_t>(n_lyr) * 6);
        for (int i = 0; i < n_lyr; i++) {
            char buf[128];

            size_t mbase = static_cast<size_t>(i) * 6;

            cg_.mod_tensors[mbase + 0] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 1);
            ggml_set_input(cg_.mod_tensors[mbase + 0]);

            cg_.mod_tensors[mbase + 1] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 1);
            ggml_set_input(cg_.mod_tensors[mbase + 1]);

            cg_.mod_tensors[mbase + 2] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 1);
            ggml_set_input(cg_.mod_tensors[mbase + 2]);

            cg_.mod_tensors[mbase + 3] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 1);
            ggml_set_input(cg_.mod_tensors[mbase + 3]);

            cg_.mod_tensors[mbase + 4] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 1);
            ggml_set_input(cg_.mod_tensors[mbase + 4]);

            cg_.mod_tensors[mbase + 5] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 1);
            ggml_set_input(cg_.mod_tensors[mbase + 5]);

            auto shift1 = cg_.mod_tensors[mbase + 0];
            auto scale1 = cg_.mod_tensors[mbase + 1];
            auto gate1  = cg_.mod_tensors[mbase + 2];
            auto shift2 = cg_.mod_tensors[mbase + 3];
            auto scale2 = cg_.mod_tensors[mbase + 4];
            auto gate2  = cg_.mod_tensors[mbase + 5];

            // ── 4a. Self-attention ────────────────────────────────────
            {
                auto h = ggml_rms_norm(ctx, cur, eps);
                std::snprintf(buf, sizeof(buf),
                              "moss_sfx_v2.blocks.%d.norm1.weight", i);
                auto nw = ggml_get_tensor(ext_ctx, buf);
                if (nw) h = ggml_mul(ctx, h, ggml_repeat(ctx, nw, h));

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
                                    q_w, k_w, v_w, o_w, qn_w, kn_w,
                                    cg_.pos_q, cg_.gf);
                if (sa)
                    cur = ggml_add(ctx, cur,
                        ggml_mul(ctx, sa, ggml_repeat(ctx, gate1, sa)));
            }

            // ── 4b. Cross-attention ──────────────────────────────────
            {
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
                                      ca_qn, ca_kn,
                                      cg_.pos_q, cg_.pos_k,
                                      cg_.gf);
                if (ca)
                    cur = ggml_add(ctx, cur,
                        ggml_mul(ctx, ca, ggml_repeat(ctx, gate1, ca)));
            }

            // ── 4c. FFN ──────────────────────────────────────────────
            {
                auto h = ggml_rms_norm(ctx, cur, eps);
                std::snprintf(buf, sizeof(buf),
                              "moss_sfx_v2.blocks.%d.norm3.weight", i);
                auto nw = ggml_get_tensor(ext_ctx, buf);
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

                auto ff = ffn_gelu(ctx, h, ffn_w0, ffn_w2);
                if (ff)
                    cur = ggml_add(ctx, cur,
                        ggml_mul(ctx, ff, ggml_repeat(ctx, gate2, ff)));
            }
        }

        // ── 5. Head ──────────────────────────────────────────────────
        {
            auto n_out = ggml_rms_norm(ctx, cur, eps);
            auto nw = weight("moss_sfx_v2.head.norm.weight");
            if (nw) n_out = ggml_mul(ctx, n_out, ggml_repeat(ctx, nw, n_out));

            if (!head_modulation_f32_.empty()) {
                auto hmod = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 2);
                ggml_set_input(hmod);
                cg_.mod_tensors.push_back(hmod);
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

            auto hw = weight("moss_sfx_v2.head.head.weight");
            auto hb = weight("moss_sfx_v2.head.head.bias");
            if (hw) {
                n_out = ggml_mul_mat(ctx, hw, n_out);
                if (hb) n_out = ggml_add(ctx, n_out, ggml_repeat(ctx, hb, n_out));
            }
            cur = n_out;
        }

        // ── 6. Output tensor ──────────────────────────────────────────
        cg_.output = cur;
        ggml_set_output(cg_.output);
        ggml_build_forward_expand(cg_.gf, cg_.output);

        // Allocate via gallocr: lifecycle-based sharing on CUDA, no CPU fallback.
        if (gallocr_) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        gallocr_ = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(cuda_backend));
        if (!gallocr_ || !ggml_gallocr_alloc_graph(gallocr_, cg_.gf)) {
            if (error) *error = "DiT: gallocr_alloc_graph failed";
            return false;
        }

        cg_.T_latent = T_latent;
        cg_.ct_len   = ct_effective;
        graph_built_ = true;
    }

    // ── Upload input data (both rebuild and reuse paths) ──────────────────
    {
        size_t xn = static_cast<size_t>(T_latent) * cfg.in_dim * sizeof(float);
        ggml_backend_tensor_set(cg_.x_t, x_t, 0, xn);

        const float* ctx_src = (cond_data && ct_len > 0) ? cond_data : nullptr;
        if (!ctx_src) {
            auto& zc = get_zero_ctx();
            size_t n = static_cast<size_t>(cond_hidden);
            if (zc.size() < n) zc.resize(n);
            ctx_src = zc.data();
        }
        size_t cn = static_cast<size_t>(ct_effective) * cond_hidden * sizeof(float);
        ggml_backend_tensor_set(cg_.ctx_emb, ctx_src, 0, cn);
    }

    // Position arrays
    {
        pos_scratch_.resize(static_cast<size_t>(T_latent) +
                            static_cast<size_t>(ct_effective));
        for (int j = 0; j < T_latent; j++)
            pos_scratch_[static_cast<size_t>(j)] = static_cast<int32_t>(j);
        ggml_backend_tensor_set(cg_.pos_q, pos_scratch_.data(), 0,
                                static_cast<size_t>(T_latent) * sizeof(int32_t));

        for (int j = 0; j < ct_effective; j++)
            pos_scratch_[static_cast<size_t>(T_latent) + j] = static_cast<int32_t>(j);
        ggml_backend_tensor_set(cg_.pos_k,
                                pos_scratch_.data() + T_latent, 0,
                                static_cast<size_t>(ct_effective) * sizeof(int32_t));
    }

    // Modulation tensors
    update_mod_tensors(mod_buf, H, n_lyr);

    // ── 7. Compute ──────────────────────────────────────────────────────
    auto t0 = ggml_time_us();
    if (ggml_backend_graph_compute(cuda_backend, cg_.gf) != 0) {
        if (error) *error = "DiT CUDA graph_compute failed";
        return false;
    }
    auto t1 = ggml_time_us();
    std::fprintf(stderr, "[dit] fwd %.1f ms\n", (t1 - t0) / 1000.0);

    int64_t n_out = static_cast<int64_t>(T_latent) * cfg.out_dim;
    ggml_backend_tensor_get(cg_.output, result, 0,
                            static_cast<size_t>(n_out) * sizeof(float));

    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
// Forward (public API)
// ══════════════════════════════════════════════════════════════════════════════

bool DiTRunner::forward(const float* x_t, const float* t,
                         const float* context, int32_t T_text,
                         int32_t B, int32_t T_latent,
                         float* output, std::string* error,
                         bool force_rebuild) {
    if (B != 1) {
        if (error) *error = "DiTRunner only supports B=1 for now";
        return false;
    }
    if (!ensure_backend(error)) return false;

    // Auto-detect rebuild: needed if graph never built, or dims changed
    int32_t ct_eff = (context && T_text > 0) ? T_text : 1;
    bool dims_changed = (T_latent != cg_.T_latent || ct_eff != cg_.ct_len);
    bool rebuild = force_rebuild || !graph_built_ || dims_changed;

    const int32_t H       = cfg_.dim;
    const int32_t freq_d  = cfg_.freq_dim;

    std::vector<float> temb(static_cast<size_t>(freq_d));
    timestep_sinusoidal(t, temb.data(), 1, freq_d);

    float mod_buf[9216];
    compute_modulation(temb.data(), freq_d, H, mod_buf);

    return run_one_forward(x_t, T_latent, H, mod_buf,
                            context, T_text, cfg_.text_dim,
                            output, error, rebuild);
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

    // Pad uncond context to T_cond so the graph structure is identical
    // for both forward calls — allows GPU graph reuse without reallocation.
    std::vector<float> padded_uncond;
    const float* uncond_data = context_uncond;
    int32_t eff_T_uncond = T_uncond;
    if (!context_uncond || T_uncond < T_cond) {
        eff_T_uncond = T_cond;
        padded_uncond.resize(static_cast<size_t>(eff_T_uncond) * cfg_.text_dim, 0.0f);
        if (context_uncond && T_uncond > 0) {
            std::memcpy(padded_uncond.data(), context_uncond,
                        static_cast<size_t>(T_uncond) * cfg_.text_dim * sizeof(float));
        }
        uncond_data = padded_uncond.data();
    }

    int64_t n_el = static_cast<int64_t>(T_latent) * cfg_.out_dim;
    std::vector<float> c_out(static_cast<size_t>(n_el));
    std::vector<float> u_out(static_cast<size_t>(n_el));

    // Cond forward: auto-rebuilds on first call, reuses on subsequent
    if (!forward(x_t, t, context_cond, T_cond, B, T_latent,
                  c_out.data(), error, /*force_rebuild=*/false)) {
        if (error) *error = "DiT: cond forward failed";
        return false;
    }
    // Uncond forward: reuse same graph (same dims after padding)
    if (!forward(x_t, t, uncond_data, eff_T_uncond, B, T_latent,
                  u_out.data(), error, /*force_rebuild=*/false)) {
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
