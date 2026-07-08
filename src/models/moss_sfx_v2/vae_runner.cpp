// vae_runner.cpp — DAC VAE decoder: latents [T_latent, 128] → PCM audio.
//
// Each operation builds a tiny ggml sub-graph, computes it, copies output to
// a CPU float vector, then discards the graph (one-at-a-time pipeline).
// When a GPU backend is available (ensure_backend()), ops are routed through
// a ggml_backend_sched that places them on GPU — ~50× faster than CPU.

#include "audiocore/models/moss_sfx_v2/vae_runner.h"
#include "audiocore/framework/ggml/backend_helper.h"  // BackendPair full def for unique_ptr destructor

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <vector>

namespace audiocore::moss_sfx_v2 {

// ── Scheduler helper: run one VAE op graph via GPU scheduler ──────────────
struct VaeInputUpload {
    ggml_tensor* t;
    const void* data;
    size_t nbytes;
    std::shared_ptr<void> owned;  // keeps data buffer alive until upload
};

// ── Log filter: suppress ggml DEBUG-level messages (cosmetic reallocs) ──
static void vae_log_filter(enum ggml_log_level level, const char* text, void* user_data) {
    (void)user_data;
    if (level >= GGML_LOG_LEVEL_INFO) {
        fputs(text, stderr);
        fflush(stderr);
    }
}

struct GgmlLogGuard {
    GgmlLogGuard()  { ggml_log_set(vae_log_filter, nullptr); }
    ~GgmlLogGuard() { ggml_log_set(nullptr, nullptr); }
};

static bool vae_sched_compute(ggml_backend_sched_t sched,
                                ggml_context* ctx, ggml_cgraph* gf,
                                ggml_tensor* out_tensor,
                                const std::vector<VaeInputUpload>& uploads,
                                float* out_data, size_t out_nbytes) {
    if (!sched) return false;
    ggml_set_output(out_tensor);
    ggml_build_forward_expand(gf, out_tensor);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) return false;
    for (const auto& u : uploads) {
        ggml_backend_tensor_set(u.t, u.data, 0, u.nbytes);
    }
    ggml_backend_sched_graph_compute(sched, gf);
    ggml_backend_sched_reset(sched);
    if (out_data && out_nbytes) {
        ggml_backend_tensor_get(out_tensor, out_data, 0, out_nbytes);
    }
    return true;
}

// ── BF16 helpers ─────────────────────────────────────────────────────────────
static float bf16_to_f32(uint16_t bits) {
    uint32_t f32_bits = static_cast<uint32_t>(bits) << 16;
    float f;
    memcpy(&f, &f32_bits, sizeof(f));
    return f;
}

static void bf16_to_f32_buf(const void* src, float* dst, int n) {
    const uint16_t* s = static_cast<const uint16_t*>(src);
    for (int i = 0; i < n; i++) dst[i] = bf16_to_f32(s[i]);
}

// ── Weight-norm: W = g * v / ||v|| ───────────────────────────────────────────
static void compute_wsconv_weight(const float* weight_g, const float* weight_v,
                                   int K, int IC, int OC, bool is_conv1d,
                                   float* out) {
    int dim0 = is_conv1d ? OC : IC;
    int fan  = is_conv1d ? K * IC : K * OC;
    const float eps = 1e-8f;
    for (int d = 0; d < dim0; d++) {
        float g = weight_g[d];
        float nsq = 0.0f;
        for (int i = 0; i < fan; i++)
            nsq += weight_v[static_cast<size_t>(d) * fan + i] *
                   weight_v[static_cast<size_t>(d) * fan + i];
        float s = g / (std::sqrt(nsq) + eps);
        for (int i = 0; i < fan; i++)
            out[static_cast<size_t>(d) * fan + i] =
                weight_v[static_cast<size_t>(d) * fan + i] * s;
    }
}

static void permute_conv_t1d_weight(const float* wsconv,
                                     int K, int IC, int OC, float* out) {
    for (int k = 0; k < K; k++)
        for (int o = 0; o < OC; o++)
            for (int ic = 0; ic < IC; ic++) {
                size_t src = static_cast<size_t>(k) + static_cast<size_t>(o) * K +
                             static_cast<size_t>(ic) * K * OC;
                size_t dst = static_cast<size_t>(ic) +
                            (static_cast<size_t>(k) * OC + static_cast<size_t>(o)) * IC;
                out[dst] = wsconv[src];
            }
}

// ══════════════════════════════════════════════════════════════════════════════
//  Conv1dOp
// ══════════════════════════════════════════════════════════════════════════════

void Conv1dOp::cache_f16() {
    if (!weight_f16.empty() || weight_f32.empty()) return;
    weight_f16.resize(weight_f32.size());
    for (size_t i = 0; i < weight_f32.size(); i++)
        weight_f16[i] = ggml_fp32_to_fp16(weight_f32[i]);
}

bool Conv1dOp::run(const float* x, int T_in, std::vector<float>& out,
                    std::string* error, ggml_backend_sched_t sched,
                    char* scratch, size_t scratch_size) const {
    // Handle unloaded weights (e.g. missing post_quant_conv) — identity pass-through
    if (weight_f32.empty()) {
        size_t n = static_cast<size_t>(T_in) * IC;
        out.assign(x, x + n);
        return true;
    }

    int T_out = (T_in + 2 * pad - dilation * (K - 1) - 1) / stride + 1;
    if (T_out <= 0) { if (error) *error = "Conv1d: T_out <= 0"; return false; }

    const bool use_sched = (sched != nullptr);
    ggml_init_params p = {scratch_size, scratch, /*no_alloc=*/use_sched};
    ggml_context* ctx = ggml_init(p);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 128, false);
    if (!ctx || !gf) { if (error) *error = "Conv1d ggml_init"; return false; }

    std::vector<VaeInputUpload> uploads;

    // Weight as F16 for ggml_conv_1d (ne=[K, IC, OC])
    const size_t n_w = static_cast<size_t>(K) * IC * OC;
    ggml_tensor* w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, K, IC, OC);
    if (use_sched) {
        // Use pre-cached F16 if available; otherwise convert on-the-fly
        if (!weight_f16.empty()) {
            ggml_set_input(w);
            uploads.push_back({w, weight_f16.data(), n_w * sizeof(ggml_fp16_t), nullptr});
        } else {
            // Each conv1d needs its OWN weight buffer — a static thread_local
            // would alias across multiple conv1d ops in the same sub-graph.
            auto w16 = std::make_shared<std::vector<ggml_fp16_t>>(n_w);
            for (size_t i = 0; i < n_w; i++) (*w16)[i] = ggml_fp32_to_fp16(weight_f32[i]);
            ggml_set_input(w);
            uploads.push_back({w, w16->data(), n_w * sizeof(ggml_fp16_t), w16});
        }
    } else {
        ggml_fp16_t* wd = static_cast<ggml_fp16_t*>(w->data);
        for (size_t i = 0; i < n_w; i++) wd[i] = ggml_fp32_to_fp16(weight_f32[i]);
    }

    // Input → ggml layout [T, IC] (col-major: row=t, col=c)
    ggml_tensor* in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_in, IC);
    ggml_set_name(in, "conv1d_in");
    if (use_sched) {
        static thread_local std::vector<float> x_t;
        x_t.resize(static_cast<size_t>(T_in) * IC);
        for (int c = 0; c < IC; c++)
            for (int t = 0; t < T_in; t++)
                x_t[t + static_cast<size_t>(c) * T_in] =
                    x[static_cast<size_t>(t) * IC + c];
        ggml_set_input(in);
        uploads.push_back({in, x_t.data(),
                           static_cast<size_t>(T_in) * IC * sizeof(float)});
    } else {
        for (int c = 0; c < IC; c++)
            for (int t = 0; t < T_in; t++)
                ((float*)in->data)[t + static_cast<size_t>(c) * T_in] =
                    x[static_cast<size_t>(t) * IC + c];
    }

    auto r = ggml_conv_1d(ctx, w, in, stride, pad, dilation);
    if (!bias_.empty()) {
        ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, OC);
        if (use_sched) {
            ggml_set_input(b);
            uploads.push_back({b, bias_.data(),
                               static_cast<size_t>(OC) * sizeof(float)});
        } else {
            memcpy(b->data, bias_.data(), static_cast<size_t>(OC) * sizeof(float));
        }
        r = ggml_add(ctx, r, ggml_repeat(ctx, b, r));
    }

    // Copy out → time-major [T_out, OC]
    out.resize(static_cast<size_t>(T_out) * OC);
    if (use_sched) {
        // GPU output is in ggml column-major [T_out, OC] (ne0=T_out).
        // Download to temp buffer then transpose to time-major row-major.
        static thread_local std::vector<float> raw;
        raw.resize(static_cast<size_t>(T_out) * OC);
        if (!vae_sched_compute(sched, ctx, gf, r, uploads,
                                raw.data(), raw.size() * sizeof(float))) {
            if (error) *error = "Conv1d sched compute failed";
            ggml_free(ctx);
            return false;
        }
        for (int t = 0; t < T_out; t++)
            for (int c = 0; c < OC; c++)
                out[static_cast<size_t>(t) * OC + c] =
                    raw[t + static_cast<size_t>(c) * T_out];
    } else {
        ggml_build_forward_expand(gf, r);
        int st = ggml_graph_compute_with_ctx(ctx, gf, 4);
        if (st != 0) { if (error) *error = "Conv1d compute failed"; ggml_free(ctx); return false; }
        for (int t = 0; t < T_out; t++)
            for (int c = 0; c < OC; c++)
                out[static_cast<size_t>(t) * OC + c] =
                    ((float*)r->data)[t + static_cast<size_t>(c) * T_out];
    }

    ggml_free(ctx);
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
//  ConvT1dOp
// ══════════════════════════════════════════════════════════════════════════════

void ConvT1dOp::cache_f16() {
    if (weight_2d.empty()) return;
    const size_t n_w = static_cast<size_t>(K) * IC * OC;

    // F16 cache (for per-op GPU path)
    if (weight_f16.empty()) {
        weight_f16.resize(n_w);
        for (int ic = 0; ic < IC; ic++)
            for (int oc = 0; oc < OC; oc++)
                for (int k = 0; k < K; k++)
                    weight_f16[k + static_cast<size_t>(oc) * K +
                               static_cast<size_t>(ic) * K * OC] =
                        ggml_fp32_to_fp16(
                            weight_2d[static_cast<size_t>(k) * OC * IC +
                                      static_cast<size_t>(oc) * IC + ic]);
    }

    // F32 transposed cache (for monolithic graph path — CUDA ConvT1d requires F32)
    if (weight_f32_t.empty()) {
        weight_f32_t.resize(n_w);
        for (int ic = 0; ic < IC; ic++)
            for (int oc = 0; oc < OC; oc++)
                for (int k = 0; k < K; k++)
                    weight_f32_t[k + static_cast<size_t>(oc) * K +
                                 static_cast<size_t>(ic) * K * OC] =
                        weight_2d[static_cast<size_t>(k) * OC * IC +
                                  static_cast<size_t>(oc) * IC + ic];
    }

    // Col2im-ordered weight: same values as weight_2d but with the
    // K*OC column dimension indexed as oc*K+k (matching col2im_1d kernel)
    // instead of k*OC+oc (matching ggml_conv_transpose_1d).
    if (weight_2d_c2i.empty()) {
        weight_2d_c2i.resize(static_cast<size_t>(IC) * K * OC);
        for (int ic = 0; ic < IC; ic++)
            for (int oc = 0; oc < OC; oc++)
                for (int k = 0; k < K; k++)
                    weight_2d_c2i[static_cast<size_t>(ic) +
                                  (static_cast<size_t>(oc) * K + k) * IC] =
                        weight_2d[static_cast<size_t>(ic) +
                                  (static_cast<size_t>(k) * OC + oc) * IC];
    }
}

bool ConvT1dOp::run(const float* x, int T_in, std::vector<float>& out,
                     std::string* error, ggml_backend_sched_t sched,
                     char* scratch, size_t scratch_size) const {
    int T_out = (T_in - 1) * stride + K - 2 * pad + output_padding;

    const bool use_sched = (sched != nullptr);
    ggml_init_params p = {scratch_size, scratch, /*no_alloc=*/use_sched};
    ggml_context* ctx = ggml_init(p);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 128, false);
    if (!ctx || !gf) { if (error) *error = "ConvT1d ggml_init"; return false; }

    std::vector<VaeInputUpload> uploads;

    // weight_2d stored in [K, OC, IC] row-major (from permute_conv_t1d_weight).
    // ggml_conv_transpose_1d expects ne=[K, OC, IC] column-major.
    // Transpose from row-major to column-major during F16 conversion.
    const size_t n_w = static_cast<size_t>(K) * IC * OC;
    ggml_tensor* w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, K, OC, IC);
    if (use_sched) {
        if (!weight_f16.empty()) {
            ggml_set_input(w);
            uploads.push_back({w, weight_f16.data(), n_w * sizeof(ggml_fp16_t), nullptr});
        } else {
            auto w16 = std::make_shared<std::vector<ggml_fp16_t>>(n_w);
            for (int ic = 0; ic < IC; ic++)
                for (int oc = 0; oc < OC; oc++)
                    for (int k = 0; k < K; k++)
                        (*w16)[k + static_cast<size_t>(oc) * K +
                            static_cast<size_t>(ic) * K * OC] =
                            ggml_fp32_to_fp16(
                                weight_2d[static_cast<size_t>(k) * OC * IC +
                                          static_cast<size_t>(oc) * IC + ic]);
            ggml_set_input(w);
            uploads.push_back({w, w16->data(), n_w * sizeof(ggml_fp16_t), w16});
        }
    } else {
        ggml_fp16_t* wd = static_cast<ggml_fp16_t*>(w->data);
        for (int ic = 0; ic < IC; ic++)
            for (int oc = 0; oc < OC; oc++)
                for (int k = 0; k < K; k++)
                    wd[k + static_cast<size_t>(oc) * K +
                       static_cast<size_t>(ic) * K * OC] =
                        ggml_fp32_to_fp16(
                            weight_2d[static_cast<size_t>(k) * OC * IC +
                                      static_cast<size_t>(oc) * IC + ic]);
    }

    // Input: ne=[T_in, IC], col-major
    ggml_tensor* in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_in, IC);
    if (use_sched) {
        static thread_local std::vector<float> x_t;
        x_t.resize(static_cast<size_t>(T_in) * IC);
        for (int c = 0; c < IC; c++)
            for (int t = 0; t < T_in; t++)
                x_t[t + static_cast<size_t>(c) * T_in] =
                    x[static_cast<size_t>(t) * IC + c];
        ggml_set_input(in);
        uploads.push_back({in, x_t.data(),
                           static_cast<size_t>(T_in) * IC * sizeof(float)});
    } else {
        for (int c = 0; c < IC; c++)
            for (int t = 0; t < T_in; t++)
                ((float*)in->data)[t + static_cast<size_t>(c) * T_in] =
                    x[static_cast<size_t>(t) * IC + c];
    }

    auto r = ggml_conv_transpose_1d(ctx, w, in, stride, 0, 1);
    if (!bias_.empty()) {
        ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, OC);
        if (use_sched) {
            ggml_set_input(b);
            uploads.push_back({b, bias_.data(),
                               static_cast<size_t>(OC) * sizeof(float)});
        } else {
            memcpy(b->data, bias_.data(), static_cast<size_t>(OC) * sizeof(float));
        }
        r = ggml_add(ctx, r, ggml_repeat(ctx, b, r));
    }

    // ggml_conv_transpose_1d with pad=0 gives: ggml_T = (T_in-1)*stride + K
    // We need: T_out = (T_in-1)*stride + K - 2*pad + output_padding
    // Crop by pad from each side — the extra output_padding falls within ggml's output since
    // pad >= output_padding for all valid stride values.
    int ggml_T = (T_in - 1) * stride + K;
    int crop_offset = pad;
    out.resize(static_cast<size_t>(T_out) * OC);
    if (use_sched) {
        // Download full output then crop
        static thread_local std::vector<float> raw;
        raw.resize(static_cast<size_t>(ggml_T) * OC);
        if (!vae_sched_compute(sched, ctx, gf, r, uploads,
                                raw.data(), raw.size() * sizeof(float))) {
            if (error) *error = "ConvT1d sched compute failed";
            ggml_free(ctx);
            return false;
        }
        for (int t = 0; t < T_out; t++)
            for (int c = 0; c < OC; c++)
                out[static_cast<size_t>(t) * OC + c] =
                    raw[(crop_offset + t) + static_cast<size_t>(c) * ggml_T];
    } else {
        ggml_build_forward_expand(gf, r);
        int st = ggml_graph_compute_with_ctx(ctx, gf, 4);
        if (st != 0) { if (error) *error = "ConvT1d compute failed"; ggml_free(ctx); return false; }
        int actual_T = static_cast<int>(r->ne[0]);
        for (int t = 0; t < T_out; t++)
            for (int c = 0; c < OC; c++)
                out[static_cast<size_t>(t) * OC + c] =
                    ((float*)r->data)[(crop_offset + t) +
                                      static_cast<size_t>(c) * actual_T];
    }

    ggml_free(ctx);
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
//  SnakeOp
// ══════════════════════════════════════════════════════════════════════════════

bool SnakeOp::run(const float* x, int T, std::vector<float>& out,
                   std::string* error, ggml_backend_sched_t sched,
                   char* scratch, size_t scratch_size) const {
    const bool use_sched = (sched != nullptr);
    ggml_init_params p = {scratch_size, scratch, /*no_alloc=*/use_sched};
    ggml_context* ctx = ggml_init(p);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 128, false);
    if (!ctx || !gf) { if (error) *error = "Snake ggml_init"; return false; }

    std::vector<VaeInputUpload> uploads;

    // Input time-major [T, C] → ggml [T, C] (ne0=T, ne1=C)
    ggml_tensor* in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, C);
    if (use_sched) {
        static thread_local std::vector<float> x_t;
        x_t.resize(static_cast<size_t>(T) * C);
        for (int c = 0; c < C; c++)
            for (int t = 0; t < T; t++)
                x_t[t + static_cast<size_t>(c) * T] =
                    x[static_cast<size_t>(t) * C + c];
        ggml_set_input(in);
        uploads.push_back({in, x_t.data(),
                           static_cast<size_t>(T) * C * sizeof(float)});
    } else {
        for (int c = 0; c < C; c++)
            for (int t = 0; t < T; t++)
                ((float*)in->data)[t + static_cast<size_t>(c) * T] =
                    x[static_cast<size_t>(t) * C + c];
    }

    // Alpha [1, C] and inv_alpha [1, C]
    ggml_tensor* a_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
    ggml_tensor* ia_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
    if (use_sched) {
        ggml_set_input(a_t);
        ggml_set_input(ia_t);
        uploads.push_back({a_t, alpha.data(),
                           static_cast<size_t>(C) * sizeof(float)});
        uploads.push_back({ia_t, inv_alpha.data(),
                           static_cast<size_t>(C) * sizeof(float)});
    } else {
        memcpy(a_t->data, alpha.data(), static_cast<size_t>(C) * sizeof(float));
        memcpy(ia_t->data, inv_alpha.data(), static_cast<size_t>(C) * sizeof(float));
    }

    auto ax   = ggml_mul(ctx, in, ggml_repeat(ctx, a_t, in));
    auto s    = ggml_sin(ctx, ax);
    auto s2   = ggml_sqr(ctx, s);
    auto term = ggml_mul(ctx, s2, ggml_repeat(ctx, ia_t, s2));
    auto r    = ggml_add(ctx, in, term);

    out.resize(static_cast<size_t>(T) * C);
    if (use_sched) {
        // GPU output is ggml column-major [T, C] (ne0=T). Transpose to row-major.
        static thread_local std::vector<float> raw;
        raw.resize(static_cast<size_t>(T) * C);
        if (!vae_sched_compute(sched, ctx, gf, r, uploads,
                                raw.data(), raw.size() * sizeof(float))) {
            if (error) *error = "Snake sched compute failed";
            ggml_free(ctx);
            return false;
        }
        for (int c = 0; c < C; c++)
            for (int t = 0; t < T; t++)
                out[static_cast<size_t>(t) * C + c] =
                    raw[t + static_cast<size_t>(c) * T];
    } else {
        ggml_build_forward_expand(gf, r);
        int st = ggml_graph_compute_with_ctx(ctx, gf, 4);
        if (st != 0) { if (error) *error = "Snake compute failed"; ggml_free(ctx); return false; }
        for (int c = 0; c < C; c++)
            for (int t = 0; t < T; t++)
                out[static_cast<size_t>(t) * C + c] =
                    ((float*)r->data)[t + static_cast<size_t>(c) * T];
    }
    ggml_free(ctx);
    return true;
}

// ── Tanh helper ─────────────────────────────────────────────────────────────

static bool op_tanh(const float* x, int T, int C,
                     std::vector<float>& out, std::string* error,
                     ggml_backend_sched_t sched,
                     char* scratch, size_t scratch_size) {
    const bool use_sched = (sched != nullptr);
    ggml_init_params p = {scratch_size, scratch, /*no_alloc=*/use_sched};
    ggml_context* ctx = ggml_init(p);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 128, false);
    if (!ctx || !gf) { if (error) *error = "Tanh ggml_init"; return false; }

    std::vector<VaeInputUpload> uploads;

    ggml_tensor* in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, C);
    if (use_sched) {
        static thread_local std::vector<float> x_t;
        x_t.resize(static_cast<size_t>(T) * C);
        for (int c = 0; c < C; c++)
            for (int t = 0; t < T; t++)
                x_t[t + static_cast<size_t>(c) * T] =
                    x[static_cast<size_t>(t) * C + c];
        ggml_set_input(in);
        uploads.push_back({in, x_t.data(),
                           static_cast<size_t>(T) * C * sizeof(float)});
    } else {
        for (int c = 0; c < C; c++)
            for (int t = 0; t < T; t++)
                ((float*)in->data)[t + static_cast<size_t>(c) * T] =
                    x[static_cast<size_t>(t) * C + c];
    }
    auto r = ggml_tanh(ctx, in);

    out.resize(static_cast<size_t>(T) * C);
    if (use_sched) {
        static thread_local std::vector<float> raw;
        raw.resize(static_cast<size_t>(T) * C);
        if (!vae_sched_compute(sched, ctx, gf, r, uploads,
                                raw.data(), raw.size() * sizeof(float))) {
            if (error) *error = "Tanh sched compute failed";
            ggml_free(ctx);
            return false;
        }
        for (int c = 0; c < C; c++)
            for (int t = 0; t < T; t++)
                out[static_cast<size_t>(t) * C + c] =
                    raw[t + static_cast<size_t>(c) * T];
    } else {
        ggml_build_forward_expand(gf, r);
        int st = ggml_graph_compute_with_ctx(ctx, gf, 4);
        if (st != 0) { if (error) *error = "Tanh compute failed"; ggml_free(ctx); return false; }
        for (int c = 0; c < C; c++)
            for (int t = 0; t < T; t++)
                out[static_cast<size_t>(t) * C + c] =
                    ((float*)r->data)[t + static_cast<size_t>(c) * T];
    }
    ggml_free(ctx);
    return true;
}

// ── GGUF helpers ────────────────────────────────────────────────────────────

static bool load_1d_bf16(ggml_context* ext_ctx, const char* name,
                          std::vector<float>& out) {
    ggml_tensor* t = ggml_get_tensor(ext_ctx, name);
    if (!t) return false;
    int n = static_cast<int>(ggml_nelements(t));
    out.resize(n);
    if (t->type == GGML_TYPE_BF16) {
        bf16_to_f32_buf(t->data, out.data(), n);
    } else if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), t->data, static_cast<size_t>(n) * sizeof(float));
    } else {
        return false;
    }
    return true;
}

static bool load_snake(ggml_context* ext_ctx, const char* name, SnakeOp* op) {
    if (!load_1d_bf16(ext_ctx, name, op->alpha)) return false;
    op->C = static_cast<int>(op->alpha.size());
    op->inv_alpha.resize(op->C);
    for (int i = 0; i < op->C; i++)
        op->inv_alpha[i] = 1.0f / (op->alpha[i] + 1e-9f);
    return true;
}

static bool load_wsconv1d(ggml_context* ext_ctx, const char* prefix,
                           int K, int IC, int OC,
                           Conv1dOp* op) {
    op->K = K; op->IC = IC; op->OC = OC;
    op->stride = 1; op->pad = 3; op->dilation = 1;

    std::string vn = std::string(prefix) + ".weight_v";
    std::string gn = std::string(prefix) + ".weight_g";
    std::string bn = std::string(prefix) + ".bias";

    std::vector<float> wv, wg;
    if (!load_1d_bf16(ext_ctx, vn.c_str(), wv)) return false;
    if (!load_1d_bf16(ext_ctx, gn.c_str(), wg)) return false;
    load_1d_bf16(ext_ctx, bn.c_str(), op->bias_);

    op->weight_f32.resize(static_cast<size_t>(K) * IC * OC);
    compute_wsconv_weight(wg.data(), wv.data(), K, IC, OC, true,
                          op->weight_f32.data());
    return true;
}

static bool load_wsconv_t1d(ggml_context* ext_ctx, const char* prefix,
                             int K, int IC, int OC, int stride,
                             ConvT1dOp* op) {
    op->K = K; op->IC = IC; op->OC = OC;
    op->stride = stride; op->pad = (stride + 1) / 2; op->output_padding = stride % 2;

    std::string vn = std::string(prefix) + ".weight_v";
    std::string gn = std::string(prefix) + ".weight_g";
    std::string bn = std::string(prefix) + ".bias";

    std::vector<float> wv, wg;
    if (!load_1d_bf16(ext_ctx, vn.c_str(), wv)) return false;
    if (!load_1d_bf16(ext_ctx, gn.c_str(), wg)) return false;
    load_1d_bf16(ext_ctx, bn.c_str(), op->bias_);

    std::vector<float> wsconv(static_cast<size_t>(K) * OC * IC);
    compute_wsconv_weight(wg.data(), wv.data(), K, IC, OC, false,
                          wsconv.data());
    op->weight_2d.resize(static_cast<size_t>(IC) * K * OC);
    permute_conv_t1d_weight(wsconv.data(), K, IC, OC, op->weight_2d.data());
    return true;
}

// Load a plain (non-weight-normalized) Conv1d weight.
// PyTorch stores Conv1d weight as [OC, IC, K] (row-major).
// ggml_conv_1d expects ne=[K, IC, OC] with element (k, ic, oc) at
// flat offset k + ic*K + oc*K*IC.  We transpose accordingly.
static bool load_conv1d_plain(ggml_context* ext_ctx, const char* prefix,
                               int K, int IC, int OC, int stride, int pad,
                               Conv1dOp* op) {
    op->K = K; op->IC = IC; op->OC = OC;
    op->stride = stride; op->pad = pad; op->dilation = 1;

    std::string wn = std::string(prefix) + ".weight";
    std::string bn = std::string(prefix) + ".bias";

    // PyTorch [OC, IC, K] → flat pt[oc * IC * K + ic * K + k]
    std::vector<float> pt_w;
    if (!load_1d_bf16(ext_ctx, wn.c_str(), pt_w)) return false;
    load_1d_bf16(ext_ctx, bn.c_str(), op->bias_);

    // ggml layout: element (k, ic, oc) at k + ic*K + oc*K*IC
    op->weight_f32.resize(static_cast<size_t>(K) * IC * OC);
    for (int oc = 0; oc < OC; oc++)
        for (int ic = 0; ic < IC; ic++)
            for (int k = 0; k < K; k++) {
                size_t pt = static_cast<size_t>(oc) * IC * K +
                            static_cast<size_t>(ic) * K + k;
                size_t gm = static_cast<size_t>(k) +
                            static_cast<size_t>(ic) * K +
                            static_cast<size_t>(oc) * K * IC;
                op->weight_f32[gm] = pt_w[pt];
            }
    return true;
}

// ══════════════════════════════════════════════════════════════════════════════
//  VAERunner
// ══════════════════════════════════════════════════════════════════════════════

VAERunner::VAERunner(ggml_context* ext_ctx, const VAEConfig& cfg)
    : ext_ctx_(ext_ctx), cfg_(cfg) {

    // Continuous-mode VAE: post_quant_conv (1×1 Conv1d, latent_dim → latent_dim)
    // applied to DiT latents BEFORE the decoder.
    if (cfg_.continuous) {
        if (!load_conv1d_plain(ext_ctx_, "moss_sfx_v2.vae.post_quant_conv",
                               1, cfg_.latent_dim, cfg_.latent_dim,
                               1, 0, &post_quant_conv_)) {
            std::fprintf(stderr,
                         "[vae] WARNING: continuous=True but post_quant_conv "
                         "weights not found in GGUF — decode will be wrong\n");
        }
    }

    // model[0]: WNConv1d(latent_dim, decoder_dim, k=7, pad=3)
    load_wsconv1d(ext_ctx_, "moss_sfx_v2.vae.0",
                   7, cfg_.latent_dim, cfg_.decoder_dim, &conv_in_);

    // Decoder strides from checkpoint metadata: decoder_rates = [8, 5, 4, 3, 2]
    // Total upsampling = 8 × 5 × 4 × 3 × 2 = 960 (matches hop_length)
    static const int strides[5] = {8, 5, 4, 3, 2};
    static const int n_blocks = 5;
    for (int bi = 0; bi < n_blocks; bi++) {
        DecoderBlockOps blk;
        int s = strides[bi];
        int in_dim  = cfg_.decoder_dim >> bi;
        int out_dim = cfg_.decoder_dim >> (bi + 1);

        char buf[128];

        // Snake
        std::snprintf(buf, sizeof(buf),
                      "moss_sfx_v2.vae.%d.block.0.alpha", bi + 1);
        load_snake(ext_ctx_, buf, &blk.snake);

        // ConvTranspose1d  (kernel = 2 × stride)
        std::snprintf(buf, sizeof(buf),
                      "moss_sfx_v2.vae.%d.block.1", bi + 1);
        load_wsconv_t1d(ext_ctx_, buf, 2 * s, in_dim, out_dim, s, &blk.conv_t);

        // 3× ResidualUnits  (dilations 1, 3, 9)
        int dilations[3] = {1, 3, 9};
        for (int ri = 0; ri < 3; ri++) {
            int dil = dilations[ri];
            int pad = ((7 - 1) * dil) / 2;
            ResUnitOps ru;

            // Snake 1
            std::snprintf(buf, sizeof(buf),
                          "moss_sfx_v2.vae.%d.block.%d.block.0.alpha",
                          bi + 1, ri + 2);
            load_snake(ext_ctx_, buf, &ru.snake1);

            // Conv1 (k=7, dilated)
            std::snprintf(buf, sizeof(buf),
                          "moss_sfx_v2.vae.%d.block.%d.block.1",
                          bi + 1, ri + 2);
            load_wsconv1d(ext_ctx_, buf, 7, out_dim, out_dim, &ru.conv1);
            ru.conv1.stride = 1;
            ru.conv1.pad = pad;
            ru.conv1.dilation = dil;

            // Snake 2
            std::snprintf(buf, sizeof(buf),
                          "moss_sfx_v2.vae.%d.block.%d.block.2.alpha",
                          bi + 1, ri + 2);
            load_snake(ext_ctx_, buf, &ru.snake2);

            // Conv2 (k=1)
            std::snprintf(buf, sizeof(buf),
                          "moss_sfx_v2.vae.%d.block.%d.block.3",
                          bi + 1, ri + 2);
            load_wsconv1d(ext_ctx_, buf, 1, out_dim, out_dim, &ru.conv2);
            ru.conv2.pad = 0;

            blk.res_units[ri] = std::move(ru);
        }
        blocks_.push_back(std::move(blk));
    }

    // model[6]: Snake (final) — index vae.6 (after 5 blocks + conv_in)
    load_snake(ext_ctx_, "moss_sfx_v2.vae.6.alpha", &snake_out_);

    // model[7]: WNConv1d(decoder_dim/32, 1, k=7, pad=3)
    load_wsconv1d(ext_ctx_, "moss_sfx_v2.vae.7",
                   7, cfg_.decoder_dim >> 5, 1, &conv_out_);

    std::fprintf(stderr, "[vae] decoder built: %d blocks, latent=%d dim=%d\n",
                 (int)blocks_.size(), cfg_.latent_dim, cfg_.decoder_dim);

    // Pre-convert all weights to F16 for GPU upload — avoids per-call conversion.
    post_quant_conv_.cache_f16();
    conv_in_.cache_f16();
    for (auto& blk : blocks_) {
        blk.conv_t.cache_f16();
        for (auto& ru : blk.res_units) {
            ru.conv1.cache_f16();
            ru.conv2.cache_f16();
        }
    }
    conv_out_.cache_f16();
    std::fprintf(stderr, "[vae] F16 weight cache built\n");
}

VAERunner::~VAERunner() {
    if (sched_) {
        ggml_backend_sched_free(sched_);
        sched_ = nullptr;
    }
    backend_pair_.reset();
}

bool VAERunner::ensure_backend() {
    if (backend_ready_) return sched_ != nullptr;

    namespace bu = audiocore::ggml_utils;
    backend_pair_  = std::make_unique<bu::BackendPair>(bu::backend_init("MSE2-VAE"));
    sched_         = bu::backend_sched_new(*backend_pair_, 8192);
    backend_ready_ = (sched_ != nullptr);
    if (backend_ready_) {
        std::fprintf(stderr, "[vae] GPU backend ready: has_gpu=%d\n",
                     backend_pair_->has_gpu ? 1 : 0);
    }
    return backend_ready_;
}

// ══════════════════════════════════════════════════════════════════════════════
//  Monolithic graph decode — builds the ENTIRE VAE as one ggml_cgraph.
//  All intermediates stay on GPU; only 1 upload (input) + 1 download (output).
//  This eliminates the ~95× per-op graph-alloc/upload/download overhead.
// ══════════════════════════════════════════════════════════════════════════════

// Create a 1D input tensor (alpha, bias, etc.) and queue for upload.
static ggml_tensor* graph_input_1d(ggml_context* ctx, const float* data,
                                     int C, std::vector<VaeInputUpload>& uploads,
                                     const char* name = nullptr) {
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
    ggml_set_input(t);
    if (name) ggml_set_name(t, name);
    uploads.push_back({t, data, static_cast<size_t>(C) * sizeof(float)});
    return t;
}

// Create a 3D F16 weight tensor from cached F16 data and queue for upload.
static ggml_tensor* graph_weight_f16(ggml_context* ctx, const mse2_fp16_t* data,
                                      int K, int D1, int D2,
                                      std::vector<VaeInputUpload>& uploads,
                                      const char* name = nullptr) {
    ggml_tensor* t = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, K, D1, D2);
    ggml_set_input(t);
    if (name) ggml_set_name(t, name);
    uploads.push_back({t, data,
                       static_cast<size_t>(K) * D1 * D2 * sizeof(mse2_fp16_t)});
    return t;
}

// Build Snake activation nodes: out = x + sin²(α·x) · inv_α
static ggml_tensor* graph_snake(ggml_context* ctx, ggml_tensor* x,
                                  const SnakeOp& sn,
                                  std::vector<VaeInputUpload>& uploads,
                                  const char* prefix = "") {
    // Alpha and inv_alpha are [C] vectors — reshape to [1, C] for broadcast
    ggml_tensor* a_t  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, sn.C);
    ggml_tensor* ia_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, sn.C);
    ggml_set_input(a_t);
    ggml_set_input(ia_t);
    uploads.push_back({a_t, sn.alpha.data(),
                       static_cast<size_t>(sn.C) * sizeof(float)});
    uploads.push_back({ia_t, sn.inv_alpha.data(),
                       static_cast<size_t>(sn.C) * sizeof(float)});

    // x has ne=[T, C]. Alpha [C] needs to broadcast across T.
    // ggml_repeat broadcasts: alpha[C] → [T, C] (matching x's shape)
    // But alpha is 1D [C], x is 2D [T, C]. Need to reshape alpha to [1, C] first.
    ggml_tensor* a_2d  = ggml_reshape_2d(ctx, a_t, 1, sn.C);
    ggml_tensor* ia_2d = ggml_reshape_2d(ctx, ia_t, 1, sn.C);

    ggml_tensor* ax    = ggml_mul(ctx, x, ggml_repeat(ctx, a_2d, x));
    ggml_tensor* s     = ggml_sin(ctx, ax);
    ggml_tensor* s2    = ggml_sqr(ctx, s);
    ggml_tensor* term  = ggml_mul(ctx, s2, ggml_repeat(ctx, ia_2d, s2));
    ggml_tensor* r     = ggml_add(ctx, x, term);
    (void)prefix;
    return r;
}

// Build Conv1d nodes: weight [K, IC, OC] F16, input [T, IC] → output [T_out, OC]
// Uses im2col + mul_mat decomposition so all ops run on CUDA (ggml_conv_1d
// as a single op has no CUDA backend, but im2col and mul_mat do).
static ggml_tensor* graph_conv1d(ggml_context* ctx, ggml_tensor* x,
                                   const Conv1dOp& conv,
                                   std::vector<VaeInputUpload>& uploads,
                                   const char* prefix = "") {
    ggml_tensor* w = graph_weight_f16(ctx, conv.weight_f16.data(),
                                       conv.K, conv.IC, conv.OC, uploads, prefix);

    // im2col: input [T_in, IC] + kernel [K, IC, OC] → [IC*K, T_out, 1] (F16)
    ggml_tensor* im2col = ggml_im2col(ctx, w, x,
                                       conv.stride, 0, conv.pad, 0,
                                       conv.dilation, 0, false, GGML_TYPE_F16);

    // mul_mat: [IC*K, T_out] @ [K*IC, OC] → [T_out, OC]
    ggml_tensor* col_2d = ggml_reshape_2d(ctx, im2col,
                                           im2col->ne[0],
                                           im2col->ne[1] * im2col->ne[2]);
    ggml_tensor* w_2d = ggml_reshape_2d(ctx, w,
                                         w->ne[0] * w->ne[1], w->ne[2]);
    ggml_tensor* r = ggml_mul_mat(ctx, col_2d, w_2d);

    // Reshape to [T_out, OC, 1]
    r = ggml_reshape_3d(ctx, r, im2col->ne[1], conv.OC, 1);

    if (!conv.bias_.empty()) {
        ggml_tensor* b = graph_input_1d(ctx, conv.bias_.data(), conv.OC, uploads);
        // Bias [OC] → reshape to [1, OC] for broadcast
        ggml_tensor* b_2d = ggml_reshape_2d(ctx, b, 1, conv.OC);
        r = ggml_add(ctx, r, ggml_repeat(ctx, b_2d, r));
    }
    return r;
}

// Build ConvTranspose1d via mul_mat + col2im_1d (both CUDA-accelerated).
// This avoids the naive CUDA ConvT1d kernel and the CPU-only ConvT1d fallback.
// x: [T_in, IC] → output: [T_out, OC]
static ggml_tensor* graph_conv_t1d(ggml_context* ctx, ggml_tensor* x,
                                     const ConvT1dOp& conv,
                                     std::vector<VaeInputUpload>& uploads,
                                     const char* prefix = "") {
    // x has ne=[T_in, IC]. Transpose to [IC, T_in] for mul_mat contraction over IC.
    ggml_tensor* x_t = ggml_cont(ctx, ggml_transpose(ctx, x));

    // Weight: weight_2d_c2i has ne=[IC, K*OC] with the K*OC column dimension
    // indexed as oc*K+k — the ordering that col2im_1d's CUDA kernel expects.
    // (The old weight_2d uses k*OC+oc, which transposes (oc,k) in col2im.)
    ggml_tensor* w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                         conv.IC, conv.K * conv.OC);
    ggml_set_input(w);
    if (prefix) ggml_set_name(w, prefix);
    uploads.push_back({w, conv.weight_2d_c2i.data(),
                       static_cast<size_t>(conv.IC) * conv.K * conv.OC * sizeof(float)});

    // mul_mat(w[IC, K*OC], x_t[IC, T_in]) → col [K*OC, T_in]
    // col row j=oc*K+k holds sum_ic W[ic,oc,k] * x[t_in,ic]
    ggml_tensor* col = ggml_mul_mat(ctx, w, x_t);

    // col2im_1d: [K*OC, T_in] → [T_out, OC]
    // Kernel reads col[(oc*K+k) + t_in*K_OC] — now matches our column ordering.
    // T_out = (T_in - 1) * stride + K - 2 * pad
    ggml_tensor* r = ggml_col2im_1d(ctx, col, conv.stride, conv.OC, conv.pad);

    if (!conv.bias_.empty()) {
        ggml_tensor* b = graph_input_1d(ctx, conv.bias_.data(), conv.OC, uploads);
        ggml_tensor* b_2d = ggml_reshape_2d(ctx, b, 1, conv.OC);
        r = ggml_add(ctx, r, ggml_repeat(ctx, b_2d, r));
    }
    return r;
}

// ══════════════════════════════════════════════════════════════════════════════
//  Per-block sub-graph decode — builds a small graph for each decoder block
//  to reduce peak memory and allow per-block timing. Each sub-graph runs
//  entirely on GPU; data transfers between blocks use CPU buffers.
// ══════════════════════════════════════════════════════════════════════════════

// Execute a sub-graph built by `builder`. Input is column-major [T_in, C_in].
// Output is returned in `output_col` (column-major [T_out, C_out]).
// The builder receives (ctx, input_tensor, uploads) and returns the output tensor.
struct SubGraphIO {
    std::vector<float> data;  // column-major [T, C]
    int T = 0, C = 0;
};

template<typename Builder>
static bool run_sub_graph(
    ggml_backend_t cuda_backend,
    std::vector<char>& scratch,
    const float* input_col, int T_in, int C_in,
    Builder builder,
    SubGraphIO* output,
    std::string* error,
    const char* label = "")
{
    ggml_init_params gip = {scratch.size(), scratch.data(), /*no_alloc=*/true};
    ggml_context* ctx = ggml_init(gip);
    if (!ctx) { if (error) *error = "sub-graph ggml_init failed"; return false; }

    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 4096, false);
    if (!gf) { if (error) *error = "sub-graph cgraph failed"; ggml_free(ctx); return false; }

    std::vector<VaeInputUpload> uploads;

    // Input tensor: column-major [T_in, C_in]
    ggml_tensor* in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_in, C_in);
    ggml_set_input(in);
    uploads.push_back({in, input_col,
                       static_cast<size_t>(T_in) * C_in * sizeof(float)});

    // Build the sub-graph
    ggml_tensor* result = builder(ctx, in, uploads);
    ggml_set_output(result);
    ggml_build_forward_expand(gf, result);

    // Allocate ALL tensors on the CUDA backend directly.
    // We bypass the scheduler entirely: the scheduler's INPUT flag handling
    // forces inputs to the last backend (CPU), preventing CUDA offload.
    // By allocating directly on CUDA, all ops run on GPU.
    auto _t0 = ggml_time_us();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cuda_backend);
    if (!buf) {
        if (error) *error = std::string(label) + ": CUDA alloc_ctx_tensors failed";
        ggml_free(ctx);
        return false;
    }
    auto _t1 = ggml_time_us();

    // Upload inputs and weights
    for (const auto& u : uploads)
        ggml_backend_tensor_set(u.t, u.data, 0, u.nbytes);
    auto _t2 = ggml_time_us();

    // Compute on CUDA
    ggml_backend_graph_compute(cuda_backend, gf);
    auto _t3 = ggml_time_us();

    // Download
    output->T = (int)result->ne[0];
    output->C = (int)result->ne[1];
    size_t n = static_cast<size_t>(output->T) * output->C;
    output->data.resize(n);
    ggml_backend_tensor_get(result, output->data.data(), 0, n * sizeof(float));
    auto _t4 = ggml_time_us();

    if (label && label[0])
        std::fprintf(stderr,
            "[vae]         %-16s alloc=%dms upload=%dms"
            " compute=%dms dl=%dms nodes=%zu\n",
            label,
            (int)((_t1-_t0)/1000), (int)((_t2-_t1)/1000),
            (int)((_t3-_t2)/1000), (int)((_t4-_t3)/1000),
            (size_t)ggml_graph_n_nodes(gf));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

bool VAERunner::decode(const float* z, int32_t B, int32_t T_latent,
                        float* out, std::string* error) {
    if (B != 1) {
        if (error) *error = "VAE only supports B=1";
        return false;
    }

    GgmlLogGuard _log_guard;
    if (!backend_ready_ && !ensure_backend()) {
        // No GPU — fall back to per-op decode
        return decode_traced(z, B, T_latent, out, nullptr, error);
    }
    if (!backend_ready_) {
        return decode_traced(z, B, T_latent, out, nullptr, error);
    }

    auto t_start = ggml_time_us();

    // Scratch buffer for graph metadata — reused across sub-graphs.
    const size_t meta_size = 128ULL * 1024 * 1024;  // 128 MB
    scratch_.resize(meta_size);

    // ── Prepare initial input: z time-major → column-major ──────────────
    SubGraphIO cur_io;
    cur_io.T = T_latent;
    cur_io.C = cfg_.latent_dim;
    cur_io.data.resize(static_cast<size_t>(T_latent) * cfg_.latent_dim);
    for (int c = 0; c < cfg_.latent_dim; c++)
        for (int t = 0; t < T_latent; t++)
            cur_io.data[static_cast<size_t>(t) + static_cast<size_t>(c) * T_latent] =
                z[static_cast<size_t>(t) * cfg_.latent_dim + c];

    // ── Sub-graph 0: post_quant_conv + conv_in ───────────────────────────
    {
        auto t0 = ggml_time_us();
        const bool has_pqc = cfg_.continuous && !post_quant_conv_.weight_f16.empty();
        SubGraphIO next;
        if (!run_sub_graph(backend_pair_->backend, scratch_,
                cur_io.data.data(), cur_io.T, cur_io.C,
                [&](ggml_context* ctx, ggml_tensor* in,
                    std::vector<VaeInputUpload>& uploads) -> ggml_tensor* {
                    ggml_tensor* r = in;
                    if (has_pqc)
                        r = graph_conv1d(ctx, r, post_quant_conv_, uploads, "pqc");
                    r = graph_conv1d(ctx, r, conv_in_, uploads, "conv_in");
                    return r;
                },
                &next, error, "pqc+conv_in")) {
            return false;
        }
        cur_io = std::move(next);
        auto t1 = ggml_time_us();
        std::fprintf(stderr, "[vae]      pqc+conv_in: %dms T=%d C=%d\n",
                     (int)((t1 - t0) / 1000), cur_io.T, cur_io.C);
    }

    // ── Sub-graphs 1..N: one per decoder block ──────────────────────────
    for (int bi = 0; bi < (int)blocks_.size(); bi++) {
        const auto& blk = blocks_[bi];
        auto t0 = ggml_time_us();
        SubGraphIO next;
        if (!run_sub_graph(backend_pair_->backend, scratch_,
                cur_io.data.data(), cur_io.T, cur_io.C,
                [&](ggml_context* ctx, ggml_tensor* in,
                    std::vector<VaeInputUpload>& uploads) -> ggml_tensor* {
                    // Snake
                    ggml_tensor* r = graph_snake(ctx, in, blk.snake, uploads);
                    // ConvT1d
                    r = graph_conv_t1d(ctx, r, blk.conv_t, uploads);
                    // 3 ResUnits
                    for (int ri = 0; ri < 3; ri++) {
                        const auto& ru = blk.res_units[ri];
                        ggml_tensor* skip = r;
                        ggml_tensor* h = graph_snake(ctx, r, ru.snake1, uploads);
                        h = graph_conv1d(ctx, h, ru.conv1, uploads);
                        h = graph_snake(ctx, h, ru.snake2, uploads);
                        h = graph_conv1d(ctx, h, ru.conv2, uploads);
                        r = ggml_add(ctx, h, skip);
                    }
                    return r;
                },
                &next, error, "block")) {
            return false;
        }
        cur_io = std::move(next);
        auto t1 = ggml_time_us();
        std::fprintf(stderr, "[vae]      block %d: %dms T=%d C=%d\n",
                     bi, (int)((t1 - t0) / 1000), cur_io.T, cur_io.C);
    }

    // ── Sub-graph N+1: snake_out + conv_out + tanh ──────────────────────
    {
        auto t0 = ggml_time_us();
        SubGraphIO next;
        if (!run_sub_graph(backend_pair_->backend, scratch_,
                cur_io.data.data(), cur_io.T, cur_io.C,
                [&](ggml_context* ctx, ggml_tensor* in,
                    std::vector<VaeInputUpload>& uploads) -> ggml_tensor* {
                    ggml_tensor* r = graph_snake(ctx, in, snake_out_, uploads);
                    r = graph_conv1d(ctx, r, conv_out_, uploads);
                    r = ggml_tanh(ctx, r);
                    return r;
                },
                &next, error, "snake_out+conv_out")) {
            return false;
        }
        cur_io = std::move(next);
        auto t1 = ggml_time_us();
        std::fprintf(stderr, "[vae]      out: %dms T=%d C=%d\n",
                     (int)((t1 - t0) / 1000), cur_io.T, cur_io.C);
    }

    // ── Copy output: column-major [T, 1] → time-major ───────────────────
    const int T_audio = cur_io.T;
    const int C_audio = cur_io.C;
    if (C_audio == 1) {
        memcpy(out, cur_io.data.data(),
               static_cast<size_t>(T_audio) * sizeof(float));
    } else {
        for (int t = 0; t < T_audio; t++)
            for (int c = 0; c < C_audio; c++)
                out[static_cast<size_t>(t) * C_audio + c] =
                    cur_io.data[static_cast<size_t>(t) + static_cast<size_t>(c) * T_audio];
    }

    auto t_end = ggml_time_us();
    std::fprintf(stderr,
        "[vae] per-block decode: total=%dms (T_latent=%d → T_audio=%d)\n",
        (int)((t_end - t_start) / 1000), T_latent, T_audio);

    return true;
}

bool VAERunner::decode_traced(const float* z, int32_t B, int32_t T_latent,
                               float* out, Trace* trace, std::string* error) {
    if (B != 1) {
        if (error) *error = "VAE only supports B=1";
        return false;
    }

    // Suppress cosmetic ggml debug messages (buffer reallocs) during decode.
    GgmlLogGuard _log_guard;

    // Lazily initialize GPU backend on first decode.
    if (!backend_ready_) ensure_backend();
    ggml_backend_sched_t sched = backend_ready_ ? sched_ : nullptr;

    // ── Shared scratch buffer ────────────────────────────────────────────
    // Allocated ONCE per decode call and reused across all ~95 ops.
    // GPU path (no_alloc=true): only stores ggml metadata → 16 MB is plenty.
    // CPU path (no_alloc=false): stores tensor data → needs several GB.
    const size_t scratch_size = sched
        ? (16ULL * 1024 * 1024)                           // 16 MB for GPU metadata
        : (4096ULL * 1024 * 1024);                        // 4 GB for CPU data
    scratch_.resize(scratch_size);
    char*  sbuf  = scratch_.data();
    size_t ssize = scratch_.size();

    std::vector<float> cur;

    // Continuous-mode: apply post_quant_conv (1×1 Conv1d) before the decoder.
    // Input z is [T_latent, latent_dim] (time-major, row=t, col=c).
    if (cfg_.continuous) {
        if (!post_quant_conv_.run(z, T_latent, cur, error, sched, sbuf, ssize)) {
            if (error) *error = "VAE: post_quant_conv failed";
            return false;
        }
    } else {
        // Discrete-mode (RVQ): latents are already quantized, pass through.
        cur.assign(z, z + static_cast<size_t>(T_latent) * cfg_.latent_dim);
    }

    // vae_dec_0 corresponds to the output of decoder.model[0] (conv_in), which
    // in Python is computed from post_quant_conv_out — NOT from raw latents.
    // The dump script applies post_quant_conv before iterating decoder.model.
    if (trace) trace->post_pqc = cur;

    // model[0]: conv_in (WNConv1d latent_dim → decoder_dim, k=7, pad=3)
    if (!conv_in_.run(cur.data(), T_latent, cur, error, sched, sbuf, ssize)) {
        if (error) *error = "VAE: conv_in failed";
        return false;
    }
    if (trace) trace->vae_dec[0] = cur;

    // DecoderBlocks: model[1]..model[5]
    for (int bi = 0; bi < (int)blocks_.size(); bi++) {
        const auto& blk = blocks_[bi];
        std::vector<float> h;
        auto blk_t0 = ggml_time_us();

        // Snake
        int T = static_cast<int>(cur.size()) / blk.snake.C;
        if (!blk.snake.run(cur.data(), T, h, error, sched, sbuf, ssize)) return false;
        if (trace && bi == 0) trace->blk1_snake = h;
        cur.swap(h);

        // ConvTranspose1d
        T = static_cast<int>(cur.size()) / blk.conv_t.IC;
        if (!blk.conv_t.run(cur.data(), T, cur, error, sched, sbuf, ssize)) return false;
        if (trace && bi == 0) trace->blk1_convt = cur;

        auto convt_t = ggml_time_us();

        // 3× ResidualUnits
        for (int ri = 0; ri < 3; ri++) {
            const auto& ru = blk.res_units[ri];
            int C = ru.conv1.IC;
            std::vector<float> skip_in = cur;  // save residual input
            int T0 = static_cast<int>(cur.size()) / C;

            // Snake 1 → Conv1 → Snake 2 → Conv2 → residual add: out = block(x) + x
            if (!ru.snake1.run(cur.data(), T0, h, error, sched, sbuf, ssize)) return false;
            if (trace && bi == 0 && ri == 0) trace->blk1_res0_s1 = h;

            T = static_cast<int>(h.size()) / C;
            if (!ru.conv1.run(h.data(), T, cur, error, sched, sbuf, ssize)) return false;
            if (trace && bi == 0 && ri == 0) trace->blk1_res0_c1 = cur;

            int T1 = static_cast<int>(cur.size()) / C;
            if (!ru.snake2.run(cur.data(), T1, h, error, sched, sbuf, ssize)) return false;
            if (trace && bi == 0 && ri == 0) trace->blk1_res0_s2 = h;

            T = static_cast<int>(h.size()) / C;
            if (!ru.conv2.run(h.data(), T, cur, error, sched, sbuf, ssize)) return false;
            if (trace && bi == 0 && ri == 0) trace->blk1_res0_c2 = cur;

            // Residual add: cur += skip_in (with center-crop to match sizes)
            int T2 = static_cast<int>(cur.size()) / C;
            int skip = (T0 - T2) / 2;
            if (skip > 0 && T0 > 0) {
                for (int i = 0; i < T2; i++)
                    for (int c = 0; c < C; c++)
                        cur[static_cast<size_t>(i) * C + c] +=
                            skip_in[static_cast<size_t>(skip + i) * C + c];
            } else {
                for (size_t i = 0; i < cur.size(); i++)
                    cur[i] += skip_in[i];
            }

            if (trace && bi == 0 && ri < 4) trace->blk1_res[ri] = cur;
        }

        auto blk_t1 = ggml_time_us();
        std::fprintf(stderr,
            "[vae]      block %d: %dms total (snake+convt=%dms, 3res=%dms)"
            " T=%d C_in=%d C_out=%d\n",
            bi,
            (int)((blk_t1 - blk_t0) / 1000),
            (int)((convt_t - blk_t0) / 1000),
            (int)((blk_t1 - convt_t) / 1000),
            (int)(cur.size() / (blk.conv_t.OC)),
            blk.conv_t.IC, blk.conv_t.OC);

        // vae_dec_[bi+1] = output of decoder.model[bi+1] (a DecoderBlock)
        if (trace) trace->vae_dec[bi + 1] = cur;
    }

    // model[6]: final Snake
    {
        std::vector<float> h;
        int T = static_cast<int>(cur.size()) / snake_out_.C;
        if (!snake_out_.run(cur.data(), T, h, error, sched, sbuf, ssize)) return false;
        cur.swap(h);
    }
    if (trace) trace->vae_dec[6] = cur;

    // model[7]: final WNConv1d (decoder_dim/32 → 1, k=7, pad=3)
    {
        int T = static_cast<int>(cur.size()) / conv_out_.IC;
        if (!conv_out_.run(cur.data(), T, cur, error, sched, sbuf, ssize)) return false;
    }
    if (trace) trace->vae_dec[7] = cur;

    // model[8]: Tanh
    {
        std::vector<float> h;
        int T = static_cast<int>(cur.size());
        if (!op_tanh(cur.data(), T, 1, h, error, sched, sbuf, ssize)) return false;
        cur.swap(h);
    }
    if (trace) { trace->vae_dec[8] = cur; trace->vae_final = cur; }

    memcpy(out, cur.data(), cur.size() * sizeof(float));
    return true;
}

}  // namespace audiocore::moss_sfx_v2
