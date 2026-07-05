// vae_runner.cpp — VAE decoder: latents [T, 64] → 48kHz stereo PCM.
//
// The VAE upscales 25Hz latent frames to 48kHz audio with 5 decoder blocks.
// Architecture (from ServeurpersoCom/acestep.cpp vae.h):
//
//   conv1(k=7,s=1,pad=3): [T,64] → [T,2048]
//   block 0: snake → conv_t1d(k=20,s=10,pad=5) → 3×ResUnit  [T*10, 1024]
//   block 1: snake → conv_t1d(k=12,s=6, pad=3) → 3×ResUnit  [T*60,  512]
//   block 2: snake → conv_t1d(k=8, s=4, pad=2) → 3×ResUnit  [T*240, 256]
//   block 3: snake → conv_t1d(k=8, s=4, pad=2) → 3×ResUnit  [T*960, 128]
//   block 4: snake → conv_t1d(k=4, s=2, pad=1) → 3×ResUnit  [T*1920,128]
//   final snake → conv2(k=7,s=1,pad=3): [T*1920,2] = stereo PCM
//
// Conv_t1d uses mul_mat(W_prepermuted, x) + col2im_1d — matches the GGUF's
// pre-permuted 2D weight layout.  Conv1d uses native ggml_conv_1d.
// Snake: x + sin²(exp_a · x) · inv_b  (exp_a, inv_b pre-computed in GGUF).
//
// GGUF weight format: ALL convolutional weights use weight standardization
// (WSConv1D) stored as bf16 weight_g + weight_v pairs. The converter
// (ServeurpersoCom/acestep.cpp → llama.cpp GGUF) produces this format.
// At load time we compose: W = weight_g · (weight_v − μ)/σ  per output
// channel. Biases and snake params are also bf16.
//
// Memory strategy: every sub-operation builds a temporary ggml context on a
// pre-allocated 256 MB ring buffer and copies its output back to a CPU float
// vector in time-major [T, C] layout.  This avoids ggml's internal dimension
// transposition (conv1d outputs ne[0]=T not ne[0]=C) and keeps peak usage low.

#include "audiocore/models/ace_step/vae_runner.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace audiocore::acestep {

// ═════════════════════════════════════════════════════════════════════════════
//  BF16 helpers
// ═════════════════════════════════════════════════════════════════════════════

static float bf16_to_f32(uint16_t bits) {
    // BF16 is F32 truncated to top 16 bits — just zero-extend.
    uint32_t f32_bits = static_cast<uint32_t>(bits) << 16;
    float f;
    memcpy(&f, &f32_bits, sizeof(f));
    return f;
}

static void bf16_to_f32_buf(const void* src, float* dst, int n) {
    const uint16_t* s = static_cast<const uint16_t*>(src);
    for (int i = 0; i < n; i++) dst[i] = bf16_to_f32(s[i]);
}

// Snake α/β load helpers — reference applies exp(alpha) and 1/exp(beta)
// at load time so the graph can use a plain mul (not exp inside the graph).
static void load_snake_exp(const ggml_tensor* t, std::vector<float>* out) {
    int C = (int)t->ne[1];
    out->resize(C);
    const uint16_t* src = static_cast<const uint16_t*>(t->data);
    for (int i = 0; i < C; i++) (*out)[i] = std::exp(bf16_to_f32(src[i]));
}
static void load_snake_inv(const ggml_tensor* t, std::vector<float>* out) {
    int C = (int)t->ne[1];
    out->resize(C);
    const uint16_t* src = static_cast<const uint16_t*>(t->data);
    for (int i = 0; i < C; i++) (*out)[i] = 1.0f / std::exp(bf16_to_f32(src[i]));
}

// ── Weight Normalization fuse: w = g * v / ||v|| ───────────────────────────
//
// The ServeurpersoCom reference (vae.h vae_fuse_wn) uses **weight
// normalization**, NOT weight standardization.  PyTorch's weight_norm(dim=0)
// normalizes each output channel's weights by their L2 norm:
//   For each output channel d:   w = g[d] * v / (sqrt(sum(v^2)) + eps)
//
// The "output channel" is PyTorch dim 0 = ggml ne[n_dims-1] (slowest axis).
// For Conv1d weights (PyTorch [OC, IC, K] → ggml ne=[K, IC, OC]) dim0 = OC.
// For ConvTranspose1d weights (PyTorch [IC, OC, K] → ggml ne=[K, OC, IC])
// dim0 = IC.
//
// The fast-varying dims (K and middle axis) are contiguous in ggml memory,
// so for each d the fan = K * middle values form a contiguous block at
// offset d*fan in the source data.
//
// Output layout matches input layout (in-place style).
static std::vector<float> compute_wsconv(const void* wv_data, const void* wg_data,
                                          int K, int OC, int IC,
                                          bool input_is_K_IC_OC,
                                          float eps = 1e-12f) {
    const uint16_t* v = static_cast<const uint16_t*>(wv_data);
    const uint16_t* g = static_cast<const uint16_t*>(wg_data);

    // dim0 = PyTorch dim 0 = ggml ne[n_dims-1].
    // fan  = K * middle_axis (the fast-varying dims).
    int dim0, fan;
    if (input_is_K_IC_OC) {
        // Conv1d: ne=[K, IC, OC] → dim0=OC, fan=K*IC
        dim0 = OC;
        fan  = K * IC;
    } else {
        // ConvTranspose1d: ne=[K, OC, IC] → dim0=IC, fan=K*OC
        dim0 = IC;
        fan  = K * OC;
    }

    const size_t total = static_cast<size_t>(K) * OC * IC;
    std::vector<float> result(total);

    for (int d = 0; d < dim0; d++) {
        float gv  = bf16_to_f32(g[d]);
        double nsq = 0.0;
        for (int i = 0; i < fan; i++) {
            float vv = bf16_to_f32(v[d * fan + i]);
            nsq += static_cast<double>(vv) * vv;
        }
        float s = gv / (static_cast<float>(std::sqrt(nsq)) + eps);
        for (int i = 0; i < fan; i++) {
            float vv = bf16_to_f32(v[d * fan + i]);
            result[d * fan + i] = vv * s;
        }
    }
    return result;
}

// Compute the conv_t1d pre-permuted 2D weight: [IC, K*OC] from WSConv 3D.
// conv_t1d op expects w as 2D [IC, K·OC] (ggml ne: ne[0]=IC, ne[1]=K*OC).
// The WSConv result is 3D [K, OC, IC] (ggml ne order, matching input layout).
// For ggml ne=[IC, K*OC] tensor, memory is data[ic + k_oc*IC].
// Permute: out[ic + (k*OC+o)*IC] = wsconv_3d[k + o*K + ic*K*OC]
static std::vector<float> permute_conv_t1d_weight(const float* wsconv_3d,
                                                    int K, int OC, int IC) {
    std::vector<float> out(static_cast<size_t>(IC) * K * OC);
    for (int k = 0; k < K; k++) {
        for (int o = 0; o < OC; o++) {
            for (int i = 0; i < IC; i++) {
                size_t src_idx = static_cast<size_t>(k) +
                                 static_cast<size_t>(o) * K +
                                 static_cast<size_t>(i) * K * OC;
                size_t dst_idx = static_cast<size_t>(i) +
                                 (static_cast<size_t>(k) * OC +
                                  static_cast<size_t>(o)) * IC;
                out[dst_idx] = wsconv_3d[src_idx];
            }
        }
    }
    return out;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Block / unit constants
// ═════════════════════════════════════════════════════════════════════════════

struct BlockCfg {
    int32_t in_ch;
    int32_t out_ch;
    int32_t stride;
    int32_t kernel;   // stride × 2
    int32_t padding;  // (kernel - stride) / 2
};

static const BlockCfg kBlocks[5] = {
    {2048, 1024, 10, 20, 5},
    {1024,  512,  6, 12, 3},
    { 512,  256,  4,  8, 2},
    { 256,  128,  4,  8, 2},
    { 128,  128,  2,  4, 1},
};

// Encoder block config (mirror of decoder, ResUnit at in_ch before strided conv)
struct EncBlockCfg {
    int32_t channel;   // ResUnit + snake operate at this channel count
    int32_t out_ch;    // after strided conv
    int32_t stride;
    int32_t kernel;    // stride × 2
    int32_t padding;   // stride / 2
};

static const EncBlockCfg kEncBlocks[5] = {
    {128,  128,  2,  4, 1},
    {128,  256,  4,  8, 2},
    {256,  512,  4,  8, 2},
    {512,  1024, 6, 12, 3},
    {1024, 2048, 10, 20, 5},
};

static const int32_t kResDilations[3] = {1, 3, 9};

// ── NaN diagnostic helper ───────────────────────────────────────────────────
static void nan_check(const char* tag, const float* p, size_t n) {
    int nan_cnt = 0;
    float mx = -1e30f, mn = 1e30f;
    for (size_t i = 0; i < n; i++) {
        float v = p[i];
        if (std::isnan(v)) nan_cnt++;
        else { if (v > mx) mx = v; if (v < mn) mn = v; }
    }
    fprintf(stderr, "[vae] %-20s NaN=%d/%zu range=[%g,%g]\n", tag, nan_cnt, n, mn, mx);
}

// ═════════════════════════════════════════════════════════════════════════════
//  Per-operation sub-graph runners
// ═════════════════════════════════════════════════════════════════════════════
//
// Every runner:
//   1. Creates a temp ggml_context from the shared pre-allocated buffer
//   2. Copies input data into a ggml tensor  (ne[0]=C, ne[1]=T — time-major)
//   3. Builds the operation graph
//   4. Calls ggml_graph_compute_with_ctx
//   5. Copies output into a vector<float> in time-major [T, C] layout
//   6. Frees the context (buffer reused)
//
// Weights are always f32 pre-computed vectors.

// ── Conv1d ───────────────────────────────────────────────────────────────────
// x: [T_in, IC] time-major, w: [K, IC, OC] f32, returns: [T_out, OC] time-major
static bool op_conv1d(const float* x, int32_t T_in, int32_t IC, int32_t OC,
                      const float* w_data, int32_t K, const float* bias,
                      int32_t stride, int32_t pad, int32_t dilation,
                      std::vector<float>* out, int32_t* out_T,
                      char* buf, size_t buf_size) {
    struct ggml_init_params gip = {buf_size, buf, false};
    ggml_context* ctx = ggml_init(gip);
    ggml_cgraph* gf  = ggml_new_graph(ctx);

    // ggml_conv_1d expects input as [T, IC] (ne0=T, ne1=IC) but our input
    // x is time-major [T, IC] in memory.  ggml_tensor with ne0=T, ne1=IC
    // has layout data[t + c*T], so we must transpose the copy.
    ggml_tensor* in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_in, IC);
    {
        float* dst = (float*)in->data;
        for (int32_t t = 0; t < T_in; t++) {
            for (int32_t c = 0; c < IC; c++) {
                dst[t + c * T_in] = x[t * IC + c];
            }
        }
    }

    // Build weight tensor from pre-computed f32 data — convert to F16
    // (im2col compute requires F16 kernel on CPU backend).
    //
    // Conv1d weights are stored in GGUF (and our compute_wsconv output) in
    // [K, IC, OC] layout, which matches ggml_conv_1d's expected ne=[K,IC,OC].
    // Linear copy is correct.
    ggml_tensor* w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, K, IC, OC);
    {
        ggml_fp16_t* wd = static_cast<ggml_fp16_t*>(w->data);
        const size_t n_w = static_cast<size_t>(K) * IC * OC;
        for (size_t i = 0; i < n_w; i++) {
            wd[i] = ggml_fp32_to_fp16(w_data[i]);
        }
    }

    // ggml_conv_1d: weight [K, IC, OC], input [T, IC] → [OL, OC, 1]
    ggml_tensor* r = ggml_conv_1d(ctx, w, in, stride, pad, dilation);
    if (bias) {
        // Bias [OC] → reshape to [1, OC] so it broadcasts across T_out in r
        ggml_tensor* bt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, OC);
        memcpy(bt->data, bias, static_cast<size_t>(OC) * sizeof(float));
        r = ggml_add(ctx, r, ggml_repeat(ctx, bt, r));
    }
    ggml_build_forward_expand(gf, r);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    *out_T = (int32_t)r->ne[0];  // OL
    int32_t T_out = *out_T;
    out->resize(static_cast<size_t>(T_out) * OC);

    // ggml output data[t + c·T_out] → our output[t·OC + c]
    const float* src = (const float*)r->data;
    float* dst = out->data();
    for (int32_t t = 0; t < T_out; t++) {
        for (int32_t c = 0; c < OC; c++) {
            dst[t * static_cast<size_t>(OC) + c] = src[t + c * static_cast<size_t>(T_out)];
        }
    }

    ggml_free(ctx);
    return true;
}

// ── Snake ────────────────────────────────────────────────────────────────────
// x: [T, C] time-major, returns: [T, C] time-major
static bool op_snake(const float* x, int32_t T, int32_t C,
                     const float* exp_a,  // [C] pre-computed
                     const float* inv_b,  // [C] pre-computed
                     std::vector<float>* out,
                     char* buf, size_t buf_size) {
    struct ggml_init_params gip = {buf_size, buf, false};
    ggml_context* ctx = ggml_init(gip);
    ggml_cgraph* gf  = ggml_new_graph(ctx);

    ggml_tensor* in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, T);
    memcpy(in->data, x, static_cast<size_t>(T) * C * sizeof(float));

    ggml_tensor* a_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
    memcpy(a_t->data, exp_a, C * sizeof(float));
    ggml_tensor* b_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
    memcpy(b_t->data, inv_b, C * sizeof(float));

    // x + sin²(α·x) · inv_beta
    ggml_tensor* ax   = ggml_mul(ctx, in, a_t);
    ggml_tensor* s    = ggml_sin(ctx, ax);
    ggml_tensor* s2   = ggml_sqr(ctx, s);
    ggml_tensor* term = ggml_mul(ctx, s2, b_t);
    ggml_tensor* r    = ggml_add(ctx, in, term);

    ggml_build_forward_expand(gf, r);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    out->resize(static_cast<size_t>(T) * C);
    memcpy(out->data(), r->data, static_cast<size_t>(T) * C * sizeof(float));

    ggml_free(ctx);
    return true;
}

// ── ConvTranspose1d (via mul_mat + col2im_1d) ────────────────────────────────
// x: [T_in, IC] time-major, returns: [T_out = stride·T_in, OC] time-major
// w_data: pre-permuted 2D [IC, K·OC] f32
static bool op_conv_t1d(const float* x, int32_t T_in, int32_t IC, int32_t OC,
                        int32_t stride, int32_t pad,
                        const float* w_data, int32_t K,
                        const float* bias,
                        std::vector<float>* out, int32_t* out_T,
                        char* buf, size_t buf_size) {
    struct ggml_init_params gip = {buf_size, buf, false};
    ggml_context* ctx = ggml_init(gip);
    ggml_cgraph* gf  = ggml_new_graph(ctx);

    // Input [T, IC] → ne[0]=IC, ne[1]=T
    ggml_tensor* in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, IC, T_in);
    memcpy(in->data, x, static_cast<size_t>(T_in) * IC * sizeof(float));

    // Weight tensor [IC, K·OC] f32
    ggml_tensor* w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, IC, K * OC);
    memcpy(w->data, w_data, static_cast<size_t>(IC) * K * OC * sizeof(float));

    // mul_mat: contracts over ne[0]=IC (both w and in have ne[0]=IC)
    // →  ne[0]=K·OC, ne[1]=T_in
    ggml_tensor* col = ggml_mul_mat(ctx, w, in);
    ggml_build_forward_expand(gf, col);

    // col2im: [K·OC, T_in] → [T_out, OC] where T_out = stride·T_in
    ggml_tensor* r = ggml_col2im_1d(ctx, col, stride, OC, pad);
    if (bias) {
        // Bias [OC] → reshape to [1, OC] so it broadcasts across T_out in r
        ggml_tensor* bt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, OC);
        memcpy(bt->data, bias, static_cast<size_t>(OC) * sizeof(float));
        r = ggml_add(ctx, r, ggml_repeat(ctx, bt, r));
    }
    ggml_build_forward_expand(gf, r);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    *out_T = (int32_t)r->ne[0];  // T_out
    int32_t T_out = *out_T;
    out->resize(static_cast<size_t>(T_out) * OC);

    // ggml output: ne[0]=T_out, ne[1]=OC, data[t + c·T_out]
    // our output:  [T, C] time-major → data[t·OC + c]
    const float* src = (const float*)r->data;
    float* dst = out->data();
    for (int32_t t = 0; t < T_out; t++) {
        for (int32_t c = 0; c < OC; c++) {
            dst[t * static_cast<size_t>(OC) + c] = src[t + c * static_cast<size_t>(T_out)];
        }
    }

    ggml_free(ctx);
    return true;
}

// ── Single ResUnit (as 4 independent sub-graph calls + CPU skip) ──────────
// x: [T, C] time-major, returns: [T, C] time-major
static bool op_resunit(const float* x, int32_t T, int32_t C,
                       const float* s1a, const float* s1b,
                       const float* c1w, int32_t c1K,
                       const float* c1b,
                       int32_t dilation,
                       const float* s2a, const float* s2b,
                       const float* c2w, int32_t c2K,
                       const float* c2b,
                       std::vector<float>* out,
                       char* buf, size_t buf_size) {
    // Save skip connection
    std::vector<float> skip(x, x + static_cast<size_t>(T) * C);

    // Snake 1
    std::vector<float> h1;
    if (!op_snake(x, T, C, s1a, s1b, &h1, buf, buf_size)) return false;
    nan_check("ru_snake1", h1.data(), h1.size());

    // Conv1 (k=7, dilated)
    std::vector<float> h2;
    int32_t T2 = 0;
    if (!op_conv1d(h1.data(), T, C, C, c1w, c1K, c1b,
                   1, 3 * dilation, dilation, &h2, &T2, buf, buf_size))
        return false;
    nan_check("ru_conv1", h2.data(), h2.size());

    // Snake 2
    std::vector<float> h3;
    if (!op_snake(h2.data(), T2, C, s2a, s2b, &h3, buf, buf_size)) return false;
    nan_check("ru_snake2", h3.data(), h3.size());

    // Conv2 (k=1)
    std::vector<float> h4;
    int32_t T4 = 0;
    if (!op_conv1d(h3.data(), T2, C, C, c2w, c2K, c2b,
                   1, 0, 1, &h4, &T4, buf, buf_size))
        return false;
    nan_check("ru_conv2", h4.data(), h4.size());

    // Skip connection element-wise add
    const int32_t T_out = std::min(T, T4);
    out->resize(static_cast<size_t>(T_out) * C);
    float* dst = out->data();
    const float* s = skip.data();
    const float* r = h4.data();
    for (int32_t t = 0; t < T_out; t++) {
        for (int32_t c = 0; c < C; c++) {
            dst[t * static_cast<size_t>(C) + c] = s[t * static_cast<size_t>(C) + c] + r[t * static_cast<size_t>(C) + c];
        }
    }
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
//  VAE weight pre-computation
// ═════════════════════════════════════════════════════════════════════════════
//
// Pre-compute all weights from the GGUF's bf16 WSConv format into f32 at
// construction time.  Stored as flat vectors on the runner.

// Helper: read snake params from bf16 tensor (ne=[1, C])
static bool read_snake_bf16(ggml_tensor* alpha_t, ggml_tensor* beta_t,
                             std::vector<float>* exp_a,
                             std::vector<float>* inv_b,
                             std::string* error) {
    if (!alpha_t || !beta_t) {
        if (error) *error = "VAE: missing snake tensor";
        return false;
    }
    int32_t C = (int32_t)alpha_t->ne[1];  // ne[0]=1, ne[1]=C
    int32_t cb = (int32_t)beta_t->ne[1];
    if (C != cb || C <= 0) {
        if (error) *error = "VAE: snake channel mismatch " +
                            std::to_string(C) + " vs " + std::to_string(cb);
        return false;
    }
    exp_a->resize(C);
    inv_b->resize(C);
    // alpha/beta are bf16, ne=[1, C]. Apply exp(alpha) and 1/exp(beta).
    load_snake_exp(alpha_t, exp_a);
    load_snake_inv(beta_t,  inv_b);
    return true;
}

// Top-level weight initialization (called from VAERunner constructor).
// Fills the runner's member vectors from ext_ctx_ tensors.
#define WV_NAME(n) ((std::string("vae.decoder.") + (n) + ".weight_v").c_str())
#define WG_NAME(n) ((std::string("vae.decoder.") + (n) + ".weight_g").c_str())
#define BIAS_NAME(n) ((std::string("vae.decoder.") + (n) + ".bias").c_str())
#define SNAKE_A(n) ((std::string("vae.decoder.") + (n) + ".alpha").c_str())
#define SNAKE_B(n) ((std::string("vae.decoder.") + (n) + ".beta").c_str())

#define WV_NAME_E(n) ((std::string("vae.encoder.") + (n) + ".weight_v").c_str())
#define WG_NAME_E(n) ((std::string("vae.encoder.") + (n) + ".weight_g").c_str())
#define BIAS_NAME_E(n) ((std::string("vae.encoder.") + (n) + ".bias").c_str())
#define SNAKE_A_E(n) ((std::string("vae.encoder.") + (n) + ".alpha").c_str())
#define SNAKE_B_E(n) ((std::string("vae.encoder.") + (n) + ".beta").c_str())

static bool load_wsconv(const void* wv, const void* wg,
                         int K, int OC, int IC,
                         std::vector<float>* out_f32) {
    *out_f32 = compute_wsconv(wv, wg, K, OC, IC, /*K_IC_OC=*/true);
    return true;
}

static bool load_wsconv_and_bias(ggml_context* ext_ctx,
                                  const std::string& name,
                                  int expected_K, int expected_OC, int expected_IC,
                                  std::vector<float>* w_f32,
                                  std::vector<float>* bias_f32,
                                  std::string* error) {
    ggml_tensor* wv = ggml_get_tensor(ext_ctx, WV_NAME(name));
    ggml_tensor* wg = ggml_get_tensor(ext_ctx, WG_NAME(name));
    if (!wv || !wg) {
        if (error) *error = "VAE: missing " + name + " weight_v/weight_g";
        return false;
    }
    int K  = (int)wv->ne[0];
    int IC = (int)wv->ne[1];
    int OC = (int)wv->ne[2];
    if (K != expected_K || OC != expected_OC || IC != expected_IC) {
        if (error) *error = "VAE: " + name + " WSConv shape mismatch: "
                            "got [" + std::to_string(K) + "," + std::to_string(OC) + "," + std::to_string(IC) +
                            "], expected [" + std::to_string(expected_K) + "," +
                            std::to_string(expected_OC) + "," + std::to_string(expected_IC) + "]";
        return false;
    }
    *w_f32 = compute_wsconv(wv->data, wg->data, K, OC, IC, /*K_IC_OC=*/true);

    // Bias (bf16)
    ggml_tensor* b = ggml_get_tensor(ext_ctx, BIAS_NAME(name));
    if (b) {
        bias_f32->resize(OC);
        bf16_to_f32_buf(b->data, bias_f32->data(), OC);
    } else {
        bias_f32->clear();
    }
    return true;
}

// For conv_t1d: compute WSConv, then permute to 2D [IC, K*OC]
static bool load_wsconv_t1weight(ggml_context* ext_ctx,
                                  const std::string& name,
                                  int expected_K, int expected_OC, int expected_IC,
                                  std::vector<float>* w_2d_f32,
                                  std::vector<float>* bias_f32,
                                  int* out_K,
                                  std::string* error) {
    ggml_tensor* wv = ggml_get_tensor(ext_ctx, WV_NAME(name));
    ggml_tensor* wg = ggml_get_tensor(ext_ctx, WG_NAME(name));
    if (!wv || !wg) {
        if (error) *error = "VAE: missing " + name + " weight_v/weight_g";
        return false;
    }
    int K  = (int)wv->ne[0];
    int OC = (int)wv->ne[1];
    int IC = (int)wv->ne[2];
    if (K != expected_K || OC != expected_OC || IC != expected_IC) {
        if (error) *error = "VAE: " + name + " WSConv shape mismatch: "
                            "got [" + std::to_string(K) + "," + std::to_string(OC) + "," + std::to_string(IC) +
                            "], expected [" + std::to_string(expected_K) + "," +
                            std::to_string(expected_OC) + "," + std::to_string(expected_IC) + "]";
        return false;
    }
    auto wsconv_3d = compute_wsconv(wv->data, wg->data, K, OC, IC, /*K_IC_OC=*/false);
    *w_2d_f32 = permute_conv_t1d_weight(wsconv_3d.data(), K, OC, IC);
    *out_K = K;

    // Bias (bf16)
    ggml_tensor* b = ggml_get_tensor(ext_ctx, BIAS_NAME(name));
    if (b) {
        bias_f32->resize(OC);
        bf16_to_f32_buf(b->data, bias_f32->data(), OC);
    } else {
        bias_f32->clear();
    }
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
//  VAERunner
// ═════════════════════════════════════════════════════════════════════════════

VAERunner::VAERunner(ggml_context* ext_ctx) : ext_ctx_(ext_ctx) {
    if (!ext_ctx_) return;
    precompute_weights(ext_ctx_);
}

VAERunner::~VAERunner() = default;

ggml_tensor* VAERunner::weight(const char* name) const {
    return ggml_get_tensor(ext_ctx_, name);
}

void VAERunner::precompute_weights(ggml_context* ext_ctx) {
    // ── conv1: [T, 64] → [T, 2048]  (k=7, s=1, p=3) ─────────────────────
    // GGUF weight_v ne=[7, 64, 2048] = [K, IC, OC] (Conv1d layout)
    {
        ggml_tensor* wv = ggml_get_tensor(ext_ctx, "vae.decoder.conv1.weight_v");
        ggml_tensor* wg = ggml_get_tensor(ext_ctx, "vae.decoder.conv1.weight_g");
        ggml_tensor* wb = ggml_get_tensor(ext_ctx, "vae.decoder.conv1.bias");
        if (wv && wg) {
            int K = (int)wv->ne[0], IC = (int)wv->ne[1], OC = (int)wv->ne[2];
            dec_conv1_w_ = compute_wsconv(wv->data, wg->data, K, OC, IC, /*K_IC_OC=*/true);
            dec_conv1_K_ = K;
            if (wb) {
                dec_conv1_b_.resize(OC);
                bf16_to_f32_buf(wb->data, dec_conv1_b_.data(), OC);
            }
        }
    }

    // ── conv2: [T, 128] → [T, 2]  (k=7, s=1, p=3) ───────────────────────
    // GGUF weight_v ne=[7, 128, 2] = [K, IC, OC] (Conv1d layout)
    {
        ggml_tensor* wv = ggml_get_tensor(ext_ctx, "vae.decoder.conv2.weight_v");
        ggml_tensor* wg = ggml_get_tensor(ext_ctx, "vae.decoder.conv2.weight_g");
        if (wv && wg) {
            int K = (int)wv->ne[0], IC = (int)wv->ne[1], OC = (int)wv->ne[2];
            dec_conv2_w_ = compute_wsconv(wv->data, wg->data, K, OC, IC, /*K_IC_OC=*/true);
            dec_conv2_K_ = K;
            // conv2 has no bias
        }
    }

    // ── Final snake ──────────────────────────────────────────────────────
    {
        ggml_tensor* sa = ggml_get_tensor(ext_ctx, "vae.decoder.snake1.alpha");
        ggml_tensor* sb = ggml_get_tensor(ext_ctx, "vae.decoder.snake1.beta");
        if (sa && sb) {
            int C = (int)sa->ne[1];
            dec_fn_exp_a_.resize(C); load_snake_exp(sa, &dec_fn_exp_a_);
            dec_fn_inv_b_.resize(C); load_snake_inv(sb, &dec_fn_inv_b_);
        }
    }

    // ── 5 decoder blocks ─────────────────────────────────────────────────
    for (int b = 0; b < 5; b++) {
        std::string p = "block." + std::to_string(b) + ".";
        const auto& bc = kBlocks[b];

        // Block-level snake (pre-conv_t1) — shape [1, in_ch] bf16
        // Tensor name: vae.decoder.block.N.snake1.alpha/beta
        auto& blk = dec_blk_[b];
        {
            ggml_tensor* sa = ggml_get_tensor(ext_ctx, SNAKE_A(p + "snake1"));
            ggml_tensor* sb = ggml_get_tensor(ext_ctx, SNAKE_B(p + "snake1"));
            if (sa && sb) {
                int C = (int)sa->ne[1];
                blk.snake_a_.resize(C); load_snake_exp(sa, &blk.snake_a_);
                blk.snake_b_.resize(C); load_snake_inv(sb, &blk.snake_b_);
            }
        }

        // Conv_t1: WSConv, then permute to 2D [IC, K*OC]
        // GGUF weight_v ne=[K, OC, IC] (ConvTranspose1d layout)
        {
            ggml_tensor* wv = ggml_get_tensor(ext_ctx, WV_NAME(p + "conv_t1"));
            ggml_tensor* wg = ggml_get_tensor(ext_ctx, WG_NAME(p + "conv_t1"));
            ggml_tensor* wb = ggml_get_tensor(ext_ctx, BIAS_NAME(p + "conv_t1"));
            if (wv && wg) {
                int K = (int)wv->ne[0], OC = (int)wv->ne[1], IC = (int)wv->ne[2];
                auto wsconv = compute_wsconv(wv->data, wg->data, K, OC, IC, /*K_IC_OC=*/false);
                blk.ct1_w_ = permute_conv_t1d_weight(wsconv.data(), K, OC, IC);
                blk.ct1_K_ = K;
                if (wb) {
                    blk.ct1_b_.resize(OC);
                    bf16_to_f32_buf(wb->data, blk.ct1_b_.data(), OC);
                }
            }
        }

        // 3× ResUnit
        for (int r = 0; r < 3; r++) {
            std::string rp = p + "res_unit" + std::to_string(r + 1) + ".";
            auto& ru = blk.res_[r];

            // Snake1
            {
                ggml_tensor* sa = ggml_get_tensor(ext_ctx, SNAKE_A(rp + "snake1"));
                ggml_tensor* sb = ggml_get_tensor(ext_ctx, SNAKE_B(rp + "snake1"));
                if (sa && sb) {
                    int C = (int)sa->ne[1];
                    ru.s1a_.resize(C); load_snake_exp(sa, &ru.s1a_);
                    ru.s1b_.resize(C); load_snake_inv(sb, &ru.s1b_);
                }
            }
            // Conv1: k=7 (dilated) — GGUF ne=[K, IC, OC] (Conv1d layout)
            {
                ggml_tensor* wv = ggml_get_tensor(ext_ctx, WV_NAME(rp + "conv1"));
                ggml_tensor* wg = ggml_get_tensor(ext_ctx, WG_NAME(rp + "conv1"));
                ggml_tensor* wb = ggml_get_tensor(ext_ctx, BIAS_NAME(rp + "conv1"));
                if (wv && wg) {
                    int K = (int)wv->ne[0], IC = (int)wv->ne[1], OC = (int)wv->ne[2];
                    ru.c1w_ = compute_wsconv(wv->data, wg->data, K, OC, IC, /*K_IC_OC=*/true);
                    ru.c1K_ = K;
                    if (wb) { ru.c1b_.resize(OC); bf16_to_f32_buf(wb->data, ru.c1b_.data(), OC); }
                }
            }
            // Snake2
            {
                ggml_tensor* sa = ggml_get_tensor(ext_ctx, SNAKE_A(rp + "snake2"));
                ggml_tensor* sb = ggml_get_tensor(ext_ctx, SNAKE_B(rp + "snake2"));
                if (sa && sb) {
                    int C = (int)sa->ne[1];
                    ru.s2a_.resize(C); load_snake_exp(sa, &ru.s2a_);
                    ru.s2b_.resize(C); load_snake_inv(sb, &ru.s2b_);
                }
            }
            // Conv2: k=1 — GGUF ne=[K, IC, OC] (Conv1d layout)
            {
                ggml_tensor* wv = ggml_get_tensor(ext_ctx, WV_NAME(rp + "conv2"));
                ggml_tensor* wg = ggml_get_tensor(ext_ctx, WG_NAME(rp + "conv2"));
                ggml_tensor* wb = ggml_get_tensor(ext_ctx, BIAS_NAME(rp + "conv2"));
                if (wv && wg) {
                    int K = (int)wv->ne[0], IC = (int)wv->ne[1], OC = (int)wv->ne[2];
                    ru.c2w_ = compute_wsconv(wv->data, wg->data, K, OC, IC, /*K_IC_OC=*/true);
                    ru.c2K_ = K;
                    if (wb) { ru.c2b_.resize(OC); bf16_to_f32_buf(wb->data, ru.c2b_.data(), OC); }
                }
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Encoder weights
    // ══════════════════════════════════════════════════════════════════════
    // conv1: [T, 2] → [T, 128] (k=7, s=1, p=3)
    // GGUF weight_v ne=[K, IC, OC] = [7, 2, 128]
    {
        ggml_tensor* wv = ggml_get_tensor(ext_ctx, "vae.encoder.conv1.weight_v");
        ggml_tensor* wg = ggml_get_tensor(ext_ctx, "vae.encoder.conv1.weight_g");
        ggml_tensor* wb = ggml_get_tensor(ext_ctx, "vae.encoder.conv1.bias");
        if (wv && wg) {
            int K = (int)wv->ne[0], IC = (int)wv->ne[1], OC = (int)wv->ne[2];
            enc_conv1_w_ = compute_wsconv(wv->data, wg->data, K, OC, IC, /*K_IC_OC=*/true);
            enc_conv1_K_ = K;
            if (wb) { enc_conv1_b_.resize(OC); bf16_to_f32_buf(wb->data, enc_conv1_b_.data(), OC); }
        }
    }
    // conv2: [T, 2048] → [T, 128] (k=3, s=1, p=1)
    // GGUF weight_v ne=[K, IC, OC] = [3, 2048, 128]
    {
        ggml_tensor* wv = ggml_get_tensor(ext_ctx, "vae.encoder.conv2.weight_v");
        ggml_tensor* wg = ggml_get_tensor(ext_ctx, "vae.encoder.conv2.weight_g");
        ggml_tensor* wb = ggml_get_tensor(ext_ctx, "vae.encoder.conv2.bias");
        if (wv && wg) {
            int K = (int)wv->ne[0], IC = (int)wv->ne[1], OC = (int)wv->ne[2];
            enc_conv2_w_ = compute_wsconv(wv->data, wg->data, K, OC, IC, /*K_IC_OC=*/true);
            enc_conv2_K_ = K;
            if (wb) { enc_conv2_b_.resize(OC); bf16_to_f32_buf(wb->data, enc_conv2_b_.data(), OC); }
        }
    }
    // encoder snake1 (final): [2048] → [2048]
    {
        ggml_tensor* sa = ggml_get_tensor(ext_ctx, "vae.encoder.snake1.alpha");
        ggml_tensor* sb = ggml_get_tensor(ext_ctx, "vae.encoder.snake1.beta");
        if (sa && sb) {
            int C = (int)sa->ne[1];
            enc_fn_exp_a_.resize(C); load_snake_exp(sa, &enc_fn_exp_a_);
            enc_fn_inv_b_.resize(C); load_snake_inv(sb, &enc_fn_inv_b_);
        }
    }
    // 5 encoder blocks
    for (int b = 0; b < 5; b++) {
        std::string p = "block." + std::to_string(b) + ".";
        const auto& bc = kEncBlocks[b];
        auto& blk = enc_blk_[b];

        // Pre-stride snake — tensor name: vae.encoder.block.N.snake1.alpha/beta
        {
            ggml_tensor* sa = ggml_get_tensor(ext_ctx, SNAKE_A_E(p + "snake1"));
            ggml_tensor* sb = ggml_get_tensor(ext_ctx, SNAKE_B_E(p + "snake1"));
            if (sa && sb) {
                int C = (int)sa->ne[1];
                blk.snake_a_.resize(C); load_snake_exp(sa, &blk.snake_a_);
                blk.snake_b_.resize(C); load_snake_inv(sb, &blk.snake_b_);
            }
        }
        // Strided conv — GGUF ne=[K, IC, OC]
        {
            ggml_tensor* wv = ggml_get_tensor(ext_ctx, WV_NAME_E(p + "conv1"));
            ggml_tensor* wg = ggml_get_tensor(ext_ctx, WG_NAME_E(p + "conv1"));
            ggml_tensor* wb = ggml_get_tensor(ext_ctx, BIAS_NAME_E(p + "conv1"));
            if (wv && wg) {
                int K = (int)wv->ne[0], IC = (int)wv->ne[1], OC = (int)wv->ne[2];
                blk.conv_w_ = compute_wsconv(wv->data, wg->data, K, OC, IC, /*K_IC_OC=*/true);
                blk.conv_K_ = K;
                if (wb) { blk.conv_b_.resize(OC); bf16_to_f32_buf(wb->data, blk.conv_b_.data(), OC); }
            }
        }
        // 3× ResUnit
        for (int r = 0; r < 3; r++) {
            std::string rp = p + "res_unit" + std::to_string(r + 1) + ".";
            auto& ru = blk.res_[r];
            int ch = bc.channel;

            ru.s1a_.resize(ch); ru.s1b_.resize(ch);
            ru.s2a_.resize(ch); ru.s2b_.resize(ch);
            {
                ggml_tensor* sa = ggml_get_tensor(ext_ctx, SNAKE_A_E(rp + "snake1"));
                ggml_tensor* sb = ggml_get_tensor(ext_ctx, SNAKE_B_E(rp + "snake1"));
                if (sa && sb) load_snake_exp(sa, &ru.s1a_);
                if (sa && sb) load_snake_inv(sb, &ru.s1b_);
            }
            {
                ggml_tensor* wv = ggml_get_tensor(ext_ctx, WV_NAME_E(rp + "conv1"));
                ggml_tensor* wg = ggml_get_tensor(ext_ctx, WG_NAME_E(rp + "conv1"));
                ggml_tensor* wb = ggml_get_tensor(ext_ctx, BIAS_NAME_E(rp + "conv1"));
                if (wv && wg) {
                    int K = (int)wv->ne[0], IC = (int)wv->ne[1], OC = (int)wv->ne[2];
                    ru.c1w_ = compute_wsconv(wv->data, wg->data, K, OC, IC, /*K_IC_OC=*/true);
                    ru.c1K_ = K;
                    if (wb) { ru.c1b_.resize(OC); bf16_to_f32_buf(wb->data, ru.c1b_.data(), OC); }
                }
            }
            {
                ggml_tensor* sa = ggml_get_tensor(ext_ctx, SNAKE_A_E(rp + "snake2"));
                ggml_tensor* sb = ggml_get_tensor(ext_ctx, SNAKE_B_E(rp + "snake2"));
                if (sa && sb) load_snake_exp(sa, &ru.s2a_);
                if (sa && sb) load_snake_inv(sb, &ru.s2b_);
            }
            {
                ggml_tensor* wv = ggml_get_tensor(ext_ctx, WV_NAME_E(rp + "conv2"));
                ggml_tensor* wg = ggml_get_tensor(ext_ctx, WG_NAME_E(rp + "conv2"));
                ggml_tensor* wb = ggml_get_tensor(ext_ctx, BIAS_NAME_E(rp + "conv2"));
                if (wv && wg) {
                    int K = (int)wv->ne[0], IC = (int)wv->ne[1], OC = (int)wv->ne[2];
                    ru.c2w_ = compute_wsconv(wv->data, wg->data, K, OC, IC, /*K_IC_OC=*/true);
                    ru.c2K_ = K;
                    if (wb) { ru.c2b_.resize(OC); bf16_to_f32_buf(wb->data, ru.c2b_.data(), OC); }
                }
            }
        }
    }
}

bool VAERunner::decode(const float* latents, int32_t n_frames,
                        std::vector<float>* pcm, std::string* error) {
    if (!ext_ctx_ || n_frames <= 0) {
        if (error) *error = "VAE: invalid state";
        return false;
    }
    fprintf(stderr, "[vae] decode: n_frames=%d\n", n_frames);

    // ── Tiled decode for long-form generation ────────────────────────────
    // The single-pass decoder tops out around T=375 (15s @ 25Hz) due to the
    // 4 GB im2col scratch budget. For longer requests we split the latent
    // sequence into overlapping tiles, decode each separately, and crossfade
    // the overlapping PCM regions with a Hann window. This unlocks full
    // songs (2-3 minute output) with no model changes.
    //
    // Tile geometry (tuned for ~10s tiles + 1s overlap):
    //   TILE_FRAMES = 250 latent frames  = 10.0s audio
    //   OVERLAP_FRAMES = 25 latent frames = 1.0s audio crossfade
    // 1920 = total VAE upsampling factor (latent_frames → pcm_samples)
    constexpr int32_t TILE_FRAMES    = 250;
    constexpr int32_t OVERLAP_FRAMES = 25;
    constexpr int32_t UPSAMPLE       = 1920;

    if (n_frames > TILE_FRAMES) {
        fprintf(stderr, "[vae] tiled decode: %d frames → tiles of %d (overlap %d)\n",
                n_frames, TILE_FRAMES, OVERLAP_FRAMES);
        const size_t pcm_total = static_cast<size_t>(n_frames) * UPSAMPLE * 2;
        pcm->assign(pcm_total, 0.0f);
        const size_t ov_pcm = static_cast<size_t>(OVERLAP_FRAMES) * UPSAMPLE;
        std::vector<float> hann_w(ov_pcm);
        for (size_t i = 0; i < ov_pcm; i++) {
            hann_w[i] = 0.5f * (1.0f - std::cos(2.0f * 3.14159265358979f * i /
                                static_cast<float>(ov_pcm - 1)));
        }
        std::vector<float> wsum(pcm_total, 0.0f);

        int32_t tile_idx = 0;
        for (int32_t t0 = 0; t0 < n_frames; t0 += (TILE_FRAMES - OVERLAP_FRAMES)) {
            int32_t t1 = std::min(t0 + TILE_FRAMES, n_frames);
            int32_t t_len = t1 - t0;
            fprintf(stderr, "[vae] tile %d: [%d..%d] (%d frames)\n",
                    tile_idx, t0, t1, t_len);

            std::vector<float> tile_pcm;
            if (!decode(latents + static_cast<size_t>(t0) * 64, t_len, &tile_pcm, error))
                return false;

            const size_t dst_off = static_cast<size_t>(t0) * UPSAMPLE * 2;
            const size_t copy_n   = std::min(tile_pcm.size(),
                                             pcm->size() - dst_off);
            for (size_t i = 0; i < copy_n; i++) {
                float w = 1.0f;
                size_t tile_sample    = i / 2;  // mono index
                size_t tile_pcm_frame = tile_sample / UPSAMPLE;
                // Fade-in region (first OVERLAP_FRAMES of tile, skip first tile)
                if (tile_pcm_frame < static_cast<size_t>(OVERLAP_FRAMES) && tile_idx > 0) {
                    size_t hann_idx = tile_pcm_frame * UPSAMPLE +
                                      (tile_sample % UPSAMPLE);
                    if (hann_idx < hann_w.size()) w = hann_w[hann_idx];
                }
                // Fade-out region (last OVERLAP_FRAMES of tile, skip last tile)
                size_t tile_total_frames = tile_pcm.size() / 2 / UPSAMPLE;
                size_t frames_from_end = (tile_total_frames >= 1)
                    ? (tile_total_frames - 1 - tile_pcm_frame) : 0;
                if (frames_from_end < static_cast<size_t>(OVERLAP_FRAMES) &&
                    t1 < n_frames) {
                    size_t hann_idx = (OVERLAP_FRAMES - 1 - frames_from_end) *
                                      UPSAMPLE + (tile_sample % UPSAMPLE);
                    if (hann_idx < hann_w.size()) w *= hann_w[hann_idx];
                }
                (*pcm)[dst_off + i] += w * tile_pcm[i];
                wsum[dst_off + i]   += w;
            }
            tile_idx++;
            if (t1 >= n_frames) break;
        }
        for (size_t i = 0; i < pcm->size(); i++) {
            if (wsum[i] > 1e-6f) (*pcm)[i] /= wsum[i];
        }
        fprintf(stderr, "[vae] tiled decode done: %zu PCM samples from %d tiles\n",
                pcm->size(), tile_idx);
        return true;
    }

    // ── Shared scratch buffer ────────────────────────────────────────────
    // Sized for the worst-case op (block 4 ResUnit conv1 @ 15s):
    //   im2col F16: T*7*128*2 ≈ 1.4 GB, plus activations ≈ 0.8 GB
    // 4 GB handles up to ~15s; longer durations need tiled decode.
    const size_t buf_size = 4096ULL * 1024 * 1024;
    fprintf(stderr, "[vae] allocating %.0f MB scratch...\n", buf_size / 1048576.0);
    std::vector<char> buf(buf_size);
    fprintf(stderr, "[vae] scratch allocated, phase 1: conv1\n");

    // ══════════════════════════════════════════════════════════════════════
    //  Phase 1: conv1  [T, 64] → [T, 2048]
    // ══════════════════════════════════════════════════════════════════════
    std::vector<float> cur;
    int32_t T = n_frames;
    int32_t C = 64;

    {
        std::vector<float> out;
        int32_t T_out = 0;
        if (!op_conv1d(latents, T, 64, 2048,
                       dec_conv1_w_.data(), dec_conv1_K_,
                       dec_conv1_b_.empty() ? nullptr : dec_conv1_b_.data(),
                       1, 3, 1,
                       &out, &T_out, buf.data(), buf_size))
            return false;
        cur = std::move(out);
        T = T_out;
        C = 2048;
        nan_check("conv1_out", cur.data(), static_cast<size_t>(T) * C);
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Phase 2: 5 decoder blocks
    // ══════════════════════════════════════════════════════════════════════
    for (int b = 0; b < 5; b++) {
        const auto& bc  = kBlocks[b];
        const auto& w   = dec_blk_[b];
        fprintf(stderr, "[vae] block %d: T=%d in=%d out=%d snake_a=%zu ct1_w=%zu\n",
                b, T, bc.in_ch, bc.out_ch, w.snake_a_.size(), w.ct1_w_.size());

        // 2a. Snake
        std::vector<float> h1;
        if (!op_snake(cur.data(), T, bc.in_ch,
                      w.snake_a_.data(), w.snake_b_.data(),
                      &h1, buf.data(), buf_size))
            return false;
        { char t[48]; std::snprintf(t, sizeof(t), "blk%d_snake", b); nan_check(t, h1.data(), h1.size()); }

        // 2b. ConvTranspose1d
        std::vector<float> h2;
        int32_t T2 = 0;
        if (!op_conv_t1d(h1.data(), T, bc.in_ch, bc.out_ch,
                         bc.stride, bc.padding,
                         w.ct1_w_.data(), w.ct1_K_,
                         w.ct1_b_.empty() ? nullptr : w.ct1_b_.data(),
                         &h2, &T2, buf.data(), buf_size))
            return false;
        fprintf(stderr, "[vae] block %d: conv_t1d done T2=%d\n", b, T2);
        { char t[48]; std::snprintf(t, sizeof(t), "blk%d_ct1d", b); nan_check(t, h2.data(), h2.size()); }

        // 2c. 3 × ResUnit — keep h_prev alive across iterations to avoid
        // use-after-free (in_data must point to a valid buffer).
        const float* in_data = h2.data();
        int32_t T_in = T2;
        std::vector<float> h_prev;  // keeps previous ResUnit output alive
        for (int r = 0; r < 3; r++) {
            const auto& ru = w.res_[r];
            fprintf(stderr, "[vae] block %d res%d: T=%d C=%d c1w=%zu c1K=%d c2w=%zu c2K=%d\n",
                    b, r, T_in, bc.out_ch, ru.c1w_.size(), ru.c1K_, ru.c2w_.size(), ru.c2K_);
            std::vector<float> h_ru;
            if (!op_resunit(in_data, T_in, bc.out_ch,
                            ru.s1a_.data(), ru.s1b_.data(),
                            ru.c1w_.data(), ru.c1K_,
                            ru.c1b_.empty() ? nullptr : ru.c1b_.data(),
                            kResDilations[r],
                            ru.s2a_.data(), ru.s2b_.data(),
                            ru.c2w_.data(), ru.c2K_,
                            ru.c2b_.empty() ? nullptr : ru.c2b_.data(),
                            &h_ru, buf.data(), buf_size))
                return false;
            { char t[48]; std::snprintf(t, sizeof(t), "blk%d_res%d", b, r); nan_check(t, h_ru.data(), h_ru.size()); }
            h_prev = std::move(h_ru);
            in_data = h_prev.data();
            T_in = (int32_t)(h_prev.size() / bc.out_ch);
        }
        fprintf(stderr, "[vae] block %d: done T=%d\n", b, T_in);

        // Copy block output to cur
        cur.assign(in_data, in_data + static_cast<size_t>(T_in) * bc.out_ch);
        T = T_in;
        C = bc.out_ch;
        char blk_tag[32]; std::snprintf(blk_tag, sizeof(blk_tag), "block%d_out", b);
        nan_check(blk_tag, cur.data(), static_cast<size_t>(T) * C);
    }

    // cur: [T, 128] at 48x oversampled rate

    // ══════════════════════════════════════════════════════════════════════
    //  Phase 3: final snake + conv2 → stereo PCM
    // ══════════════════════════════════════════════════════════════════════

    // Final snake
    std::vector<float> after_fn;
    if (!op_snake(cur.data(), T, 128,
                  dec_fn_exp_a_.data(), dec_fn_inv_b_.data(),
                  &after_fn, buf.data(), buf_size))
        return false;

    // Final conv2: [T, 128] → [T, 2] stereo
    std::vector<float> audio;
    int32_t T_audio = 0;
    if (!op_conv1d(after_fn.data(), T, 128, 2,
                   dec_conv2_w_.data(), dec_conv2_K_,
                   nullptr,
                   1, 3, 1,
                   &audio, &T_audio, buf.data(), buf_size))
        return false;

    pcm->resize(static_cast<size_t>(T_audio) * 2);
    memcpy(pcm->data(), audio.data(),
           static_cast<size_t>(T_audio) * 2 * sizeof(float));

    nan_check("conv2_out(pcm)", pcm->data(), pcm->size());

    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
//  VAE Encoder: stereo PCM [T, 2] → latent [T/1920, 64]
// ═════════════════════════════════════════════════════════════════════════════

bool VAERunner::encode(const float* pcm_stereo, int32_t n_samples,
                        std::vector<float>* latents, std::string* error) {
    if (!ext_ctx_ || n_samples <= 0) {
        if (error) *error = "VAE encode: invalid state";
        return false;
    }

    // ── Shared scratch buffer (4 GB — matches decoder capacity) ──────────
    const size_t buf_size = 4096ULL * 1024 * 1024;
    std::vector<char> buf(buf_size);

    // ── conv1: [T, 2] → [T, 128] (k=7, s=1, p=3) ─────────────────────────
    std::vector<float> cur;
    int32_t T = n_samples;
    int32_t C = 2;
    {
        std::vector<float> out;
        int32_t T_out = 0;
        if (!op_conv1d(pcm_stereo, T, 2, 128,
                       enc_conv1_w_.data(), enc_conv1_K_,
                       enc_conv1_b_.empty() ? nullptr : enc_conv1_b_.data(),
                       1, 3, 1,
                       &out, &T_out, buf.data(), buf_size))
            return false;
        cur = std::move(out);
        T = T_out;
        C = 128;
    }

    // ── 5 encoder blocks ───────────────────────────────────────────────────
    for (int b = 0; b < 5; b++) {
        const auto& bc = kEncBlocks[b];
        const auto& blk = enc_blk_[b];

        // 2a. 3× ResUnit (at bc.channel)
        const float* in = cur.data();
        int32_t T_in = T;
        std::vector<float> h_prev;
        for (int r = 0; r < 3; r++) {
            const auto& ru = blk.res_[r];
            std::vector<float> h;
            if (!op_resunit(in, T_in, bc.channel,
                            ru.s1a_.data(), ru.s1b_.data(),
                            ru.c1w_.data(), ru.c1K_,
                            ru.c1b_.empty() ? nullptr : ru.c1b_.data(),
                            kResDilations[r],
                            ru.s2a_.data(), ru.s2b_.data(),
                            ru.c2w_.data(), ru.c2K_,
                            ru.c2b_.empty() ? nullptr : ru.c2b_.data(),
                            &h, buf.data(), buf_size))
                return false;
            h_prev = std::move(h);
            in = h_prev.data();
            T_in = static_cast<int32_t>(h_prev.size() / bc.channel);
        }

        // 2b. Snake (at bc.channel)
        std::vector<float> after_snake;
        if (!op_snake(in, T_in, bc.channel,
                       blk.snake_a_.data(), blk.snake_b_.data(),
                       &after_snake, buf.data(), buf_size))
            return false;

        // 2c. Strided conv1d: bc.channel → bc.out_ch
        std::vector<float> after_conv;
        int32_t T_conv = 0;
        if (!op_conv1d(after_snake.data(), T_in, bc.channel, bc.out_ch,
                        blk.conv_w_.data(), blk.conv_K_,
                        blk.conv_b_.empty() ? nullptr : blk.conv_b_.data(),
                        bc.stride, bc.padding, 1,
                        &after_conv, &T_conv, buf.data(), buf_size))
            return false;

        // Copy block output
        cur.assign(after_conv.data(),
                   after_conv.data() + static_cast<size_t>(T_conv) * bc.out_ch);
        T = T_conv;
        C = bc.out_ch;
    }

    // cur: [T, 2048] at sample-rate/1920

    // ── Final snake (encoder.snake1.*) ────────────────────────────────────
    {
        std::vector<float> after_fn;
        if (!op_snake(cur.data(), T, 2048,
                       enc_fn_exp_a_.data(), enc_fn_inv_b_.data(),
                       &after_fn, buf.data(), buf_size))
            return false;
        cur = std::move(after_fn);
    }

    // ── Final conv2: [T, 2048] → [T, 128] (k=3, s=1, p=1) ───────────────
    std::vector<float> encoded_128;
    int32_t T_128 = 0;
    if (!op_conv1d(cur.data(), T, 2048, 128,
                    enc_conv2_w_.data(), enc_conv2_K_,
                    enc_conv2_b_.empty() ? nullptr : enc_conv2_b_.data(),
                    1, 1, 1,
                    &encoded_128, &T_128, buf.data(), buf_size))
        return false;

    // ── Extract mean: channels 0..63 → [T, 64] time-major ────────────────
    const int32_t T_latent = T_128;
    latents->resize(static_cast<size_t>(T_latent) * 64);
    float* dst = latents->data();
    const float* src = encoded_128.data();
    for (int32_t t = 0; t < T_latent; t++) {
        for (int32_t c = 0; c < 64; c++) {
            dst[static_cast<size_t>(t) * 64 + c] =
                src[static_cast<size_t>(t) * 128 + c];
        }
    }

    return true;
}

}  // namespace audiocore::acestep
