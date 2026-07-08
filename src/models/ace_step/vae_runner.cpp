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
#include "audiocore/framework/ggml/backend_helper.h"  // BackendPair full def for unique_ptr destructor

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
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
    // The GGUF stores weight_v in PyTorch's natural C-order of the ORIGINAL
    // PyTorch shape. The ggml tensor's ne array is the REVERSED PyTorch shape,
    // but the byte layout matches PyTorch C-order.
    //
    // Conv1d: PyTorch shape (OC, IC, K), so v_pytorch[oc, ic, k] lives at
    //         flat[oc*IC*K + ic*K + k] in the raw buffer.
    //
    // ConvTranspose1d: PyTorch shape (IC, OC, K), so v_pytorch[ic, oc, k]
    //         lives at flat[ic*OC*K + oc*K + k].
    //
    // PyTorch weight_norm normalizes per dim-0 of the ORIGINAL PyTorch shape:
    //   w = g * v / ||v||_dim0
    //
    // We OUTPUT in ggml natural flat for the reversed shape:
    //   Conv1d         [K, IC, OC]: result[k + ic*K + oc*K*IC]
    //   ConvTranspose1d [K, OC, IC]: result[k + oc*K + ic*K*OC]
    // so that downstream ggml ops (ggml_conv_1d expects ne=[K,IC,OC];
    // permute_conv_t1d_weight indexes wsconv_3d[k + oc*K + ic*K*OC])
    // read the right values.
    const uint16_t* v = static_cast<const uint16_t*>(wv_data);
    const uint16_t* g = static_cast<const uint16_t*>(wg_data);
    const size_t total = static_cast<size_t>(K) * OC * IC;
    std::vector<float> result(total);

    if (input_is_K_IC_OC) {
        // Conv1d: PyTorch shape (OC, IC, K). Normalize per oc (PyTorch dim 0).
        // Source: v[((oc*IC + ic)*K) + k] = v[oc*IC*K + ic*K + k].
        // Output: result[k + ic*K + oc*K*IC] (ggml natural flat for [K,IC,OC]).
        for (int oc = 0; oc < OC; oc++) {
            float gv  = bf16_to_f32(g[oc]);
            double nsq = 0.0;
            for (int ic = 0; ic < IC; ic++) {
                for (int k = 0; k < K; k++) {
                    float vv = bf16_to_f32(v[(static_cast<size_t>(oc) * IC + ic) * K + k]);
                    nsq += static_cast<double>(vv) * vv;
                }
            }
            float s = gv / (static_cast<float>(std::sqrt(nsq)) + eps);
            for (int ic = 0; ic < IC; ic++) {
                for (int k = 0; k < K; k++) {
                    float vv = bf16_to_f32(v[(static_cast<size_t>(oc) * IC + ic) * K + k]);
                    result[static_cast<size_t>(k) + ic * K +
                           static_cast<size_t>(oc) * K * IC] = vv * s;
                }
            }
        }
    } else {
        // ConvTranspose1d: PyTorch shape (IC, OC, K). Normalize per ic (dim 0).
        // Source: v[((ic*OC + oc)*K) + k] = v[ic*OC*K + oc*K + k].
        // Output: result[k + oc*K + ic*K*OC] (ggml natural flat for [K,OC,IC]).
        for (int ic = 0; ic < IC; ic++) {
            float gv  = bf16_to_f32(g[ic]);
            double nsq = 0.0;
            for (int oc = 0; oc < OC; oc++) {
                for (int k = 0; k < K; k++) {
                    float vv = bf16_to_f32(v[(static_cast<size_t>(ic) * OC + oc) * K + k]);
                    nsq += static_cast<double>(vv) * vv;
                }
            }
            float s = gv / (static_cast<float>(std::sqrt(nsq)) + eps);
            for (int oc = 0; oc < OC; oc++) {
                for (int k = 0; k < K; k++) {
                    float vv = bf16_to_f32(v[(static_cast<size_t>(ic) * OC + oc) * K + k]);
                    result[static_cast<size_t>(k) + oc * K +
                           static_cast<size_t>(ic) * K * OC] = vv * s;
                }
            }
        }
    }
    return result;
}

// Compute the conv_t1d pre-permuted 2D weight: [IC, K*OC] from WSConv 3D.
// conv_t1d op expects w as 2D [IC, K·OC] (ggml ne: ne[0]=IC, ne[1]=K*OC).
// The WSConv result is 3D [K, OC, IC] (ggml ne order, matching input layout).
// For ggml ne=[IC, K*OC] tensor, memory is data[ic + k_oc*IC].
//
// CRITICAL: ggml_col2im_1d reads the columns tensor as col[(oc*K + k) + t_in*K*OC],
// i.e. the column index encodes (oc, k) as oc*K + k (k fast, oc slow). So the
// 2D weight column index must also be oc*K + k for the matmul to place each
// W[ic, oc, k] contribution in the slot col2im_1d will read for (oc, k).
// Permute: out[ic + (o*K + k)*IC] = wsconv_3d[k + o*K + ic*K*OC]
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
                                 (static_cast<size_t>(o) * K +
                                  static_cast<size_t>(k)) * IC;
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
// Gated behind ACE_STEP_VAE_DEBUG: nan_check scans every element of tensors
// that can hold millions of floats (block 4: T=120000 C=128 = 15.4M), and is
// called ~20x per tile. Unconditionally that CPU scan + stderr I/O cost ~6.5s
// per generation (measured: VAE reported 7876ms but actual CUDA compute was
// only ~1350ms). Default off → VAE decode drops to ~1.4s.
static bool vae_debug_enabled() {
    static const bool v = std::getenv("ACE_STEP_VAE_DEBUG") != nullptr;
    return v;
}

static void nan_check(const char* tag, const float* p, size_t n) {
    if (!vae_debug_enabled()) return;
    int nan_cnt = 0;
    float mx = -1e30f, mn = 1e30f;
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        float v = p[i];
        if (std::isnan(v)) nan_cnt++;
        else {
            if (v > mx) mx = v;
            if (v < mn) mn = v;
            sum_sq += (double)v * v;
        }
    }
    double rms = (n > 0) ? std::sqrt(sum_sq / n) : 0.0;
    fprintf(stderr, "[vae] %-20s NaN=%d/%zu range=[%g,%g] RMS=%.4f\n",
            tag, nan_cnt, n, mn, mx, rms);
    // Optional tensor dump (set ACE_STEP_DUMP_VAE=dir)
    if (const char* dir = std::getenv("ACE_STEP_DUMP_VAE")) {
        std::string path = std::string(dir) + "/" + tag + ".bin";
        if (FILE* f = std::fopen(path.c_str(), "wb")) {
            // Write as [T, C] row-major (our layout); numpy can mmap via np.fromfile
            uint64_t n64 = (uint64_t)n;
            std::fwrite(&n64, sizeof(n64), 1, f);
            std::fwrite(p, sizeof(float), n, f);
            std::fclose(f);
        }
    }
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

// ── Direct-backend helper: run one VAE op graph on the CUDA backend ──────
// Each VAE op builds a small graph with input tensors marked ggml_set_input.
// This helper bypasses the scheduler entirely (which forces INPUT-flagged
// tensors to the last backend = CPU) and instead allocates ALL tensors
// directly on the CUDA backend, computes there, and downloads the result.
struct VaeInputUpload {
    ggml_tensor* t;
    const void* data;
    size_t nbytes;
    // Optional owned storage — keeps `data` alive until after upload.
    // Used when `data` points to a per-call allocated buffer (e.g. F16 weight
    // conversion) that must outlive the builder function scope.
    std::shared_ptr<void> owned;
};

// ── VAE profiling accumulator ─────────────────────────────────────────────
struct VaeProfiler {
    double t_alloc = 0, t_upload = 0, t_compute = 0, t_download = 0;
    size_t bytes_up = 0, bytes_down = 0;
    int n_calls = 0;
    void acc(const std::vector<VaeInputUpload>& uploads, size_t out_nb,
             double a, double u, double c, double d) {
        t_alloc += a; t_upload += u; t_compute += c; t_download += d;
        for (const auto& up : uploads) bytes_up += up.nbytes;
        bytes_down += out_nb;
        n_calls++;
    }
    void reset() { *this = {}; }
    void print(const char* label) const {
        if (n_calls == 0) return;
        fprintf(stderr, "[vae-prof] %s: %d ops  alloc=%.0fms  upload=%.0fms"
                "  compute=%.0fms  download=%.0fms  total=%.0fms"
                "  up=%.1fMB  down=%.1fMB\n",
                label, n_calls,
                t_alloc, t_upload, t_compute, t_download,
                t_alloc + t_upload + t_compute + t_download,
                bytes_up / 1048576.0, bytes_down / 1048576.0);
    }
};
static thread_local VaeProfiler vae_prof;

static bool vae_cuda_compute(ggml_backend_t backend,
                               ggml_context* ctx, ggml_cgraph* gf,
                               ggml_tensor* out_tensor,
                               const std::vector<VaeInputUpload>& uploads,
                               float* out_data, size_t out_nbytes) {
    if (!backend) return false;
    ggml_set_output(out_tensor);
    ggml_build_forward_expand(gf, out_tensor);

    auto _t0 = ggml_time_us();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) return false;
    auto _t1 = ggml_time_us();

    for (const auto& u : uploads)
        ggml_backend_tensor_set(u.t, u.data, 0, u.nbytes);
    auto _t2 = ggml_time_us();

    ggml_backend_graph_compute(backend, gf);
    auto _t3 = ggml_time_us();

    if (out_data && out_nbytes)
        ggml_backend_tensor_get(out_tensor, out_data, 0, out_nbytes);
    auto _t4 = ggml_time_us();

    // Accumulate timing for profiling
    vae_prof.acc(uploads, out_nbytes,
                 (_t1-_t0)/1000.0, (_t2-_t1)/1000.0,
                 (_t3-_t2)/1000.0, (_t4-_t3)/1000.0);

    ggml_backend_buffer_free(buf);
    return true;
}

// ── Conv1d ───────────────────────────────────────────────────────────────────
// x: [T_in, IC] time-major, w: [K, IC, OC] f32, returns: [T_out, OC] time-major
static bool op_conv1d(const float* x, int32_t T_in, int32_t IC, int32_t OC,
                      const float* w_data, int32_t K, const float* bias,
                      int32_t stride, int32_t pad, int32_t dilation,
                      std::vector<float>* out, int32_t* out_T,
                      char* buf, size_t buf_size,
                      ggml_backend_t backend = nullptr) {
    // GPU path: no_alloc ctx, mark inputs, run via direct CUDA backend.
    // CPU path: no_alloc=false ctx, direct memcpy, ggml_graph_compute_with_ctx.
    const bool use_gpu = (backend != nullptr);
    struct ggml_init_params gip = {buf_size, buf, use_gpu};
    ggml_context* ctx = ggml_init(gip);
    ggml_cgraph* gf  = ggml_new_graph(ctx);

    std::vector<VaeInputUpload> uploads;

    // ggml_conv_1d expects input as [T, IC] (ne0=T, ne1=IC) but our input
    // x is time-major [T, IC] in memory.  ggml_tensor with ne0=T, ne1=IC
    // has layout data[t + c*T], so we must transpose the copy.
    ggml_tensor* in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_in, IC);
    ggml_set_name(in, "conv1d_in");
    if (use_gpu) {
        // Build transposed buffer on CPU, then upload after alloc.
        static thread_local std::vector<float> x_t;
        x_t.resize(static_cast<size_t>(T_in) * IC);
        for (int32_t t = 0; t < T_in; t++)
            for (int32_t c = 0; c < IC; c++)
                x_t[t + c * T_in] = x[t * IC + c];
        ggml_set_input(in);
        uploads.push_back({in, x_t.data(),
                           static_cast<size_t>(T_in) * IC * sizeof(float)});
    } else {
        float* dst = (float*)in->data;
        for (int32_t t = 0; t < T_in; t++)
            for (int32_t c = 0; c < IC; c++)
                dst[t + c * T_in] = x[t * IC + c];
    }

    // Build weight tensor from pre-computed f32 data — convert to F16
    // (im2col compute requires F16 kernel on CPU backend).
    // Conv1d weights are stored in GGUF (and our compute_wsconv output) in
    // [K, IC, OC] layout, which matches ggml_conv_1d's expected ne=[K,IC,OC].
    ggml_tensor* w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, K, IC, OC);
    if (use_gpu) {
        static thread_local std::vector<ggml_fp16_t> w16;
        const size_t n_w = static_cast<size_t>(K) * IC * OC;
        w16.resize(n_w);
        for (size_t i = 0; i < n_w; i++) w16[i] = ggml_fp32_to_fp16(w_data[i]);
        ggml_set_input(w);
        uploads.push_back({w, w16.data(), n_w * sizeof(ggml_fp16_t)});
    } else {
        ggml_fp16_t* wd = static_cast<ggml_fp16_t*>(w->data);
        const size_t n_w = static_cast<size_t>(K) * IC * OC;
        for (size_t i = 0; i < n_w; i++) wd[i] = ggml_fp32_to_fp16(w_data[i]);
    }

    // ggml_conv_1d: weight [K, IC, OC], input [T, IC] → [OL, OC, 1]
    ggml_tensor* r = ggml_conv_1d(ctx, w, in, stride, pad, dilation);
    if (bias) {
        ggml_tensor* bt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, OC);
        if (use_gpu) {
            ggml_set_input(bt);
            uploads.push_back({bt, bias, static_cast<size_t>(OC) * sizeof(float)});
        } else {
            memcpy(bt->data, bias, static_cast<size_t>(OC) * sizeof(float));
        }
        r = ggml_add(ctx, r, ggml_repeat(ctx, bt, r));
    }

    *out_T = (int32_t)r->ne[0];  // OL
    int32_t T_out = *out_T;
    out->resize(static_cast<size_t>(T_out) * OC);

    if (use_gpu) {
        if (!vae_cuda_compute(backend, ctx, gf, r, uploads,
                               out->data(),
                               static_cast<size_t>(T_out) * OC * sizeof(float))) {
            ggml_free(ctx); return false;
        }
        // ggml output data[t + c·T_out] → our output[t·OC + c] (transpose)
        static thread_local std::vector<float> tmp;
        tmp.resize(static_cast<size_t>(T_out) * OC);
        memcpy(tmp.data(), out->data(), tmp.size() * sizeof(float));
        for (int32_t t = 0; t < T_out; t++)
            for (int32_t c = 0; c < OC; c++)
                (*out)[t * static_cast<size_t>(OC) + c] = tmp[t + c * static_cast<size_t>(T_out)];
    } else {
        ggml_build_forward_expand(gf, r);
        ggml_graph_compute_with_ctx(ctx, gf, 1);
        const float* src = (const float*)r->data;
        for (int32_t t = 0; t < T_out; t++)
            for (int32_t c = 0; c < OC; c++)
                (*out)[t * static_cast<size_t>(OC) + c] = src[t + c * static_cast<size_t>(T_out)];
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
                     char* buf, size_t buf_size,
                     ggml_backend_t backend = nullptr) {
    const bool use_gpu = (backend != nullptr);
    struct ggml_init_params gip = {buf_size, buf, use_gpu};
    ggml_context* ctx = ggml_init(gip);
    ggml_cgraph* gf  = ggml_new_graph(ctx);

    std::vector<VaeInputUpload> uploads;

    ggml_tensor* in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, T);
    ggml_tensor* a_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
    ggml_tensor* b_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
    if (use_gpu) {
        ggml_set_input(in); uploads.push_back({in, x, static_cast<size_t>(T) * C * sizeof(float)});
        ggml_set_input(a_t); uploads.push_back({a_t, exp_a, C * sizeof(float)});
        ggml_set_input(b_t); uploads.push_back({b_t, inv_b, C * sizeof(float)});
    } else {
        memcpy(in->data, x, static_cast<size_t>(T) * C * sizeof(float));
        memcpy(a_t->data, exp_a, C * sizeof(float));
        memcpy(b_t->data, inv_b, C * sizeof(float));
    }

    // x + sin²(α·x) · inv_beta
    ggml_tensor* ax   = ggml_mul(ctx, in, a_t);
    ggml_tensor* s    = ggml_sin(ctx, ax);
    ggml_tensor* s2   = ggml_sqr(ctx, s);
    ggml_tensor* term = ggml_mul(ctx, s2, b_t);
    ggml_tensor* r    = ggml_add(ctx, in, term);

    out->resize(static_cast<size_t>(T) * C);
    if (use_gpu) {
        if (!vae_cuda_compute(backend, ctx, gf, r, uploads, out->data(),
                               static_cast<size_t>(T) * C * sizeof(float))) {
            ggml_free(ctx); return false;
        }
    } else {
        ggml_build_forward_expand(gf, r);
        ggml_graph_compute_with_ctx(ctx, gf, 1);
        memcpy(out->data(), r->data, static_cast<size_t>(T) * C * sizeof(float));
    }

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
                        char* buf, size_t buf_size,
                        ggml_backend_t backend = nullptr) {
    const bool use_gpu = (backend != nullptr);
    struct ggml_init_params gip = {buf_size, buf, use_gpu};
    ggml_context* ctx = ggml_init(gip);
    ggml_cgraph* gf  = ggml_new_graph(ctx);

    std::vector<VaeInputUpload> uploads;

    // Input [T, IC] → ne[0]=IC, ne[1]=T
    ggml_tensor* in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, IC, T_in);
    // Weight tensor [IC, K·OC] f32
    ggml_tensor* w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, IC, K * OC);
    if (use_gpu) {
        ggml_set_input(in);
        uploads.push_back({in, x, static_cast<size_t>(T_in) * IC * sizeof(float)});
        ggml_set_input(w);
        uploads.push_back({w, w_data, static_cast<size_t>(IC) * K * OC * sizeof(float)});
    } else {
        memcpy(in->data, x, static_cast<size_t>(T_in) * IC * sizeof(float));
        memcpy(w->data, w_data, static_cast<size_t>(IC) * K * OC * sizeof(float));
    }

    // mul_mat: contracts over ne[0]=IC (both w and in have ne[0]=IC)
    // →  ne[0]=K·OC, ne[1]=T_in
    ggml_tensor* col = ggml_mul_mat(ctx, w, in);

    // col2im: [K·OC, T_in] → [T_out, OC] where T_out = stride·T_in
    ggml_tensor* r = ggml_col2im_1d(ctx, col, stride, OC, pad);
    if (bias) {
        ggml_tensor* bt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, OC);
        if (use_gpu) {
            ggml_set_input(bt);
            uploads.push_back({bt, bias, static_cast<size_t>(OC) * sizeof(float)});
        } else {
            memcpy(bt->data, bias, static_cast<size_t>(OC) * sizeof(float));
        }
        r = ggml_add(ctx, r, ggml_repeat(ctx, bt, r));
    }

    *out_T = (int32_t)r->ne[0];  // T_out
    int32_t T_out = *out_T;
    out->resize(static_cast<size_t>(T_out) * OC);

    if (use_gpu) {
        if (!vae_cuda_compute(backend, ctx, gf, r, uploads, out->data(),
                               static_cast<size_t>(T_out) * OC * sizeof(float))) {
            ggml_free(ctx); return false;
        }
        // Transpose from ggml [T,C] to time-major [T,C]
        static thread_local std::vector<float> tmp;
        tmp.resize(static_cast<size_t>(T_out) * OC);
        memcpy(tmp.data(), out->data(), tmp.size() * sizeof(float));
        for (int32_t t = 0; t < T_out; t++)
            for (int32_t c = 0; c < OC; c++)
                (*out)[t * static_cast<size_t>(OC) + c] = tmp[t + c * static_cast<size_t>(T_out)];
    } else {
        ggml_build_forward_expand(gf, col);
        ggml_build_forward_expand(gf, r);
        ggml_graph_compute_with_ctx(ctx, gf, 1);
        const float* src = (const float*)r->data;
        for (int32_t t = 0; t < T_out; t++)
            for (int32_t c = 0; c < OC; c++)
                (*out)[t * static_cast<size_t>(OC) + c] = src[t + c * static_cast<size_t>(T_out)];
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
                       char* buf, size_t buf_size,
                       ggml_backend_t backend = nullptr) {
    // Save skip connection
    std::vector<float> skip(x, x + static_cast<size_t>(T) * C);

    // Snake 1
    std::vector<float> h1;
    if (!op_snake(x, T, C, s1a, s1b, &h1, buf, buf_size, backend)) return false;
    nan_check("ru_snake1", h1.data(), h1.size());

    // Conv1 (k=7, dilated)
    std::vector<float> h2;
    int32_t T2 = 0;
    if (!op_conv1d(h1.data(), T, C, C, c1w, c1K, c1b,
                   1, 3 * dilation, dilation, &h2, &T2, buf, buf_size, backend))
        return false;
    nan_check("ru_conv1", h2.data(), h2.size());

    // Snake 2
    std::vector<float> h3;
    if (!op_snake(h2.data(), T2, C, s2a, s2b, &h3, buf, buf_size, backend)) return false;
    nan_check("ru_snake2", h3.data(), h3.size());

    // Conv2 (k=1)
    std::vector<float> h4;
    int32_t T4 = 0;
    if (!op_conv1d(h3.data(), T2, C, C, c2w, c2K, c2b,
                   1, 0, 1, &h4, &T4, buf, buf_size, backend))
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

VAERunner::~VAERunner() {
    // No scheduler to free — we compute directly on the backend.
    cuda_backend_ = nullptr;
    backend_pair_.reset();
}

bool VAERunner::ensure_backend() {
    if (backend_ready_) return cuda_backend_ != nullptr;

    namespace bu = audiocore::ggml_utils;
    backend_pair_  = std::make_unique<bu::BackendPair>(bu::backend_init("VAE"));
    cuda_backend_  = backend_pair_->backend;  // CUDA when available, CPU otherwise
    backend_ready_ = (cuda_backend_ != nullptr);
    // VAE weights are NOT in ext_ctx_ — they're pre-computed into per-op
    // f32 vectors at construction time. Each op builds its own small graph
    // with input tensors that get uploaded via ggml_backend_tensor_set.
    return backend_ready_;
}

// Helper: run a single-op graph through the VAE scheduler.
// Each VAE op (conv1d, snake, conv_t1d, etc.) builds a small graph with
// input tensors marked via ggml_set_input(). This helper:
//   1. Allocates the graph on the scheduler (GPU if available)
//   2. Uploads input data from CPU
//   3. Computes
//   4. Downloads output to CPU
// Returns true on success.
// (VaeInputUpload struct + vae_cuda_compute defined earlier, before op_* fns.)

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

// ═════════════════════════════════════════════════════════════════════════════
//  Per-block sub-graph decode (fast CUDA path)
// ═════════════════════════════════════════════════════════════════════════════
//
// Instead of 73 separate op calls (each with alloc/upload/download/transpose),
// we build 7 sub-graphs: conv1, 5 decoder blocks, final snake+conv2.
// Data stays on the GPU within each block — no CPU transposes or copies.
//
// All builders use column-major layout: ne0=T, ne1=C.

struct SubGraphIO {
    std::vector<float> data;  // column-major [T, C]: data[t + c*T]
    int T = 0, C = 0;
};

// ── Graph builders ──────────────────────────────────────────────────────────

// Snake: x + sin²(α·x) · inv_β   (element-wise, layout [T, C])
static ggml_tensor* gb_snake(ggml_context* ctx, ggml_tensor* x,
                               const float* exp_a, const float* inv_b, int C,
                               std::vector<VaeInputUpload>& uploads)
{
    ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
    ggml_set_input(a);
    uploads.push_back({a, exp_a, static_cast<size_t>(C) * sizeof(float)});
    ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
    ggml_set_input(b);
    uploads.push_back({b, inv_b, static_cast<size_t>(C) * sizeof(float)});

    ggml_tensor* a_2d = ggml_reshape_2d(ctx, a, 1, C);
    ggml_tensor* b_2d = ggml_reshape_2d(ctx, b, 1, C);

    ggml_tensor* ax   = ggml_mul(ctx, x, ggml_repeat(ctx, a_2d, x));
    ggml_tensor* s    = ggml_sin(ctx, ax);
    ggml_tensor* s2   = ggml_sqr(ctx, s);
    ggml_tensor* term = ggml_mul(ctx, s2, ggml_repeat(ctx, b_2d, s2));
    return ggml_add(ctx, x, term);
}

// Conv1d via ggml_conv_1d (decomposes to im2col + mul_mat, both CUDA-supported).
// x: [T, IC] column-major → output: [T_out, OC] column-major
static ggml_tensor* gb_conv1d(ggml_context* ctx, ggml_tensor* x,
                                const float* w_f32, int K, int IC, int OC,
                                const float* bias,
                                int stride, int pad, int dilation,
                                std::vector<VaeInputUpload>& uploads)
{
    // Weight [K, IC, OC] → F16. Each conv1d owns its F16 buffer via shared_ptr
    // so it survives until upload (after the entire graph is built). Previously
    // a static thread_local buffer was reused across calls, causing all convs
    // in a multi-op sub-graph to receive the LAST conv's weights.
    const size_t n_w = static_cast<size_t>(K) * IC * OC;
    auto w16 = std::make_shared<std::vector<ggml_fp16_t>>(n_w);
    for (size_t i = 0; i < n_w; i++) (*w16)[i] = ggml_fp32_to_fp16(w_f32[i]);

    ggml_tensor* w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, K, IC, OC);
    ggml_set_input(w);
    uploads.push_back({w, w16->data(), n_w * sizeof(ggml_fp16_t), w16});

    ggml_tensor* r = ggml_conv_1d(ctx, w, x, stride, pad, dilation);

    if (bias) {
        ggml_tensor* bt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, OC);
        ggml_set_input(bt);
        uploads.push_back({bt, bias, static_cast<size_t>(OC) * sizeof(float)});
        ggml_tensor* bt_2d = ggml_reshape_2d(ctx, bt, 1, OC);
        r = ggml_add(ctx, r, ggml_repeat(ctx, bt_2d, r));
    }
    return r;
}

// ConvTranspose1d via mul_mat + col2im_1d (both CUDA-supported).
// x: [T_in, IC] column-major → output: [T_out, OC] column-major
static ggml_tensor* gb_conv_t1d(ggml_context* ctx, ggml_tensor* x,
                                  const float* w_2d, int IC, int OC, int K,
                                  int stride, int pad, const float* bias,
                                  std::vector<VaeInputUpload>& uploads)
{
    ggml_tensor* x_t = ggml_cont(ctx, ggml_transpose(ctx, x));  // [IC, T_in]

    ggml_tensor* w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, IC, K * OC);
    ggml_set_input(w);
    uploads.push_back({w, w_2d,
                       static_cast<size_t>(IC) * K * OC * sizeof(float)});

    ggml_tensor* col = ggml_mul_mat(ctx, w, x_t);  // [K*OC, T_in]
    ggml_tensor* r = ggml_col2im_1d(ctx, col, stride, OC, pad);  // [T_out, OC]

    if (bias) {
        ggml_tensor* bt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, OC);
        ggml_set_input(bt);
        uploads.push_back({bt, bias, static_cast<size_t>(OC) * sizeof(float)});
        ggml_tensor* bt_2d = ggml_reshape_2d(ctx, bt, 1, OC);
        r = ggml_add(ctx, r, ggml_repeat(ctx, bt_2d, r));
    }
    return r;
}

// ResUnit: snake1 → conv1(k=7,dilated) → snake2 → conv2(k=1) + skip
static ggml_tensor* gb_resunit(ggml_context* ctx, ggml_tensor* x, int C,
                                 const VAERunner::ResUnitWeights& ru,
                                 int dilation,
                                 std::vector<VaeInputUpload>& uploads)
{
    ggml_tensor* h1 = gb_snake(ctx, x, ru.s1a_.data(), ru.s1b_.data(), C, uploads);
    ggml_tensor* h2 = gb_conv1d(ctx, h1, ru.c1w_.data(), ru.c1K_, C, C,
                                  ru.c1b_.empty() ? nullptr : ru.c1b_.data(),
                                  1, 3 * dilation, dilation, uploads);
    ggml_tensor* h3 = gb_snake(ctx, h2, ru.s2a_.data(), ru.s2b_.data(), C, uploads);
    ggml_tensor* h4 = gb_conv1d(ctx, h3, ru.c2w_.data(), ru.c2K_, C, C,
                                  ru.c2b_.empty() ? nullptr : ru.c2b_.data(),
                                  1, 0, 1, uploads);
    return ggml_add(ctx, x, h4);  // skip connection
}

// Decoder block: snake → conv_t1d → 3×resunit
static ggml_tensor* gb_block(ggml_context* ctx, ggml_tensor* x,
                               int block_idx,
                               const VAERunner::BlockWeights& bw,
                               std::vector<VaeInputUpload>& uploads)
{
    const BlockCfg& bc = kBlocks[block_idx];
    ggml_tensor* h1 = gb_snake(ctx, x, bw.snake_a_.data(), bw.snake_b_.data(),
                                bc.in_ch, uploads);
    ggml_tensor* h2 = gb_conv_t1d(ctx, h1, bw.ct1_w_.data(),
                                   bc.in_ch, bc.out_ch, bw.ct1_K_,
                                   bc.stride, bc.padding,
                                   bw.ct1_b_.empty() ? nullptr : bw.ct1_b_.data(),
                                   uploads);
    ggml_tensor* cur = h2;
    for (int r = 0; r < 3; r++)
        cur = gb_resunit(ctx, cur, bc.out_ch, bw.res_[r],
                          kResDilations[r], uploads);
    return cur;
}

// ── Sub-graph executor (direct CUDA backend, no scheduler) ──────────────────
template<typename Builder>
static bool run_block_subgraph(
    ggml_backend_t backend,
    const float* input_col, int T_in, int C_in,
    Builder builder,
    SubGraphIO* output,
    std::string* error, const char* label = "")
{
    // Scratch buffer for tensor metadata only (data lives on CUDA).
    constexpr size_t META_BUF = 1 << 20;  // 1 MB
    static thread_local std::vector<char> meta_buf;
    meta_buf.assign(META_BUF, 0);

    ggml_init_params gip = {meta_buf.size(), meta_buf.data(), /*no_alloc=*/true};
    ggml_context* ctx = ggml_init(gip);
    if (!ctx) { if (error) *error = std::string(label) + ": ggml_init failed"; return false; }

    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 8192, false);

    std::vector<VaeInputUpload> uploads;

    ggml_tensor* in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_in, C_in);
    ggml_set_input(in);
    uploads.push_back({in, input_col,
                       static_cast<size_t>(T_in) * C_in * sizeof(float)});

    ggml_tensor* result = builder(ctx, in, uploads);
    ggml_set_output(result);
    ggml_build_forward_expand(gf, result);

    auto _t0 = ggml_time_us();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        if (error) *error = std::string(label) + ": CUDA alloc failed";
        ggml_free(ctx);
        return false;
    }
    auto _t1 = ggml_time_us();

    for (const auto& u : uploads)
        ggml_backend_tensor_set(u.t, u.data, 0, u.nbytes);
    auto _t2 = ggml_time_us();

    ggml_backend_graph_compute(backend, gf);
    auto _t3 = ggml_time_us();

    output->T = (int)result->ne[0];
    output->C = (int)result->ne[1];
    size_t n = static_cast<size_t>(output->T) * output->C;
    output->data.resize(n);
    ggml_backend_tensor_get(result, output->data.data(), 0, n * sizeof(float));
    auto _t4 = ggml_time_us();

    if (std::getenv("AUDIOCORE_VAE_PROFILE"))
        fprintf(stderr, "[vae-blk] %-12s alloc=%dms upload=%dms compute=%dms"
                " dl=%dms nodes=%zu → [%d, %d]\n",
                label, (int)((_t1-_t0)/1000), (int)((_t2-_t1)/1000),
                (int)((_t3-_t2)/1000), (int)((_t4-_t3)/1000),
                (size_t)ggml_graph_n_nodes(gf), output->T, output->C);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

// ── Per-block decode (fast path) ────────────────────────────────────────────
static const char* kBlkNames[5] = {"block0", "block1", "block2", "block3", "block4"};

bool VAERunner::decode_blocks(const float* latents, int32_t n_frames,
                                std::vector<float>* pcm, std::string* error)
{
    const int32_t T = n_frames;

    // Transpose latents from time-major [T, 64] to column-major [T, 64]
    SubGraphIO io;
    io.T = T;
    io.C = 64;
    io.data.resize(static_cast<size_t>(T) * 64);
    for (int32_t t = 0; t < T; t++)
        for (int32_t c = 0; c < 64; c++)
            io.data[t + c * T] = latents[t * 64 + c];

    // ── Sub-graph 0: conv1 [T, 64] → [T, 2048] ────────────────────────────
    {
        SubGraphIO out;
        if (!run_block_subgraph(cuda_backend_, io.data.data(), io.T, io.C,
                [&](ggml_context* ctx, ggml_tensor* in,
                    std::vector<VaeInputUpload>& uploads) {
                    return gb_conv1d(ctx, in,
                        dec_conv1_w_.data(), dec_conv1_K_, 64, 2048,
                        dec_conv1_b_.empty() ? nullptr : dec_conv1_b_.data(),
                        1, 3, 1, uploads);
                },
                &out, error, "conv1"))
            return false;
        // Diagnostic: dump first few conv1 output values
        if (std::getenv("ACE_STEP_VAE_DEBUG"))
            fprintf(stderr, "[vae_diag] conv1 out T=%d C=%d [0..3]=%.4f,%.4f,%.4f,%.4f RMS=%.4f\n",
                    out.T, out.C, out.data[0], out.data[1], out.data[2], out.data[3],
                    std::sqrt([&]{ double s=0; for(float v: out.data) s+=v*v; return s/out.data.size(); }()));
        io = std::move(out);
    }

    // ── Sub-graphs 1-5: decoder blocks ────────────────────────────────────
    // When ACE_STEP_VAE_BISECT is set, block 0 is split into sub-graphs
    // to pinpoint divergence vs slow path.
    //   BISECT=1: each op separately (snake, ct1d, res0, res1, res2)
    //   BISECT=2: snake+ct1d as one, res0 as one, res1 as one, res2 as one
    //   BISECT=3: snake+ct1d+res0 as one, res1+res2 as one
    const bool bisect = std::getenv("ACE_STEP_VAE_BISECT") != nullptr;
    int bisect_mode = bisect ? atoi(std::getenv("ACE_STEP_VAE_BISECT")) : 0;
    if (bisect_mode == 0) bisect_mode = 1;
    for (int b = 0; b < 5; b++) {
        const int bi = b;
        if (bisect && b == 0) {
            auto rms_dump = [](const SubGraphIO& o, const char* tag) {
                double sq = 0; for (float v : o.data) sq += v*v;
                fprintf(stderr, "[vae_diag] blk0 %s: T=%d C=%d RMS=%.4f\n",
                        tag, o.T, o.C, std::sqrt(sq / o.data.size()));
            };
            if (bisect_mode == 1) {
                // Each op as its own sub-graph
                SubGraphIO out;
                // snake
                if (!run_block_subgraph(cuda_backend_, io.data.data(), io.T, io.C,
                        [&](ggml_context* ctx, ggml_tensor* in, std::vector<VaeInputUpload>& u) {
                            return gb_snake(ctx, in, dec_blk_[0].snake_a_.data(),
                                dec_blk_[0].snake_b_.data(), kBlocks[0].in_ch, u); },
                        &out, error, "blk0_snake")) return false;
                rms_dump(out, "snake"); io = std::move(out);
                // conv_t1d
                if (!run_block_subgraph(cuda_backend_, io.data.data(), io.T, io.C,
                        [&](ggml_context* ctx, ggml_tensor* in, std::vector<VaeInputUpload>& u) {
                            return gb_conv_t1d(ctx, in, dec_blk_[0].ct1_w_.data(),
                                kBlocks[0].in_ch, kBlocks[0].out_ch, dec_blk_[0].ct1_K_,
                                kBlocks[0].stride, kBlocks[0].padding,
                                dec_blk_[0].ct1_b_.empty()?nullptr:dec_blk_[0].ct1_b_.data(), u); },
                        &out, error, "blk0_ct1d")) return false;
                rms_dump(out, "conv_t1d"); io = std::move(out);
                // res0, res1, res2
                for (int r = 0; r < 3; r++) {
                    const int ri = r;
                    if (!run_block_subgraph(cuda_backend_, io.data.data(), io.T, io.C,
                            [&](ggml_context* ctx, ggml_tensor* in, std::vector<VaeInputUpload>& u) {
                                return gb_resunit(ctx, in, kBlocks[0].out_ch,
                                    dec_blk_[0].res_[ri], kResDilations[ri], u); },
                            &out, error, "blk0_res")) return false;
                    char tag[16]; snprintf(tag, sizeof(tag), "res%d", ri);
                    rms_dump(out, tag); io = std::move(out);
                }
                continue;
            } else if (bisect_mode == 2) {
                // snake+ct1d fused, then each resunit separate
                SubGraphIO out;
                if (!run_block_subgraph(cuda_backend_, io.data.data(), io.T, io.C,
                        [&](ggml_context* ctx, ggml_tensor* in, std::vector<VaeInputUpload>& u) {
                            auto h = gb_snake(ctx, in, dec_blk_[0].snake_a_.data(),
                                dec_blk_[0].snake_b_.data(), kBlocks[0].in_ch, u);
                            return gb_conv_t1d(ctx, h, dec_blk_[0].ct1_w_.data(),
                                kBlocks[0].in_ch, kBlocks[0].out_ch, dec_blk_[0].ct1_K_,
                                kBlocks[0].stride, kBlocks[0].padding,
                                dec_blk_[0].ct1_b_.empty()?nullptr:dec_blk_[0].ct1_b_.data(), u); },
                        &out, error, "blk0_snake_ct1d")) return false;
                rms_dump(out, "snake+ct1d"); io = std::move(out);
                for (int r = 0; r < 3; r++) {
                    const int ri = r;
                    if (!run_block_subgraph(cuda_backend_, io.data.data(), io.T, io.C,
                            [&](ggml_context* ctx, ggml_tensor* in, std::vector<VaeInputUpload>& u) {
                                return gb_resunit(ctx, in, kBlocks[0].out_ch,
                                    dec_blk_[0].res_[ri], kResDilations[ri], u); },
                            &out, error, "blk0_res")) return false;
                    char tag[16]; snprintf(tag, sizeof(tag), "res%d", ri);
                    rms_dump(out, tag); io = std::move(out);
                }
                continue;
            } else if (bisect_mode == 3) {
                // snake+ct1d+res0 fused, res1+res2 fused
                SubGraphIO out;
                if (!run_block_subgraph(cuda_backend_, io.data.data(), io.T, io.C,
                        [&](ggml_context* ctx, ggml_tensor* in, std::vector<VaeInputUpload>& u) {
                            auto h = gb_snake(ctx, in, dec_blk_[0].snake_a_.data(),
                                dec_blk_[0].snake_b_.data(), kBlocks[0].in_ch, u);
                            h = gb_conv_t1d(ctx, h, dec_blk_[0].ct1_w_.data(),
                                kBlocks[0].in_ch, kBlocks[0].out_ch, dec_blk_[0].ct1_K_,
                                kBlocks[0].stride, kBlocks[0].padding,
                                dec_blk_[0].ct1_b_.empty()?nullptr:dec_blk_[0].ct1_b_.data(), u);
                            return gb_resunit(ctx, h, kBlocks[0].out_ch,
                                dec_blk_[0].res_[0], kResDilations[0], u); },
                        &out, error, "blk0_snake_ct1d_res0")) return false;
                rms_dump(out, "snake+ct1d+res0"); io = std::move(out);
                // res1+res2 fused
                if (!run_block_subgraph(cuda_backend_, io.data.data(), io.T, io.C,
                        [&](ggml_context* ctx, ggml_tensor* in, std::vector<VaeInputUpload>& u) {
                            auto h = gb_resunit(ctx, in, kBlocks[0].out_ch,
                                dec_blk_[0].res_[1], kResDilations[1], u);
                            return gb_resunit(ctx, h, kBlocks[0].out_ch,
                                dec_blk_[0].res_[2], kResDilations[2], u); },
                        &out, error, "blk0_res12")) return false;
                rms_dump(out, "res1+res2"); io = std::move(out);
                continue;
            }
        }
        SubGraphIO out;
        if (!run_block_subgraph(cuda_backend_, io.data.data(), io.T, io.C,
                [&](ggml_context* ctx, ggml_tensor* in,
                    std::vector<VaeInputUpload>& uploads) {
                    return gb_block(ctx, in, bi, dec_blk_[bi], uploads);
                },
                &out, error, kBlkNames[b]))
            return false;
        // Diagnostic: RMS after each block
        if (std::getenv("ACE_STEP_VAE_DEBUG")) {
            double sq = 0; for (float v : out.data) sq += v*v;
            fprintf(stderr, "[vae_diag] after block %d: T=%d C=%d RMS=%.4f\n",
                    b, out.T, out.C, std::sqrt(sq / out.data.size()));
        }
        io = std::move(out);
    }

    // ── Sub-graph 6: final snake + conv2 [T, 128] → [T, 2] ────────────────
    {
        SubGraphIO out;
        if (!run_block_subgraph(cuda_backend_, io.data.data(), io.T, io.C,
                [&](ggml_context* ctx, ggml_tensor* in,
                    std::vector<VaeInputUpload>& uploads) {
                    ggml_tensor* h = gb_snake(ctx, in,
                        dec_fn_exp_a_.data(), dec_fn_inv_b_.data(), 128, uploads);
                    return gb_conv1d(ctx, h,
                        dec_conv2_w_.data(), dec_conv2_K_, 128, 2,
                        nullptr, 1, 3, 1, uploads);
                },
                &out, error, "final"))
            return false;
        io = std::move(out);
    }

    // ── Convert column-major [T, 2] → interleaved stereo [T*2] ────────────
    pcm->resize(static_cast<size_t>(io.T) * 2);
    for (int32_t t = 0; t < io.T; t++) {
        (*pcm)[t * 2]     = io.data[t];              // L
        (*pcm)[t * 2 + 1] = io.data[t + io.T];       // R
    }

    return true;
}

// (kBlkNames defined above, before decode_blocks)

bool VAERunner::decode(const float* latents, int32_t n_frames,
                        std::vector<float>* pcm, std::string* error) {
    if (!ext_ctx_ || n_frames <= 0) {
        if (error) *error = "VAE: invalid state";
        return false;
    }
    // Initialize GPU backend + scheduler on first call.
    if (!ensure_backend()) {
        if (error) *error = "VAE: backend init failed";
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
    // Tile geometry (tuned for per-block VRAM budget):
    //   TILE_FRAMES = 100 latent frames  = 4.0s audio
    //     At 100 frames, block 4 (the final ×2 upsampling stage, T_in=96000,
    //     C=128) allocates ~7 GB for im2col + intermediates — fits alongside
    //     resident DiT weights on a 24 GB GPU. At 150 frames block 4 needs
    //     ~11.8 GB and OOMs, forcing the slow per-op fallback (73 individual
    //     ops with per-op CPU overhead = ~6s/tile vs ~0.7s/tile for the
    //     7-sub-graph fast path).
    //   OVERLAP_FRAMES = 20 latent frames = 0.8s audio crossfade
    // 1920 = total VAE upsampling factor (latent_frames → pcm_samples)
    constexpr int32_t TILE_FRAMES    = 100;
    constexpr int32_t OVERLAP_FRAMES = 20;
    constexpr int32_t UPSAMPLE       = 1920;

    if (n_frames > TILE_FRAMES) {
        fprintf(stderr, "[vae] tiled decode: %d frames → tiles of %d (overlap %d)\n",
                n_frames, TILE_FRAMES, OVERLAP_FRAMES);
        // Reset the per-block fast-path failure flag for each new generation.
        // Within a single tiled decode, tile 0's failure still prevents retries
        // on tiles 1/2 (they face the same VRAM conditions). But across
        // generations — especially after a model swap frees VRAM — the fast
        // path MUST be retried. Without this reset, one OOM (e.g. transient
        // fragmentation after model swap) permanently disables the fast path,
        // turning every subsequent decode from ~1.1s into ~6.4s.
        blocks_failed_once_ = false;
        const size_t pcm_total = static_cast<size_t>(n_frames) * UPSAMPLE * 2;
        pcm->assign(pcm_total, 0.0f);
        const size_t ov_pcm = static_cast<size_t>(OVERLAP_FRAMES) * UPSAMPLE;
        // HALF-Hann window: goes 0 → 1 (monotonically increasing).
        // A FULL Hann (2π) goes 0 → 1 → 0, which makes BOTH tiles
        // silent at overlap boundaries — the source of the crackle.
        // The half-Hann ensures fade-in goes 0→1 and, with the reversed
        // fade-out index, fade-out goes 1→0.  Their sum is always > 0.
        std::vector<float> hann_w(ov_pcm);
        for (size_t i = 0; i < ov_pcm; i++) {
            hann_w[i] = 0.5f * (1.0f - std::cos(3.14159265358979f * i /
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
                // Fade-out region (last OVERLAP_FRAMES of tile, skip last tile).
                // The outgoing tile must be at FULL volume when entering the
                // overlap (frames_from_end = OVERLAP-1 → hann ≈ 1.0) and SILENT
                // when leaving (frames_from_end = 0 → hann = 0).  This
                // COMPLEMENTS the fade-in (which goes 0 → 1) so the weights
                // always sum to ≈1 in the overlap — a proper crossfade.
                //
                // BUGFIX: the old index was (OVERLAP-1 - frames_from_end),
                // which is the same direction as fade-in.  Both tiles went to
                // 0 at the overlap START (silence → click) and to 1.0 at the
                // overlap END (double volume → pop).  This was the root cause
                // of the periodic crackle heard every ~3.2 s in 10 s clips.
                size_t tile_total_frames = tile_pcm.size() / 2 / UPSAMPLE;
                size_t frames_from_end = (tile_total_frames >= 1)
                    ? (tile_total_frames - 1 - tile_pcm_frame) : 0;
                if (frames_from_end < static_cast<size_t>(OVERLAP_FRAMES) &&
                    t1 < n_frames) {
                    size_t hann_idx = frames_from_end * UPSAMPLE +
                                      (tile_sample % UPSAMPLE);
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

    // ── Fast path: per-block sub-graph decode (CUDA) ───────────────────────
    // Builds 7 sub-graphs instead of 73 per-op calls, keeping data on GPU.
    // If CUDA alloc fails (VRAM exhaustion on large tiles), falls back to
    // the per-op path which has lower peak VRAM (one op at a time).
    // Set ACE_STEP_VAE_SLOW=1 to force the per-op path (with diagnostics).
    const bool force_slow = std::getenv("ACE_STEP_VAE_SLOW") != nullptr;
    // blocks_failed_once_: if the per-block fast path has failed for a tile in
    // THIS tiled decode, don't retry for subsequent tiles — they face the same
    // VRAM conditions and will fail the same way, wasting ~200ms/tile. The flag
    // is reset at the top of each tiled decode so a transient OOM doesn't
    // permanently cripple the fast path for all future decodes.
    if (cuda_backend_ && !force_slow && !blocks_failed_once_) {
        std::string blk_err;
        if (decode_blocks(latents, n_frames, pcm, &blk_err)) {
            return true;
        }
        fprintf(stderr, "[vae] per-block decode failed (%s), falling back to per-op"
                        " (will skip fast-path for remaining tiles in this decode)\n",
                blk_err.c_str());
        blocks_failed_once_ = true;
        if (error) error->clear();
    }

    // ── Fallback: per-op decode ───────────────────────────────────────────
    // Each op builds its own small graph. Lower peak VRAM than per-block.
    //
    // Scratch buffer: with use_gpu=true (no_alloc=true) this pool holds ONLY
    // tensor metadata (~KB per op), not compute data — the actual compute
    // happens in CUDA scratch. 256MB is vast overkill for metadata but safe.
    // STATIC + thread_local so the buffer is allocated ONCE and reused across
    // all tiles + all decode() calls. Previously this was a 4GB vector<char>
    // value-initialized (zero-filled) and destroyed per tile — that memset +
    // munmap cost ~2-4s/tile, dominating the VAE decode time (was 9.5s for a
    // 10s clip; ~8s of that was this allocation, not compute).
    const size_t buf_size = 256ULL * 1024 * 1024;
    static thread_local std::vector<char> buf;
    if (buf.size() < buf_size) {
        fprintf(stderr, "[vae] allocating %.0f MB scratch (once)...\n", buf_size / 1048576.0);
        buf.resize(buf_size);  // no zero-fill needed — ggml_init doesn't read it
        fprintf(stderr, "[vae] scratch ready, phase 1: conv1\n");
    }

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
                       &out, &T_out, buf.data(), buf_size, cuda_backend_))
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
                      &h1, buf.data(), buf_size, cuda_backend_))
            return false;
        { char t[48]; std::snprintf(t, sizeof(t), "blk%d_snake", b); nan_check(t, h1.data(), h1.size()); }

        // 2b. ConvTranspose1d
        std::vector<float> h2;
        int32_t T2 = 0;
        if (!op_conv_t1d(h1.data(), T, bc.in_ch, bc.out_ch,
                         bc.stride, bc.padding,
                         w.ct1_w_.data(), w.ct1_K_,
                         w.ct1_b_.empty() ? nullptr : w.ct1_b_.data(),
                         &h2, &T2, buf.data(), buf_size, cuda_backend_))
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
                            &h_ru, buf.data(), buf_size, cuda_backend_))
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
                  &after_fn, buf.data(), buf_size, cuda_backend_))
        return false;
    nan_check("final_snake", after_fn.data(), after_fn.size());

    // Final conv2: [T, 128] → [T, 2] stereo
    std::vector<float> audio;
    int32_t T_audio = 0;
    if (!op_conv1d(after_fn.data(), T, 128, 2,
                   dec_conv2_w_.data(), dec_conv2_K_,
                   nullptr,
                   1, 3, 1,
                   &audio, &T_audio, buf.data(), buf_size, cuda_backend_))
        return false;
    nan_check("final_conv2_out", audio.data(), audio.size());

    pcm->resize(static_cast<size_t>(T_audio) * 2);
    memcpy(pcm->data(), audio.data(),
           static_cast<size_t>(T_audio) * 2 * sizeof(float));

    nan_check("conv2_out(pcm)", pcm->data(), pcm->size());

    vae_prof.print("decode");
    vae_prof.reset();

    return true;
}
// ═════════════════════════════════════════════════════════════════════════════

bool VAERunner::encode(const float* pcm_stereo, int32_t n_samples,
                        std::vector<float>* latents, std::string* error) {
    if (!ext_ctx_ || n_samples <= 0) {
        if (error) *error = "VAE encode: invalid state";
        return false;
    }
    if (!ensure_backend()) {
        if (error) *error = "VAE: backend init failed";
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
                       &out, &T_out, buf.data(), buf_size, cuda_backend_))
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
                            &h, buf.data(), buf_size, cuda_backend_))
                return false;
            h_prev = std::move(h);
            in = h_prev.data();
            T_in = static_cast<int32_t>(h_prev.size() / bc.channel);
        }

        // 2b. Snake (at bc.channel)
        std::vector<float> after_snake;
        if (!op_snake(in, T_in, bc.channel,
                       blk.snake_a_.data(), blk.snake_b_.data(),
                       &after_snake, buf.data(), buf_size, cuda_backend_))
            return false;

        // 2c. Strided conv1d: bc.channel → bc.out_ch
        std::vector<float> after_conv;
        int32_t T_conv = 0;
        if (!op_conv1d(after_snake.data(), T_in, bc.channel, bc.out_ch,
                        blk.conv_w_.data(), blk.conv_K_,
                        blk.conv_b_.empty() ? nullptr : blk.conv_b_.data(),
                        bc.stride, bc.padding, 1,
                        &after_conv, &T_conv, buf.data(), buf_size, cuda_backend_))
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
                       &after_fn, buf.data(), buf_size, cuda_backend_))
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
                    &encoded_128, &T_128, buf.data(), buf_size, cuda_backend_))
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
