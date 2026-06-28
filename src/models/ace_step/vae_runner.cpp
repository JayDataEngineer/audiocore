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

static const int32_t kResDilations[3] = {1, 3, 9};

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
// This gives us clean per-block memory and avoids dealing with ggml's internal
// dimension transposition in conv1d output (which is [OL, OC, 1] = ne[0]=OL).

// ── Conv1d ───────────────────────────────────────────────────────────────────
// x: [T_in, IC] time-major, returns: [T_out, OC] time-major
static bool op_conv1d(const float* x, int32_t T_in, int32_t IC, int32_t OC,
                      ggml_tensor* w, ggml_tensor* bias,
                      int32_t stride, int32_t pad, int32_t dilation,
                      std::vector<float>* out, int32_t* out_T,
                      char* buf, size_t buf_size) {
    struct ggml_init_params gip = {buf_size, buf, false};
    ggml_context* ctx = ggml_init(gip);
    ggml_cgraph* gf  = ggml_new_graph(ctx);

    ggml_tensor* in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, IC, T_in);
    memcpy(in->data, x, static_cast<size_t>(T_in) * IC * sizeof(float));

    // ggml_conv_1d: weight [kH, IC, OC], input [T, IC] → [OL, OC, 1]
    ggml_tensor* r = ggml_conv_1d(ctx, w, in, stride, pad, dilation);
    if (bias) {
        r = ggml_add(ctx, r, bias);
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
static bool op_conv_t1d(const float* x, int32_t T_in, int32_t IC, int32_t OC,
                        int32_t stride, int32_t pad,
                        ggml_tensor* w,   // 2D [IC, K·OC] F16, pre-permuted
                        ggml_tensor* bias,
                        std::vector<float>* out, int32_t* out_T,
                        char* buf, size_t buf_size) {
    struct ggml_init_params gip = {buf_size, buf, false};
    ggml_context* ctx = ggml_init(gip);
    ggml_cgraph* gf  = ggml_new_graph(ctx);

    // Input [T, IC] → ne[0]=IC, ne[1]=T
    ggml_tensor* in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, IC, T_in);
    memcpy(in->data, x, static_cast<size_t>(T_in) * IC * sizeof(float));

    // mul_mat: contracts over ne[0]=IC (both w and in have ne[0]=IC)
    // w:  ne[0]=IC, ne[1]=K·OC
    // in: ne[0]=IC, ne[1]=T_in
    // →  ne[0]=K·OC, ne[1]=T_in
    ggml_tensor* col = ggml_mul_mat(ctx, w, in);
    ggml_build_forward_expand(gf, col);

    // col2im: [K·OC, T_in] → [T_out, OC] where T_out = stride·T_in
    ggml_tensor* r = ggml_col2im_1d(ctx, col, stride, OC, pad);
    if (bias) {
        r = ggml_add(ctx, r, bias);
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
                       ggml_tensor* c1w, ggml_tensor* c1b, int32_t dilation,
                       const float* s2a, const float* s2b,
                       ggml_tensor* c2w, ggml_tensor* c2b,
                       std::vector<float>* out,
                       char* buf, size_t buf_size) {
    // Save skip connection
    std::vector<float> skip(x, x + static_cast<size_t>(T) * C);

    // Snake 1
    std::vector<float> h1;
    if (!op_snake(x, T, C, s1a, s1b, &h1, buf, buf_size)) return false;

    // Conv1 (k=7, dilated)
    std::vector<float> h2;
    int32_t T2 = 0;
    if (!op_conv1d(h1.data(), T, C, C, c1w, c1b,
                   1, 3 * dilation, dilation, &h2, &T2, buf, buf_size))
        return false;

    // Snake 2
    std::vector<float> h3;
    if (!op_snake(h2.data(), T2, C, s2a, s2b, &h3, buf, buf_size)) return false;

    // Conv2 (k=1)
    std::vector<float> h4;
    int32_t T4 = 0;
    if (!op_conv1d(h3.data(), T2, C, C, c2w, c2b,
                   1, 0, 1, &h4, &T4, buf, buf_size))
        return false;

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
//  VAERunner
// ═════════════════════════════════════════════════════════════════════════════

VAERunner::VAERunner(ggml_context* ext_ctx) : ext_ctx_(ext_ctx) {}
VAERunner::~VAERunner() = default;

ggml_tensor* VAERunner::weight(const char* name) const {
    return ggml_get_tensor(ext_ctx_, name);
}

// Read snake alpha/beta from weight tensors (stored as exp(alpha) / 1/exp(beta)).
static bool load_snake_params(ggml_tensor* alpha_t, ggml_tensor* beta_t,
                              std::vector<float>* exp_a,
                              std::vector<float>* inv_b,
                              int32_t expected_C,
                              std::string* error) {
    if (!alpha_t || !beta_t) {
        if (error) *error = "VAE: missing snake tensor";
        return false;
    }
    int32_t ca = (int32_t)alpha_t->ne[0];
    int32_t cb = (int32_t)beta_t->ne[0];
    if (ca != expected_C || cb != expected_C) {
        if (error) *error = "VAE: snake channels " + std::to_string(ca) +
                            "/" + std::to_string(cb) + " != expected " +
                            std::to_string(expected_C);
        return false;
    }
    const float* a = (const float*)alpha_t->data;
    const float* b = (const float*)beta_t->data;
    exp_a->assign(a, a + ca);
    inv_b->assign(b, b + cb);
    return true;
}

bool VAERunner::decode(const float* latents, int32_t n_frames,
                        std::vector<float>* pcm, std::string* error) {
    if (!ext_ctx_ || n_frames <= 0) {
        if (error) *error = "VAE: invalid state";
        return false;
    }

    // ── Shared scratch buffer (256 MB) ─────────────────────────────────────
    const size_t buf_size = 256u * 1024u * 1024u;
    std::vector<char> buf(buf_size);

    // ── Load all weights ──────────────────────────────────────────────────

    // conv1
    ggml_tensor* conv1_w = weight("vae.decoder.conv1");
    ggml_tensor* conv1_b = weight("vae.decoder.conv1_bias");
    if (!conv1_w || !conv1_b) {
        if (error) *error = "VAE: missing vae.decoder.conv1";
        return false;
    }

    // final snake + conv2
    std::vector<float> fn_exp_a, fn_inv_b;
    if (!load_snake_params(weight("vae.decoder.snake1.alpha"),
                           weight("vae.decoder.snake1.beta"),
                           &fn_exp_a, &fn_inv_b, 128, error))
        return false;
    ggml_tensor* conv2_w = weight("vae.decoder.conv2");
    if (!conv2_w) {
        if (error) *error = "VAE: missing vae.decoder.conv2";
        return false;
    }

    // Block weights
    struct ResData {
        std::vector<float> s1a, s1b;
        ggml_tensor* c1w = nullptr, * c1b = nullptr;
        std::vector<float> s2a, s2b;
        ggml_tensor* c2w = nullptr, * c2b = nullptr;
    };
    struct BlkData {
        std::vector<float> sa, sb;
        ggml_tensor* ctw = nullptr, * ctb = nullptr;
        ResData ru[3];
    };
    BlkData blk[5];

    for (int b = 0; b < 5; b++) {
        std::string p = "vae.decoder.block." + std::to_string(b) + ".";
        const auto& bc = kBlocks[b];

        if (!load_snake_params(weight((p + "snake1.alpha").c_str()),
                               weight((p + "snake1.beta").c_str()),
                               &blk[b].sa, &blk[b].sb, bc.in_ch, error))
            return false;

        blk[b].ctw = weight((p + "conv_t1").c_str());
        blk[b].ctb = weight((p + "conv_t1_bias").c_str());
        if (!blk[b].ctw) { if (error) *error = "VAE: missing " + p + "conv_t1"; return false; }

        for (int r = 0; r < 3; r++) {
            std::string rp = p + "res_unit" + std::to_string(r + 1) + ".";
            int32_t oc = bc.out_ch;

            if (!load_snake_params(weight((rp + "snake1.alpha").c_str()),
                                   weight((rp + "snake1.beta").c_str()),
                                   &blk[b].ru[r].s1a, &blk[b].ru[r].s1b, oc, error))
                return false;

            blk[b].ru[r].c1w = weight((rp + "conv1").c_str());
            blk[b].ru[r].c1b = weight((rp + "conv1_bias").c_str());
            if (!blk[b].ru[r].c1w) { if (error) *error = "VAE: missing " + rp + "conv1"; return false; }

            if (!load_snake_params(weight((rp + "snake2.alpha").c_str()),
                                   weight((rp + "snake2.beta").c_str()),
                                   &blk[b].ru[r].s2a, &blk[b].ru[r].s2b, oc, error))
                return false;

            blk[b].ru[r].c2w = weight((rp + "conv2").c_str());
            blk[b].ru[r].c2b = weight((rp + "conv2_bias").c_str());
            if (!blk[b].ru[r].c2w) { if (error) *error = "VAE: missing " + rp + "conv2"; return false; }
        }
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
                       conv1_w, conv1_b, 1, 3, 1,
                       &out, &T_out, buf.data(), buf_size))
            return false;
        cur = std::move(out);
        T = T_out;
        C = 2048;
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Phase 2: 5 decoder blocks
    // ══════════════════════════════════════════════════════════════════════
    for (int b = 0; b < 5; b++) {
        const auto& bc  = kBlocks[b];
        const auto& w   = blk[b];

        // 2a. Snake
        std::vector<float> h1;
        if (!op_snake(cur.data(), T, bc.in_ch,
                      w.sa.data(), w.sb.data(), &h1, buf.data(), buf_size))
            return false;

        // 2b. ConvTranspose1d
        std::vector<float> h2;
        int32_t T2 = 0;
        if (!op_conv_t1d(h1.data(), T, bc.in_ch, bc.out_ch,
                         bc.stride, bc.padding,
                         w.ctw, w.ctb, &h2, &T2, buf.data(), buf_size))
            return false;

        // 2c. 3 × ResUnit
        const float* in_data = h2.data();
        int32_t T_in = T2;
        for (int r = 0; r < 3; r++) {
            const auto& ru = w.ru[r];
            std::vector<float> h_ru;
            if (!op_resunit(in_data, T_in, bc.out_ch,
                            ru.s1a.data(), ru.s1b.data(),
                            ru.c1w, ru.c1b, kResDilations[r],
                            ru.s2a.data(), ru.s2b.data(),
                            ru.c2w, ru.c2b,
                            &h_ru, buf.data(), buf_size))
                return false;
            in_data = h_ru.data();
            T_in = (int32_t)(h_ru.size() / bc.out_ch);
        }

        // Copy block output to cur
        cur.assign(in_data, in_data + static_cast<size_t>(T_in) * bc.out_ch);
        T = T_in;
        C = bc.out_ch;
    }

    // cur: [T, 128] at 48x oversampled rate

    // ══════════════════════════════════════════════════════════════════════
    //  Phase 3: final snake + conv2 → stereo PCM
    // ══════════════════════════════════════════════════════════════════════

    // Final snake
    std::vector<float> after_fn;
    if (!op_snake(cur.data(), T, 128,
                  fn_exp_a.data(), fn_inv_b.data(),
                  &after_fn, buf.data(), buf_size))
        return false;

    // Final conv2: [T, 128] → [T, 2] stereo
    std::vector<float> audio;
    int32_t T_audio = 0;
    if (!op_conv1d(after_fn.data(), T, 128, 2,
                   conv2_w, nullptr, 1, 3, 1,
                   &audio, &T_audio, buf.data(), buf_size))
        return false;

    pcm->resize(static_cast<size_t>(T_audio) * 2);
    memcpy(pcm->data(), audio.data(),
           static_cast<size_t>(T_audio) * 2 * sizeof(float));

    return true;
}

}  // namespace audiocore::acestep
