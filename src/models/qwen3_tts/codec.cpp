// codec.cpp — Qwen3-TTS-Tokenizer-12Hz decoder, ggml port.
//
// SPDX-License-Identifier: MIT
//
// Adapted from CrispStrobe/CrispASR (MIT) src/qwen3_tts.cpp into audiocore's
// Backend / TensorStorage / WeightLoader abstractions. The original copyright
// belongs to the CrispASR contributors; this file is the audiocore-side port
// described in docs/CODEC_PORTS.md §2.
//
// Decoder graph (12.5 fps codec frame rate → 24 kHz mono PCM):
//
//   codes (n_q=16, T_codec) int32
//     │
//     │ Step 1: RVQ decode
//     │   cb0 lookup → out_proj (causal_conv1d k=1) → [512, T]
//     │   sum(cb1..15 lookups) → out_proj → [512, T]
//     │   add → [512, T]
//     ▼
//   pre_conv (causal_conv1d k=3) → [1024, T]
//     │
//     │ Step 2: xfmr input_proj (mul_mat + b) → [512, T]
//     │
//     │ Step 3: 8 transformer layers (pre-attn RMSNorm + self-attn +
//     │         LayerScale + residual; pre-FFN RMSNorm + SwiGLU +
//     │         LayerScale + residual) + final RMSNorm + output_proj → [1024, T]
//     ▼
//   2 ConvNeXt upsample stages (each stride 2): tconv → dw_causal_conv →
//     LayerNorm → pwconv1 → GELU → pwconv2 → gamma → residual → [1024, 4T]
//     │
//     │ Step 5: in_conv (causal_conv1d k=3) → [1536, 4T]
//     │
//     │ Step 6: 4 decoder blocks (stride 8/5/4/3): snake_beta →
//     │         transposed_conv → 3× res_unit (dilations 1/3/9) → [1536, 1920·T]
//     ▼
//   out_snake_beta → out_conv (causal_conv1d k=7) → clamp → [1, 1920·T]
//     │
//     ▼
//   reshape_1d → waveform (T_codec * 1920,)
//
// All convs are causal (left-pad only). The transformer uses pre-LN RMSNorm,
// non-fused Q/K/V matmuls, NEOX RoPE (beta_fast=32, beta_slow=1),
// ggml_flash_attn_ext with a causal mask, SwiGLU FFN, and per-channel
// LayerScale on the residual branches. SnakeBeta activation:
//   y = x + exp(-β) · sin²(x · exp(α))
//
// Single-pass decode: no sliding-window chunking (CrispASR's windowed mode is
// a perf overlay that keeps VRAM constant for long sequences; correctness is
// identical to the full-sequence path the codec's causal structure makes
// bit-equivalent to chunked decode with sufficient left-context).
//
// Backend note: CrispASR ships two transposed-conv1d implementations. The
// "decomposed" path (mul_mat + col2im_1d) is the GPU default; the "crop"
// path (ggml_conv_transpose_1d + tail crop) is documented as "stable, works
// on CPU without the col2im op." This port uses the crop path exclusively —
// it avoids the host-side weight permutation step and works on every
// backend ggml_conv_transpose_1d supports. The Metal-specific ggml_conv_*
// hang documented in CrispASR's conv.h comment is a Metal-only issue; CPU
// and CUDA paths are unaffected.

#include "audiocore/models/qwen3_tts/codec.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace audiocore::qwen3_tts {

namespace {

// f16 bit constants for the causal mask. Same encoding the MOSS codec port
// uses (Stage 16) — ggml_flash_attn_ext accepts an F16 mask and treats
// -inf entries as masked-out.
constexpr uint16_t F16_ZERO    = 0x0000;
constexpr uint16_t F16_NEG_INF = 0xFC00;

}  // namespace

// ───────────────────────────────────────────────────────────────────────────
// Lifecycle
// ───────────────────────────────────────────────────────────────────────────

Qwen3TtsCodecGraphs::~Qwen3TtsCodecGraphs() {
    if (galloc_) ggml_gallocr_free(galloc_);
}

ggml_tensor* Qwen3TtsCodecGraphs::tensor_(const std::string& name) const {
    ggml_tensor* t = ggml_get_tensor(source_ctx_, name.c_str());
    if (!t) {
        throw std::runtime_error("Qwen3TtsCodecGraphs: missing codec tensor: " + name);
    }
    return t;
}

ggml_tensor* Qwen3TtsCodecGraphs::tensor_or_null_(const std::string& name) const {
    return ggml_get_tensor(source_ctx_, name.c_str());
}

// ───────────────────────────────────────────────────────────────────────────
// Tensor resolution
//
// Names match the cstr/qwen3-tts-tokenizer-12hz-GGUF layout exactly — same
// convention CrispASR's load_codec() req() lookups use (CrispASR qwen3_tts.cpp
// lines 3877-3955). Codec tensors live in a SEPARATE GGUF from the talker;
// the loader discovers it via extras["codec_path"] or by probing the talker
// directory for tokenizer-{f16,q8_0}.gguf.
// ───────────────────────────────────────────────────────────────────────────

void Qwen3TtsCodecGraphs::resolve_tensors_() {
    // RVQ front-end.
    rvq_first_cb_         = tensor_("codec.dec.rvq_first.codebook");
    rvq_first_out_w_      = tensor_("codec.dec.rvq_first.out_proj_w");
    rvq_rest_out_w_       = tensor_("codec.dec.rvq_rest.out_proj_w");
    for (int q = 0; q < 15; ++q) {
        rvq_rest_cb_[q]   = tensor_("codec.dec.rvq_rest." + std::to_string(q) + ".codebook");
    }

    // pre_conv + transformer.
    pre_conv_w_           = tensor_("codec.dec.pre_conv_w");
    pre_conv_b_           = tensor_("codec.dec.pre_conv_b");
    xfmr_in_proj_w_       = tensor_("codec.dec.xfmr.in_proj_w");
    xfmr_in_proj_b_       = tensor_("codec.dec.xfmr.in_proj_b");
    xfmr_norm_w_          = tensor_("codec.dec.xfmr.norm_w");
    xfmr_out_proj_w_      = tensor_("codec.dec.xfmr.out_proj_w");
    xfmr_out_proj_b_      = tensor_("codec.dec.xfmr.out_proj_b");

    xfmr_layers_.resize(hp.n_layers);
    for (uint32_t il = 0; il < hp.n_layers; ++il) {
        const std::string p = "codec.dec.xfmr.blk." + std::to_string(il) + ".";
        XfmrLayer& L = xfmr_layers_[il];
        L.attn_norm_w = tensor_(p + "attn_norm_w");
        L.ffn_norm_w  = tensor_(p + "ffn_norm_w");
        L.attn_q_w    = tensor_(p + "attn_q_w");
        L.attn_k_w    = tensor_(p + "attn_k_w");
        L.attn_v_w    = tensor_(p + "attn_v_w");
        L.attn_o_w    = tensor_(p + "attn_o_w");
        L.attn_ls_w   = tensor_(p + "attn_ls_w");
        L.ffn_gate_w  = tensor_(p + "ffn_gate_w");
        L.ffn_up_w    = tensor_(p + "ffn_up_w");
        L.ffn_down_w  = tensor_(p + "ffn_down_w");
        L.ffn_ls_w    = tensor_(p + "ffn_ls_w");
    }

    // 2 ConvNeXt upsample stages.
    for (int s = 0; s < 2; ++s) {
        const std::string p = "codec.dec.up." + std::to_string(s) + ".";
        UpStage& U = up_[s];
        U.tconv_w = tensor_(p + "tconv_w");
        U.tconv_b = tensor_(p + "tconv_b");
        U.dw_w    = tensor_(p + "cnx.dw_w");
        U.dw_b    = tensor_(p + "cnx.dw_b");
        U.norm_w  = tensor_(p + "cnx.norm_w");
        U.norm_b  = tensor_(p + "cnx.norm_b");
        U.pw1_w   = tensor_(p + "cnx.pw1_w");
        U.pw1_b   = tensor_(p + "cnx.pw1_b");
        U.pw2_w   = tensor_(p + "cnx.pw2_w");
        U.pw2_b   = tensor_(p + "cnx.pw2_b");
        U.gamma   = tensor_(p + "cnx.gamma");
    }

    // Decoder body.
    in_conv_w_   = tensor_("codec.dec.in_conv_w");
    in_conv_b_   = tensor_("codec.dec.in_conv_b");
    for (int b = 0; b < 4; ++b) {
        const std::string p = "codec.dec.blk." + std::to_string(b) + ".";
        DecBlock& B = blocks_[b];
        B.snake_a = tensor_(p + "snake_a");
        B.snake_b = tensor_(p + "snake_b");
        B.tconv_w = tensor_(p + "tconv_w");
        B.tconv_b = tensor_(p + "tconv_b");
        for (int u = 0; u < 3; ++u) {
            const std::string rp = p + "res." + std::to_string(u) + ".";
            ResUnit& R = B.res[u];
            R.act1_a  = tensor_(rp + "act1_a");
            R.act1_b  = tensor_(rp + "act1_b");
            R.act2_a  = tensor_(rp + "act2_a");
            R.act2_b  = tensor_(rp + "act2_b");
            R.conv1_w = tensor_(rp + "conv1_w");
            R.conv1_b = tensor_(rp + "conv1_b");
            R.conv2_w = tensor_(rp + "conv2_w");
            R.conv2_b = tensor_(rp + "conv2_b");
        }
    }
    out_snake_a_ = tensor_("codec.dec.out_snake_a");
    out_snake_b_ = tensor_("codec.dec.out_snake_b");
    out_conv_w_  = tensor_("codec.dec.out_conv_w");
    out_conv_b_  = tensor_("codec.dec.out_conv_b");
}

void Qwen3TtsCodecGraphs::parse_hp_from_kv_() {
    // Hook for future use: when the caller hasn't pre-populated `hp`, the
    // loader can hand the GgufReader's KV pairs to this method. Today the
    // loader itself reads qwen3tts_codec.dec.* keys via GgufReader::get_kv_*
    // (keeping the codec class independent of GgufReader), so this is a
    // no-op. The defaults baked into the HP struct already match the
    // CrispASR reference build (the only known-good configuration today).
}

bool Qwen3TtsCodecGraphs::bind(ggml_context* source_ctx,
                                ggml_backend_t backend,
                                std::string* error) {
    if (present_) return true;
    source_ctx_ = source_ctx;
    backend_    = backend;
    if (!source_ctx_ || !backend_) {
        if (error) *error = "Qwen3TtsCodecGraphs::bind: nullptr source_ctx or backend";
        return false;
    }
    try {
        resolve_tensors_();
        resolve_cenc_tensors_();
        parse_hp_from_kv_();
    } catch (const std::exception& e) {
        if (error) *error = std::string("Qwen3TtsCodecGraphs::bind: ") + e.what();
        return false;
    }
    galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (!galloc_) {
        if (error) *error = "Qwen3TtsCodecGraphs::bind: gallocr_new failed";
        return false;
    }
    present_ = true;
    return true;
}

// ───────────────────────────────────────────────────────────────────────────
// Causal / transposed conv1d helpers (CrispASR conv.h + qwen3_tts.cpp lines
// 3531-3651). All operate on channels-first tensors [C, T].
// ───────────────────────────────────────────────────────────────────────────

// Causal Conv1d (k, stride, dilation). Left-pads by (K-1)*dilation (minus
// stride-1 if strided) so output time = ceil(T/stride). Used by RVQ out_proj
// (k=1), pre_conv (k=3), res_unit convs (k=7 dilated, k=1), in_conv (k=3),
// out_conv (k=7). CrispASR codec_causal_conv1d verbatim.
ggml_tensor* Qwen3TtsCodecGraphs::causal_conv1d_(ggml_context* ctx,
                                                   ggml_tensor* x,
                                                   ggml_tensor* w,
                                                   ggml_tensor* b,
                                                   int stride,
                                                   int dilation) const {
    const int K = (int)w->ne[0];
    int pad_left = (K - 1) * dilation;
    if (stride > 1) pad_left -= (stride - 1);
    if (pad_left < 0) pad_left = 0;

    x = ggml_cont(ctx, ggml_transpose(ctx, x));                    // [T, C]
    if (pad_left > 0) {
        x = ggml_pad_ext(ctx, x, pad_left, 0, 0, 0, 0, 0, 0, 0);
        // ggml_pad_ext always emits a 4D tensor; reshape back to 2D so
        // ggml_conv_1d's internal im2col step sees a standard 2D input.
        x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);
    }
    x = ggml_conv_1d(ctx, w, x, stride, /*p0=*/0, dilation);
    x = ggml_cont(ctx, ggml_transpose(ctx, x));                    // [C_out, T_out]
    if (b) x = ggml_add(ctx, x, b);
    return x;
}

// Causal depthwise Conv1d for ConvNeXt blocks. w shape: [K, 1, C]. CrispASR
// codec_dw_causal_conv1d verbatim.
ggml_tensor* Qwen3TtsCodecGraphs::dw_causal_conv1d_(ggml_context* ctx,
                                                      ggml_tensor* x,
                                                      ggml_tensor* w,
                                                      ggml_tensor* b) const {
    const int K = (int)w->ne[0];
    const int pad_left = K - 1;

    x = ggml_cont(ctx, ggml_transpose(ctx, x));                    // [T, C]
    if (pad_left > 0) {
        x = ggml_pad_ext(ctx, x, pad_left, 0, 0, 0, 0, 0, 0, 0);
        x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);
    }
    x = ggml_conv_1d_dw(ctx, w, x, /*s0=*/1, /*p0=*/0, /*d0=*/1);
    if (ggml_n_dims(x) > 2) {
        x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1] * x->ne[2]);
    }
    x = ggml_cont(ctx, ggml_transpose(ctx, x));                    // [C, T]
    if (b) x = ggml_add(ctx, x, b);
    return x;
}

// Causal transposed Conv1d (crop path). CrispASR codec_transposed_conv1d
// fallback → core_convt::convt1d_crop. Crops K-stride samples from the right
// tail so output time = T_in * stride.
ggml_tensor* Qwen3TtsCodecGraphs::transposed_conv1d_(ggml_context* ctx,
                                                        ggml_tensor* x,
                                                        ggml_tensor* w,
                                                        ggml_tensor* b,
                                                        int stride) const {
    const int K = (int)w->ne[0];
    const int crop_right = (K > stride) ? (K - stride) : 0;
    const int Cout = (int)w->ne[1];

    ggml_tensor* xT = ggml_cont(ctx, ggml_transpose(ctx, x));      // [T_in, Cin]
    ggml_tensor* y  = ggml_conv_transpose_1d(ctx, w, xT, stride, /*p0=*/0, /*d0=*/1);
    const int64_t T_unpad = y->ne[0];
    const int64_t T_out   = T_unpad - crop_right;
    y = ggml_reshape_2d(ctx, y, T_unpad, Cout);
    if (crop_right > 0) {
        // View [T_out, Cout] from the uncropped buffer, offset 0 (left aligned).
        y = ggml_view_2d(ctx, y, T_out, Cout,
                         (size_t)T_unpad * sizeof(float),
                         /*offset=*/0);
        y = ggml_cont(ctx, y);                                      // [T_out, Cout]
    }
    y = ggml_cont(ctx, ggml_transpose(ctx, y));                    // [Cout, T_out]
    if (b) y = ggml_add(ctx, y, b);
    return y;
}

// ───────────────────────────────────────────────────────────────────────────
// SnakeBeta: y = x + exp(-β) · sin²(x · exp(α)).  CrispASR core_act::snake_beta
// verbatim. Per-channel α and β broadcast over time.
// ───────────────────────────────────────────────────────────────────────────

ggml_tensor* Qwen3TtsCodecGraphs::snake_beta_(ggml_context* ctx,
                                                 ggml_tensor* x,
                                                 ggml_tensor* alpha,
                                                 ggml_tensor* beta) const {
    ggml_tensor* ea    = ggml_exp(ctx, alpha);                     // (C,)
    ggml_tensor* inv_eb = ggml_exp(ctx, ggml_neg(ctx, beta));      // (C,) = 1/exp(β)
    ggml_tensor* xa    = ggml_mul(ctx, x, ea);                     // (C, T)
    ggml_tensor* s     = ggml_sin(ctx, xa);
    ggml_tensor* s2    = ggml_mul(ctx, s, s);                      // (C, T)
    ggml_tensor* scaled = ggml_mul(ctx, s2, inv_eb);
    return ggml_add(ctx, x, scaled);
}

// ───────────────────────────────────────────────────────────────────────────
// ConvNeXt block (upsample stage inner block). CrispASR codec_convnext_block
// verbatim. LayerNorm over channels; LayerScale gamma; residual.
// ───────────────────────────────────────────────────────────────────────────

ggml_tensor* Qwen3TtsCodecGraphs::convnext_block_(ggml_context* ctx,
                                                     ggml_tensor* x,
                                                     const UpStage& up) const {
    constexpr float ln_eps = 1e-5f;
    ggml_tensor* residual = x;

    x = dw_causal_conv1d_(ctx, x, up.dw_w, up.dw_b);               // [C, T]
    x = ggml_norm(ctx, x, ln_eps);                                 // normalize over ne[0] = C
    x = ggml_mul(ctx, x, up.norm_w);
    x = ggml_add(ctx, x, up.norm_b);
    x = ggml_add(ctx, ggml_mul_mat(ctx, up.pw1_w, x), up.pw1_b);   // C → 4C
    x = ggml_gelu(ctx, x);
    x = ggml_add(ctx, ggml_mul_mat(ctx, up.pw2_w, x), up.pw2_b);   // 4C → C
    x = ggml_mul(ctx, x, up.gamma);                                // LayerScale
    return ggml_add(ctx, residual, x);
}

// Residual unit: snake1 → dilated_conv(k=7) → snake2 → conv(k=1) → add.
// CrispASR codec_res_unit verbatim.
ggml_tensor* Qwen3TtsCodecGraphs::res_unit_(ggml_context* ctx,
                                              ggml_tensor* x,
                                              const ResUnit& ru,
                                              int dilation) const {
    ggml_tensor* residual = x;
    x = snake_beta_(ctx, x, ru.act1_a, ru.act1_b);
    x = causal_conv1d_(ctx, x, ru.conv1_w, ru.conv1_b, /*stride=*/1, dilation);
    x = snake_beta_(ctx, x, ru.act2_a, ru.act2_b);
    x = causal_conv1d_(ctx, x, ru.conv2_w, ru.conv2_b, /*stride=*/1, /*dilation=*/1);
    return ggml_add(ctx, residual, x);
}

// Decoder block: snake → tconv(stride) → 3× res_unit (dilations 1/3/9).
// CrispASR codec_dec_block verbatim.
ggml_tensor* Qwen3TtsCodecGraphs::dec_block_(ggml_context* ctx,
                                                ggml_tensor* x,
                                                const DecBlock& blk,
                                                int stride) const {
    x = snake_beta_(ctx, x, blk.snake_a, blk.snake_b);
    x = transposed_conv1d_(ctx, x, blk.tconv_w, blk.tconv_b, stride);
    static const int dilations[3] = {1, 3, 9};
    for (int u = 0; u < 3; ++u) {
        x = res_unit_(ctx, x, blk.res[u], dilations[u]);
    }
    return x;
}

// ───────────────────────────────────────────────────────────────────────────
// Self-attention (CrispASR core_attn::llama_self_attn, codec-shaped).
// Non-fused Q/K/V matmuls (codec stores them separately), NEOX RoPE
// (beta_fast=32, beta_slow=1), ggml_flash_attn_ext, no output bias.
// ───────────────────────────────────────────────────────────────────────────

ggml_tensor* Qwen3TtsCodecGraphs::self_attn_(ggml_context* ctx,
                                                ggml_tensor* x,
                                                const XfmrLayer& L,
                                                ggml_tensor* pos,
                                                ggml_tensor* mask) const {
    const int    hd      = (int)hp.head_dim;
    const int    n_q     = (int)hp.n_heads;
    const int    n_ctx_o = (int)hp.max_pos;
    const float  theta   = hp.rope_theta;
    const float  scale   = 1.0f / std::sqrt((float)hd);
    const int64_t T      = x->ne[1];

    ggml_tensor* Q = ggml_mul_mat(ctx, L.attn_q_w, x);
    ggml_tensor* K = ggml_mul_mat(ctx, L.attn_k_w, x);
    ggml_tensor* V = ggml_mul_mat(ctx, L.attn_v_w, x);

    Q = ggml_reshape_3d(ctx, Q, hd, n_q, T);
    K = ggml_reshape_3d(ctx, K, hd, n_q, T);
    V = ggml_reshape_3d(ctx, V, hd, n_q, T);

    Q = ggml_rope_ext(ctx, Q, pos, /*factors=*/nullptr, hd,
                      GGML_ROPE_TYPE_NEOX, n_ctx_o, theta,
                      /*freq_scale=*/1.0f, /*ext_factor=*/0.0f,
                      /*attn_factor=*/1.0f, /*beta_fast=*/32.0f, /*beta_slow=*/1.0f);
    K = ggml_rope_ext(ctx, K, pos, nullptr, hd,
                      GGML_ROPE_TYPE_NEOX, n_ctx_o, theta,
                      1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

    // Flash-attention layout: (head_dim, T, n_heads).
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
    V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

    ggml_tensor* attn = ggml_flash_attn_ext(ctx, Q, K, V, mask, scale,
                                              /*max_bias=*/0.0f, /*logit_softcap=*/0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    // Output (head_dim, n_heads, T) → flatten heads into d_model.
    attn = ggml_reshape_2d(ctx, attn, hd * n_q, T);
    return ggml_mul_mat(ctx, L.attn_o_w, attn);
}

// SwiGLU FFN: out = W_down @ (silu(W_gate @ x) * (W_up @ x)).  No biases.
// CrispASR core_ffn::swiglu verbatim.
ggml_tensor* Qwen3TtsCodecGraphs::swiglu_(ggml_context* ctx,
                                            ggml_tensor* x,
                                            const XfmrLayer& L) const {
    ggml_tensor* gate = ggml_mul_mat(ctx, L.ffn_gate_w, x);
    ggml_tensor* up   = ggml_mul_mat(ctx, L.ffn_up_w, x);
    ggml_tensor* mlp  = ggml_mul(ctx, ggml_silu(ctx, gate), up);
    return ggml_mul_mat(ctx, L.ffn_down_w, mlp);
}

// ───────────────────────────────────────────────────────────────────────────
// Causal mask (F16 [T, T], upper triangle = -inf).
// ───────────────────────────────────────────────────────────────────────────

ggml_tensor* Qwen3TtsCodecGraphs::make_causal_mask_(ggml_context* ctx, int64_t T) {
    ggml_tensor* m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, T, T);
    ggml_set_input(m);
    return m;
}

void Qwen3TtsCodecGraphs::fill_causal_mask_(ggml_tensor* mask) {
    const int64_t T = mask->ne[0];
    std::vector<uint16_t> buf(size_t(T) * size_t(T));
    for (int64_t q = 0; q < T; ++q) {
        uint16_t* row = buf.data() + size_t(q) * size_t(T);
        for (int64_t kv = 0; kv < T; ++kv) {
            row[kv] = (kv <= q) ? F16_ZERO : F16_NEG_INF;
        }
    }
    ggml_backend_tensor_set(mask, buf.data(), 0, buf.size() * sizeof(uint16_t));
}

// ───────────────────────────────────────────────────────────────────────────
// Decode entry point (CrispASR build_graph_codec_decode + codec_decode_window
// fused into a single-pass entry — lines 3660-3807 + 4192-4274).
// ───────────────────────────────────────────────────────────────────────────

std::vector<float> Qwen3TtsCodecGraphs::decode(const int32_t* codes,
                                                int32_t n_q,
                                                int32_t T_codec) {
    if (!present_) {
        throw std::runtime_error("Qwen3TtsCodecGraphs::decode: not bound (codec tensors missing?)");
    }
    if (n_q != (int32_t)hp.n_q) {
        throw std::runtime_error("Qwen3TtsCodecGraphs::decode: expected n_q="
                                  + std::to_string(hp.n_q) + ", got "
                                  + std::to_string(n_q));
    }
    if (T_codec <= 0) return {};

    // Refuse runaway generations early. The transformer's T×T F16 causal
    // mask is the largest single allocation; cap it well below OOM territory.
    {
        const int64_t mask_bytes = int64_t(T_codec) * int64_t(T_codec) * int64_t(sizeof(uint16_t));
        const int64_t cap_bytes  = int64_t(8) * 1024 * 1024 * 1024;  // 8 GiB
        if (mask_bytes > cap_bytes) {
            const double secs = T_codec * 1920.0 / 24000.0;
            throw std::runtime_error(
                "Qwen3TtsCodecGraphs::decode: refusing to decode " + std::to_string(T_codec) +
                " frames (~" + std::to_string(int(secs)) + "s): the codec attention "
                "mask alone would need ~" + std::to_string(mask_bytes >> 30) +
                " GiB. This usually means generation never emitted an end-of-speech "
                "token (try a lower --max-new-tokens / different sampling).");
        }
    }

    const int n_samples = T_codec * 1920;

    // ── Build graph ─────────────────────────────────────────────────────
    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * 65536 + ggml_graph_overhead_custom(65536, false);
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context* ctx = ggml_init(ip);
    if (!ctx) throw std::runtime_error("Qwen3TtsCodecGraphs::decode: ggml_init failed");

    // codes_inp: 2D I32 [T, n_q] (ne[0]=T innermost, ne[1]=n_q). Flat layout
    // is [n_q, T] row-major — one codebook per row — matching what the
    // Qwen3TtsSession flattens its (T_codec, n_q) code matrix into before
    // calling. CrispASR build_graph_codec_decode line 3678.
    ggml_tensor* codes_inp = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, T_codec, n_q);
    ggml_set_name(codes_inp, "codec_codes");
    ggml_set_input(codes_inp);

    // Positions [0..T-1].
    ggml_tensor* positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_codec);
    ggml_set_name(positions, "codec_positions");
    ggml_set_input(positions);

    // Causal mask (F16 [T, T]). For T==1 the attention is trivial and
    // flash_attn_ext accepts a nullptr mask; we still allocate one for
    // uniformity — it's one entry.
    ggml_tensor* mask = make_causal_mask_(ctx, T_codec);
    ggml_set_name(mask, "codec_mask");

    // ── Step 1: RVQ decode ──────────────────────────────────────────────
    // rvq_first: codebook 0 lookup → out_proj (causal_conv1d k=1) → [512, T].
    ggml_tensor* cb0_ids = ggml_view_1d(ctx, codes_inp, T_codec, 0);
    ggml_tensor* emb_first = ggml_get_rows(ctx, rvq_first_cb_, cb0_ids);
    emb_first = causal_conv1d_(ctx, emb_first, rvq_first_out_w_, nullptr, 1, 1);

    // rvq_rest: sum codebooks 1..15 lookups → out_proj → [512, T].
    ggml_tensor* emb_rest = ggml_get_rows(
        ctx, rvq_rest_cb_[0],
        ggml_view_1d(ctx, codes_inp, T_codec, (size_t)1 * T_codec * sizeof(int32_t)));
    for (int q = 1; q < 15; ++q) {
        ggml_tensor* ids_q = ggml_view_1d(ctx, codes_inp, T_codec,
                                          (size_t)(q + 1) * T_codec * sizeof(int32_t));
        emb_rest = ggml_add(ctx, emb_rest, ggml_get_rows(ctx, rvq_rest_cb_[q], ids_q));
    }
    emb_rest = causal_conv1d_(ctx, emb_rest, rvq_rest_out_w_, nullptr, 1, 1);

    ggml_tensor* h = ggml_add(ctx, emb_first, emb_rest);           // [512, T]

    // ── Step 2: pre_conv (causal_conv1d k=3) ────────────────────────────
    h = causal_conv1d_(ctx, h, pre_conv_w_, pre_conv_b_, 1, 1);    // [1024, T]

    // ── Step 3: transformer ─────────────────────────────────────────────
    h = ggml_add(ctx, ggml_mul_mat(ctx, xfmr_in_proj_w_, h), xfmr_in_proj_b_);
    for (size_t il = 0; il < xfmr_layers_.size(); ++il) {
        const XfmrLayer& L = xfmr_layers_[il];
        ggml_tensor* residual = h;

        ggml_tensor* a = ggml_rms_norm(ctx, h, hp.rms_norm_eps);
        a = ggml_mul(ctx, a, L.attn_norm_w);
        a = self_attn_(ctx, a, L, positions, mask);
        a = ggml_mul(ctx, a, L.attn_ls_w);                         // LayerScale
        h = ggml_add(ctx, residual, a);

        residual = h;
        ggml_tensor* f = ggml_rms_norm(ctx, h, hp.rms_norm_eps);
        f = ggml_mul(ctx, f, L.ffn_norm_w);
        f = swiglu_(ctx, f, L);
        f = ggml_mul(ctx, f, L.ffn_ls_w);                          // LayerScale
        h = ggml_add(ctx, residual, f);
    }
    h = ggml_rms_norm(ctx, h, hp.rms_norm_eps);
    h = ggml_mul(ctx, h, xfmr_norm_w_);
    h = ggml_add(ctx, ggml_mul_mat(ctx, xfmr_out_proj_w_, h), xfmr_out_proj_b_);

    // ── Step 4: 2 ConvNeXt upsample stages (each stride 2) ──────────────
    for (int s = 0; s < 2; ++s) {
        h = transposed_conv1d_(ctx, h, up_[s].tconv_w, up_[s].tconv_b, 2);
        h = convnext_block_(ctx, h, up_[s]);
    }

    // ── Step 5: in_conv (causal_conv1d k=3) → [1536, 4T] ───────────────
    h = causal_conv1d_(ctx, h, in_conv_w_, in_conv_b_, 1, 1);

    // ── Step 6: 4 decoder blocks (strides 8/5/4/3) ─────────────────────
    for (int b = 0; b < 4; ++b) {
        h = dec_block_(ctx, h, blocks_[b], hp.upsample_rates[b]);
    }

    // ── Step 7: final snake + out_conv (causal_conv1d k=7) + clamp ─────
    h = snake_beta_(ctx, h, out_snake_a_, out_snake_b_);
    h = causal_conv1d_(ctx, h, out_conv_w_, out_conv_b_, 1, 1);
    h = ggml_clamp(ctx, h, -1.0f, 1.0f);

    // Reshape [1, 1920·T] → 1D [1920·T].
    ggml_tensor* waveform = ggml_reshape_1d(ctx, h, n_samples);
    ggml_set_name(waveform, "waveform");
    ggml_set_output(waveform);

    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 65536, false);
    ggml_build_forward_expand(graph, waveform);

    if (!ggml_gallocr_alloc_graph(galloc_, graph)) {
        ggml_free(ctx);
        throw std::runtime_error("Qwen3TtsCodecGraphs::decode: gallocr_alloc_graph failed");
    }

    // ── Upload inputs ───────────────────────────────────────────────────
    // codes is already (n_q, T_codec) row-major, which matches codes_inp's
    // flat layout (ne[0]=T innermost, ne[1]=n_q outer).
    ggml_backend_tensor_set(codes_inp, codes, 0,
                             size_t(n_q) * size_t(T_codec) * sizeof(int32_t));

    std::vector<int32_t> pos_vals(T_codec);
    for (int32_t t = 0; t < T_codec; ++t) pos_vals[size_t(t)] = t;
    ggml_backend_tensor_set(positions, pos_vals.data(), 0,
                             pos_vals.size() * sizeof(int32_t));

    fill_causal_mask_(mask);

    if (ggml_backend_graph_compute(backend_, graph) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        throw std::runtime_error("Qwen3TtsCodecGraphs::decode: graph_compute failed");
    }

    std::vector<float> wav;
    wav.resize(size_t(n_samples));
    ggml_backend_tensor_get(waveform, wav.data(), 0, wav.size() * sizeof(float));

    ggml_free(ctx);
    return wav;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Encoder: tensor resolution
// ═════════════════════════════════════════════════════════════════════════════

void Qwen3TtsCodecGraphs::resolve_cenc_tensors_() {
    EncSEANet& E = enc_seanet_;
    E.init.w = tensor_or_null_("codec.enc.seanet.init_w");
    E.init.b = tensor_or_null_("codec.enc.seanet.init_b");
    E.final.w = tensor_or_null_("codec.enc.seanet.final_w");
    E.final.b = tensor_or_null_("codec.enc.seanet.final_b");
    for (int i = 0; i < 4; ++i) {
        const std::string p = "codec.enc.seanet.blk." + std::to_string(i) + ".";
        E.resblk[i].shortcut.w = tensor_or_null_(p + "short_w");
        E.resblk[i].shortcut.b = tensor_or_null_(p + "short_b");
        E.resblk[i].expand.w   = tensor_or_null_(p + "exp_w");
        E.resblk[i].expand.b   = tensor_or_null_(p + "exp_b");
        E.ds[i].w = tensor_or_null_(p + "ds." + std::to_string(i) + ".w");
        E.ds[i].b = tensor_or_null_(p + "ds." + std::to_string(i) + ".b");
    }
    if (!E.init.w) { enc_present_ = false; return; }

    enc_xfmr_layers_.resize(8);
    for (size_t i = 0; i < enc_xfmr_layers_.size(); ++i) {
        const std::string p = "codec.enc.xfmr.blk." + std::to_string(i) + ".";
        EncXfmrLayer& L = enc_xfmr_layers_[i];
        L.norm1_w  = tensor_(p + "norm1_w");  L.norm1_b = tensor_or_null_(p + "norm1_b");
        L.norm2_w  = tensor_(p + "norm2_w");  L.norm2_b = tensor_or_null_(p + "norm2_b");
        L.attn_q_w = tensor_(p + "attn_q_w"); L.attn_k_w = tensor_(p + "attn_k_w");
        L.attn_v_w = tensor_(p + "attn_v_w"); L.attn_o_w = tensor_(p + "attn_o_w");
        L.attn_ls  = tensor_or_null_(p + "attn_ls");
        L.fc1_w    = tensor_(p + "fc1_w");
        L.fc2_w    = tensor_(p + "fc2_w");
        L.ffn_ls   = tensor_or_null_(p + "ffn_ls");
    }

    enc_ds_.w = tensor_or_null_("codec.enc.ds_w");

    enc_rvq_.sem_in_w = tensor_or_null_("codec.enc.rvq.sem.in_w");
    enc_rvq_.sem_cb   = tensor_or_null_("codec.enc.rvq.sem.cb");
    enc_rvq_.ac_in_w  = tensor_or_null_("codec.enc.rvq.ac.in_w");
    for (int q = 0; q < 15; ++q)
        enc_rvq_.ac_cb[q] = tensor_or_null_("codec.enc.rvq.ac." + std::to_string(q) + ".cb");

    enc_present_ = true;
}

// ── cenc_conv1d_ext: replicate-padded conv1d ───────────────────────────────
ggml_tensor* Qwen3TtsCodecGraphs::cenc_conv1d_ext_(ggml_context* ctx,
                                                     ggml_tensor* x,
                                                     ggml_tensor* w,
                                                     ggml_tensor* b,
                                                     int stride,
                                                     bool pad_replicate) {
    if (pad_replicate && w) {
        const int K = (int)w->ne[0];
        const int T = (int)x->ne[1];
        const int pad = K - 1;
        ggml_tensor* padded = ggml_new_tensor_2d(ctx, x->type, T + 2 * pad, x->ne[0]);
        ggml_set_name(padded, "cenc_padded");
        ggml_set_input(padded);
        x = padded;
    }
    if (!w) return x;
    x = ggml_conv_1d(ctx, x, w, 0, 1, stride);
    if (b) x = ggml_add(ctx, x, b);
    return x;
}

// ── build_cenc_seanet: SEANet encoder graph ────────────────────────────────
ggml_tensor* Qwen3TtsCodecGraphs::build_cenc_seanet_(ggml_context* ctx,
                                                       ggml_tensor* pcm) const {
    const EncSEANet& E = enc_seanet_;
    ggml_tensor* h = causal_conv1d_(ctx, pcm, E.init.w, E.init.b, 1, 1);
    h = ggml_elu(ctx, h);
    for (int i = 0; i < 4; ++i) {
        ggml_tensor* r = causal_conv1d_(ctx, h, E.resblk[i].shortcut.w,
                                         E.resblk[i].shortcut.b, 1, 1);
        ggml_tensor* e = ggml_conv_1d(ctx, h, E.resblk[i].expand.w, 0, 1, 1);
        if (E.resblk[i].expand.b) e = ggml_add(ctx, e, E.resblk[i].expand.b);
        h = ggml_add(ctx, r, e);
        h = ggml_elu(ctx, h);
        const int s = (i == 0) ? 4 : (i == 1) ? 5 : (i == 2) ? 6 : 8;
        h = causal_conv1d_(ctx, h, E.ds[i].w, E.ds[i].b, s, 1);
        h = ggml_elu(ctx, h);
    }
    h = causal_conv1d_(ctx, h, E.final.w, E.final.b, 1, 1);
    return h;
}

// ── build_cenc_xfmr: encoder transformer ───────────────────────────────────
ggml_tensor* Qwen3TtsCodecGraphs::build_cenc_xfmr_(ggml_context* ctx,
                                                     ggml_tensor* x,
                                                     int32_t T_enc) const {
    std::vector<int32_t> pos_vals;
    pos_vals.resize(size_t(T_enc));
    for (int32_t i = 0; i < T_enc; ++i) pos_vals[size_t(i)] = i;
    ggml_tensor* positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_enc);
    ggml_set_name(positions, "cenc_pos");
    ggml_set_input(positions);

    ggml_tensor* mask = make_causal_mask_(ctx, T_enc);
    ggml_set_name(mask, "cenc_mask");
    ggml_set_input(mask);

    const int hd = (int)hp.head_dim;
    const int n_q = (int)hp.n_heads;
    const int n_ctx_o = (int)hp.max_pos;
    const float theta = hp.rope_theta;
    const float scale = 1.0f / std::sqrt((float)hd);

    for (size_t il = 0; il < enc_xfmr_layers_.size(); ++il) {
        const EncXfmrLayer& L = enc_xfmr_layers_[il];
        ggml_tensor* residual = x;

        ggml_tensor* a = ggml_norm(ctx, x, hp.rms_norm_eps);
        a = ggml_mul(ctx, a, L.norm1_w);
        if (L.norm1_b) a = ggml_add(ctx, a, L.norm1_b);
        {
            const int64_t T = a->ne[1];
            ggml_tensor* Qv = ggml_mul_mat(ctx, L.attn_q_w, a);
            ggml_tensor* Kv = ggml_mul_mat(ctx, L.attn_k_w, a);
            ggml_tensor* Vv = ggml_mul_mat(ctx, L.attn_v_w, a);
            Qv = ggml_reshape_3d(ctx, Qv, hd, n_q, T);
            Kv = ggml_reshape_3d(ctx, Kv, hd, n_q, T);
            Vv = ggml_reshape_3d(ctx, Vv, hd, n_q, T);
            Qv = ggml_rope_ext(ctx, Qv, positions, nullptr, hd,
                               GGML_ROPE_TYPE_NEOX, n_ctx_o, theta,
                               1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
            Kv = ggml_rope_ext(ctx, Kv, positions, nullptr, hd,
                               GGML_ROPE_TYPE_NEOX, n_ctx_o, theta,
                               1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
            Qv = ggml_cont(ctx, ggml_permute(ctx, Qv, 0, 2, 1, 3));
            Kv = ggml_cont(ctx, ggml_permute(ctx, Kv, 0, 2, 1, 3));
            Vv = ggml_cont(ctx, ggml_permute(ctx, Vv, 0, 2, 1, 3));
            ggml_tensor* attn = ggml_flash_attn_ext(ctx, Qv, Kv, Vv, mask, scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
            attn = ggml_reshape_2d(ctx, attn, hd * n_q, T);
            a = ggml_mul_mat(ctx, L.attn_o_w, attn);
        }
        if (L.attn_ls) a = ggml_mul(ctx, a, L.attn_ls);
        x = ggml_add(ctx, residual, a);

        residual = x;
        ggml_tensor* f = ggml_norm(ctx, x, hp.rms_norm_eps);
        f = ggml_mul(ctx, f, L.norm2_w);
        if (L.norm2_b) f = ggml_add(ctx, f, L.norm2_b);
        f = ggml_mul_mat(ctx, L.fc1_w, f);
        f = ggml_gelu(ctx, f);
        f = ggml_mul_mat(ctx, L.fc2_w, f);
        if (L.ffn_ls) f = ggml_mul(ctx, f, L.ffn_ls);
        x = ggml_add(ctx, residual, f);
    }
    return x;
}

// ── build_cenc_downsample: stride-2 conv with replicate padding ────────────
ggml_tensor* Qwen3TtsCodecGraphs::build_cenc_downsample_(ggml_context* ctx,
                                                           ggml_tensor* x) const {
    if (!enc_ds_.w) return x;
    return cenc_conv1d_ext_(ctx, x, enc_ds_.w, nullptr, 2, true);
}

// ── cenc_rvq_encode: CPU RVQ quantize ──────────────────────────────────────
bool Qwen3TtsCodecGraphs::cenc_rvq_encode_(const float* emb, int32_t T_frames,
                                             std::vector<int32_t>* codes_out) const {
    if (!enc_rvq_.sem_in_w || !enc_rvq_.sem_cb || !enc_rvq_.ac_in_w) return false;
    for (int q = 0; q < 15; ++q) if (!enc_rvq_.ac_cb[q]) return false;

    codes_out->resize(size_t(16) * size_t(T_frames), 0);
    constexpr int d_half = 256;
    constexpr int C_emb = 512;

    auto rvq_nn = [](const float* z, const float* cb, int cb_size, int dim) -> int {
        int best = 0;
        float best_d = 0.0f;
        for (int i = 0; i < cb_size; ++i) {
            float d = 0.0f;
            for (int j = 0; j < dim; ++j) {
                float diff = z[j] - cb[size_t(i) * size_t(dim) + j];
                d += diff * diff;
            }
            if (i == 0 || d < best_d) { best_d = d; best = i; }
        }
        return best;
    };

    auto matmul = [](const float* x, const float* w, int C_in, int C_out,
                     int T, float* out) {
        for (int t = 0; t < T; ++t) {
            for (int o = 0; o < C_out; ++o) {
                float s = 0.0f;
                for (int i = 0; i < C_in; ++i)
                    s += x[size_t(t) * size_t(C_in) + i] *
                         w[size_t(o) + size_t(i) * size_t(C_out)];
                out[size_t(t) * size_t(C_out) + o] = s;
            }
        }
    };

    // Transpose emb from [C, T] to [T, C] for CPU matmul
    std::vector<float> emb_T(size_t(T_frames) * C_emb);
    for (int t = 0; t < T_frames; ++t)
        for (int c = 0; c < C_emb; ++c)
            emb_T[size_t(t) * C_emb + c] = emb[size_t(c) * T_frames + t];

    std::vector<float> z(size_t(T_frames) * d_half);

    // Semantic book (codebook 0): project 512->256, NN search
    const float* siw = (const float*)enc_rvq_.sem_in_w->data;
    const int sid = (int)enc_rvq_.sem_in_w->ne[1];
    const int sod = (int)enc_rvq_.sem_in_w->ne[0];
    matmul(emb_T.data(), siw, sid, sod, T_frames, z.data());
    const float* scb = (const float*)enc_rvq_.sem_cb->data;
    const int scs = (int)enc_rvq_.sem_cb->ne[0];
    for (int t = 0; t < T_frames; ++t)
        (*codes_out)[t] = rvq_nn(&z[size_t(t) * d_half], scb, scs, d_half);

    // Acoustic books (1..15): project 512->256, residual NN search
    const float* aiw = (const float*)enc_rvq_.ac_in_w->data;
    const int aid = (int)enc_rvq_.ac_in_w->ne[1];
    const int aod = (int)enc_rvq_.ac_in_w->ne[0];
    matmul(emb_T.data(), aiw, aid, aod, T_frames, z.data());

    for (int q = 1; q < 16; ++q) {
        const int q_idx = q - 1;
        const float* cb = (const float*)enc_rvq_.ac_cb[q_idx]->data;
        const int cbs = (int)enc_rvq_.ac_cb[q_idx]->ne[0];
        for (int t = 0; t < T_frames; ++t) {
            int code = rvq_nn(&z[size_t(t) * d_half], cb, cbs, d_half);
            (*codes_out)[size_t(q) * T_frames + t] = code;
            // Subtract quantized embedding from residual
            for (int j = 0; j < d_half; ++j)
                z[size_t(t) * d_half + j] -= cb[size_t(code) * d_half + j];
        }
    }
    return true;
}

// ── encode: PCM 24kHz → [16, T_frames] codes row-major ─────────────────────
std::vector<int32_t> Qwen3TtsCodecGraphs::encode(const float* pcm,
                                                   int32_t n_samples) {
    if (!enc_present_ || !pcm || n_samples <= 0) return {};

    // Round n_samples down to nearest multiple of 960 (SEANet stride product)
    const int T_pad = (n_samples + 959) / 960 * 960;
    const int T_enc = T_pad / 960;

    // Build encoder compute graph
    ggml_context* ctx = nullptr;
    struct ggml_init_params params = {
        /*.mem_size   =*/ 64ull * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx = ggml_init(params);
    if (!ctx) throw std::runtime_error("Qwen3TtsCodecGraphs::encode: ggml_init failed");

    struct ggml_tensor* pcm_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_pad, 1);
    ggml_set_name(pcm_t, "cenc_pcm");
    ggml_set_input(pcm_t);

    ggml_tensor* h = build_cenc_seanet_(ctx, pcm_t);
    h = build_cenc_xfmr_(ctx, h, T_enc);
    h = build_cenc_downsample_(ctx, h);
    // h is [latent_dim=512, T_frames] after downsample
    ggml_set_name(h, "cenc_out");
    ggml_set_output(h);

    // Build compute graph
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 65536, false);
    ggml_build_forward_expand(graph, h);

    // Allocate + compute
    if (!ggml_gallocr_alloc_graph(galloc_, graph)) {
        ggml_free(ctx);
        throw std::runtime_error("Qwen3TtsCodecGraphs::encode: gallocr_alloc_graph failed");
    }
    ggml_backend_tensor_set(pcm_t, pcm, 0, size_t(n_samples) * sizeof(float));
    if (T_pad > n_samples) {
        std::vector<float> zero(size_t(T_pad - n_samples), 0.0f);
        ggml_backend_tensor_set(pcm_t, zero.data(),
                                size_t(n_samples) * sizeof(float),
                                size_t(T_pad - n_samples) * sizeof(float));
    }
    if (ggml_backend_graph_compute(backend_, graph) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        throw std::runtime_error("Qwen3TtsCodecGraphs::encode: graph_compute failed");
    }

    // Read output
    const int64_t T_frames = h->ne[1];
    const int64_t C_latent = h->ne[0];
    std::vector<float> h_cpu(size_t(C_latent) * size_t(T_frames));
    ggml_backend_tensor_get(h, h_cpu.data(), 0, h_cpu.size() * sizeof(float));

    ggml_free(ctx);

    // RVQ quantize to codes
    std::vector<int32_t> codes;
    if (!cenc_rvq_encode_(h_cpu.data(), (int32_t)T_frames, &codes))
        return {};
    return codes;
}

}  // namespace audiocore::qwen3_tts
