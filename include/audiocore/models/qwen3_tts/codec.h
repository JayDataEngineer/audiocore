// codec.h — Qwen3-TTS-Tokenizer-12Hz decoder, ggml port.
//
// Adapted from CrispStrobe/CrispASR (MIT) src/qwen3_tts.cpp into audiocore's
// Backend / TensorStorage / WeightLoader abstractions. Architecture overview
// and port-plan in docs/CODEC_PORTS.md §2.
//
// Decoder path: 16-codebook code matrix (T_codec, 16) → 24 kHz mono PCM of
// length T_codec * 1920 (12.5 fps codec frame rate). The WavTokenizer-class
// decoder runs entirely in ggml on the same backend as the talker.
//
// Encode is intentionally NOT ported (decode-only, matching Stage 16's
// MOSS port): audiocore consumes codes the talker + MTP predictor already
// produced. CrispASR's encoder lives in qwen3_tts.cpp for reference if a
// future Voice-Clone-from-audio path ever needs to produce codes locally.
//
// Single-pass decode only. CrispASR's chunked decode (an optimization for
// long sequences that keeps VRAM constant by decoding [ctx + chunk] windows
// and discarding the left-context PCM) is intentionally omitted — it's a
// perf overlay, not a correctness concern, and can be added later as a
// follow-up stage if needed.

#ifndef AUDIOCORE_MODELS_QWEN3_TTS_CODEC_H
#define AUDIOCORE_MODELS_QWEN3_TTS_CODEC_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend;
using ggml_backend_t = ggml_backend*;
struct ggml_gallocr;
using ggml_gallocr_t = ggml_gallocr*;

namespace audiocore::qwen3_tts {

// Qwen3TtsCodecGraphs owns no weights itself — the codec.* tensors live in
// whatever ggml_context the caller points the class at (typically a fresh
// context audiocore's loader creates when mmap-ing the codec GGUF). This
// class resolves the codec-specific tensors by name, then builds + runs the
// decode graph on demand.
//
// Tensor names match the cstr/qwen3-tts-tokenizer-12hz-GGUF layout (and
// CrispASR's `req()` lookups in load_codec). Prefix is `codec.dec.*`.
class Qwen3TtsCodecGraphs {
public:
    Qwen3TtsCodecGraphs() = default;
    ~Qwen3TtsCodecGraphs();

    Qwen3TtsCodecGraphs(const Qwen3TtsCodecGraphs&) = delete;
    Qwen3TtsCodecGraphs& operator=(const Qwen3TtsCodecGraphs&) = delete;

    // Resolve every decoder tensor by name from `source_ctx`, parse codec
    // hyperparameters from the GGUF KV via the `load_*` accessors, and
    // create the ggml_gallocr used by decode(). Idempotent.
    //
    // The caller is responsible for parsing hyperparameter KVs (see
    // src/models/qwen3_tts/codec.cpp `bind()` for the canonical list).
    // Pass them in via the HP struct so the same code works for any
    // upstream GGUF that deviates from the CrispASR defaults.
    bool bind(ggml_context* source_ctx,
              ggml_backend_t backend,
              std::string* error);

    bool is_present() const { return present_; }

    // Decode (n_q, T_codec) codes (n_q must be 16) → mono PCM float32
    // at the codec's native 24 kHz. Output length = T_codec * 1920.
    // Empty vector when T_codec <= 0. Throws std::runtime_error on failure.
    //
    // `codes` is (n_q, T_codec) row-major — one codebook per row, T_codec
    // columns. The Qwen3TtsSession flattens its (T_codec, n_q) code matrix
    // into this layout before calling.
    std::vector<float> decode(const int32_t* codes,
                              int32_t n_q,
                              int32_t T_codec);

    // Hyperparameters. Public so the loader can populate them from KV
    // metadata before calling bind(). Defaults match the CrispASR
    // reference (which itself matches the cstr/qwen3-tts-tokenizer-12hz-GGUF
    // build). Fields with non-zero defaults are the architecture's only
    // known-good configuration today.
    struct HP {
        uint32_t n_layers       = 8;
        uint32_t d_model        = 512;
        uint32_t n_heads        = 16;
        uint32_t head_dim       = 64;
        uint32_t ff_dim         = 1024;
        uint32_t n_q            = 16;
        uint32_t codebook_size  = 2048;
        uint32_t latent_dim     = 1024;
        uint32_t decoder_dim    = 1536;
        uint32_t sliding_window = 72;     // reserved — single-pass decode uses full causal mask
        uint32_t max_pos        = 8000;
        float    rope_theta     = 10000.0f;
        float    rms_norm_eps   = 1e-5f;
        int      upsample_rates[4] = {8, 5, 4, 3};  // 4 decoder blocks
        int      upsampling_ratios[2] = {2, 2};     // 2 ConvNeXt upsample stages
    };
    HP hp;

private:
    struct XfmrLayer {
        ggml_tensor* attn_norm_w = nullptr;
        ggml_tensor* ffn_norm_w  = nullptr;
        ggml_tensor* attn_q_w    = nullptr;
        ggml_tensor* attn_k_w    = nullptr;
        ggml_tensor* attn_v_w    = nullptr;
        ggml_tensor* attn_o_w    = nullptr;
        ggml_tensor* attn_ls_w   = nullptr;
        ggml_tensor* ffn_gate_w  = nullptr;
        ggml_tensor* ffn_up_w    = nullptr;
        ggml_tensor* ffn_down_w  = nullptr;
        ggml_tensor* ffn_ls_w    = nullptr;
    };
    struct UpStage {
        ggml_tensor* tconv_w = nullptr;
        ggml_tensor* tconv_b = nullptr;
        ggml_tensor* dw_w    = nullptr;
        ggml_tensor* dw_b    = nullptr;
        ggml_tensor* norm_w  = nullptr;
        ggml_tensor* norm_b  = nullptr;
        ggml_tensor* pw1_w   = nullptr;
        ggml_tensor* pw1_b   = nullptr;
        ggml_tensor* pw2_w   = nullptr;
        ggml_tensor* pw2_b   = nullptr;
        ggml_tensor* gamma   = nullptr;
    };
    struct ResUnit {
        ggml_tensor* act1_a  = nullptr;
        ggml_tensor* act1_b  = nullptr;
        ggml_tensor* act2_a  = nullptr;
        ggml_tensor* act2_b  = nullptr;
        ggml_tensor* conv1_w = nullptr;
        ggml_tensor* conv1_b = nullptr;
        ggml_tensor* conv2_w = nullptr;
        ggml_tensor* conv2_b = nullptr;
    };
    struct DecBlock {
        ggml_tensor* snake_a = nullptr;
        ggml_tensor* snake_b = nullptr;
        ggml_tensor* tconv_w = nullptr;
        ggml_tensor* tconv_b = nullptr;
        ResUnit     res[3]   = {};
    };

    // ── State ───────────────────────────────────────────────────────────
    bool            present_    = false;
    ggml_context*   source_ctx_ = nullptr;  // not owned
    ggml_backend_t  backend_    = nullptr;  // not owned
    ggml_gallocr_t  galloc_     = nullptr;  // owned

    // RVQ front-end.
    ggml_tensor*    rvq_first_cb_     = nullptr;
    ggml_tensor*    rvq_first_out_w_  = nullptr;
    ggml_tensor*    rvq_rest_cb_[15]  = {};
    ggml_tensor*    rvq_rest_out_w_   = nullptr;

    // Pre-conv + transformer.
    ggml_tensor*    pre_conv_w_       = nullptr;
    ggml_tensor*    pre_conv_b_       = nullptr;
    ggml_tensor*    xfmr_in_proj_w_   = nullptr;
    ggml_tensor*    xfmr_in_proj_b_   = nullptr;
    ggml_tensor*    xfmr_norm_w_      = nullptr;
    ggml_tensor*    xfmr_out_proj_w_  = nullptr;
    ggml_tensor*    xfmr_out_proj_b_  = nullptr;
    std::vector<XfmrLayer> xfmr_layers_;

    // 2 ConvNeXt upsample stages.
    UpStage up_[2] = {};

    // Decoder body.
    ggml_tensor*    in_conv_w_   = nullptr;
    ggml_tensor*    in_conv_b_   = nullptr;
    DecBlock        blocks_[4]   = {};
    ggml_tensor*    out_snake_a_ = nullptr;
    ggml_tensor*    out_snake_b_ = nullptr;
    ggml_tensor*    out_conv_w_  = nullptr;
    ggml_tensor*    out_conv_b_  = nullptr;

    // ── Helpers ─────────────────────────────────────────────────────────
    ggml_tensor* tensor_(const std::string& name) const;
    ggml_tensor* tensor_or_null_(const std::string& name) const;

    void resolve_tensors_();
    void parse_hp_from_kv_();  // optional — caller may pre-populate `hp`

    // Graph builders — adapted verbatim from CrispASR.
    ggml_tensor* causal_conv1d_(ggml_context* ctx, ggml_tensor* x,
                                  ggml_tensor* w, ggml_tensor* b,
                                  int stride, int dilation) const;
    ggml_tensor* dw_causal_conv1d_(ggml_context* ctx, ggml_tensor* x,
                                     ggml_tensor* w, ggml_tensor* b) const;
    ggml_tensor* transposed_conv1d_(ggml_context* ctx, ggml_tensor* x,
                                      ggml_tensor* w, ggml_tensor* b,
                                      int stride) const;
    ggml_tensor* snake_beta_(ggml_context* ctx, ggml_tensor* x,
                               ggml_tensor* alpha, ggml_tensor* beta) const;
    ggml_tensor* convnext_block_(ggml_context* ctx, ggml_tensor* x,
                                   const UpStage& up) const;
    ggml_tensor* res_unit_(ggml_context* ctx, ggml_tensor* x,
                             const ResUnit& ru, int dilation) const;
    ggml_tensor* dec_block_(ggml_context* ctx, ggml_tensor* x,
                              const DecBlock& blk, int stride) const;
    ggml_tensor* self_attn_(ggml_context* ctx, ggml_tensor* x,
                              const XfmrLayer& L, ggml_tensor* pos,
                              ggml_tensor* mask) const;
    ggml_tensor* swiglu_(ggml_context* ctx, ggml_tensor* x,
                           const XfmrLayer& L) const;
    static ggml_tensor* make_causal_mask_(ggml_context* ctx, int64_t T);
    static void         fill_causal_mask_(ggml_tensor* mask);
};

}  // namespace audiocore::qwen3_tts

#endif  // AUDIOCORE_MODELS_QWEN3_TTS_CODEC_H
