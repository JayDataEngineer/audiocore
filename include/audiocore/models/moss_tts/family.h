// family.h — MOSS-TTS family session.
//
// MOSS is Qwen3-8B + an audio codec. We do NOT reimplement Qwen3 — the
// backbone runs through the unified qwen3::Runner (libllama). This file
// covers everything that ISN'T the transformer:
//
//   • audio_embed.{i}.weight — projects codec tokens into Qwen3 hidden space
//   • audio_head.{i}.weight  — projects Qwen3 hidden states back to codec
//                              logits, one head per RVQ stream
//   • codec.enc/dec/quantizer.* — wav ↔ codec-token round-trip
//
// Verified tensor names come from pwilkin/openmoss/src/model.cpp. See
// docs/GGUF_FORMAT.md → "MOSS-TTS" section for the canonical list.

#ifndef AUDIOCORE_MODELS_MOSS_TTS_FAMILY_H
#define AUDIOCORE_MODELS_MOSS_TTS_FAMILY_H

#include <memory>
#include <string>
#include <vector>

#include "audiocore/framework/core/session.h"
#include "audiocore/framework/io/gguf_reader.h"   // complete type — methods take GgufReader&
#include "audiocore/models/qwen3/runner.h"

struct ggml_context;
struct ggml_tensor;

namespace audiocore::moss {

// Parsed from MOSS GGUF KV metadata. These keys are stable across OpenMOSS
// releases (the same names are read by openmoss/src/model.cpp).
struct MossConfig {
    int32_t n_vq              = 32;     // moss.n_vq — RVQ stream count
    int32_t audio_vocab_size  = 0;      // moss.audio_vocab_size
    int32_t audio_pad_code    = 0;      // moss.audio_pad_code
    int32_t sampling_rate     = 24000;  // moss.sampling_rate
    int32_t downsample_rate   = 0;      // moss.downsample_rate
    int32_t tok_audio_start   = 0;      // moss.token.audio_start
    int32_t tok_audio_end     = 0;      // moss.token.audio_end
    int32_t tok_user_slot     = 0;      // moss.token.user_slot
    int32_t tok_audio_gen     = 0;      // moss.token.audio_gen_slot
    int32_t tok_audio_delay   = 0;      // moss.token.audio_delay_slot
    int32_t tok_im_start      = 0;      // moss.token.im_start
    int32_t tok_im_end        = 0;      // moss.token.im_end
    int32_t tok_pad           = 0;      // moss.token.pad
    bool    codec_present     = false;  // moss.codec.present
};

// Concrete request/response shapes. The base Session uses void* because the
// framework can't know family types; MossSession down-casts inside run_tts.
struct TtsRequest {
    std::string text;
    std::string language   = "en";
    std::string voice_path;             // optional ref audio for cloning
    int32_t     seed       = 0;
    float       temperature= 0.8f;
    float       top_p      = 0.9f;
    int32_t     max_tokens = 0;         // 0 → model default (n_vq * max_secs)
};

struct TtsResponse {
    std::vector<float> pcm_mono;       // sampling_rate Hz, mono
    int32_t            sampling_rate = 24000;
    std::string        error;
};

class MossSession : public Session {
public:
    ~MossSession() override;
    std::string family_name() const override { return "moss_tts"; }
    bool load(const std::string& model_path,
              const LoadOptions& opts,
              const BackendConfig& backend_cfg,
              std::string* error = nullptr) override;
    bool run_tts(const void* request, void* response,
                 std::string* error = nullptr) override;

    const MossConfig& config() const { return cfg_; }

private:
    // Open the GGUF, parse MOSS KV metadata, bind moss.* extension tensors
    // into ext_ctx_, and start the Qwen3 backbone via the unified runner.
    bool bind_extension_tensors(const GgufReader& r,
                                std::string* error);

    // Build a single-row input embedding by summing text-token embeddings
    // (from libllama) with any in-context audio embeddings (via
    // audio_embed.{i}.weight * one_hot(codec_token[i])). Output shape:
    //   (n_tokens, hidden_size) row-major float32.
    bool build_input_embeddings(const int32_t* text_tokens, int32_t n_text,
                                const int32_t* audio_tokens, int32_t n_audio,
                                std::vector<float>* embd_out,
                                std::string* error);

    // Apply audio_head.{i}.weight to a hidden-state row → logits per stream.
    // hidden: (n_tokens, hidden_size). logits_out: (n_tokens, n_vq, vocab+1).
    bool project_to_audio_logits(const float* hidden, int32_t n_tokens,
                                 std::vector<float>* logits_out,
                                 std::string* error);

    // Decode codec tokens → PCM via moss.codec.dec.* graphs. Codec tokens:
    // (n_audio_tokens, n_vq). PCM out: mono float32 at config().sampling_rate.
    //
    // The codec graph is ported from openmoss/src/codec.cpp. Marked TODO
    // until that port lands.
    bool decode_codec(const int32_t* codec_tokens, int32_t n_tokens,
                      std::vector<float>* pcm_out,
                      std::string* error);

    std::unique_ptr<qwen3::Runner> backbone_;     // Qwen3-8B via libllama
    ggml_context*  ext_ctx_   = nullptr;          // moss.* weights
    ggml_tensor*   audio_embed_[32] = {};         // moss.audio_embed.{i}.weight
    ggml_tensor*   audio_head_[32]  = {};         // moss.audio_head.{i}.weight
    // Codec weights — anchored here, wired in codec.cpp.
    ggml_tensor*   codec_dec_root_ = nullptr;     // moss.codec.dec.<root>
    MossConfig     cfg_;
    bool           owns_ext_ctx_ = false;
};

}  // namespace audiocore::moss

#endif  // AUDIOCORE_MODELS_MOSS_TTS_FAMILY_H
