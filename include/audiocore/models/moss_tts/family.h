// family.h — MOSS-TTS family session.
//
// MOSS is Qwen3-8B + an audio codec. We do NOT reimplement Qwen3 — the
// backbone runs through the unified qwen3::Runner (libllama). This file
// covers everything that ISN'T the transformer:
//
//   • audio_embed.{i}.weight — projects codec tokens into Qwen3 hidden space
//   • audio_head.{i}.weight  — projects Qwen3 hidden states back to codec
//                              logits, one head per RVQ stream
//   • moss.codec.dec.*        — codec-token → PCM decoder (ggml port pending)
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
#include "audiocore/framework/runtime/tasks.h"    // TtsRequest / TtsResponse
#include "audiocore/models/qwen3/runner.h"
#include "audiocore/models/moss_tts/codec.h"      // MossCodecGraphs (Stage 16 port)

struct ggml_context;
struct ggml_tensor;

namespace audiocore::moss {

// MOSS-TTS uses the unified audiocore::TtsRequest / TtsResponse directly.
// These aliases keep existing references in session.cpp / server.cpp
// working without touching every call site.
using TtsRequest  = ::audiocore::TtsRequest;
using TtsResponse = ::audiocore::TtsResponse;

// Parsed from MOSS GGUF KV metadata. These keys are stable across OpenMOSS
// releases (the same names are read by openmoss/src/model.cpp).
struct MossConfig {
    // Values read from GGUF KV metadata (moss.*). When absent (common for
    // community GGUFs that only have the backbone), we fall back to the
    // hardcoded defaults matching the OpenMOSS upstream constants.
    int32_t n_vq              = 32;     // moss.n_vq — RVQ stream count
    int32_t audio_vocab_size  = 1024;   // moss.audio_vocab_size (0-1023 + pad=1024)
    int32_t audio_pad_code    = 1024;   // moss.audio_pad_code
    int32_t sampling_rate     = 24000;  // moss.sampling_rate
    int32_t downsample_rate   = 320;    // moss.downsample_rate (codec)
    float   frame_rate        = 0.0f;   // moss.frame_rate — computed from sr/dsr when 0
    int32_t n_quantized_embd  = 0;      // moss.n_quantized_embd (optional)
    // Token IDs matching _constants.py / delay_state.h:
    int32_t tok_audio_start   = 151652;
    int32_t tok_audio_end     = 151653;
    int32_t tok_user_slot     = 151654;
    int32_t tok_audio_gen     = 151656;
    int32_t tok_audio_delay   = 151662;
    int32_t tok_im_start      = 151644;
    int32_t tok_im_end        = 151645;
    int32_t tok_pad           = 151643;
    bool    codec_present     = false;  // moss.codec.present
};

// Concrete request/response shapes are unified across every TTS family —
// see include/audiocore/framework/runtime/tasks.h. MossSession down-casts
// the void* it gets from Session::run_tts to TtsRequest* / TtsResponse*.

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

    // Load audio embedding and head tables from .npy files. Used when the
    // GGUF only contains the Qwen3 backbone (the common community pattern).
    //   emb_dir: path to embeddings/  directory (embed_tokens.npy + emb_ext_*.npy)
    //   lm_dir:  path to lm_heads/    directory (lm_head_audio_*.npy)
    bool load_npy_extras(const std::string& emb_dir,
                         const std::string& lm_dir,
                         std::string* error);

    // Look up raw text-token embeddings from the GGUF's `token_embd.weight`
    // (which libllama has its own copy of internally; we keep ours for raw
    // gather without a forward pass). F32/F16 only — quantized tables need
    // the ggml_cgraph production path. Output: (n_tokens, hidden_size).
    bool embed_text_tokens(const int32_t* text_tokens, int32_t n_tokens,
                           std::vector<float>* embd_out,
                           std::string* error);

    // Build a single-row input embedding by summing text-token embeddings
    // (from embed_text_tokens) with any in-context audio embeddings (via
    // audio_embed.{i}.weight * one_hot(codec_token[i])). Output shape:
    //   (n_tokens, hidden_size) row-major float32.
    bool build_input_embeddings(const int32_t* text_tokens, int32_t n_text,
                                const int32_t* audio_tokens, int32_t n_audio,
                                std::vector<float>* embd_out,
                                std::string* error);

    // Embed a single (1+n_vq) token vector into one hidden_size row.
    // tokens[0] is the text token, tokens[1..n_vq] are audio codec tokens.
    bool embed_one_step(const int32_t* tokens, std::vector<float>* embd_out,
                        std::string* error);

    // Apply audio_head.{i}.weight to a hidden-state row → logits per stream.
    // hidden: (n_tokens, hidden_size). logits_out: (n_tokens, n_vq, vocab+1).
    bool project_to_audio_logits(const float* hidden, int32_t n_tokens,
                                 std::vector<float>* logits_out,
                                 std::string* error);

    // Decode codec tokens → PCM. Codec tokens: (n_audio_tokens, n_vq).
    // PCM out: mono float32 at config().sampling_rate.
    //
    // Stage 16: if MossCodecGraphs is bound (codec_graphs_.is_present()),
    // this routes the codes through the ggml port of
    // openmoss/src/codec.cpp. Otherwise it emits 1 s of silence at the
    // configured sample rate — the documented fallback for GGUFs that
    // don't carry the moss.codec.* tensors (see GAPS.md §1.3).
    bool decode_codec(const int32_t* codec_tokens, int32_t n_tokens,
                      std::vector<float>* pcm_out,
                      std::string* error);

    // Buffers for data materialized from GGUF files (used when tensor_data_ptr
    // returns null, e.g. some extras GGUFs have incorrect data offsets).
    // The ggml_tensor::data pointers point into these. Kept alive for the
    // session lifetime.
    std::vector<std::vector<uint8_t>>  gguf_buffers_;

    // NV: npy-loaded data buffers. The ggml_tensor::data pointers for npy-
    // loaded tensors point into these. Kept alive for the session lifetime.
    std::vector<std::vector<uint8_t>>  npy_buffers_;

    std::unique_ptr<qwen3::Runner> backbone_;     // Qwen3-8B via libllama
    ggml_context*  ext_ctx_   = nullptr;          // moss.* weights
    ggml_tensor*   audio_embed_[32] = {};         // moss.audio_embed.{i}.weight
    ggml_tensor*   audio_head_[32]  = {};         // moss.audio_head.{i}.weight
    // The Qwen3 token embedding table — we read it from the GGUF directly
    // (parallel to libllama's internal copy) for raw gathers without a
    // forward pass. Bound at load time via the GgufReader.
    ggml_tensor*   token_embd_ = nullptr;         // token_embd.weight
    // Codec weights — populated by bind_extension_tensors() when the GGUF
    // carries moss.codec.dec.* tensors. Used by MossCodecGraphs to build
    // and run the decode graph on the active backend.
    ggml_tensor*     codec_dec_root_ = nullptr;    // moss.codec.dec.<root>
    MossCodecGraphs  codec_graphs_;                // Stage 16 ggml port
    MossConfig     cfg_;
    bool           owns_ext_ctx_ = false;
};



}  // namespace audiocore::moss

#endif  // AUDIOCORE_MODELS_MOSS_TTS_FAMILY_H
