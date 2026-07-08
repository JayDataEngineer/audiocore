// family.h — Qwen3-TTS model family.
//
// Architecture (official Qwen3-TTS):
//   1. Talker (qwen3tts arch) — Qwen3 backbone with dual embedding:
//      text_embedding [text_vocab, 2048] + text_projection (2048→1024) for text
//      codec_embedding [3072, 1024] for codec tokens
//      Both are summed element-wise at the input.
//   2. Code Predictor (qwen3tts_cp arch) — 5/8-layer transformer with 31
//      separate codec_embedding tables + 31 lm_heads. Generates fine codebooks
//      1-31 via MTP-style autoregressive sub-steps.
//   3. Speaker Encoder (ECAPA-TDNN) — extracts speaker embedding from
//      reference audio. (Stage 17b: ggml port wired — `Qwen3TtsSpeakerEncoder`
//      in speaker_encoder.h, adapted from CrispStrobe/CrispASR's ECAPA-TDNN
//      section. Loads `speaker.*` tensors from the talker GGUF. Unblocks
//      Voice Clone mode.)
//   4. Speech Tokenizer — decodes [16 × T] code matrix to waveform.
//      (Stage 17: ggml port wired — `Qwen3TtsCodecGraphs` in codec.h,
//      adapted from CrispStrobe/CrispASR's qwen3_tts.cpp codec section.
//      Auto-activates when the codec sidecar GGUF is discovered. Stage 17b
//      wires the ECAPA-TDNN speaker encoder — `Qwen3TtsSpeakerEncoder` in
//      speaker_encoder.h — loaded from the talker GGUF's `speaker.*` tensors.)
//
// Official weights: QwenLM/Qwen3-TTS on HuggingFace.
// Use tools/convert_qwen3tts (C++) to produce GGUFs from safetensors.
//
// Inference pipeline:
//   text/instruct → text_embedding + text_projection [2048→1024] →
//   sum with codec prefill tokens [codec_embedding] →
//   Talker forward (embd mode) → hidden states → codec_head → codebook 0 →
//   Code Predictor MTP loop (31 fine codebooks) →
//   [32 × T] code matrix → Speech Tokenizer → PCM audio

#ifndef AUDIOCORE_MODELS_QWEN3_TTS_FAMILY_H
#define AUDIOCORE_MODELS_QWEN3_TTS_FAMILY_H

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "audiocore/framework/core/backend.h"
#include "audiocore/framework/core/session.h"
#include "audiocore/framework/runtime/tasks.h"    // TtsRequest / TtsResponse
#include "audiocore/models/qwen3/runner.h"
#include "audiocore/models/qwen3_tts/codec.h"     // Stage 17: Qwen3TtsCodecGraphs
#include "audiocore/models/qwen3_tts/speaker_encoder.h"  // Stage 17b: Qwen3TtsSpeakerEncoder

// Forward decl — full GgufReader brings sys/mman.h into the TU; family.h is
// included widely enough that we keep it opaque here. The loader.cpp TU sees
// the full definition and owns the reader instance.
namespace audiocore { class GgufReader; struct TensorStorage; }

namespace audiocore::qwen3_tts {

// Qwen3-TTS uses the unified audiocore::TtsRequest / TtsResponse directly.
// These aliases keep existing references in session.cpp / server.cpp
// working without touching every call site.
using TtsRequest  = ::audiocore::TtsRequest;
using TtsResponse = ::audiocore::TtsResponse;

// ── Config ──────────────────────────────────────────────────────────────────

// Which training variant of Qwen3-TTS we loaded. Comes from extras["variant"]
// or is inferred from the talker path. Drives per-variant defaults:
//   Base         — plain TTS, no speaker token required
//   CustomVoice  — speaker_name resolves to one of the 9 default speakers
//   VoiceDesign  — voice_design mode is supported on this backbone
// Until config.json parsing lands we can't actually detect the variant from
// the GGUF itself; extras["variant"] is the source of truth. Unknown =
// behave like Base (the safest default — every variant accepts plain TTS).
enum class Qwen3TtsVariant {
    Unknown = 0,
    Base,
    CustomVoice,
    VoiceDesign,
};

// Per-request mode, mirroring Qwen3-TTS's five user-facing modes. Comes from
// the unified TtsRequest.mode field. The default ("tts") works on every
// variant. "voice_design" is intended for the VoiceDesign variant;
// "voice_clone" runs the ECAPA-TDNN speaker encoder (Stage 17b) when the
// speaker encoder is loaded; falls back to a fail-fast error otherwise.
enum class Qwen3TtsMode {
    TtsBatch = 0,    // default plain batch TTS (also covers instruct style)
    VoiceDesign,     // describe a voice → generate with that voice
    VoiceClone,      // reference_audio → speaker embedding (ECAPA-TDNN, Stage 17b)
    Streaming,       // per-frame streaming via stream callback (Stage 19)
};

struct Qwen3TtsConfig {
    std::string talker_path;          // Path to talker GGUF
    std::string predictor_path;       // Path to predictor GGUF
    std::string codec_path;           // Stage 17: path to codec GGUF
                                       // (cstr/qwen3-tts-tokenizer-12hz-GGUF);
                                       // empty if no codec sidecar found.
    std::string speaker_encoder_path; // Stage 17b: path to standalone
                                       // ECAPA-TDNN speaker-encoder GGUF
                                       // (qwen3tts-speaker-encoder.gguf from
                                       // marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B).
                                       // Empty → fall back to talker_path
                                       // (for older bundled-tensor layouts).
    int  n_codebooks        = 32;
    int  n_ctx_talker       = 8192;
    int  n_ctx_predictor    = 4096;
    int  n_gpu_layers       = -1;
    bool flash_attn         = true;
    float temperature       = 0.7f;
    float top_p             = 0.9f;
    int   max_new_tokens    = 4096;
    Qwen3TtsVariant variant = Qwen3TtsVariant::Unknown;
    // Populated from extras["model_size_b"] when known (1.7 or 0.6). Used
    // only for logging / future size-aware defaults. Zero = unknown.
    float model_size_b      = 0.0f;
    // Stage 17: true once the codec GGUF was discovered, opened, and
    // Qwen3TtsCodecGraphs::bind() succeeded. Drives the run_inference
    // Phase 4 branch: bound → real codec decode; unbound → silence fallback.
    bool codec_present      = false;
    // Stage 17b: true once the ECAPA-TDNN speaker encoder was loaded
    // (either from speaker_encoder_path or, for legacy layouts, from
    // speaker.* tensors bundled inside the talker GGUF). Drives the Voice
    // Clone branch in run_inference: bound → real ECAPA embedding;
    // unbound → fail-fast with GAPS.md pointer.
    bool speaker_present    = false;
    // WDELTA (Base → CustomVoice weight patching) — see
    // Runner::apply_wdelta_patch docstring. Populated by Qwen3TtsSession::load
    // when a CV talker is loaded and a sibling Base talker GGUF is found.
    // wdelta_applied=true means text_proj biases + codec_embedding in the
    // loaded CV talker have been overwritten with Base's versions, unlocking
    // the 4-feature pipeline (custom embedding + instruct + text + quality)
    // on the CV variant. Drives the speaker-embedding path in synthesize().
    bool wdelta_applied     = false;
    std::string wdelta_base_path;
};

// Parse a mode string ("tts" | "voice_design" | "voice_clone" | "streaming")
// from TtsRequest.mode into the enum. Unknown / "tts" / "" → TtsBatch.
Qwen3TtsMode parse_mode(const std::string& s);

// Inline helper for variant display in logs.
inline const char* variant_name(Qwen3TtsVariant v) {
    switch (v) {
        case Qwen3TtsVariant::Base:         return "Base";
        case Qwen3TtsVariant::CustomVoice:  return "CustomVoice";
        case Qwen3TtsVariant::VoiceDesign:  return "VoiceDesign";
        case Qwen3TtsVariant::Unknown:      break;
    }
    return "Unknown";
}

// ── Request / Response ──────────────────────────────────────────────────────
//
// Unified across every TTS family — see
// include/audiocore/framework/runtime/tasks.h. The aliases above bring them
// into this namespace so session.cpp reads naturally.

// ── Session ─────────────────────────────────────────────────────────────────

class Qwen3TtsSession : public Session {
public:
    Qwen3TtsSession();
    ~Qwen3TtsSession() override;

    std::string family_name() const override { return "qwen3_tts"; }

    bool load(const std::string& model_path,
              const LoadOptions& opts,
              const BackendConfig& backend_cfg,
              std::string* error = nullptr) override;

    bool run_tts(const void* request, void* response,
                 std::string* error = nullptr) override;

    // ── Speaker embedding extract / synthesize API ────────────────────────
    // Compute a speaker embedding from a reference WAV file for later reuse.
    // Returns the embedding as a float vector (size = d_model, 1024).
    std::vector<float> extract_speaker_embedding(const std::string& wav_path,
                                                  std::string* error = nullptr);

    // Base Session override — delegates to extract_speaker_embedding so
    // Python bindings can call compute_embedding() generically across all
    // families (returns empty + error on families that don't implement it).
    std::vector<float> compute_embedding(const std::string& wav_path,
                                          std::string* error = nullptr) override {
        return extract_speaker_embedding(wav_path, error);
    }

    // Synthesize with a pre-computed speaker embedding (bypasses WAV load).
    // `embedding` must be the output of extract_speaker_embedding or match
    // the talker's n_embd dimension.
    bool synthesize_with_embedding(const TtsRequest& req_base,
                                   const float* embedding, size_t emb_dim,
                                   TtsResponse& resp,
                                   std::string* error = nullptr);

    // True when the ECAPA-TDNN speaker encoder GGUF was loaded alongside
    // the talker. Required for extract_speaker_embedding / voice-clone mode.
    bool speaker_encoder_loaded() const { return config_.speaker_present; }

private:
    std::unique_ptr<qwen3::Runner> talker_;     // qwen3tts backbone + dual-embedding extras
    std::unique_ptr<qwen3::Runner> predictor_;  // qwen3tts_cp + MTP extras
    Qwen3TtsConfig config_;

    // Stage 17: codec graphs + the GGUF reader that owns the mmap'd codec
    // weight bytes. The reader must outlive codec_graphs_ — order matters in
    // the destructor (codec_graphs_' tensors alias the reader's mmap).
    // backend_ matches the device the talker chose so codec + talker share
    // VRAM rather than round-tripping through host RAM.
    std::unique_ptr<GgufReader>     codec_reader_;
    std::unique_ptr<Backend>        codec_backend_;
    Qwen3TtsCodecGraphs             codec_graphs_;

    // Stage 17b: ECAPA-TDNN speaker encoder. The speaker.* tensors live in
    // the talker GGUF, not a separate file — we open a separate GgufReader
    // on the same file to load them (llama.cpp skips unrecognized tensor
    // names). The reader must outlive speaker_encoder_ (same dtor ordering).
    // speaker_backend_ shares the talker's device so weight tensors live
    // in VRAM alongside the talker/predictor.
    std::unique_ptr<GgufReader>         speaker_reader_;
    std::unique_ptr<Backend>            speaker_backend_;
    Qwen3TtsSpeakerEncoder              speaker_encoder_;

    bool run_inference(const TtsRequest& req, TtsResponse& resp,
                       std::string* error);
};

}  // namespace audiocore::qwen3_tts

#endif  // AUDIOCORE_MODELS_QWEN3_TTS_FAMILY_H
