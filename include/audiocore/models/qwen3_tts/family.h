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
//      reference audio. (Future GGUF port; ONNX version removed.)
//   4. Speech Tokenizer — decodes [32 × T] code matrix to waveform.
//      (Future ggml port; ONNX version removed.)
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
// "voice_clone" is rejected up-front until the ECAPA-TDNN port lands.
enum class Qwen3TtsMode {
    TtsBatch = 0,    // default plain batch TTS (also covers instruct style)
    VoiceDesign,     // describe a voice → generate with that voice
    VoiceClone,      // reference_audio → speaker embedding (NOT IMPLEMENTED)
    Streaming,       // chunked output (NOT IMPLEMENTED)
};

struct Qwen3TtsConfig {
    std::string talker_path;          // Path to talker GGUF
    std::string predictor_path;       // Path to predictor GGUF
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

private:
    std::unique_ptr<qwen3::Runner> talker_;     // qwen3tts backbone + dual-embedding extras
    std::unique_ptr<qwen3::Runner> predictor_;  // qwen3tts_cp + MTP extras
    Qwen3TtsConfig config_;

    bool run_inference(const TtsRequest& req, TtsResponse& resp,
                       std::string* error);
};

}  // namespace audiocore::qwen3_tts

#endif  // AUDIOCORE_MODELS_QWEN3_TTS_FAMILY_H
