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
#include "audiocore/models/qwen3/runner.h"

namespace audiocore::qwen3_tts {

// ── Config ──────────────────────────────────────────────────────────────────

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
};

// ── Request / Response ──────────────────────────────────────────────────────

struct TtsRequest {
    // Core
    std::string text;               // Input text to synthesize
    std::string language   = "";    // "en", "zh", "ja", etc. Empty = auto
    float       speed      = 1.0f;

    // Sampling
    float       temperature = 0.7f;
    float       top_p      = 0.9f;
    int         max_new_tokens = 4096;

    // Voice design / emotion instruct (for "voice_design" / "custom_voice")
    std::string instruct   = "";    // Natural language style instruction
    std::string speaker_name = "";  // Speaker ID ("Vivian", "Ryan", etc.)

    // Voice cloning (for "base" model)
    std::string reference_audio;    // Path to reference audio
    std::string reference_text;     // Reference text for ICL cloning
    std::string speaker_embedding;  // Pre-computed speaker embedding (base64)
};

struct TtsResponse {
    std::vector<float> pcm_mono;
    int                sampling_rate = 24000;  // 24kHz output
};

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
