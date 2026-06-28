// family.h — ZONOS2 subprocess TTS family.
//
// ZONOS2 is an 8B-parameter MoE decoder-only transformer (SonicMoE, ~900M
// active params) with 9-codebook DAC audio output at 44.1 kHz. Because the
// architecture is too complex to reimplement in C++ via ggml, this family
// spawns a Python subprocess running the official ZONOS2 Mini-SGLang server
// and communicates via HTTP. The subprocess is started in load() and killed
// in ~Zonos2Session().
//
// Architecture:
//   Text → UTF-8 byte tokenization → MoE backbone →
//   DAC decoder → 44.1 kHz mono PCM
//
// Speaker conditioning:
//   ECAPA-TDNN embedding (128-dim) extracted from reference audio.
//   Passed per-request via the speaker_embedding field (base64-encoded).
//
// Weight sources:
//   - HuggingFace: Zyphra/ZONOS2  (official, full PyTorch weights)
//   - GGUF:        cagyirey/ZONOS2-GGUF  (via mistral.rs, not llama.cpp)
//
// Runtime dependencies (subprocess):
//   Python 3.10+, torch, transformers, zonos, dac
//
// See https://huggingface.co/Zyphra/ZONOS2
//     https://github.com/Zyphra/ZONOS2

#ifndef AUDIOCORE_MODELS_ZONOS2_FAMILY_H
#define AUDIOCORE_MODELS_ZONOS2_FAMILY_H

#include <memory>
#include <string>
#include <vector>

#include "audiocore/framework/core/session.h"

namespace audiocore::zonos2 {

// Parsed from LoadOptions + extras at load time.
struct Zonos2Config {
    std::string model_path;            // HuggingFace model ID or local path
    std::string python_bin = "python3"; // Python interpreter for subprocess
    std::string device = "cuda:0";     // device string passed to subprocess
    int         port = 0;              // 0 = auto-select free port
    int         ready_timeout_sec = 180; // max wait for subprocess health
};

// Request shape for ZONOS2 TTS.
struct TtsRequest {
    std::string text;                      // input text (UTF-8)
    std::string language = "en";           // language for text normalization
    std::string speaker_embedding;         // optional, base64 128-dim float32
    float       speed = 1.0f;              // speaking speed multiplier
    float       temperature = 0.7f;        // sampling temperature
    float       top_p = 0.9f;              // nucleus sampling threshold
    float       cfg_scale = 2.0f;          // classifier-free guidance scale
    int         max_new_tokens = 0;        // 0 = auto (model default)
};

struct TtsResponse {
    std::vector<float> pcm_mono;          // 44.1 kHz mono float32 PCM
    int32_t            sampling_rate = 44100;
    std::string        error;
};

class Zonos2Session : public Session {
public:
    Zonos2Session();
    ~Zonos2Session() override;

    Zonos2Session(const Zonos2Session&) = delete;
    Zonos2Session& operator=(const Zonos2Session&) = delete;

    std::string family_name() const override { return "zonos2"; }

    bool load(const std::string& model_path,
              const LoadOptions& opts,
              const BackendConfig& backend_cfg,
              std::string* error = nullptr) override;

    bool run_tts(const void* request, void* response,
                 std::string* error = nullptr) override;

    const Zonos2Config& config() const { return cfg_; }

private:
    // Start the Python subprocess on an available port.
    bool start_subprocess(std::string* error);

    // Poll GET /health until the subprocess responds.
    bool wait_for_ready(std::string* error);

    // Kill the subprocess (SIGTERM, then SIGKILL after timeout). Also sends
    // a graceful /shutdown POST before the signals.
    void stop_subprocess();

    // Parse WAV bytes from the subprocess response into PCM float samples.
    bool parse_wav_response(const std::string& wav_bytes,
                            std::vector<float>& pcm_out,
                            int32_t& sampling_rate_out,
                            std::string* error);

    // Find a free TCP port on 127.0.0.1.
    static int find_free_port();

    Zonos2Config cfg_;
    pid_t        subprocess_pid_ = -1;
    int          subprocess_port_ = 0;
    std::string  subprocess_url_;
    bool         subprocess_running_ = false;
};

}  // namespace audiocore::zonos2

#endif  // AUDIOCORE_MODELS_ZONOS2_FAMILY_H
