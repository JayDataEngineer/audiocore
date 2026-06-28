// family.h — Kokoro ONNX TTS family session.
//
// Kokoro is a compact TTS model (~82M params, ~300MB full / ~80MB quantized)
// running on ONNX Runtime. It takes (phoneme tokens, style vector, speed) and
// directly outputs 24kHz waveform — no codec, no LLM backbone.
//
// Architecture:
//   Text → espeak-ng phonemization → vocab tokenization →
//   ONNX model (style-conditioned decoder-only) → 24kHz PCM
//
// Weight files:
//   - kokoro-v1.0.onnx          The ONNX model
//   - voices-v1.0.bin           Voice style embeddings (.npz archive)
//
// See https://github.com/thewh1teagle/kokoro-onnx

#ifndef AUDIOCORE_MODELS_KOKORO_FAMILY_H
#define AUDIOCORE_MODELS_KOKORO_FAMILY_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "audiocore/framework/core/session.h"

struct OrtSession;
struct OrtEnv;
struct OrtSessionOptions;
struct OrtMemoryInfo;
struct OrtAllocator;
struct OrtValue;

namespace audiocore::kokoro {

// Forward declarations.
class PhonemeTokenizer;

// Voice style data: each voice has a 2D table of shape (510, 256) where
// each row is the style vector for a given phoneme sequence length.
struct VoiceData {
    std::vector<float> data;  // flattened (510, 256) row-major
    int32_t n_styles = 510;   // first dimension (rows = max phoneme len)
    int32_t style_dim = 256;  // second dimension (style vector size)
};

// Parsed from config + extras at load time.
struct KokoroConfig {
    std::string model_path;         // path to kokoro-v1.0.onnx
    std::string voices_path;        // path to voices-v1.0.bin (.npz)
    std::string default_voice = "af_heart";
    int32_t     max_phoneme_len = 510;
};

// Request shape for Kokoro TTS.
struct TtsRequest {
    std::string text;              // input text
    std::string voice = "af_heart"; // voice name (e.g. "af_heart", "am_adam", "bf_emma")
    float       speed = 1.0f;      // 0.5–2.0
    std::string language = "en-us"; // espeak language tag ("en-us", "en-gb", "ja", etc.)
    bool        is_phonemes = false; // if true, `text` is already phoneme string
    bool        trim = true;        // trim leading/trailing silence
};

struct TtsResponse {
    std::vector<float> pcm_mono;   // 24kHz mono float32 PCM
    int32_t            sampling_rate = 24000;
    std::string        error;
};

class KokoroSession : public Session {
public:
    KokoroSession();
    ~KokoroSession() override;

    KokoroSession(const KokoroSession&) = delete;
    KokoroSession& operator=(const KokoroSession&) = delete;

    std::string family_name() const override { return "kokoro"; }

    bool load(const std::string& model_path,
              const LoadOptions& opts,
              const BackendConfig& backend_cfg,
              std::string* error = nullptr) override;

    bool run_tts(const void* request, void* response,
                 std::string* error = nullptr) override;

    const KokoroConfig& config() const { return cfg_; }

    // List available voice names.
    std::vector<std::string> list_voices() const;

private:
    // Load voicepack from .npz archive (zip with stored .npy entries).
    bool load_voices(const std::string& voices_path, std::string* error);

    // Load the ONNX model via ONNX Runtime.
    bool load_onnx(const std::string& onnx_path, bool use_gpu,
                   std::string* error);

    // Run ONNX inference: tokens → waveform.
    //   tokens: padded token IDs [0, ..., 0]
    //   style:  (1, 256) style vector
    //   speed:  speaking speed
    //   pcm_out: output waveform
    bool infer(const std::vector<int64_t>& tokens,
               const std::vector<float>& style,
               float speed,
               std::vector<float>& pcm_out,
               std::string* error);

    KokoroConfig cfg_;

    // ONNX Runtime state
    void*       ort_env_ = nullptr;
    void*       ort_session_ = nullptr;
    void*       ort_mem_info_ = nullptr;
    bool        onnx_loaded_ = false;

    // Voice data: name → style table
    std::unordered_map<std::string, VoiceData> voices_;

    // Phoneme tokenizer (lazily initialized on first run_tts call)
    std::unique_ptr<PhonemeTokenizer> tokenizer_;
    bool tokenizer_initialized_ = false;
};

}  // namespace audiocore::kokoro

#endif  // AUDIOCORE_MODELS_KOKORO_FAMILY_H
