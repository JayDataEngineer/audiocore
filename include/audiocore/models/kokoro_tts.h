// kokoro_tts.h — Kokoro-82M TTS family.
//
// Architecture (Kokoro/StyleTTS-2 variant):
//   1. ALBERT encoder — shared-parameter transformer with 12 recurrences
//   2. Duration predictor — LSTM + ALBERT + F0/N prosody prediction
//   3. Text encoder — Conv layers + LSTM
//   4. Decoder/generator — BigVGAN-style with noise blocks, residual blocks,
//      harmonic sin generation, and STFT-based reconstruction
//
// 82M params, ~7× smaller than Qwen3-TTS 0.6B. Output: 24 kHz mono.
//
// Ported from mmwillet/TTS.cpp (MIT license). The graph-building code
// (~1500 lines) is kept self-contained in src/models/kokoro_tts/internal/
// to preserve the original include structure. This header provides the
// audiocore Session interface.
//
// Phonemization via espeak-ng (linked at build time). Voice packs are
// stored in the GGUF as `kokoro.voice_tensors.*` tensors.

#ifndef AUDIOCORE_MODELS_KOKORO_TTS_H
#define AUDIOCORE_MODELS_KOKORO_TTS_H

#include <memory>
#include <string>

#include "audiocore/framework/core/session.h"

namespace audiocore::kokoro_tts {

class KokoroTtsSession : public Session {
public:
    KokoroTtsSession();
    ~KokoroTtsSession() override;

    std::string family() const override { return "kokoro_tts"; }

    bool load(const std::string& model_path,
              const LoadOptions& opts,
              const BackendConfig& backend_cfg,
              std::string* error = nullptr) override;

    bool run_tts(const void* request, void* response,
                 std::string* error = nullptr) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace audiocore::kokoro_tts

#endif  // AUDIOCORE_MODELS_KOKORO_TTS_H
