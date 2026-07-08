// silero_vad.h — Silero VAD v6.2 voice activity detection family.
//
// Silero VAD is a tiny (~1.8 MB) conv-recurrent model that classifies
// short audio chunks as speech/non-speech. Architecture:
//   1. STFT magnitude (129 bins, 16 ms windows at 16 kHz)
//   2. 4× Conv1D + ReLU: 129 → 128 → 64 → 64 → 128
//   3. LSTM cell (input 128, hidden 128), stateful across chunks
//   4. Linear head: 128 → 1 (speech probability)
//
// Output: speech segment timestamps. See VadRequest/VadResponse in
// audiocore/framework/runtime/tasks.h.
//
// Ported from 0xShug0/audio.cpp (release-0.1, MIT license). The graph-
// building code lives in src/models/silero_vad/runtime.cpp.
//
// Weights source: snakers4/silero-vad on HuggingFace, converted to GGUF
// via tools/convert_silero_vad.py. Stored as silero_vad.* tensors.

#ifndef AUDIOCORE_MODELS_SILERO_VAD_H
#define AUDIOCORE_MODELS_SILERO_VAD_H

#include <memory>
#include <string>

#include "audiocore/framework/core/session.h"

namespace audiocore::silero_vad {

class SileroVadSession : public Session {
public:
    SileroVadSession();
    ~SileroVadSession() override;

    std::string family_name() const override { return "silero_vad"; }

    bool load(const std::string& model_path,
              const LoadOptions& opts,
              const BackendConfig& backend_cfg,
              std::string* error = nullptr) override;

    bool run_vad(const void* request, void* response,
                 std::string* error = nullptr) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace audiocore::silero_vad

#endif  // AUDIOCORE_MODELS_SILERO_VAD_H
