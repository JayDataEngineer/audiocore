// runtime.h — Silero VAD runtime: load weights, run inference, segment.
//
// CPU reference implementation in runtime.cpp. Plain C++ loops (no ggml
// graph) — the model is tiny and runs far above realtime on CPU. GPU
// acceleration is a clean follow-up.

#ifndef AUDIOCORE_MODELS_SILERO_VAD_RUNTIME_H
#define AUDIOCORE_MODELS_SILERO_VAD_RUNTIME_H

#include <memory>
#include <string>

#include "audiocore/framework/core/backend.h"
#include "audiocore/framework/io/weight_loader.h"
#include "audiocore/framework/runtime/tasks.h"

namespace audiocore::silero_vad {

class SileroVadRuntime {
public:
    SileroVadRuntime();
    ~SileroVadRuntime();

    // Load GGUF weights via WeightLoader (audiocore convention: never
    // call gguf_* directly from family code).
    bool load(const std::string& model_path,
              const BackendConfig& backend_cfg,
              std::string* error = nullptr);

    // Run VAD over a WAV file, produce speech segments.
    bool detect(const VadRequest& req, VadResponse& resp,
                std::string* error = nullptr);

private:
    // Forward one [context(64) | chunk(512)] = 576-sample input through the
    // conv/LSTM/linear stack, return speech probability in [0, 1].
    // Updates internal LSTM hidden/cell state.
    float forward_chunk(const float* input_576);

    struct Impl;
    std::unique_ptr<Impl> impl_;
    // Owns the WeightLoader so the mmap backing any captured tensor
    // pointers stays alive for the session's lifetime.
    std::unique_ptr<WeightLoader> loader_;
};

}  // namespace audiocore::silero_vad

#endif  // AUDIOCORE_MODELS_SILERO_VAD_RUNTIME_H
