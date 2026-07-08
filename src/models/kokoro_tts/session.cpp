// session.cpp — audiocore Session wrapper for Kokoro-82M TTS.
//
// Bridges audiocore's Session interface to mmwillet's Kokoro runner.
// The runner handles: phonemization → duration prediction → audio generation.

#include "audiocore/models/kokoro_tts.h"
#include "audiocore/framework/runtime/tasks.h"

#include "models/kokoro/model.h"      // kokoro_model, kokoro_loader
#include "models/loaders.h"    // runner_from_file
#include "common.h"     // generation_configuration, tts_response

#include <cstring>
#include <filesystem>

namespace audiocore::kokoro_tts {

struct KokoroTtsSession::Impl {
    std::unique_ptr<tts_generation_runner> runner;
    std::string model_path;
};

KokoroTtsSession::KokoroTtsSession()
    : impl_(std::make_unique<Impl>()) {}

KokoroTtsSession::~KokoroTtsSession() = default;

bool KokoroTtsSession::load(const std::string& model_path,
                            const LoadOptions& opts,
                            const BackendConfig& backend_cfg,
                            std::string* error) {
    impl_->model_path = model_path;

    // Determine voice from options
    std::string voice = opts.extras.count("voice") ?
        opts.extras.at("voice") : "af_heart";

    // Build generation configuration
    generation_configuration config(voice);

    // Create the runner from the GGUF file
    impl_->runner = runner_from_file(model_path.c_str(), backend_cfg.n_threads,
                                     config, /*cpu_only=*/true);
    if (!impl_->runner) {
        if (error) *error = "Failed to load Kokoro model from: " + model_path;
        return false;
    }

    loaded_ = true;
    return true;
}

bool KokoroTtsSession::run_tts(const void* request, void* response,
                               std::string* error) {
    if (!impl_ || !impl_->runner) {
        if (error) *error = "Kokoro model not loaded";
        return false;
    }

    const auto* req = static_cast<const TtsRequest*>(request);
    auto* resp = static_cast<TtsResponse*>(response);

    // Update voice if specified
    generation_configuration config(req->speaker_name.empty() ? "af_heart" : req->speaker_name);

    // Generate speech
    tts_response tts_resp{};
    tts_resp.data = nullptr;
    tts_resp.n_outputs = 0;

    impl_->runner->generate(req->text.c_str(), tts_resp, config);

    if (tts_resp.n_outputs == 0 || !tts_resp.data) {
        if (error) *error = "Kokoro generated no audio output";
        return false;
    }

    // Copy to response
    resp->pcm_mono.assign(tts_resp.data, tts_resp.data + tts_resp.n_outputs);
    resp->sampling_rate = 24000;

    // Free the runner-allocated buffer
    free(tts_resp.data);

    return true;
}

}  // namespace audiocore::kokoro_tts
