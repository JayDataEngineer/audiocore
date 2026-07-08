// session.cpp — audiocore Session wrapper for Silero VAD.
//
// Bridges audiocore's Session interface to the Silero VAD runtime.
// The runtime handles: WAV load → chunk → STFT → conv/LSTM → segmenter.

#include "audiocore/models/silero_vad.h"
#include "audiocore/framework/runtime/tasks.h"

#include "runtime.h"

namespace audiocore::silero_vad {

struct SileroVadSession::Impl {
    std::unique_ptr<SileroVadRuntime> runtime;
    std::string model_path;
};

SileroVadSession::SileroVadSession()
    : impl_(std::make_unique<Impl>()) {}

SileroVadSession::~SileroVadSession() = default;

bool SileroVadSession::load(const std::string& model_path,
                            const LoadOptions& opts,
                            const BackendConfig& backend_cfg,
                            std::string* error) {
    (void)opts;
    impl_->model_path = model_path;
    impl_->runtime = std::make_unique<SileroVadRuntime>();
    return impl_->runtime->load(model_path, backend_cfg, error);
}

bool SileroVadSession::run_vad(const void* request, void* response,
                                std::string* error) {
    if (!impl_->runtime) {
        if (error) *error = "silero_vad: session not loaded";
        return false;
    }
    const auto* req = static_cast<const VadRequest*>(request);
    auto* resp = static_cast<VadResponse*>(response);
    return impl_->runtime->detect(*req, *resp, error);
}

}  // namespace audiocore::silero_vad
