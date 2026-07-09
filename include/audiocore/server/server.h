#ifndef AUDIOCORE_SERVER_SERVER_H
#define AUDIOCORE_SERVER_SERVER_H

#include <httplib.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "audiocore/framework/runtime/registry.h"
#include "audiocore/server/acestep_proxy.h"

namespace audiocore {

// One loaded model + the mutex that serializes concurrent requests against
// the same backend.
struct ModelSlot {
    std::shared_ptr<ILoadedModel> model;
    IOfflineTaskSession* session = nullptr;  // non-owning; lifetime tied to model
    std::mutex mtx;
    bool loaded = true;
    ModelLoadRequest load_req;  // cached for /v1/models/load re-loading
};

// Build a configured httplib::Server with all routes wired against `slots`.
// clips_dir is the directory for audio clip storage (upload/delete/raw).
// If empty, clip management routes are not registered.
// registry + weights_dir are needed for the /v1/models/load endpoint.
// model_configs is the initial config list (for re-loading by id).
std::shared_ptr<httplib::Server> build_server(
        std::shared_ptr<std::unordered_map<std::string, std::shared_ptr<ModelSlot>>> slots,
        const std::string& clips_dir = {},
        std::shared_ptr<ModelRegistry> registry = {},
        const std::string& weights_dir = {},
        std::shared_ptr<std::unordered_map<std::string, AceStepProxyConfig>> acestep_proxies = {});

// WAV encoders (exposed for unit testing).
std::string pcm_mono_to_wav(const std::vector<float>& pcm, int32_t sr);
std::string pcm_stereo_to_wav(const std::vector<float>& pcm, int32_t sr);

}  // namespace audiocore

#endif  // AUDIOCORE_SERVER_SERVER_H
