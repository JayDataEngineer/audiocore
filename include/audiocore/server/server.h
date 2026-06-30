#ifndef AUDIOCORE_SERVER_SERVER_H
#define AUDIOCORE_SERVER_SERVER_H

#include <httplib.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "audiocore/framework/runtime/registry.h"

namespace audiocore {

// One loaded model + the mutex that serializes concurrent requests against
// the same backend. The new IOfflineTaskSession interface replaces the old
// Session base class.
struct ModelSlot {
    std::shared_ptr<ILoadedModel> model;
    std::unique_ptr<IOfflineTaskSession> session;
    std::mutex mtx;
};

// Build a configured httplib::Server with all routes wired against `slots`.
std::shared_ptr<httplib::Server> build_server(
        std::shared_ptr<std::unordered_map<std::string, std::shared_ptr<ModelSlot>>> slots);

// WAV encoders (exposed for unit testing).
std::string pcm_mono_to_wav(const std::vector<float>& pcm, int32_t sr);
std::string pcm_stereo_to_wav(const std::vector<float>& pcm, int32_t sr);

}  // namespace audiocore

#endif  // AUDIOCORE_SERVER_SERVER_H
