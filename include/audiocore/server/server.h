// server.h — factory for the audiocore HTTP server.
//
// Extracted from src/server/main.cpp so tests can spin up a configured
// httplib::Server against a mock ModelSlot map without forking the binary.
// The binary just wires this to argv parsing + listening.

#ifndef AUDIOCORE_SERVER_SERVER_H
#define AUDIOCORE_SERVER_SERVER_H

#include <httplib.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "audiocore/framework/core/session.h"

namespace audiocore {

// One loaded model + the mutex that serializes concurrent requests against
// the same backend. The backend itself isn't thread-safe; this is the
// serialization point.
struct ModelSlot {
    std::unique_ptr<Session> session;
    std::mutex               mtx;
};

// Build a configured httplib::Server with all routes wired against `slots`.
// The caller owns the server and the slots map; both must outlive any
// request the server handles.
//
// Routes:
//   GET  /health
//   GET  /v1/models
//   POST /v1/audio/speech          (kokoro / zonos2 / moss_tts)
//   POST /v1/audio/music           (ace_step)
std::shared_ptr<httplib::Server> build_server(
        std::shared_ptr<std::unordered_map<std::string, std::shared_ptr<ModelSlot>>> slots);

// ---- WAV encoders (exposed for unit testing) -----------------------------
// Mono / stereo PCM float → 16-bit PCM WAV bytes.
std::string pcm_mono_to_wav(const std::vector<float>& pcm, int32_t sr);
std::string pcm_stereo_to_wav(const std::vector<float>& pcm, int32_t sr);

}  // namespace audiocore

#endif  // AUDIOCORE_SERVER_SERVER_H
