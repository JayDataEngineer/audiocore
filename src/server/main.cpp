// audiocore_server entry point.
//
// Loads server.json, instantiates one Session per configured model id via
// the FamilyRegistry, and serves OpenAI-compatible endpoints:
//
//   GET  /health
//   GET  /v1/models
//   POST /v1/audio/speech          OpenAI TTS (moss_tts)
//   POST /v1/audio/transcriptions  OpenAI ASR (future)
//   POST /v1/audio/music           ACE-Step music generation (audiocore-specific)
//   POST /v1/audio/music/submit    async submit (returns job id)
//   GET  /v1/audio/music/job?id=…  poll for music gen result
//
// Single-process, one loaded model per id. Concurrency = N configured models;
// requests on the same model serialize through the session's backend.

#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>

#include "audiocore/framework/core/backend.h"
#include "audiocore/framework/core/session.h"
#include "audiocore/framework/runtime/registry.h"
#include "engine/framework/io/json.h"   // vendored cJSON wrapper (Phase 2)

// Link against audio.cpp-style HTTP library (Phase 2 milestone).
// For now this is a stub that demonstrates config loading + session wiring.

namespace {

struct ConfigModel {
    std::string id;
    std::string family;
    std::string path;
    std::string task;
    std::string mode = "offline";
    std::string backend = "ggml_cuda";
    audiocore::LoadOptions load_options;
};

struct ServerConfig {
    std::string host = "127.0.0.1";
    int port = 8080;
    int device = 0;
    int threads = 1;
    std::vector<ConfigModel> models;
};

bool load_config(const std::string& path, ServerConfig& out) {
    // TODO(Phase 2): JSON parse via vendored cJSON.
    (void)path; (void)out;
    return false;
}

audiocore::BackendKind parse_backend(const std::string& s) {
    if (s == "ggml_cuda")   return audiocore::BackendKind::ggml_cuda;
    if (s == "ggml_cpu")    return audiocore::BackendKind::ggml_cpu;
    if (s == "ggml_vulkan") return audiocore::BackendKind::ggml_vulkan;
    if (s == "ggml_metal")  return audiocore::BackendKind::ggml_metal;
    if (s == "onnxruntime") return audiocore::BackendKind::onnxruntime;
    return audiocore::BackendKind::ggml_cuda;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s --config server.json\n", argv[0]);
        return 1;
    }

    std::string config_path;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) config_path = argv[++i];
    }
    if (config_path.empty()) {
        std::fprintf(stderr, "missing --config\n");
        return 1;
    }

    ServerConfig cfg;
    if (!load_config(config_path, cfg)) {
        std::fprintf(stderr, "failed to load config '%s'\n", config_path.c_str());
        return 1;
    }

    // Instantiate one Session per configured model id.
    std::unordered_map<std::string, std::unique_ptr<audiocore::Session>> sessions;
    for (const ConfigModel& m : cfg.models) {
        auto sess = audiocore::FamilyRegistry::instance().create(m.family);
        if (!sess) {
            std::fprintf(stderr, "unknown family '%s' for model id '%s'\n",
                         m.family.c_str(), m.id.c_str());
            return 1;
        }
        audiocore::BackendConfig bc{
            .kind      = parse_backend(m.backend),
            .device_id = cfg.device,
            .n_threads = cfg.threads,
        };
        std::string err;
        if (!sess->load(m.path, m.load_options, bc, &err)) {
            std::fprintf(stderr, "load failed for '%s': %s\n",
                         m.id.c_str(), err.c_str());
            return 1;
        }
        sessions[m.id] = std::move(sess);
    }

    std::fprintf(stderr, "audiocore_server: %zu model(s) loaded\n",
                 sessions.size());
    // TODO(Phase 2): bind HTTP server on (cfg.host, cfg.port) and dispatch
    // requests by model id → sessions[id]->run_{tts,music,asr}().
    return 0;
}
