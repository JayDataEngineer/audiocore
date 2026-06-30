// audiocore_server entry point.
//
// Loads server.json, instantiates one Session per configured model id via
// the FamilyRegistry, and serves:
//
//   GET  /health
//   GET  /v1/models
//   POST /v1/audio/speech          OpenAI-compatible TTS (moss_tts)
//   POST /v1/audio/music           ACE-Step music generation (audiocore-specific)
//
// Scope: TTS + music generation only. ASR / transcriptions are explicitly
// out of scope (use CrispASR or whisper.cpp directly).
//
// Single-process, one loaded model per id. Concurrency = N configured models;
// requests on the same model serialize through the session's backend.
//
// The HTTP routing, WAV encoders, and handler logic live in server.cpp so
// tests can drive them without forking this binary.

#include <atomic>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>
#include <httplib.h>

#include "audiocore/framework/core/backend.h"
#include "audiocore/framework/core/session.h"
#include "audiocore/framework/runtime/discovery.h"
#include "audiocore/framework/runtime/registry.h"
#include "audiocore/models/ace_step/family.h"
#include "audiocore/models/moss_tts/family.h"
#include "audiocore/models/qwen3_tts/family.h"
#include "audiocore/server/server.h"

namespace {

using audiocore::Session;
using audiocore::LoadOptions;
using audiocore::BackendConfig;
using audiocore::BackendKind;
using audiocore::FamilyRegistry;
using audiocore::ModelSlot;
using nlohmann::json;

// Each family loader.cpp exposes one of these. Calling them explicitly here
// defeats the linker stripping the static registrars out of the framework
// archive (which would leave FamilyRegistry empty at runtime). Add one line
// per new family.
extern "C" void audiocore_register_moss_tts();
extern "C" void audiocore_register_ace_step();
extern "C" void audiocore_register_qwen3_tts();

void register_all_families() {
    audiocore_register_moss_tts();
    audiocore_register_ace_step();
    audiocore_register_qwen3_tts();
}

struct ConfigModel {
    std::string id;
    std::string family;
    std::string path;
    std::string backend = "ggml_cuda";
    LoadOptions load_options;
};

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    int device = 0;
    int threads = 1;
    // llama.cpp-style auto-discovery root. When `models` is empty and this
    // is set, the server walks the directory and registers one model per
    // subdir whose name matches a registered family. See discovery.h.
    std::string model_dir;
    std::vector<ConfigModel> models;
};

BackendKind parse_backend(const std::string& s) {
    if (s == "ggml_cuda")   return BackendKind::ggml_cuda;
    if (s == "ggml_cpu")    return BackendKind::ggml_cpu;
    if (s == "ggml_vulkan") return BackendKind::ggml_vulkan;
    if (s == "ggml_metal")  return BackendKind::ggml_metal;
    return BackendKind::ggml_cuda;
}

bool load_config(const std::string& path, ServerConfig& out) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::fprintf(stderr, "could not open config '%s'\n", path.c_str());
        return false;
    }
    json j;
    try { f >> j; }
    catch (const std::exception& e) {
        std::fprintf(stderr, "config parse error: %s\n", e.what());
        return false;
    }
    out.host   = j.value("host", out.host);
    out.port   = j.value("port", out.port);
    out.device = j.value("device", out.device);
    out.threads= j.value("threads", out.threads);
    out.model_dir = j.value("model_dir", "");
    // `models` is optional when `model_dir` is set — discovery fills it in
    // after load_config returns. If `models` IS present it must be an array.
    if (j.contains("models")) {
        if (!j["models"].is_array()) {
            std::fprintf(stderr, "config 'models' must be an array\n");
            return false;
        }
        for (const auto& m : j["models"]) {
            ConfigModel cm;
            cm.id       = m.value("id", "");
            cm.family   = m.value("family", "");
            cm.path     = m.value("path", "");
            cm.backend  = m.value("backend", cm.backend);
            if (m.contains("voice_path"))
                cm.load_options.voice_path = m["voice_path"].get<std::string>();
            if (m.contains("language"))
                cm.load_options.language = m["language"].get<std::string>();
            if (m.contains("extras") && m["extras"].is_object()) {
                for (auto& [key, val] : m["extras"].items())
                    cm.load_options.extras[key] = val.get<std::string>();
            }
            if (cm.id.empty() || cm.family.empty() || cm.path.empty()) {
                std::fprintf(stderr, "model entry missing id/family/path\n");
                return false;
            }
            out.models.push_back(std::move(cm));
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string config_path;
    std::string model_dir_override;
    std::string model_override;      // --model: llama.cpp-style primary path
    std::string family_override;     // --family: bypass family inference
    std::string alias_override;      // --alias: rename the loaded model's id
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--config" && i + 1 < argc)        config_path        = argv[++i];
        else if (a == "--model-dir" && i + 1 < argc) model_dir_override = argv[++i];
        else if (a == "--model" && i + 1 < argc)     model_override     = argv[++i];
        else if (a == "--family" && i + 1 < argc)    family_override    = argv[++i];
        else if (a == "--alias" && i + 1 < argc)     alias_override     = argv[++i];
    }
    if (config_path.empty()) {
        std::fprintf(stderr,
            "usage: %s --config server.json [--model /path] [--family name] "
            "[--model-dir /path] [--alias name]\n",
            argv[0]);
        return 1;
    }

    ServerConfig cfg;
    if (!load_config(config_path, cfg)) return 1;

    // CLI overrides beat JSON. Lets operators point at a different models
    // root without editing the config (mirrors llama.cpp's flag precedence).
    if (!model_dir_override.empty()) cfg.model_dir = model_dir_override;

    register_all_families();   // pulls in moss_tts, ace_step, …

    // ── Model resolution ────────────────────────────────────────────────
    // llama.cpp contract: --model is the primary path. Exactly one model
    // loads; to swap, restart the container with a different path. When set,
    // --model overrides both the JSON models array and --model-dir.
    if (!model_override.empty()) {
        std::string err;
        auto m = audiocore::from_single_path(model_override, family_override, &err);
        if (!m) {
            std::fprintf(stderr, "model: %s\n", err.c_str());
            return 1;
        }
        ConfigModel cm;
        cm.id      = std::move(m->id);
        cm.family  = std::move(m->family);
        cm.path    = std::move(m->path);
        cm.backend = std::move(m->backend);
        cm.load_options = std::move(m->load_options);
        cfg.models.clear();
        cfg.models.push_back(std::move(cm));
        cfg.model_dir.clear();
        std::fprintf(stderr,
            "audiocore_server: --model %s (family=%s, id=%s)\n",
            model_override.c_str(),
            cfg.models.front().family.c_str(),
            cfg.models.front().id.c_str());
    }
    // Auto-discovery fallback: walks --model-dir when no explicit --model
    // and no JSON models array are configured. The per-family loaders do
    // their own file resolution within each discovered directory.
    else if (cfg.models.empty() && !cfg.model_dir.empty()) {
        std::string err;
        auto discovered = audiocore::discover_models(cfg.model_dir, &err);
        if (!err.empty()) {
            std::fprintf(stderr, "discovery: %s\n", err.c_str());
            return 1;
        }
        for (auto& d : discovered) {
            ConfigModel cm;
            cm.id           = std::move(d.id);
            cm.family       = std::move(d.family);
            cm.path         = std::move(d.path);
            cm.backend      = std::move(d.backend);
            cm.load_options = std::move(d.load_options);
            cfg.models.push_back(std::move(cm));
        }
        std::fprintf(stderr, "audiocore_server: discovered %zu model(s) under %s\n",
                     cfg.models.size(), cfg.model_dir.c_str());
    }

    // Phase 4: --alias renames the loaded model's id. Matches llama.cpp's
    // --alias: useful for OpenAI client compatibility where the client
    // hardcodes a model name like "tts-1" or "whisper-1". Only meaningful
    // in single-model mode — multi-model configs need per-entry ids and
    // can't share one alias.
    if (!alias_override.empty()) {
        if (cfg.models.size() != 1) {
            std::fprintf(stderr,
                "--alias requires exactly one model; current config has %zu. "
                "Drop the alias, or switch to --model for single-model mode.\n",
                cfg.models.size());
            return 1;
        }
        std::fprintf(stderr, "audiocore_server: aliasing '%s' → '%s'\n",
                     cfg.models.front().id.c_str(), alias_override.c_str());
        cfg.models.front().id = alias_override;
    }

    // Instantiate one Session per configured model id. Each lives behind a
    // mutex so concurrent requests on the same model serialize.
    auto slots = std::make_shared<
        std::unordered_map<std::string, std::shared_ptr<ModelSlot>>>();
    for (const ConfigModel& m : cfg.models) {
        auto sess = FamilyRegistry::instance().create(m.family);
        if (!sess) {
            std::fprintf(stderr, "unknown family '%s' for model id '%s'\n",
                         m.family.c_str(), m.id.c_str());
            return 1;
        }
        BackendConfig bc{
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
        auto slot = std::make_shared<ModelSlot>();
        slot->session = std::move(sess);
        (*slots)[m.id] = slot;
        std::fprintf(stderr, "audiocore_server: loaded '%s' (%s)\n",
                     m.id.c_str(), m.family.c_str());
    }
    if (slots->empty()) {
        // Boots anyway — /health and /v1/models work, /v1/audio/* will return
        // 404 until a model is loaded. Matches the llama.cpp server pattern:
        // listen first, reject per-request when no model is bound. Lets the
        // container image start cleanly for smoke tests (AUDIOCORE_ALLOW_EMPTY)
        // and for orchestrators that mount models after first boot.
        std::fprintf(stderr,
            "audiocore_server: WARNING — no models configured. "
            "Booting anyway; /v1/audio/* will return 404 until a model is loaded.\n");
    }

    auto svr = audiocore::build_server(slots);
    svr->set_logger([](const httplib::Request& req, const httplib::Response&) {
        std::fprintf(stderr, "%s %s\n", req.method.c_str(), req.path.c_str());
    });

    std::fprintf(stderr, "audiocore_server: listening on %s:%d\n",
                 cfg.host.c_str(), cfg.port);
    if (!svr->listen(cfg.host, cfg.port)) {
        std::fprintf(stderr, "failed to bind %s:%d\n",
                     cfg.host.c_str(), cfg.port);
        return 1;
    }
    return 0;
}
