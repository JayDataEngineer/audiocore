#include <atomic>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>
#include <httplib.h>

#include "audiocore/framework/core/session.h"       // LoadOptions (ConfigModel uses it)
#include "audiocore/framework/runtime/discovery.h"
#include "audiocore/framework/runtime/registry.h"
#include "audiocore/models/ace_step/family.h"
#include "audiocore/models/moss_tts/family.h"
#include "audiocore/models/qwen3_tts/family.h"
#include "audiocore/server/server.h"

namespace {

using audiocore::ModelRegistry;
using audiocore::ILoadedModel;
using audiocore::IOfflineTaskSession;
using audiocore::ModelSlot;
using audiocore::VoiceTaskKind;
using nlohmann::json;

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
    audiocore::LoadOptions load_options;
};

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    int device = 0;
    int threads = 1;
    std::string model_dir;
    std::string clips_dir;
    std::vector<ConfigModel> models;
};

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
    out.clips_dir = j.value("clips_dir", "");
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
    std::string model_override;
    std::string family_override;
    std::string alias_override;
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
    if (!model_dir_override.empty()) cfg.model_dir = model_dir_override;

    // Default clips_dir to a "clips" subdirectory next to the binary.
    if (cfg.clips_dir.empty()) {
        char exe_buf[4096];
        ssize_t len = readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
        if (len > 0) {
            exe_buf[len] = '\0';
            cfg.clips_dir = std::string(exe_buf).substr(
                0, std::string(exe_buf).rfind('/')) + "/clips";
        } else {
            cfg.clips_dir = "clips";
        }
    }

    register_all_families();

    // ── Build the ModelRegistry ───────────────────────────────────────────
    auto registry = std::make_shared<audiocore::ModelRegistry>(
        audiocore::make_default_registry());
    std::fprintf(stderr, "audiocore_server: registry: %zu loader(s) [",
                 registry->size());
    for (const auto& f : registry->families())
        std::fprintf(stderr, " %s", f.c_str());
    std::fprintf(stderr, " ]\n");

    // ── Resolve model paths ───────────────────────────────────────────────
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

    // ── Load models via ModelRegistry → ILoadedModel → IOfflineTaskSession ──
    auto slots = std::make_shared<
        std::unordered_map<std::string, std::shared_ptr<ModelSlot>>>();
    for (const ConfigModel& m : cfg.models) {
        audiocore::ModelLoadRequest req;
        req.model_path = m.path;
        req.family_hint = m.family;
        for (const auto& [k, v] : m.load_options.extras)
            req.options[k] = v;
        if (!m.load_options.voice_path.empty())
            req.options["voice_path"] = m.load_options.voice_path;
        if (!m.load_options.language.empty())
            req.options["language"] = m.load_options.language;
        req.options["backend"] = m.backend;
        req.options["device"] = std::to_string(cfg.device);

        try {
            auto loaded = registry->load(req);
            auto session = loaded->create_session(
                {VoiceTaskKind::Tts}, {});

            auto slot = std::make_shared<ModelSlot>();
            slot->model = std::move(loaded);
            slot->session = std::move(session);
            (*slots)[m.id] = slot;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "load failed for '%s': %s\n",
                         m.id.c_str(), e.what());
            return 1;
        }
        std::fprintf(stderr, "audiocore_server: loaded '%s' (%s)\n",
                     m.id.c_str(), m.family.c_str());
    }

    if (slots->empty()) {
        std::fprintf(stderr,
            "audiocore_server: WARNING — no models configured. "
            "Booting anyway; /v1/audio/* will return 404 until a model is loaded.\n");
    }

    auto svr = audiocore::build_server(slots, cfg.clips_dir);
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
