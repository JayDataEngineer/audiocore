#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <sys/wait.h>
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
extern "C" void audiocore_register_moss_sfx_v2();

void register_all_families() {
    audiocore_register_moss_tts();
    audiocore_register_ace_step();
    audiocore_register_qwen3_tts();
    audiocore_register_moss_sfx_v2();
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
    // Lazy mode: register every model but load NONE at startup. Models are
    // materialised on first /v1/models/load request from the webapp. Keeps
    // VRAM free on small boxes and lets the server boot past a broken file.
    bool lazy_models = false;
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
    out.lazy_models = j.value("lazy_models", false);
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
    bool lazy_flag = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--config" && i + 1 < argc)        config_path        = argv[++i];
        else if (a == "--model-dir" && i + 1 < argc) model_dir_override = argv[++i];
        else if (a == "--model" && i + 1 < argc)     model_override     = argv[++i];
        else if (a == "--family" && i + 1 < argc)    family_override    = argv[++i];
        else if (a == "--alias" && i + 1 < argc)     alias_override     = argv[++i];
        else if (a == "--lazy")                      lazy_flag          = true;
    }
    if (config_path.empty()) {
        std::fprintf(stderr,
            "usage: %s --config server.json [--model /path] [--family name] "
            "[--model-dir /path] [--alias name] [--lazy]\n",
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
    // ALL models are registered in the slots map. For each FAMILY, only the
    // first model is loaded eagerly; subsequent models of the same family
    // are registered but NOT loaded — the webapp loads them on demand via
    // /v1/models/load (after unloading the current one to free VRAM).
    // This applies to ACE-Step (turbo/base/scrag), qwen3_tts
    // (CustomVoice/VoiceDesign), and any other multi-variant family.
    // Single-model families (e.g. moss_sfx) are always eager.
    //
    auto slots = std::make_shared<
        std::unordered_map<std::string, std::shared_ptr<ModelSlot>>>();
    // --lazy / "lazy_models": defer ALL models (load on demand from webapp).
    const bool lazy_mode = cfg.lazy_models || lazy_flag;

    // Track which families already have an eagerly-loaded representative.
    std::set<std::string> eager_families;

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

        // First model of each family is eager; the rest wait for webapp load.
        // In lazy mode ALL models defer — nothing touches VRAM until the
        // webapp requests a model via /v1/models/load.
        const bool defer = lazy_mode || !eager_families.insert(m.family).second;

        auto slot = std::make_shared<ModelSlot>();
        slot->load_req = req;
        slot->loaded = false;

        if (!defer) {
            try {
                auto loaded = registry->load(req);
                slot->model = std::move(loaded);
                slot->session = slot->model->create_session(
                    {VoiceTaskKind::Tts}, {});
                slot->loaded = true;
            } catch (const std::exception& e) {
                // Non-fatal: a corrupt/partial GGUF must NOT take the server
                // down. Register the slot as load-on-demand so the webapp can
                // surface the error per-model and everything else still works.
                std::fprintf(stderr,
                    "audiocore_server: WARNING load failed for '%s': %s "
                    "(registered load-on-demand; booting anyway)\n",
                    m.id.c_str(), e.what());
            }
            if (slot->loaded)
                std::fprintf(stderr, "audiocore_server: loaded '%s' (%s)\n",
                             m.id.c_str(), m.family.c_str());
            else
                std::fprintf(stderr,
                    "audiocore_server: registered '%s' (%s) [load on demand]\n",
                    m.id.c_str(), m.family.c_str());
        } else {
            std::fprintf(stderr,
                "audiocore_server: registered '%s' (%s) [load on demand]\n",
                m.id.c_str(), m.family.c_str());
        }
        (*slots)[m.id] = slot;
    }

    if (slots->empty()) {
        std::fprintf(stderr,
            "audiocore_server: WARNING — no models configured. "
            "Booting anyway; /v1/audio/* will return 404 until a model is loaded.\n");
    }


    // Pass registry + model dir so the webapp can load/unload models at runtime.
    // cfg.model_dir is the discovery scan root (may be empty in single-model mode).
    auto svr = audiocore::build_server(slots, cfg.clips_dir, registry, cfg.model_dir);

    // Logger: skip per-byte noise from static-asset GETs (the browser fetches
    // /, /style.css, /app.js on every page load) and from /health probes
    // (which fire periodically from the webapp). Everything else — API
    // calls, uploads, errors — still logs one line per request.
    svr->set_logger([](const httplib::Request& req, const httplib::Response& res) {
        if (req.method == "GET") {
            if (req.path == "/" || req.path == "/style.css" || req.path == "/app.js" ||
                req.path == "/health" || req.path == "/favicon.ico") {
                return;
            }
            if (req.path.rfind("/v1/clips/raw/", 0) == 0 ||
                req.path.rfind("/v1/voices/raw/", 0) == 0) {
                return;  // audio playback range requests would flood the log
            }
        }
        // Annotate non-2xx so failures are easy to grep in production logs.
        const int s = res.status;
        if (s >= 400) {
            std::fprintf(stderr, "[%d] %s %s\n", s, req.method.c_str(), req.path.c_str());
        } else {
            std::fprintf(stderr, "%s %s\n", req.method.c_str(), req.path.c_str());
        }
    });

    std::fprintf(stderr, "\n  audiocore server\n");
    std::fprintf(stderr, "  ─────────────────────────────────────────\n");
    std::fprintf(stderr, "  version   : %s\n",
#ifndef AUDIOCORE_VERSION
        "0.0.0-dev"
#else
        AUDIOCORE_VERSION
#endif
    );
    std::fprintf(stderr, "  listen    : http://%s:%d\n",
                 cfg.host.c_str(), cfg.port);
    std::fprintf(stderr, "  webapp    : ");
#ifdef AUDIOCORE_HAS_WEBAPP
    std::fprintf(stderr, "embedded (/)\n");
#else
    std::fprintf(stderr, "not built (define ENGINE_BUILD_WEBAPP=ON)\n");
#endif
    std::fprintf(stderr, "  clips_dir : %s\n",
                 cfg.clips_dir.empty() ? "(none)" : cfg.clips_dir.c_str());
    std::fprintf(stderr, "  models    : %zu configured (%zu loaded)\n",
                 cfg.models.size(),
                 std::count_if(slots->begin(), slots->end(),
                               [](const auto& kv) { return kv.second->loaded; }));
    std::fprintf(stderr, "  (Ctrl-C to stop)\n\n");

    // ── Graceful shutdown ─────────────────────────────────────────────────
    // SIGINT/SIGTERM ask httplib to stop accepting and finish in-flight
    // responses, then listen() returns and we exit 0. This matters in
    // containers: the orchestrator sends SIGTERM and waits a grace period
    // before SIGKILL — without this handler, long music/TTS generations get
    // hard-killed and the client sees a broken connection.
    //
    // We can't call svr->stop() directly from the signal handler (it's not
    // async-signal-safe — httplib acquires mutexes internally), so a tiny
    // watcher thread polls a sig_atomic flag set by the handler and invokes
    // stop() from normal context.
    static std::atomic<bool> g_shutdown_requested{false};
    auto relay_signal = [](int sig) {
        (void)sig;
        g_shutdown_requested.store(true);
    };
    std::signal(SIGINT, relay_signal);
    std::signal(SIGTERM, relay_signal);
    std::thread shutdown_watcher([&svr]() {
        while (!g_shutdown_requested.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        std::fprintf(stderr, "\n  shutdown requested — finishing in-flight requests…\n");
        svr->stop();
    });
    shutdown_watcher.detach();

    if (!svr->listen(cfg.host, cfg.port)) {
        std::fprintf(stderr, "failed to bind %s:%d\n",
                     cfg.host.c_str(), cfg.port);
        return 1;
    }
    return 0;
}
