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
#include "audiocore/server/acestep_proxy.h"
#include "audiocore/server/qwen3tts_proxy.h"

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
    // Reference ace-server proxy configuration.
    struct AceProxyCfg {
        std::string server_url = "http://localhost:8085";
        std::string binary;      // path to ace-server executable
        std::string models_dir;  // directory of GGUF files for ace-server
        std::string library_dir; // LD_LIBRARY_PATH for ace-server
        std::vector<std::string> extra_args;
    };
    bool has_proxy = false;
    AceProxyCfg proxy;
    // Reference qwen_tts server subprocess configuration.
    struct Qwen3TtsProxyCfg {
        std::string server_url = "http://localhost:8086";
        std::string binary;      // path to qwen_tts executable
        std::string model_dir;   // model directory for qwen_tts (-d)
        std::string library_dir; // LD_LIBRARY_PATH for qwen_tts
        std::vector<std::string> extra_args;
        // CLI runner paths (for voice-bearing requests via shell-out).
        std::string customvoice_dir;  // 1.7B-CustomVoice model dir
        std::string voicedesign_dir;  // VoiceDesign model dir
        std::string base_dir;         // 1.7B-Base model dir
        std::string voices_dir;       // saved .qvoice files directory
        bool        use_gpu = false;  // --backend cuda for CLI calls
        int         gpu_device = 0;   // CUDA_VISIBLE_DEVICES
    };
    bool has_qwen3tts_proxy = false;
    Qwen3TtsProxyCfg qwen3tts_proxy;
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
    // Parse acestep_proxy config (reference ace-server subprocess).
    if (j.contains("acestep_proxy") && j["acestep_proxy"].is_object()) {
        const auto& p = j["acestep_proxy"];
        out.has_proxy = true;
        out.proxy.server_url  = p.value("server_url", out.proxy.server_url);
        out.proxy.binary      = p.value("binary", "");
        out.proxy.models_dir  = p.value("models_dir", "");
        out.proxy.library_dir = p.value("library_dir", "");
        if (p.contains("extra_args") && p["extra_args"].is_array()) {
            for (const auto& a : p["extra_args"])
                out.proxy.extra_args.push_back(a.get<std::string>());
        }
    }
    // Parse qwen3tts_proxy config (reference qwen_tts subprocess).
    if (j.contains("qwen3tts_proxy") && j["qwen3tts_proxy"].is_object()) {
        const auto& p = j["qwen3tts_proxy"];
        out.has_qwen3tts_proxy = true;
        out.qwen3tts_proxy.server_url  = p.value("server_url", out.qwen3tts_proxy.server_url);
        out.qwen3tts_proxy.binary      = p.value("binary", "");
        out.qwen3tts_proxy.model_dir   = p.value("model_dir", "");
        out.qwen3tts_proxy.library_dir = p.value("library_dir", "");
        if (p.contains("extra_args") && p["extra_args"].is_array()) {
            for (const auto& a : p["extra_args"])
                out.qwen3tts_proxy.extra_args.push_back(a.get<std::string>());
        }
        // CLI runner paths (voice-bearing requests).
        out.qwen3tts_proxy.customvoice_dir = p.value("customvoice_dir", "");
        out.qwen3tts_proxy.voicedesign_dir = p.value("voicedesign_dir", "");
        out.qwen3tts_proxy.base_dir        = p.value("base_dir", "");
        out.qwen3tts_proxy.voices_dir      = p.value("voices_dir", "");
        out.qwen3tts_proxy.use_gpu         = p.value("use_gpu", false);
        out.qwen3tts_proxy.gpu_device      = p.value("gpu_device", 0);
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
    // Models with backend="acestep_proxy" are NOT loaded locally — they are
    // proxied to the reference ace-server. They are registered in the proxy
    // map instead of the slots map.
    auto slots = std::make_shared<
        std::unordered_map<std::string, std::shared_ptr<ModelSlot>>>();
    auto proxies = std::make_shared<
        std::unordered_map<std::string, audiocore::AceStepProxyConfig>>();
    auto qwen3_proxies = std::make_shared<
        std::unordered_map<std::string, audiocore::Qwen3TtsProxyConfig>>();
    // --lazy / "lazy_models": defer ALL models (load on demand from webapp).
    const bool lazy_mode = cfg.lazy_models || lazy_flag;

    // Track which families already have an eagerly-loaded representative.
    std::set<std::string> eager_families;

    for (const ConfigModel& m : cfg.models) {
        // Proxy models skip local loading entirely.
        if (m.backend == "acestep_proxy") {
            audiocore::AceStepProxyConfig pc;
            pc.ace_server_url = cfg.has_proxy ? cfg.proxy.server_url : "http://localhost:8085";
            // Extract dit_variant from extras if present.
            auto it = m.load_options.extras.find("dit_variant");
            if (it != m.load_options.extras.end())
                pc.dit_variant = it->second;
            // Extract vae_override from extras if present (ScragVAE swap-in).
            auto vit = m.load_options.extras.find("vae_override");
            if (vit != m.load_options.extras.end())
                pc.vae_override = vit->second;
            (*proxies)[m.id] = pc;
            std::fprintf(stderr,
                "audiocore_server: proxy '%s' → ace-server (%s, dit=%s, vae=%s)\n",
                m.id.c_str(), pc.ace_server_url.c_str(),
                pc.dit_variant.empty() ? "auto" : pc.dit_variant.c_str(),
                pc.vae_override.empty() ? "default" : pc.vae_override.c_str());
            continue;
        }
        if (m.backend == "qwen3tts_proxy") {
            audiocore::Qwen3TtsProxyConfig pc;
            pc.server_url = cfg.has_qwen3tts_proxy
                ? cfg.qwen3tts_proxy.server_url
                : "http://localhost:8086";
            pc.model_dir  = cfg.has_qwen3tts_proxy
                ? cfg.qwen3tts_proxy.model_dir
                : m.path;
            // CLI runner paths (voice-bearing requests).
            if (cfg.has_qwen3tts_proxy) {
                pc.binary_path     = cfg.qwen3tts_proxy.binary;
                pc.customvoice_dir = cfg.qwen3tts_proxy.customvoice_dir;
                pc.voicedesign_dir = cfg.qwen3tts_proxy.voicedesign_dir;
                pc.base_dir        = cfg.qwen3tts_proxy.base_dir;
                pc.voices_dir      = cfg.qwen3tts_proxy.voices_dir;
                pc.use_gpu         = cfg.qwen3tts_proxy.use_gpu;
                pc.gpu_device      = cfg.qwen3tts_proxy.gpu_device;
            }
            (*qwen3_proxies)[m.id] = pc;
            std::fprintf(stderr,
                "audiocore_server: proxy '%s' → qwen_tts-server (%s)\n",
                m.id.c_str(), pc.server_url.c_str());
            continue;
        }
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

    // ── Start the reference ace-server subprocess (if configured) ──────────
    // The proxy routes /v1/audio/music for backend="acestep_proxy" models to
    // this external process. We fork/exec it with the configured GGUF
    // directory + LD_LIBRARY_PATH, then block until /health responds.
    pid_t ace_server_pid = -1;
    if (cfg.has_proxy && !cfg.proxy.binary.empty()) {
        std::fprintf(stderr,
            "audiocore_server: starting ace-server subprocess\n"
            "  binary     : %s\n"
            "  models_dir : %s\n"
            "  server_url : %s\n",
            cfg.proxy.binary.c_str(),
            cfg.proxy.models_dir.c_str(),
            cfg.proxy.server_url.c_str());

        ace_server_pid = fork();
        if (ace_server_pid < 0) {
            std::perror("fork");
            return 1;
        }
        if (ace_server_pid == 0) {
            // ── Child process ──
            // Set LD_LIBRARY_PATH so ace-server finds its bundled ggml backend.
            if (!cfg.proxy.library_dir.empty()) {
                const char* old = std::getenv("LD_LIBRARY_PATH");
                std::string lp = cfg.proxy.library_dir;
                if (old && *old) lp += std::string(":") + old;
                setenv("LD_LIBRARY_PATH", lp.c_str(), 1);
            }
            // Parse the port out of server_url to pass via --port.
            // ace-server listens on 8085 by default; honor the configured URL.
            std::string port_str;
            {
                auto colon = cfg.proxy.server_url.rfind(':');
                if (colon != std::string::npos) {
                    // Strip any trailing slash from the port portion.
                    std::string tail = cfg.proxy.server_url.substr(colon + 1);
                    auto slash = tail.find('/');
                    port_str = slash == std::string::npos ? tail : tail.substr(0, slash);
                }
            }

            std::vector<std::string> argv_str;
            argv_str.push_back(cfg.proxy.binary);
            if (!cfg.proxy.models_dir.empty()) {
                argv_str.push_back("--models");
                argv_str.push_back(cfg.proxy.models_dir);
            }
            if (!port_str.empty()) {
                argv_str.push_back("--port");
                argv_str.push_back(port_str);
            }
            for (const auto& a : cfg.proxy.extra_args)
                argv_str.push_back(a);

            std::vector<char*> argv_c;
            argv_c.reserve(argv_str.size() + 1);
            for (auto& s : argv_str) argv_c.push_back(&s[0]);
            argv_c.push_back(nullptr);

            // Redirect child stdout/stderr to the parent's so ace-server logs
            // surface in our terminal alongside our own.
            execv(cfg.proxy.binary.c_str(), argv_c.data());
            // execv only returns on failure.
            std::perror("execv(ace-server)");
            std::_Exit(127);
        }

        // ── Parent: wait for ace-server /health ──
        audiocore::AceStepProxyConfig hc;
        hc.ace_server_url = cfg.proxy.server_url;
        hc.timeout_seconds = 60;
        bool healthy = false;
        for (int attempt = 0; attempt < 120; ++attempt) {
            // Bail early if the child died.
            int status = 0;
            pid_t w = waitpid(ace_server_pid, &status, WNOHANG);
            if (w == ace_server_pid || (w == -1 && errno != EINTR)) {
                std::fprintf(stderr,
                    "audiocore_server: ace-server process exited (status=%d) — "
                    "check model paths and library_dir\n", status);
                ace_server_pid = -1;
                break;
            }
            if (audiocore::acestep_proxy_health(hc)) {
                healthy = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        if (healthy) {
            std::fprintf(stderr,
                "audiocore_server: ace-server healthy (pid=%d)\n", ace_server_pid);
        } else if (ace_server_pid > 0) {
            std::fprintf(stderr,
                "audiocore_server: WARNING — ace-server did not become healthy "
                "within 60s; proxy requests will fail until it starts.\n");
        }
    } else if (!proxies->empty()) {
        // Proxies are configured but no binary — assume ace-server runs
        // out-of-band (managed by a supervisor / separate terminal).
        audiocore::AceStepProxyConfig hc;
        hc.ace_server_url = cfg.has_proxy ? cfg.proxy.server_url : "http://localhost:8085";
        if (audiocore::acestep_proxy_health(hc)) {
            std::fprintf(stderr,
                "audiocore_server: ace-server already running at %s\n",
                hc.ace_server_url.c_str());
        } else {
            std::fprintf(stderr,
                "audiocore_server: WARNING — %zu proxy model(s) configured but "
                "ace-server is not reachable at %s. Start it manually or add an "
                "\"acestep_proxy.binary\" to the config.\n",
                proxies->size(), hc.ace_server_url.c_str());
        }
    }

    // ── Start the reference qwen_tts subprocess (if configured) ──────────
    // The proxy routes /v1/audio/speech for backend="qwen3tts_proxy" models to
    // this external process. We fork/exec it with the model dir, then block
    // until /v1/health responds.
    pid_t qwen3tts_server_pid = -1;
    if (cfg.has_qwen3tts_proxy && !cfg.qwen3tts_proxy.binary.empty()) {
        std::fprintf(stderr,
            "audiocore_server: starting qwen_tts-server subprocess\n"
            "  binary     : %s\n"
            "  model_dir  : %s\n"
            "  server_url : %s\n",
            cfg.qwen3tts_proxy.binary.c_str(),
            cfg.qwen3tts_proxy.model_dir.c_str(),
            cfg.qwen3tts_proxy.server_url.c_str());

        qwen3tts_server_pid = fork();
        if (qwen3tts_server_pid < 0) {
            std::perror("fork");
            return 1;
        }
        if (qwen3tts_server_pid == 0) {
            // ── Child process ──
            if (!cfg.qwen3tts_proxy.library_dir.empty()) {
                const char* old = std::getenv("LD_LIBRARY_PATH");
                std::string lp = cfg.qwen3tts_proxy.library_dir;
                if (old && *old) lp += std::string(":") + old;
                setenv("LD_LIBRARY_PATH", lp.c_str(), 1);
            }
            // Parse the port out of server_url to pass via --serve.
            std::string port_str;
            {
                std::string u = cfg.qwen3tts_proxy.server_url;
                auto colon = u.rfind(':');
                if (colon != std::string::npos) {
                    std::string tail = u.substr(colon + 1);
                    auto slash = tail.find('/');
                    port_str = slash == std::string::npos ? tail : tail.substr(0, slash);
                }
            }

            std::vector<std::string> argv_str;
            argv_str.push_back(cfg.qwen3tts_proxy.binary);
            if (!cfg.qwen3tts_proxy.model_dir.empty()) {
                argv_str.push_back("-d");
                argv_str.push_back(cfg.qwen3tts_proxy.model_dir);
            }
            if (!port_str.empty()) {
                argv_str.push_back("--serve");
                argv_str.push_back(port_str);
            }
            for (const auto& a : cfg.qwen3tts_proxy.extra_args)
                argv_str.push_back(a);

            std::vector<char*> argv_c;
            argv_c.reserve(argv_str.size() + 1);
            for (auto& s : argv_str) argv_c.push_back(&s[0]);
            argv_c.push_back(nullptr);

            execv(cfg.qwen3tts_proxy.binary.c_str(), argv_c.data());
            std::perror("execv(qwen_tts)");
            std::_Exit(127);
        }

        // ── Parent: wait for qwen_tts /v1/health ──
        audiocore::Qwen3TtsProxyConfig hc;
        hc.server_url = cfg.qwen3tts_proxy.server_url;
        hc.timeout_seconds = 120;
        bool healthy = false;
        for (int attempt = 0; attempt < 240; ++attempt) {
            int status = 0;
            pid_t w = waitpid(qwen3tts_server_pid, &status, WNOHANG);
            if (w == qwen3tts_server_pid || (w == -1 && errno != EINTR)) {
                std::fprintf(stderr,
                    "audiocore_server: qwen_tts-server process exited (status=%d) — "
                    "check model paths\n", status);
                qwen3tts_server_pid = -1;
                break;
            }
            if (audiocore::qwen3tts_proxy_health(hc)) {
                healthy = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        if (healthy) {
            std::fprintf(stderr,
                "audiocore_server: qwen_tts-server healthy (pid=%d)\n",
                qwen3tts_server_pid);
        } else if (qwen3tts_server_pid > 0) {
            std::fprintf(stderr,
                "audiocore_server: WARNING — qwen_tts-server did not become healthy "
                "within 120s; proxy requests will fail until it starts.\n");
        }
    } else if (!qwen3_proxies->empty()) {
        audiocore::Qwen3TtsProxyConfig hc;
        hc.server_url = cfg.has_qwen3tts_proxy ? cfg.qwen3tts_proxy.server_url : "http://localhost:8086";
        if (audiocore::qwen3tts_proxy_health(hc)) {
            std::fprintf(stderr,
                "audiocore_server: qwen_tts-server already running at %s\n",
                hc.server_url.c_str());
        } else {
            std::fprintf(stderr,
                "audiocore_server: WARNING — %zu qwen3tts proxy model(s) configured "
                "but qwen_tts-server is not reachable at %s. Start it manually or "
                "add a \"qwen3tts_proxy.binary\" to the config.\n",
                qwen3_proxies->size(), hc.server_url.c_str());
        }
    }

    // Pass registry + model dir so the webapp can load/unload models at runtime.
    // cfg.model_dir is the discovery scan root (may be empty in single-model mode).
    auto svr = audiocore::build_server(slots, cfg.clips_dir, registry, cfg.model_dir,
                                       proxies, qwen3_proxies);

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
