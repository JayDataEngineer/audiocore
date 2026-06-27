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

#include <atomic>
#include <cstdio>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>
#include <httplib.h>

#include "audiocore/framework/core/backend.h"
#include "audiocore/framework/core/session.h"
#include "audiocore/framework/runtime/registry.h"
#include "audiocore/models/ace_step/family.h"
#include "audiocore/models/moss_tts/family.h"

namespace {

using audiocore::Session;
using audiocore::LoadOptions;
using audiocore::BackendConfig;
using audiocore::BackendKind;
using audiocore::FamilyRegistry;
using nlohmann::json;

// Each family loader.cpp exposes one of these. Calling them explicitly here
// defeats the linker stripping the static registrars out of the framework
// archive (which would leave FamilyRegistry empty at runtime). Add one line
// per new family.
extern "C" void audiocore_register_moss_tts();
extern "C" void audiocore_register_ace_step();

void register_all_families() {
    audiocore_register_moss_tts();
    audiocore_register_ace_step();
}

struct ConfigModel {
    std::string id;
    std::string family;
    std::string path;
    std::string backend = "ggml_cuda";
    LoadOptions load_options;
};

struct ServerConfig {
    std::string host = "127.0.0.1";
    int port = 8080;
    int device = 0;
    int threads = 1;
    std::vector<ConfigModel> models;
};

BackendKind parse_backend(const std::string& s) {
    if (s == "ggml_cuda")   return BackendKind::ggml_cuda;
    if (s == "ggml_cpu")    return BackendKind::ggml_cpu;
    if (s == "ggml_vulkan") return BackendKind::ggml_vulkan;
    if (s == "ggml_metal")  return BackendKind::ggml_metal;
    if (s == "onnxruntime") return BackendKind::onnxruntime;
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
    if (!j.contains("models") || !j["models"].is_array()) {
        std::fprintf(stderr, "config missing 'models' array\n");
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
        if (cm.id.empty() || cm.family.empty() || cm.path.empty()) {
            std::fprintf(stderr, "model entry missing id/family/path\n");
            return false;
        }
        out.models.push_back(std::move(cm));
    }
    return true;
}

// Serializes a single session — we serialize requests per model because the
// underlying backend isn't thread-safe. The server can still handle N models
// in parallel (one request per model id).
struct ModelSlot {
    std::unique_ptr<Session> session;
    std::mutex               mtx;
};

// Tiny WAV writer for mono PCM16 → response body. Keeps the server binary
// dependency-free; we encode MP3 only when libmp3lame is linked (Phase 2).
std::string pcm_mono_to_wav(const std::vector<float>& pcm, int32_t sr) {
    std::ostringstream o;
    auto w16 = [&](uint16_t v) { o.put(v & 0xff); o.put((v >> 8) & 0xff); };
    auto w32 = [&](uint32_t v) {
        for (int i = 0; i < 4; i++) o.put((v >> (i * 8)) & 0xff);
    };
    const uint16_t channels = 1, bps = 16;
    const uint32_t data_bytes = static_cast<uint32_t>(pcm.size()) * 2;
    const uint32_t byte_rate  = sr * channels * bps / 8;
    o.write("RIFF", 4);
    w32(36 + data_bytes);
    o.write("WAVE", 4);
    o.write("fmt ", 4);  w32(16);  w16(1);
    w16(channels);  w32(sr);  w32(byte_rate);  w16(channels * bps / 8);  w16(bps);
    o.write("data", 4);  w32(data_bytes);
    for (float s : pcm) {
        if (s >  1.0f) s =  1.0f;
        if (s < -1.0f) s = -1.0f;
        w16(static_cast<int16_t>(s * 32767.0f));
    }
    return o.str();
}

std::string pcm_stereo_to_wav(const std::vector<float>& pcm, int32_t sr) {
    std::ostringstream o;
    auto w16 = [&](uint16_t v) { o.put(v & 0xff); o.put((v >> 8) & 0xff); };
    auto w32 = [&](uint32_t v) {
        for (int i = 0; i < 4; i++) o.put((v >> (i * 8)) & 0xff);
    };
    const uint16_t channels = 2, bps = 16;
    const uint32_t data_bytes = static_cast<uint32_t>(pcm.size()) * 2;
    const uint32_t byte_rate  = sr * channels * bps / 8;
    o.write("RIFF", 4);
    w32(36 + data_bytes);
    o.write("WAVE", 4);
    o.write("fmt ", 4);  w32(16);  w16(1);
    w16(channels);  w32(sr);  w32(byte_rate);  w16(channels * bps / 8);  w16(bps);
    o.write("data", 4);  w32(data_bytes);
    for (float s : pcm) {
        if (s >  1.0f) s =  1.0f;
        if (s < -1.0f) s = -1.0f;
        w16(static_cast<int16_t>(s * 32767.0f));
    }
    return o.str();
}

}  // namespace

int main(int argc, char** argv) {
    std::string config_path;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) config_path = argv[++i];
    }
    if (config_path.empty()) {
        std::fprintf(stderr, "usage: %s --config server.json\n", argv[0]);
        return 1;
    }

    ServerConfig cfg;
    if (!load_config(config_path, cfg)) return 1;

    register_all_families();   // pulls in moss_tts, ace_step, …

    // Instantiate one Session per configured model id. Each lives behind a
    // mutex so concurrent requests on the same model serialize.
    std::unordered_map<std::string, std::shared_ptr<ModelSlot>> slots;
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
        slots[m.id] = slot;
        std::fprintf(stderr, "audiocore_server: loaded '%s' (%s)\n",
                     m.id.c_str(), m.family.c_str());
    }
    if (slots.empty()) {
        std::fprintf(stderr, "no models configured — exiting\n");
        return 1;
    }

    httplib::Server svr;
    std::atomic<bool> shutting_down{false};

    svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        json j = {{"status", "ok"}};
        res.set_content(j.dump(), "application/json");
    });

    svr.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        json arr = json::array();
        for (const auto& [id, slot] : slots) {
            std::lock_guard<std::mutex> g(slot->mtx);
            arr.push_back({
                {"id", id},
                {"family", slot->session->family_name()},
                {"loaded", slot->session->loaded()},
            });
        }
        res.set_content(json{{"object", "list"}, {"data", arr}}.dump(),
                        "application/json");
    });

    // OpenAI-compatible TTS. Maps to moss_tts family.
    svr.Post("/v1/audio/speech", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }
        const std::string model_id = body.value("model", "");
        auto it = slots.find(model_id);
        if (it == slots.end()) {
            res.status = 404;
            res.set_content(R"({"error":"unknown model"})", "application/json");
            return;
        }
        auto& slot = it->second;
        std::lock_guard<std::mutex> g(slot->mtx);

        audiocore::moss::TtsRequest tr{};
        tr.text     = body.value("input", "");
        tr.language = body.value("language", "en");
        tr.voice_path = body.value("voice", "");
        if (body.contains("seed"))     tr.seed       = body["seed"].get<int32_t>();
        if (body.contains("temperature")) tr.temperature = body["temperature"].get<float>();
        if (body.contains("top_p"))    tr.top_p      = body["top_p"].get<float>();

        audiocore::moss::TtsResponse tresp;
        std::string err;
        if (!slot->session->run_tts(&tr, &tresp, &err)) {
            res.status = 500;
            json e = {{"error", err}};
            res.set_content(e.dump(), "application/json");
            return;
        }
        // OpenAI expects audio/mpeg by default; we ship WAV until MP3 is wired.
        res.set_content(pcm_mono_to_wav(tresp.pcm_mono, tresp.sampling_rate),
                        "audio/wav");
    });

    // ACE-Step music generation. Maps to ace_step family.
    svr.Post("/v1/audio/music", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }
        const std::string model_id = body.value("model", "");
        auto it = slots.find(model_id);
        if (it == slots.end()) {
            res.status = 404;
            res.set_content(R"({"error":"unknown model"})", "application/json");
            return;
        }
        auto& slot = it->second;
        std::lock_guard<std::mutex> g(slot->mtx);

        audiocore::acestep::MusicRequest mr;
        mr.caption  = body.value("caption", "");
        mr.lyrics   = body.value("lyrics", "");
        mr.duration = body.value("duration", 30.0f);
        if (body.contains("seed"))           mr.seed              = body["seed"].get<int32_t>();
        if (body.contains("guidance_scale")) mr.guidance_scale    = body["guidance_scale"].get<float>();
        if (body.contains("steps"))          mr.n_diffusion_steps = body["steps"].get<int32_t>();

        audiocore::acestep::MusicResponse mresp;
        std::string err;
        if (!slot->session->run_music(&mr, &mresp, &err)) {
            res.status = 500;
            json e = {{"error", err}};
            res.set_content(e.dump(), "application/json");
            return;
        }
        res.set_content(pcm_stereo_to_wav(mresp.pcm_stereo, mresp.sampling_rate),
                        "audio/wav");
    });

    svr.set_logger([](const httplib::Request& req, const httplib::Response&) {
        std::fprintf(stderr, "%s %s\n", req.method.c_str(), req.path.c_str());
    });

    std::fprintf(stderr, "audiocore_server: listening on %s:%d\n",
                 cfg.host.c_str(), cfg.port);
    if (!svr.listen(cfg.host, cfg.port)) {
        std::fprintf(stderr, "failed to bind %s:%d\n",
                     cfg.host.c_str(), cfg.port);
        return 1;
    }
    return 0;
}
