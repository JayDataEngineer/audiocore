// server.cpp — HTTP server factory + WAV encoders (see server.h).

#include "audiocore/server/server.h"

#include <atomic>
#include <cstdio>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include "audiocore/models/ace_step/family.h"
#include "audiocore/models/moss_tts/family.h"

namespace audiocore {

using nlohmann::json;

// ---- WAV encoders --------------------------------------------------------
// Tiny WAV writer for PCM16 → response body. Keeps the server binary
// dependency-free; we encode MP3 only when libmp3lame is linked (Phase 2).

std::string pcm_mono_to_wav(const std::vector<float>& pcm, int32_t sr) {
    std::ostringstream o;
    auto w16 = [&](uint16_t v) { o.put(v & 0xff); o.put((v >> 8) & 0xff); };
    auto w32 = [&](uint32_t v) {
        for (int i = 0; i < 4; i++) o.put((v >> (i * 8)) & 0xff);
    };
    const uint16_t channels = 1, bps = 16;
    const uint32_t data_bytes = static_cast<uint32_t>(pcm.size()) * 2;
    const uint32_t byte_rate  = static_cast<uint32_t>(sr) * channels * bps / 8;
    o.write("RIFF", 4);
    w32(36 + data_bytes);
    o.write("WAVE", 4);
    o.write("fmt ", 4);  w32(16);  w16(1);
    w16(channels);  w32(static_cast<uint32_t>(sr));  w32(byte_rate);
    w16(channels * bps / 8);  w16(bps);
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
    const uint32_t byte_rate  = static_cast<uint32_t>(sr) * channels * bps / 8;
    o.write("RIFF", 4);
    w32(36 + data_bytes);
    o.write("WAVE", 4);
    o.write("fmt ", 4);  w32(16);  w16(1);
    w16(channels);  w32(static_cast<uint32_t>(sr));  w32(byte_rate);
    w16(channels * bps / 8);  w16(bps);
    o.write("data", 4);  w32(data_bytes);
    for (float s : pcm) {
        if (s >  1.0f) s =  1.0f;
        if (s < -1.0f) s = -1.0f;
        w16(static_cast<int16_t>(s * 32767.0f));
    }
    return o.str();
}

// ---- HTTP server factory -------------------------------------------------

std::shared_ptr<httplib::Server> build_server(
        std::shared_ptr<std::unordered_map<std::string, std::shared_ptr<ModelSlot>>> slots) {
    auto svr = std::make_shared<httplib::Server>();
    // Capture the slots map by value (shared_ptr copy) so the lambda's
    // lifetime matches the server's, not the caller's stack frame.
    auto slots_ref = slots;

    svr->Get("/health", [slots_ref](const httplib::Request&, httplib::Response& res) {
        json j = {{"status", "ok"}};
        res.set_content(j.dump(), "application/json");
    });

    svr->Get("/v1/models", [slots_ref](const httplib::Request&, httplib::Response& res) {
        json arr = json::array();
        for (const auto& [id, slot] : *slots_ref) {
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

    svr->Post("/v1/audio/speech",
              [slots_ref](const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }
        const std::string model_id = body.value("model", "");
        auto it = slots_ref->find(model_id);
        if (it == slots_ref->end()) {
            res.status = 404;
            res.set_content(R"({"error":"unknown model"})", "application/json");
            return;
        }
        auto& slot = it->second;
        std::lock_guard<std::mutex> g(slot->mtx);

        // The request shape is whatever the loaded family's run_tts expects.
        // We pass a generic bag of strings; families that don't use a
        // particular field just ignore it. The mock family in tests reads
        // "input" directly.
        //
        // NOTE: production families (moss_tts, ace_step) down-cast inside
        // their run_tts to their concrete request struct. We mirror the
        // fields moss_tts reads here so a generic Session* is sufficient
        // for the framework path. Any family-specific extras go through the
        // TtsRequest struct in the family header.
        moss::TtsRequest tr{};
        tr.text       = body.value("input", "");
        tr.language   = body.value("language", "en");
        tr.voice_path = body.value("voice", "");
        if (body.contains("seed"))        tr.seed       = body["seed"].get<int32_t>();
        if (body.contains("temperature")) tr.temperature= body["temperature"].get<float>();
        if (body.contains("top_p"))       tr.top_p      = body["top_p"].get<float>();

        // The ONNX decoder path can be overridden per-request.
        // Pass it via LoadOptions::extras so the session picks it up at load time.
        // NOTE: the session was already loaded by the time this handler runs;
        // the per-request handling is stubbed. For true per-request override,
        // add decoder_onnx to the TtsRequest struct itself. Here we just
        // accept the field for logging; the actual path comes from load.
        if (body.contains("decoder_onnx")) {
            // ignored at runtime unless the session re-reads it
        }

        moss::TtsResponse tresp;
        std::string err;
        if (!slot->session->run_tts(&tr, &tresp, &err)) {
            res.status = 500;
            json e = {{"error", err}};
            res.set_content(e.dump(), "application/json");
            return;
        }
        res.set_content(pcm_mono_to_wav(tresp.pcm_mono, tresp.sampling_rate),
                        "audio/wav");
    });

    svr->Post("/v1/audio/music",
              [slots_ref](const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"invalid json"})", "application/json");
            return;
        }
        const std::string model_id = body.value("model", "");
        auto it = slots_ref->find(model_id);
        if (it == slots_ref->end()) {
            res.status = 404;
            res.set_content(R"({"error":"unknown model"})", "application/json");
            return;
        }
        auto& slot = it->second;
        std::lock_guard<std::mutex> g(slot->mtx);

        acestep::MusicRequest mr;
        mr.caption  = body.value("caption", "");
        mr.lyrics   = body.value("lyrics", "");
        mr.duration = body.value("duration", 30.0f);
        if (body.contains("seed"))           mr.seed              = body["seed"].get<int32_t>();
        if (body.contains("guidance_scale")) mr.guidance_scale    = body["guidance_scale"].get<float>();
        if (body.contains("steps"))          mr.n_diffusion_steps = body["steps"].get<int32_t>();

        acestep::MusicResponse mresp;
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

    return svr;
}

}  // namespace audiocore
