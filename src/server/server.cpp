// server.cpp — HTTP server factory + WAV encoders (see server.h).

#include "audiocore/server/server.h"

#include <atomic>
#include <cstdio>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include "audiocore/models/ace_step/family.h"
#include "audiocore/models/moss_tts/family.h"
#include "audiocore/models/qwen3_tts/family.h"

namespace audiocore {

using nlohmann::json;

// ---- WAV encoders --------------------------------------------------------
// Tiny WAV writer for PCM16 → response body. Keeps the server binary
// dependency-free; we encode MP3 only when libmp3lame is linked (Phase 2).

namespace {
std::string pcm_to_wav_impl(const std::vector<float>& pcm, int32_t sr,
                             uint16_t channels) {
    std::ostringstream o;
    auto w16 = [&](uint16_t v) { o.put(v & 0xff); o.put((v >> 8) & 0xff); };
    auto w32 = [&](uint32_t v) {
        for (int i = 0; i < 4; i++) o.put((v >> (i * 8)) & 0xff);
    };
    const uint16_t bps = 16;
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
}  // namespace

std::string pcm_mono_to_wav(const std::vector<float>& pcm, int32_t sr) {
    return pcm_to_wav_impl(pcm, sr, 1);
}

std::string pcm_stereo_to_wav(const std::vector<float>& pcm, int32_t sr) {
    return pcm_to_wav_impl(pcm, sr, 2);
}

// ---- HTTP server factory -------------------------------------------------

namespace {

// Parse JSON body, resolve model-id → slot, and lock. Returns nullopt (with
// `res` set to the appropriate error) on failure, or a locked SlotGuard on
// success. The lock is held for the lifetime of the SlotGuard.
struct SlotGuard {
    json body;
    std::shared_ptr<ModelSlot> slot;
    std::unique_lock<std::mutex> lock;
};

std::optional<SlotGuard> resolve_slot(
        const httplib::Request& req,
        const std::unordered_map<std::string, std::shared_ptr<ModelSlot>>& slots,
        httplib::Response& res) {
    json body;
    try { body = json::parse(req.body); }
    catch (...) {
        res.status = 400;
        res.set_content(R"({"error":"invalid json"})", "application/json");
        return std::nullopt;
    }
    const std::string model_id = body.value("model", "");
    auto it = slots.find(model_id);
    if (it == slots.end()) {
        res.status = 404;
        res.set_content(R"({"error":"unknown model"})", "application/json");
        return std::nullopt;
    }
    auto& slot = it->second;
    return SlotGuard{std::move(body), slot, std::unique_lock<std::mutex>(slot->mtx)};
}

// Helper: set res to a 500 JSON error and log the failure.
void fail_with(httplib::Response& res, const std::string& err) {
    res.status = 500;
    json e = {{"error", err}};
    res.set_content(e.dump(), "application/json");
}

}  // namespace

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
        auto sg = resolve_slot(req, *slots_ref, res);
        if (!sg) return;
        const auto& body = sg->body;
        auto& slot = sg->slot;

        const std::string family = slot->session->family_name();

        if (family == "qwen3_tts") {
            // ── Qwen3-TTS ─────────────────────────────────────────────────
            qwen3_tts::TtsRequest tr{};
            tr.text          = body.value("input", "");
            tr.language      = body.value("language", "");
            tr.speed         = body.value("speed", 1.0f);
            tr.temperature   = body.value("temperature", 0.7f);
            tr.top_p         = body.value("top_p", 0.9f);
            tr.instruct      = body.value("instruct", "");
            tr.speaker_name  = body.value("speaker", "");
            tr.reference_audio = body.value("reference_audio", "");
            tr.reference_text  = body.value("reference_text", "");
            if (body.contains("max_new_tokens"))
                tr.max_new_tokens = body["max_new_tokens"].get<int>();

            qwen3_tts::TtsResponse tresp;
            std::string err;
            if (!slot->session->run_tts(&tr, &tresp, &err)) {
                fail_with(res, err);
                return;
            }
            res.set_content(pcm_mono_to_wav(tresp.pcm_mono, tresp.sampling_rate),
                            "audio/wav");
            return;
        }

        // ── MOSS TTS (default / fallback) ──────────────────────────────────
        moss::TtsRequest tr{};
        tr.text       = body.value("input", "");
        tr.language   = body.value("language", "en");
        tr.voice_path = body.value("voice", "");
        tr.mode       = body.value("mode", "tts");
        if (body.contains("seed"))        tr.seed       = body["seed"].get<int32_t>();
        if (body.contains("temperature")) tr.temperature= body["temperature"].get<float>();
        if (body.contains("top_p"))       tr.top_p      = body["top_p"].get<float>();
        if (body.contains("max_tokens"))  tr.max_tokens = body["max_tokens"].get<int32_t>();

        moss::TtsResponse tresp;
        std::string err;
        if (!slot->session->run_tts(&tr, &tresp, &err)) {
            fail_with(res, err);
            return;
        }
        res.set_content(pcm_mono_to_wav(tresp.pcm_mono, tresp.sampling_rate),
                        "audio/wav");
    });

    svr->Post("/v1/audio/music",
              [slots_ref](const httplib::Request& req, httplib::Response& res) {
        auto sg = resolve_slot(req, *slots_ref, res);
        if (!sg) return;
        const auto& body = sg->body;
        auto& slot = sg->slot;

        acestep::MusicRequest mr;
        mr.caption  = body.value("caption", "");
        mr.lyrics   = body.value("lyrics", "");
        mr.duration = body.value("duration", 30.0f);
        if (body.contains("seed"))           mr.seed              = body["seed"].get<int32_t>();
        if (body.contains("guidance_scale")) mr.guidance_scale    = body["guidance_scale"].get<float>();
        if (body.contains("steps"))          mr.n_diffusion_steps = body["steps"].get<int32_t>();
        if (body.contains("temperature"))    mr.temperature       = body["temperature"].get<float>();
        if (body.contains("top_p"))          mr.top_p             = body["top_p"].get<float>();

        acestep::MusicResponse mresp;
        std::string err;
        if (!slot->session->run_music(&mr, &mresp, &err)) {
            fail_with(res, err);
            return;
        }
        res.set_content(pcm_stereo_to_wav(mresp.pcm_stereo, mresp.sampling_rate),
                        "audio/wav");
    });

    return svr;
}

}  // namespace audiocore
