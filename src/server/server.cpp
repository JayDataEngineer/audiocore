// server.cpp — HTTP server factory + WAV encoders (see server.h).

#include "audiocore/server/server.h"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#include "audiocore/models/ace_step/family.h"
#include "audiocore/models/moss_tts/family.h"
#include "audiocore/models/qwen3_tts/family.h"
#include "audiocore/framework/runtime/tasks.h"
#ifdef AUDIOCORE_ENABLE_MP3
#include "audiocore/framework/io/mp3_encoder.h"
#endif

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

        // One request shape for every TTS family. The unified TtsRequest is
        // a strict superset of the old moss::TtsRequest and qwen3_tts::
        // TtsRequest fields, so we parse everything the client might send
        // and let the family read whatever subset it cares about. No more
        // per-family if/else branch in the routing layer.
        TtsRequest tr;
        tr.text            = body.value("input", "");
        tr.language        = body.value("language", "");
        tr.voice_path      = body.value("voice", "");
        tr.mode            = body.value("mode", "tts");
        tr.speed           = body.value("speed", 1.0f);
        tr.instruct        = body.value("instruct", "");
        tr.quality         = body.value("quality", "");
        tr.speaker_name    = body.value("speaker", "");
        tr.reference_audio = body.value("reference_audio", "");
        tr.reference_text  = body.value("reference_text", "");
        if (body.contains("seed"))             tr.seed             = body["seed"].get<int32_t>();
        if (body.contains("temperature"))      tr.temperature      = body["temperature"].get<float>();
        if (body.contains("top_p"))            tr.top_p            = body["top_p"].get<float>();
        if (body.contains("text_temperature")) tr.text_temperature = body["text_temperature"].get<float>();
        if (body.contains("text_top_p"))       tr.text_top_p       = body["text_top_p"].get<float>();
        if (body.contains("text_top_k"))       tr.text_top_k       = body["text_top_k"].get<int32_t>();
        if (body.contains("duration_tokens"))     tr.duration_tokens     = body["duration_tokens"].get<int32_t>();
        if (body.contains("max_new_tokens"))      tr.max_new_tokens      = body["max_new_tokens"].get<int32_t>();
        if (body.contains("max_tokens"))          tr.max_new_tokens      = body["max_tokens"].get<int32_t>();
        if (body.contains("repetition_penalty"))  tr.repetition_penalty  = body["repetition_penalty"].get<float>();
        // Parse pre-computed speaker embedding (base64-encoded float array)
        if (body.contains("speaker_embedding") && body["speaker_embedding"].is_string()) {
            std::string b64 = body["speaker_embedding"].get<std::string>();
            if (!b64.empty()) {
                static const char kDec[256] = {
                    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
                    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
                    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
                    52,53,54,55,56,57,58,59,60,61,-1,-1,-1, 0,-1,-1,
                    -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
                    15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
                    -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
                    41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
                };
                std::vector<uint8_t> raw;
                raw.reserve(b64.size() * 3 / 4);
                int pad = 0;
                uint8_t buf[4];
                int nbuf = 0;
                for (char c : b64) {
                    if (c == '=') { pad++; continue; }
                    int v = kDec[static_cast<uint8_t>(c)];
                    if (v < 0) continue;
                    buf[nbuf++] = static_cast<uint8_t>(v);
                    if (nbuf == 4) {
                        raw.push_back((buf[0] << 2) | (buf[1] >> 4));
                        raw.push_back((buf[1] << 4) | (buf[2] >> 2));
                        raw.push_back((buf[2] << 6) | buf[3]);
                        nbuf = 0;
                    }
                }
                if (nbuf >= 2) raw.push_back((buf[0] << 2) | (buf[1] >> 4));
                if (nbuf >= 3) raw.push_back((buf[1] << 4) | (buf[2] >> 2));
                // Interpret raw bytes as float array (little-endian)
                size_t n_floats = raw.size() / sizeof(float);
                tr.speaker_embedding.resize(n_floats);
                std::memcpy(tr.speaker_embedding.data(), raw.data(), n_floats * sizeof(float));
            }
        }
        // Parse multi-turn messages (OpenAI-compatible format)
        if (body.contains("messages") && body["messages"].is_array()) {
            for (const auto& m : body["messages"]) {
                ChatMessage cm;
                cm.role    = m.value("role", "");
                cm.content = m.value("content", "");
                if (!cm.role.empty() && !cm.content.empty())
                    tr.messages.push_back(std::move(cm));
            }
        }

        TtsResponse tresp;
        std::string err;
        if (!slot->session->run_tts(&tr, &tresp, &err)) {
            fail_with(res, err);
            return;
        }
        if (tr.response_format == "mp3") {
#ifdef AUDIOCORE_ENABLE_MP3
            auto mp3 = pcm_mono_to_mp3(tresp.pcm_mono.data(),
                                        tresp.pcm_mono.size(),
                                        tresp.sampling_rate);
            res.set_content(reinterpret_cast<const char*>(mp3.data()),
                            mp3.size(), "audio/mpeg");
#else
            fail_with(res, "MP3 output not compiled (enable ENGINE_ENABLE_MP3)");
#endif
        } else {
            res.set_content(pcm_mono_to_wav(tresp.pcm_mono, tresp.sampling_rate),
                            "audio/wav");
        }
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
        mr.mode     = body.value("mode", "text_to_music");
        mr.mask_start = body.value("mask_start", 0.0f);
        mr.mask_end   = body.value("mask_end", 1.0f);
        mr.response_format = body.value("response_format", "wav");
        if (body.contains("seed"))           mr.seed              = body["seed"].get<int32_t>();
        if (body.contains("guidance_scale")) mr.guidance_scale    = body["guidance_scale"].get<float>();
        if (body.contains("steps"))          mr.n_diffusion_steps = body["steps"].get<int32_t>();
        if (body.contains("temperature"))    mr.temperature       = body["temperature"].get<float>();
        if (body.contains("top_p"))          mr.top_p             = body["top_p"].get<float>();

        // ── Parse input_audio from base64 WAV ─────────────────────────────
        if (body.contains("input_audio") && body["input_audio"].is_string()) {
            std::string b64 = body["input_audio"].get<std::string>();
            if (!b64.empty()) {
                // Simple base64 decode (RFC 4648)
                static const char kDec[256] = {
                    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
                    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
                    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
                    52,53,54,55,56,57,58,59,60,61,-1,-1,-1, 0,-1,-1,
                    -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
                    15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
                    -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
                    41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
                };
                std::vector<uint8_t> raw;
                raw.reserve(b64.size() * 3 / 4);
                int pad = 0;
                uint8_t buf[4];
                int nbuf = 0;
                for (char c : b64) {
                    if (c == '=') { pad++; continue; }
                    int v = kDec[static_cast<uint8_t>(c)];
                    if (v < 0) continue;  // skip whitespace
                    buf[nbuf++] = static_cast<uint8_t>(v);
                    if (nbuf == 4) {
                        raw.push_back((buf[0] << 2) | (buf[1] >> 4));
                        raw.push_back((buf[1] << 4) | (buf[2] >> 2));
                        raw.push_back((buf[2] << 6) | buf[3]);
                        nbuf = 0;
                    }
                }
                if (nbuf >= 2) raw.push_back((buf[0] << 2) | (buf[1] >> 4));
                if (nbuf >= 3) raw.push_back((buf[1] << 4) | (buf[2] >> 2));

                // Parse WAV: 44-byte header, PCM16 little-endian
                if (raw.size() > 44) {
                    const uint8_t* d = raw.data();
                    uint16_t channels = d[22] | (static_cast<uint16_t>(d[23]) << 8);
                    uint32_t sr = static_cast<uint32_t>(d[24]) |
                                  (static_cast<uint32_t>(d[25]) << 8) |
                                  (static_cast<uint32_t>(d[26]) << 16) |
                                  (static_cast<uint32_t>(d[27]) << 24);
                    // Find data chunk
                    size_t data_off = 44;  // minimal header
                    for (size_t p = 44; p + 8 <= raw.size(); p += 4) {
                        if (raw[p] == 'd' && raw[p+1] == 'a' && raw[p+2] == 't' && raw[p+3] == 'a') {
                            data_off = p + 8;
                            break;
                        }
                        uint32_t chunk_size = static_cast<uint32_t>(raw[p+4]) |
                                              (static_cast<uint32_t>(raw[p+5]) << 8) |
                                              (static_cast<uint32_t>(raw[p+6]) << 16) |
                                              (static_cast<uint32_t>(raw[p+7]) << 24);
                        p += 4 + chunk_size;
                    }
                    size_t n_pcm_bytes = raw.size() - data_off;
                    size_t n_samples = n_pcm_bytes / 2;
                    if (channels == 0) channels = 1;
                    size_t n_per_ch = n_samples / channels;
                    mr.input_audio.resize(n_per_ch * 2);  // stereo interleaved
                    for (size_t i = 0; i < n_per_ch * 2 && i < n_per_ch * channels; i++) {
                        int16_t s = static_cast<int16_t>(raw[data_off + i * 2]) |
                                    (static_cast<int16_t>(raw[data_off + i * 2 + 1]) << 8);
                        mr.input_audio[i] = s / 32768.0f;
                    }
                    // If mono, duplicate to stereo
                    if (channels == 1 && n_per_ch > 0) {
                        mr.input_audio.resize(n_per_ch * 2);
                        for (size_t i = n_per_ch; i > 0; i--) {
                            mr.input_audio[i * 2 - 1] = mr.input_audio[i - 1];
                            mr.input_audio[i * 2 - 2] = mr.input_audio[i - 1];
                        }
                    }
                    // Resample if not 48kHz
                    (void)sr;  // TODO: resample to 48kHz if needed
                }
            }
        }

        acestep::MusicResponse mresp;
        std::string err;
        if (!slot->session->run_music(&mr, &mresp, &err)) {
            fail_with(res, err);
            return;
        }
        if (mr.response_format == "mp3") {
#ifdef AUDIOCORE_ENABLE_MP3
            auto mp3 = pcm_stereo_to_mp3(mresp.pcm_stereo.data(),
                                          mresp.pcm_stereo.size(),
                                          mresp.sampling_rate);
            res.set_content(reinterpret_cast<const char*>(mp3.data()),
                            mp3.size(), "audio/mpeg");
#else
            fail_with(res, "MP3 output not compiled (enable ENGINE_ENABLE_MP3)");
#endif
        } else {
            res.set_content(pcm_stereo_to_wav(mresp.pcm_stereo, mresp.sampling_rate),
                            "audio/wav");
        }
    });

    // ── POST /v1/audio/speech/stream ────────────────────────────────────────
    //
    // Chunked-transfer variant of /v1/audio/speech. Two modes:
    //
    // 1. Streaming callback mode (preferred): the family calls
    //    TtsRequest::stream::on_audio with PCM chunks as audio is produced.
    //    Each chunk is WAV-encoded and emitted immediately via the chunked
    //    content provider. This gives true progressive audio.
    //
    // 2. Fallback (no callback): the family generates the full PCM, then we
    //    WAV-encode and chunk-emit the result.
    //
    // Mode 1 requires the session to run on a background thread so the
    // callback can push data while the chunked content provider pumps it.
    svr->Post("/v1/audio/speech/stream",
              [slots_ref](const httplib::Request& req, httplib::Response& res) {
        auto sg = resolve_slot(req, *slots_ref, res);
        if (!sg) return;
        const auto& body = sg->body;
        auto& slot = sg->slot;

        // Same field set as /v1/audio/speech.
        TtsRequest tr;
        tr.text            = body.value("input", "");
        tr.language        = body.value("language", "");
        tr.voice_path      = body.value("voice", "");
        tr.mode            = body.value("mode", "tts");
        tr.speed           = body.value("speed", 1.0f);
        tr.instruct        = body.value("instruct", "");
        tr.quality         = body.value("quality", "");
        tr.speaker_name    = body.value("speaker", "");
        tr.reference_audio = body.value("reference_audio", "");
        tr.reference_text  = body.value("reference_text", "");
        tr.response_format = body.value("response_format", "wav");
        if (body.contains("seed"))             tr.seed             = body["seed"].get<int32_t>();
        if (body.contains("temperature"))      tr.temperature      = body["temperature"].get<float>();
        if (body.contains("top_p"))            tr.top_p            = body["top_p"].get<float>();
        if (body.contains("text_temperature")) tr.text_temperature = body["text_temperature"].get<float>();
        if (body.contains("text_top_p"))       tr.text_top_p       = body["text_top_p"].get<float>();
        if (body.contains("text_top_k"))       tr.text_top_k       = body["text_top_k"].get<int32_t>();
        if (body.contains("duration_tokens"))  tr.duration_tokens  = body["duration_tokens"].get<int32_t>();
        if (body.contains("max_new_tokens"))      tr.max_new_tokens      = body["max_new_tokens"].get<int32_t>();
        if (body.contains("max_tokens"))          tr.max_new_tokens      = body["max_tokens"].get<int32_t>();
        if (body.contains("repetition_penalty"))  tr.repetition_penalty  = body["repetition_penalty"].get<float>();
        if (body.contains("speaker_embedding") && body["speaker_embedding"].is_string()) {
            std::string b64 = body["speaker_embedding"].get<std::string>();
            if (!b64.empty()) {
                static const char kDec[256] = {
                    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
                    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
                    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
                    52,53,54,55,56,57,58,59,60,61,-1,-1,-1, 0,-1,-1,
                    -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
                    15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
                    -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
                    41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
                };
                std::vector<uint8_t> raw;
                raw.reserve(b64.size() * 3 / 4);
                int nbuf = 0;
                uint8_t buf[4];
                for (char c : b64) {
                    if (c == '=') continue;
                    int v = kDec[static_cast<uint8_t>(c)];
                    if (v < 0) continue;
                    buf[nbuf++] = static_cast<uint8_t>(v);
                    if (nbuf == 4) {
                        raw.push_back((buf[0] << 2) | (buf[1] >> 4));
                        raw.push_back((buf[1] << 4) | (buf[2] >> 2));
                        raw.push_back((buf[2] << 6) | buf[3]);
                        nbuf = 0;
                    }
                }
                if (nbuf >= 2) raw.push_back((buf[0] << 2) | (buf[1] >> 4));
                if (nbuf >= 3) raw.push_back((buf[1] << 4) | (buf[2] >> 2));
                size_t n_floats = raw.size() / sizeof(float);
                tr.speaker_embedding.resize(n_floats);
                std::memcpy(tr.speaker_embedding.data(), raw.data(), n_floats * sizeof(float));
            }
        }
        if (body.contains("messages") && body["messages"].is_array()) {
            for (const auto& m : body["messages"]) {
                ChatMessage cm;
                cm.role    = m.value("role", "");
                cm.content = m.value("content", "");
                if (!cm.role.empty() && !cm.content.empty())
                    tr.messages.push_back(std::move(cm));
            }
        }

        // ── Streaming state shared between generation thread and HTTP pump ──
        struct StreamBuf {
            std::mutex              mtx;
            std::string             buffer;  // WAV bytes ready to send
            bool                    done    = false;
            bool                    ok      = true;
            std::string             error;
            std::condition_variable cv;
        };
        auto sbuf = std::make_shared<StreamBuf>();

        // ── Heap-allocated context for streaming ────────────────────────────
        // Captures MUST be by value (shared_ptr) so they survive the route-
        // handler return. Stack locals (tr, tresp, asc, slot-ref) would dangle
        // once the detached worker thread starts running.
        struct StreamCtx {
            TtsRequest          tr;
            TtsResponse         tresp;
            AudioStreamCallbacks asc;
        };
        auto ctx = std::make_shared<StreamCtx>();
        ctx->tr = std::move(tr);

        const std::string fmt = ctx->tr.response_format;
        ctx->asc.on_audio = [sbuf, fmt](const float* pcm, size_t n) -> bool {
            std::vector<float> chunk(pcm, pcm + n);
            std::string encoded;
            if (fmt == "mp3") {
#ifdef AUDIOCORE_ENABLE_MP3
                auto mp3 = pcm_mono_to_mp3(chunk.data(), chunk.size(), 24000);
                if (!mp3.empty())
                    encoded.assign(reinterpret_cast<const char*>(mp3.data()), mp3.size());
#else
                (void)0;
#endif
            }
            if (encoded.empty())
                encoded = pcm_mono_to_wav(chunk, 24000);
            std::lock_guard<std::mutex> lk(sbuf->mtx);
            sbuf->buffer.append(encoded);
            sbuf->cv.notify_one();
            return true;
        };
        ctx->tr.stream = &ctx->asc;

        // Run inference on a background thread so the callback fires
        // concurrently with the HTTP chunked provider pump.
        // Capture slot by value (shared_ptr copy) so the slot stays alive.
        auto slot_guard = sg->slot;
        std::thread worker([slot = std::move(slot_guard), ctx, sbuf]() {
            std::string err;
            bool ok = slot->session->run_tts(&ctx->tr, &ctx->tresp, &err);
            std::lock_guard<std::mutex> lk(sbuf->mtx);
            sbuf->done = true;
            sbuf->ok   = ok;
            if (!ok) sbuf->error = std::move(err);
            sbuf->cv.notify_one();
        });
        worker.detach();

        // Chunked content provider: pumps WAV/MP3 bytes from StreamBuf
        const char* ct = (fmt == "mp3") ? "audio/mpeg" : "audio/wav";
        res.set_chunked_content_provider(
            ct,
            [sbuf](size_t /*offset*/, httplib::DataSink& sink) {
                std::unique_lock<std::mutex> lk(sbuf->mtx);

                sbuf->cv.wait(lk, [sbuf]() {
                    return !sbuf->buffer.empty() || sbuf->done;
                });

                if (!sbuf->buffer.empty()) {
                    if (!sink.write(sbuf->buffer.data(), sbuf->buffer.size()))
                        return false;
                    sbuf->buffer.clear();
                }

                if (sbuf->done) {
                    sink.done();
                    return sbuf->ok;
                }
                return true;
            });
    });

    return svr;
}

}  // namespace audiocore
