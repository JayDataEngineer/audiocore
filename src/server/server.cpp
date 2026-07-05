#include "audiocore/server/server.h"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#include "audiocore/framework/runtime/tasks.h"
#include "audiocore/framework/audio/dsp.h"   // pitch_shift, change_speed
#include "audiocore/models/ace_step/family.h"    // MusicRequest / MusicResponse
#ifdef AUDIOCORE_ENABLE_MP3
#include "audiocore/framework/io/mp3_encoder.h"
#endif

#ifdef AUDIOCORE_HAS_WEBAPP
#include "index.html.hpp"
#include "style.css.hpp"
#include "app.js.hpp"
#endif

namespace audiocore {

using nlohmann::json;
namespace fs = std::filesystem;

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

namespace {

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

void fail_with(httplib::Response& res, const std::string& err) {
    res.status = 500;
    json e = {{"error", err}};
    res.set_content(e.dump(), "application/json");
}

// Resolve the `voice` field on a TtsRequest. .voice files (QWEN3VOICE
// binary, dim-N ECAPA-TDNN embedding) live under <clips_dir>/voices/.
// Anything else is treated as a WAV path for ICL reference cloning.
//
// On match: reads the binary, populates tr.speaker_embedding, and clears
// tr.voice_path so the session doesn't try to interpret it as a WAV.
// On no match: leaves tr.voice_path untouched (legacy behaviour).
void resolve_voice_field(TtsRequest& tr, const std::string& clips_dir) {
    if (tr.voice_path.empty()) return;
    if (tr.voice_path.size() < 6) return;
    if (tr.voice_path.compare(tr.voice_path.size() - 6, 6, ".voice") != 0)
        return;
    std::string vname = tr.voice_path;
    size_t slash = vname.find_last_of('/');
    if (slash != std::string::npos) vname = vname.substr(slash + 1);
    if (vname.find("..") != std::string::npos) return;
    if (clips_dir.empty()) return;
    std::string vpath = clips_dir + "/voices/" + vname;
    std::ifstream vf(vpath, std::ios::binary);
    if (!vf) return;
    char hdr[32];
    if (!vf.read(hdr, 32)) return;
    if (std::memcmp(hdr, "QWEN3VOICE", 10) != 0) return;
    uint32_t dim = 0;
    std::memcpy(&dim, hdr + 20, 4);
    if (dim == 0 || dim > 65536) return;  // sanity
    std::vector<uint8_t> raw(static_cast<size_t>(dim) * 4);
    if (!vf.read(reinterpret_cast<char*>(raw.data()),
                  static_cast<std::streamsize>(raw.size())))
        return;
    tr.speaker_embedding.resize(dim);
    std::memcpy(tr.speaker_embedding.data(), raw.data(), dim * 4);
    tr.voice_path.clear();
    if (tr.mode.empty() || tr.mode == "tts") tr.mode = "voice_clone";

    // ── Load DSP sidecar `<name>.voice.json` if present ────────────────
    // The Voice Maker writes pitch_shift / speed here so any Synthesize
    // call that picks this voice automatically inherits the saved shaping.
    std::string mpath = vpath + ".json";
    std::ifstream mf(mpath);
    if (mf) {
        std::string mtext((std::istreambuf_iterator<char>(mf)),
                           std::istreambuf_iterator<char>());
        try {
            auto meta = json::parse(mtext);
            // Range-clamp semitones to a safe ±12; speed to (0.25, 4.0].
            float ps = meta.value("pitch_shift", 0.0f);
            if (ps < -12.0f) ps = -12.0f;
            if (ps >  12.0f) ps =  12.0f;
            float sp = meta.value("speed", 1.0f);
            if (sp < 0.25f) sp = 0.25f;
            if (sp > 4.0f)  sp = 4.0f;
            tr.voice_pitch_shift = ps;
            tr.voice_speed       = sp;
            tr.has_voice_meta    = true;
        } catch (...) {
            // Malformed sidecar — silently ignore.
        }
    }
}

// ── Clip helpers ──────────────────────────────────────────────────────────

static const std::unordered_set<std::string> kAudioExts = {
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".weba"
};

std::string mime_for(const std::string& path) {
    auto dot = path.rfind('.');
    if (dot == std::string::npos) return "application/octet-stream";
    std::string ext = path.substr(dot);
    for (auto& c : ext) c = static_cast<char>(std::tolower(c));
    if (ext == ".wav")  return "audio/wav";
    if (ext == ".mp3")  return "audio/mpeg";
    if (ext == ".flac") return "audio/flac";
    if (ext == ".ogg")  return "audio/ogg";
    if (ext == ".m4a")  return "audio/mp4";
    if (ext == ".opus") return "audio/opus";
    if (ext == ".weba") return "audio/webm";
    if (ext == ".js")   return "application/javascript; charset=utf-8";
    if (ext == ".css")  return "text/css; charset=utf-8";
    if (ext == ".html") return "text/html; charset=utf-8";
    return "application/octet-stream";
}

std::string sanitize_filename(const std::string& name) {
    std::string safe;
    safe.reserve(name.size());
    for (char c : name) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '-' || c == '_')
            safe += c;
        else
            safe += '_';
    }
    if (safe.empty() || !std::isalpha(static_cast<unsigned char>(safe[0])))
        safe = "clip_" + safe;
    return safe;
}

float wav_duration_seconds(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return -1.0f;
    char hdr[44];
    if (!f.read(hdr, 44)) return -1.0f;
    if (std::memcmp(hdr, "RIFF", 4) != 0 || std::memcmp(hdr + 8, "WAVE", 4) != 0)
        return -1.0f;
    uint32_t sr = static_cast<uint32_t>(static_cast<uint8_t>(hdr[24])) |
                  (static_cast<uint32_t>(static_cast<uint8_t>(hdr[25])) << 8) |
                  (static_cast<uint32_t>(static_cast<uint8_t>(hdr[26])) << 16) |
                  (static_cast<uint32_t>(static_cast<uint8_t>(hdr[27])) << 24);
    uint16_t ch = static_cast<uint16_t>(static_cast<uint8_t>(hdr[22])) |
                  (static_cast<uint16_t>(static_cast<uint8_t>(hdr[23])) << 8);
    if (sr == 0 || ch == 0) return -1.0f;
    // Find data chunk
    f.seekg(12);
    char chunk[8];
    while (f.read(chunk, 8)) {
        uint32_t sz = static_cast<uint32_t>(static_cast<uint8_t>(chunk[4])) |
                      (static_cast<uint32_t>(static_cast<uint8_t>(chunk[5])) << 8) |
                      (static_cast<uint32_t>(static_cast<uint8_t>(chunk[6])) << 16) |
                      (static_cast<uint32_t>(static_cast<uint8_t>(chunk[7])) << 24);
        if (std::memcmp(chunk, "data", 4) == 0) {
            return static_cast<float>(sz) / (sr * ch * 2);
        }
        f.seekg(static_cast<std::streamoff>(sz), std::ios::cur);
    }
    return -1.0f;
}

json clip_info(const fs::directory_entry& entry) {
    auto status = entry.status();
    auto ftime = fs::last_write_time(entry);
    auto sctp = std::chrono::time_point_cast<std::chrono::seconds>(
        std::chrono::system_clock::now() +
        (ftime - fs::file_time_type::clock::now()));
    int64_t mtime_s = sctp.time_since_epoch().count();
    json c;
    c["name"] = entry.path().filename().string();
    c["size"] = static_cast<uint64_t>(entry.file_size());
    c["mtime"] = mtime_s;
    float dur = wav_duration_seconds(entry.path());
    if (dur > 0) c["duration"] = dur;
    return c;
}

#ifdef AUDIOCORE_HAS_WEBAPP
std::string_view asset_data(const std::string& name) {
    if (name == "index.html")
        return std::string_view(reinterpret_cast<const char*>(index_html), index_html_len);
    if (name == "style.css")
        return std::string_view(reinterpret_cast<const char*>(style_css), style_css_len);
    if (name == "app.js")
        return std::string_view(reinterpret_cast<const char*>(app_js), app_js_len);
    return {};
}
#endif

}  // namespace

std::shared_ptr<httplib::Server> build_server(
        std::shared_ptr<std::unordered_map<std::string, std::shared_ptr<ModelSlot>>> slots,
        const std::string& clips_dir) {
    auto svr = std::make_shared<httplib::Server>();
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
                {"family", slot->session->family()},
                {"loaded", true},
            });
        }
        res.set_content(json{{"object", "list"}, {"data", arr}}.dump(),
                        "application/json");
    });

    svr->Post("/v1/audio/speech",
              [slots_ref, clips_dir](const httplib::Request& req, httplib::Response& res) {
        auto sg = resolve_slot(req, *slots_ref, res);
        if (!sg) return;
        const auto& body = sg->body;
        auto& slot = sg->slot;

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
        resolve_voice_field(tr, clips_dir);
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

        TtsResponse tresp;
        std::string err;
        if (!slot->session->run_tts(&tr, &tresp, &err)) {
            fail_with(res, err);
            return;
        }

        // ── Optional post-processing: pitch shift + speed change. ──────
        // WSOLA-based, model-agnostic. See framework/audio/dsp.h.
        // These are independent of any TTS family — pure signal processing
        // on the float32 PCM before WAV/MP3 encoding. Useful for "fine-
        // tune the voice after generation" sliders in the UI.
        //
        // Resolution order: (1) `<name>.voice.json` sidecar defaults
        // (set by resolve_voice_field from the Voice Maker), then (2) the
        // explicit `pitch_shift` / `speed` keys in this request body —
        // the body wins so users can always override per-call.
        float pitch_shift_semi = tr.has_voice_meta ? tr.voice_pitch_shift : 0.0f;
        float speed_mult       = tr.has_voice_meta ? tr.voice_speed       : 1.0f;
        if (body.contains("pitch_shift")) pitch_shift_semi = body["pitch_shift"].get<float>();
        if (body.contains("speed"))       speed_mult       = body["speed"].get<float>();
        if (!tresp.pcm_mono.empty()) {
            if (std::fabs(pitch_shift_semi) > 0.001f) {
                auto shifted = audiocore::pitch_shift(
                    tresp.pcm_mono.data(), tresp.pcm_mono.size(),
                    static_cast<double>(pitch_shift_semi),
                    tresp.sampling_rate);
                tresp.pcm_mono = std::move(shifted);
            }
            if (std::fabs(speed_mult - 1.0f) > 0.001f && speed_mult > 0.0f) {
                auto sped = audiocore::change_speed(
                    tresp.pcm_mono.data(), tresp.pcm_mono.size(),
                    static_cast<double>(speed_mult),
                    tresp.sampling_rate);
                tresp.pcm_mono = std::move(sped);
            }
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

        if (body.contains("input_audio") && body["input_audio"].is_string()) {
            std::string b64 = body["input_audio"].get<std::string>();
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

                if (raw.size() > 44) {
                    const uint8_t* d = raw.data();
                    uint16_t channels = d[22] | (static_cast<uint16_t>(d[23]) << 8);
                    uint32_t sr = static_cast<uint32_t>(d[24]) |
                                  (static_cast<uint32_t>(d[25]) << 8) |
                                  (static_cast<uint32_t>(d[26]) << 16) |
                                  (static_cast<uint32_t>(d[27]) << 24);
                    size_t data_off = 44;
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
                    mr.input_audio.resize(n_per_ch * 2);
                    for (size_t i = 0; i < n_per_ch * 2 && i < n_per_ch * channels; i++) {
                        int16_t s = static_cast<int16_t>(raw[data_off + i * 2]) |
                                    (static_cast<int16_t>(raw[data_off + i * 2 + 1]) << 8);
                        mr.input_audio[i] = s / 32768.0f;
                    }
                    if (channels == 1 && n_per_ch > 0) {
                        mr.input_audio.resize(n_per_ch * 2);
                        for (size_t i = n_per_ch; i > 0; i--) {
                            mr.input_audio[i * 2 - 1] = mr.input_audio[i - 1];
                            mr.input_audio[i * 2 - 2] = mr.input_audio[i - 1];
                        }
                    }
                    (void)sr;
                }
            }
        }

        acestep::MusicResponse mresp;
        std::string err;
        if (!slot->session->run_music(&mr, &mresp, &err)) {
            fail_with(res, err);
            return;
        }

        // Optional pitch shift + speed change post-processing (same WSOLA
        // pipeline as the speech endpoint — useful for transposing a
        // generated loop to match a key or fitting it to a video length).
        // The DSP is mono, so we deinterleave → process per-channel →
        // reinterleave for the stereo output.
        const float pitch_shift_semi = body.value("pitch_shift", 0.0f);
        const float speed_ratio      = body.value("speed", 1.0f);
        if (!mresp.pcm_stereo.empty() && mresp.pcm_stereo.size() % 2 == 0) {
            const size_t n_per_ch = mresp.pcm_stereo.size() / 2;
            const int   sr        = mresp.sampling_rate;
            auto run_per_channel = [&](auto fn) {
                std::vector<float> l(n_per_ch), r(n_per_ch);
                for (size_t i = 0; i < n_per_ch; i++) {
                    l[i] = mresp.pcm_stereo[i * 2];
                    r[i] = mresp.pcm_stereo[i * 2 + 1];
                }
                l = fn(l.data(), l.size());
                r = fn(r.data(), r.size());
                std::vector<float> out(l.size() * 2);
                for (size_t i = 0; i < l.size() && i < r.size(); i++) {
                    out[i * 2]     = l[i];
                    out[i * 2 + 1] = r[i];
                }
                return out;
            };
            if (std::fabs(pitch_shift_semi) > 0.001f) {
                mresp.pcm_stereo = run_per_channel(
                    [&](const float* p, size_t n) {
                        return audiocore::pitch_shift(
                            p, n, static_cast<double>(pitch_shift_semi), sr);
                    });
            }
            if (std::fabs(speed_ratio - 1.0f) > 0.001f && speed_ratio > 0.0f) {
                mresp.pcm_stereo = run_per_channel(
                    [&](const float* p, size_t n) {
                        return audiocore::change_speed(
                            p, n, static_cast<double>(speed_ratio), sr);
                    });
            }
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

    // ── POST /v1/audio/speech/stream ──────────────────────────────────────
    svr->Post("/v1/audio/speech/stream",
              [slots_ref, clips_dir](const httplib::Request& req, httplib::Response& res) {
        auto sg = resolve_slot(req, *slots_ref, res);
        if (!sg) return;
        const auto& body = sg->body;
        auto& slot = sg->slot;

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
        resolve_voice_field(tr, clips_dir);
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

        struct StreamBuf {
            std::mutex              mtx;
            std::string             buffer;
            bool                    done    = false;
            bool                    ok      = true;
            std::string             error;
            std::condition_variable cv;
        };
        auto sbuf = std::make_shared<StreamBuf>();

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

    // ── Clip management routes ────────────────────────────────────────────
    if (!clips_dir.empty()) {
        // Ensure clips directory exists.
        std::error_code ec;
        fs::create_directories(clips_dir, ec);

        // GET /v1/clips — list audio clips as JSON
        svr->Get("/v1/clips", [clips_dir](const httplib::Request&, httplib::Response& res) {
            json arr = json::array();
            std::error_code ec;
            for (auto& entry : fs::directory_iterator(clips_dir, ec)) {
                if (!entry.is_regular_file()) continue;
                auto ext = entry.path().extension().string();
                for (auto& c : ext) c = static_cast<char>(std::tolower(c));
                if (kAudioExts.count(ext))
                    arr.push_back(clip_info(entry));
            }
            // Sort by mtime descending (newest first).
            std::sort(arr.begin(), arr.end(), [](const json& a, const json& b) {
                return a["mtime"].get<int64_t>() > b["mtime"].get<int64_t>();
            });
            res.set_content(json{{"clips", arr}}.dump(), "application/json");
        });

        // GET /v1/clips/raw/<name> — serve audio file
        svr->Get(R"(/v1/clips/raw/([^/?]+))",
                 [clips_dir](const httplib::Request& req, httplib::Response& res) {
            std::string name = req.matches[1];
            // Prevent path traversal.
            if (name.find("..") != std::string::npos || name.find('/') != std::string::npos) {
                res.status = 400;
                res.set_content(R"({"error":"invalid name"})", "application/json");
                return;
            }
            fs::path target = fs::path(clips_dir) / name;
            if (!fs::is_regular_file(target)) {
                res.status = 404;
                res.set_content(R"({"error":"not found"})", "application/json");
                return;
            }
            auto ext = target.extension().string();
            for (auto& c : ext) c = static_cast<char>(std::tolower(c));
            if (!kAudioExts.count(ext)) {
                res.status = 403;
                res.set_content(R"({"error":"not an audio file"})", "application/json");
                return;
            }
            std::ifstream f(target, std::ios::binary);
            std::string data((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
            res.set_content(std::move(data), mime_for(name));
        });

        // POST /v1/clips/upload — multipart file upload
        svr->Post("/v1/clips/upload",
                  [clips_dir](const httplib::Request& req, httplib::Response& res) {
            json resp;
            if (!req.is_multipart_form_data()) {
                res.status = 400;
                resp["error"] = "expected multipart/form-data";
                res.set_content(resp.dump(), "application/json");
                return;
            }
            auto files = req.get_file_values("file");
            json saved = json::array();
            for (auto& f : files) {
                if (f.filename.empty()) continue;
                auto ext = fs::path(f.filename).extension().string();
                for (auto& c : ext) c = static_cast<char>(std::tolower(c));
                if (!kAudioExts.count(ext)) continue;
                std::string safe = sanitize_filename(f.filename);
                fs::path dest = fs::path(clips_dir) / safe;
                std::ofstream out(dest, std::ios::binary);
                out.write(f.content.data(), f.content.size());
                saved.push_back(safe);
            }
            resp["ok"] = true;
            resp["saved"] = std::move(saved);
            res.set_content(resp.dump(), "application/json");
        });

        // POST /v1/clips/delete — delete a clip by name
        svr->Post("/v1/clips/delete",
                  [clips_dir](const httplib::Request& req, httplib::Response& res) {
            json resp;
            std::string name;
            if (req.has_param("name")) {
                name = req.get_param_value("name");
            } else {
                // Try form body parsing.
                std::regex re("name=([^&]+)");
                std::smatch m;
                if (std::regex_search(req.body, m, re))
                    name = m[1];
            }
            // Also try JSON body.
            if (name.empty()) {
                try {
                    auto j = json::parse(req.body);
                    name = j.value("name", "");
                } catch (...) {}
            }
            if (name.empty() || name.find("..") != std::string::npos ||
                name.find('/') != std::string::npos) {
                res.status = 400;
                resp["error"] = "invalid name";
                res.set_content(resp.dump(), "application/json");
                return;
            }
            fs::path target = fs::path(clips_dir) / name;
            if (fs::is_regular_file(target)) {
                fs::remove(target);
                resp["ok"] = true;
            } else {
                resp["ok"] = false;
                resp["error"] = "not found";
            }
            res.set_content(resp.dump(), "application/json");
        });
    }

    // ── Voice library routes (.voice files = QWEN3VOICE embeddings) ────────
    // Voices live under <clips_dir>/voices/. The frontend uses these routes
    // to populate the "Voices" tab and let users save / upload / delete
    // 2048-d ECAPA-TDNN embeddings for reuse in the Synthesize tab.
    {
        std::string voices_dir;
        if (!clips_dir.empty()) voices_dir = clips_dir + "/voices";
        if (!voices_dir.empty()) {
            std::error_code ec;
            fs::create_directories(voices_dir, ec);

            // Read a QWEN3VOICE file's dim + L2 norm (for metadata in lists).
            // Also reads an optional `<name>.voice.json` sidecar (written by
            // the Voice Maker via PUT /v1/voices/<n>/meta) that carries
            // default DSP shaping — `pitch_shift` (semitones) and `speed`.
            auto voice_meta = [](const fs::path& p) -> json {
                std::ifstream f(p, std::ios::binary);
                if (!f) return {};
                char hdr[32];
                if (!f.read(hdr, 32)) return {};
                if (std::memcmp(hdr, "QWEN3VOICE", 10) != 0) return {};
                uint32_t dim = 0;
                std::memcpy(&dim, hdr + 20, 4);
                // Read a chunk to estimate L2 (only first 1024 floats —
                // accurate enough for display, faster than full scan).
                std::vector<float> sample(std::min<uint32_t>(dim, 2048));
                f.read(reinterpret_cast<char*>(sample.data()),
                       static_cast<std::streamsize>(sample.size() * 4));
                double l2 = 0.0;
                for (float v : sample) l2 += static_cast<double>(v) * v;
                l2 = std::sqrt(l2 * (static_cast<double>(dim) /
                                      static_cast<double>(sample.size())));
                json v;
                v["name"]    = p.filename().string();
                v["path"]    = p.filename().string();
                v["size"]    = static_cast<uint64_t>(fs::file_size(p));
                v["dim"]     = static_cast<uint64_t>(dim);
                v["l2_norm"] = l2;

                // Voice-shaping sidecar: <name>.voice.json next to the .voice.
                std::string sidecar = p.string() + ".json";
                std::ifstream sf(sidecar);
                if (sf) {
                    std::string mtext((std::istreambuf_iterator<char>(sf)),
                                       std::istreambuf_iterator<char>());
                    try {
                        auto meta = json::parse(mtext);
                        json dsp;
                        dsp["pitch_shift"] = meta.value("pitch_shift", 0.0f);
                        dsp["speed"]       = meta.value("speed",       1.0f);
                        v["dsp"] = std::move(dsp);
                    } catch (...) { /* malformed sidecar — ignore */ }
                }
                return v;
            };

            // GET /v1/voices — list .voice files
            svr->Get("/v1/voices", [voices_dir, voice_meta](const httplib::Request&, httplib::Response& res) {
                json arr = json::array();
                std::error_code ec;
                for (auto& entry : fs::directory_iterator(voices_dir, ec)) {
                    if (!entry.is_regular_file()) continue;
                    if (entry.path().extension() != ".voice") continue;
                    auto meta = voice_meta(entry.path());
                    if (!meta.is_null()) arr.push_back(meta);
                }
                std::sort(arr.begin(), arr.end(), [](const json& a, const json& b) {
                    return a["name"].get<std::string>() < b["name"].get<std::string>();
                });
                res.set_content(json{{"voices", arr}}.dump(), "application/json");
            });

            // GET /v1/voices/raw/<name> — serve the .voice binary
            svr->Get(R"(/v1/voices/raw/([^/?]+))",
                     [voices_dir](const httplib::Request& req, httplib::Response& res) {
                std::string name = req.matches[1];
                if (name.find("..") != std::string::npos || name.find('/') != std::string::npos) {
                    res.status = 400;
                    res.set_content(R"({"error":"invalid name"})", "application/json");
                    return;
                }
                fs::path target = fs::path(voices_dir) / name;
                if (!fs::is_regular_file(target)) {
                    res.status = 404;
                    res.set_content(R"({"error":"not found"})", "application/json");
                    return;
                }
                std::ifstream f(target, std::ios::binary);
                std::string data((std::istreambuf_iterator<char>(f)),
                                  std::istreambuf_iterator<char>());
                res.set_content(std::move(data), "application/octet-stream");
            });

            // ── Voice shaping sidecar endpoints ──────────────────────────
            // <voices_dir>/<name>.voice.json — written by the Voice Maker
            // Knob 4 ("Shape voice") and read by resolve_voice_field() so
            // any Synthesize call using this voice automatically picks up
            // the saved pitch_shift + speed defaults.

            // PUT /v1/voices/<name>/meta — save DSP sidecar JSON
            svr->Put(R"(/v1/voices/([^/?]+)/meta)",
                     [voices_dir](const httplib::Request& req, httplib::Response& res) {
                json resp;
                std::string name = req.matches[1];
                if (name.empty() || name.size() < 6 ||
                    name.compare(name.size() - 6, 6, ".voice") != 0 ||
                    name.find("..") != std::string::npos ||
                    name.find('/') != std::string::npos) {
                    res.status = 400;
                    resp["error"] = "name must end in .voice and be a flat filename";
                    res.set_content(resp.dump(), "application/json");
                    return;
                }
                fs::path target = fs::path(voices_dir) / name;
                if (!fs::is_regular_file(target)) {
                    res.status = 404;
                    resp["error"] = "voice not found — upload the .voice first";
                    res.set_content(resp.dump(), "application/json");
                    return;
                }
                json body;
                try { body = json::parse(req.body); }
                catch (...) {
                    res.status = 400;
                    resp["error"] = "invalid json";
                    res.set_content(resp.dump(), "application/json");
                    return;
                }
                float ps = body.value("pitch_shift", 0.0f);
                float sp = body.value("speed",       1.0f);
                if (ps < -12.0f) ps = -12.0f;
                if (ps >  12.0f) ps =  12.0f;
                if (sp < 0.25f)  sp = 0.25f;
                if (sp > 4.0f)   sp = 4.0f;
                json out;
                out["pitch_shift"] = ps;
                out["speed"]       = sp;
                fs::path sidecar = target;
                sidecar += ".json";
                std::ofstream of(sidecar);
                if (!of) {
                    res.status = 500;
                    resp["error"] = "cannot write sidecar";
                    res.set_content(resp.dump(), "application/json");
                    return;
                }
                of << out.dump(2);
                of.close();
                resp["ok"]   = true;
                resp["name"] = name;
                resp["dsp"]  = std::move(out);
                res.set_content(resp.dump(), "application/json");
            });

            // GET /v1/voices/<name>/meta — read sidecar (or defaults)
            svr->Get(R"(/v1/voices/([^/?]+)/meta)",
                     [voices_dir](const httplib::Request& req, httplib::Response& res) {
                json resp;
                std::string name = req.matches[1];
                if (name.empty() || name.find("..") != std::string::npos ||
                    name.find('/') != std::string::npos) {
                    res.status = 400;
                    resp["error"] = "invalid name";
                    res.set_content(resp.dump(), "application/json");
                    return;
                }
                fs::path sidecar = fs::path(voices_dir) / name;
                sidecar += ".json";
                std::ifstream sf(sidecar);
                if (!sf) {
                    resp["pitch_shift"] = 0.0f;
                    resp["speed"]       = 1.0f;
                    resp["has_meta"]    = false;
                } else {
                    std::string mtext((std::istreambuf_iterator<char>(sf)),
                                       std::istreambuf_iterator<char>());
                    try {
                        auto meta = json::parse(mtext);
                        resp["pitch_shift"] = meta.value("pitch_shift", 0.0f);
                        resp["speed"]       = meta.value("speed",       1.0f);
                        resp["has_meta"]    = true;
                    } catch (...) {
                        resp["pitch_shift"] = 0.0f;
                        resp["speed"]       = 1.0f;
                        resp["has_meta"]    = false;
                    }
                }
                res.set_content(resp.dump(), "application/json");
            });

            // DELETE /v1/voices/<name>/meta — remove sidecar
            svr->Delete(R"(/v1/voices/([^/?]+)/meta)",
                        [voices_dir](const httplib::Request& req, httplib::Response& res) {
                json resp;
                std::string name = req.matches[1];
                if (name.empty() || name.find("..") != std::string::npos ||
                    name.find('/') != std::string::npos) {
                    res.status = 400;
                    resp["error"] = "invalid name";
                    res.set_content(resp.dump(), "application/json");
                    return;
                }
                fs::path sidecar = fs::path(voices_dir) / name;
                sidecar += ".json";
                if (fs::is_regular_file(sidecar)) {
                    fs::remove(sidecar);
                    resp["ok"] = true;
                } else {
                    resp["ok"] = false;
                    resp["error"] = "no sidecar";
                }
                res.set_content(resp.dump(), "application/json");
            });

            // POST /v1/voices/upload — multipart file upload (.voice files)
            svr->Post("/v1/voices/upload",
                      [voices_dir](const httplib::Request& req, httplib::Response& res) {
                json resp;
                if (!req.is_multipart_form_data()) {
                    res.status = 400;
                    resp["error"] = "expected multipart/form-data";
                    res.set_content(resp.dump(), "application/json");
                    return;
                }
                auto files = req.get_file_values("file");
                json saved = json::array();
                for (auto& f : files) {
                    if (f.filename.empty()) continue;
                    auto ext = fs::path(f.filename).extension().string();
                    for (auto& c : ext) c = static_cast<char>(std::tolower(c));
                    if (ext != ".voice") continue;
                    std::string safe = sanitize_filename(f.filename);
                    fs::path dest = fs::path(voices_dir) / safe;
                    std::ofstream out(dest, std::ios::binary);
                    out.write(f.content.data(), f.content.size());
                    saved.push_back(safe);
                }
                resp["ok"] = true;
                resp["saved"] = std::move(saved);
                res.set_content(resp.dump(), "application/json");
            });

            // POST /v1/voices/delete — delete a voice by name
            svr->Post("/v1/voices/delete",
                      [voices_dir](const httplib::Request& req, httplib::Response& res) {
                json resp;
                std::string name;
                try {
                    auto j = json::parse(req.body);
                    name = j.value("name", "");
                } catch (...) {}
                if (name.empty() || name.find("..") != std::string::npos ||
                    name.find('/') != std::string::npos) {
                    res.status = 400;
                    resp["error"] = "invalid name";
                    res.set_content(resp.dump(), "application/json");
                    return;
                }
                fs::path target = fs::path(voices_dir) / name;
                if (fs::is_regular_file(target)) {
                    fs::remove(target);
                    resp["ok"] = true;
                } else {
                    resp["ok"] = false;
                    resp["error"] = "not found";
                }
                res.set_content(resp.dump(), "application/json");
            });

            // ── Voice arithmetic endpoints ───────────────────────────────
            // These match the qwen_voice CLI subcommands and exist so the
            // Voice Maker tab in the webapp can do all three voice-editing
            // knobs (linear mix, SLERP, steering-vector shift) without a
            // shell round-trip.

            // Helper: load .voice file by name (or path) from voices_dir.
            auto load_voice_by_name = [voices_dir](const std::string& name,
                                                     std::vector<float>& out,
                                                     std::string& err) -> bool {
                if (name.empty() || name.find("..") != std::string::npos ||
                    name.find('/') != std::string::npos) {
                    err = "invalid name"; return false;
                }
                fs::path p = fs::path(voices_dir) / name;
                std::ifstream f(p, std::ios::binary);
                if (!f) { err = "voice not found: " + name; return false; }
                char hdr[32];
                if (!f.read(hdr, 32)) { err = "short read"; return false; }
                if (std::memcmp(hdr, "QWEN3VOICE", 10) != 0) {
                    err = "bad magic"; return false;
                }
                uint32_t dim = 0;
                std::memcpy(&dim, hdr + 20, 4);
                out.resize(dim);
                f.read(reinterpret_cast<char*>(out.data()),
                       static_cast<std::streamsize>(dim * 4));
                if (!f) { err = "short data"; return false; }
                return true;
            };
            auto write_voice_to_disk = [voices_dir](const std::string& name,
                                                     const std::vector<float>& v) -> bool {
                if (name.empty() || name.find("..") != std::string::npos ||
                    name.find('/') != std::string::npos) return false;
                fs::path p = fs::path(voices_dir) / name;
                std::ofstream f(p, std::ios::binary);
                if (!f) return false;
                f.write("QWEN3VOICE\0\0\0\0\0\0", 16);
                uint32_t ver = 1, dim = static_cast<uint32_t>(v.size()),
                         flags = 0, res = 0;
                f.write(reinterpret_cast<const char*>(&ver), 4);
                f.write(reinterpret_cast<const char*>(&dim), 4);
                f.write(reinterpret_cast<const char*>(&flags), 4);
                f.write(reinterpret_cast<const char*>(&res), 4);
                f.write(reinterpret_cast<const char*>(v.data()),
                       static_cast<std::streamsize>(v.size() * 4));
                return f.good();
            };

            // POST /v1/voices/blend — linear mix or SLERP between two voices.
            //   { a, b, t: [0,1], mode: "linear"|"slerp", out: "name.voice" }
            svr->Post("/v1/voices/blend",
                      [load_voice_by_name, write_voice_to_disk](
                          const httplib::Request& req, httplib::Response& res) {
                json resp;
                json body;
                try { body = json::parse(req.body); }
                catch (...) {
                    res.status = 400;
                    resp["error"] = "invalid json";
                    res.set_content(resp.dump(), "application/json");
                    return;
                }
                std::string a_name = body.value("a", "");
                std::string b_name = body.value("b", "");
                std::string out_name = body.value("out", "");
                std::string mode = body.value("mode", "linear");
                float t = body.value("t", 0.5f);
                if (t < 0.0f) t = 0.0f;
                if (t > 1.0f) t = 1.0f;
                if (a_name.empty() || b_name.empty() || out_name.empty()) {
                    res.status = 400;
                    resp["error"] = "requires a, b, out";
                    res.set_content(resp.dump(), "application/json");
                    return;
                }
                std::vector<float> a, b;
                std::string err;
                if (!load_voice_by_name(a_name, a, err) ||
                    !load_voice_by_name(b_name, b, err)) {
                    res.status = 404;
                    resp["error"] = err;
                    res.set_content(resp.dump(), "application/json");
                    return;
                }
                if (a.size() != b.size()) {
                    res.status = 400;
                    resp["error"] = "dim mismatch";
                    res.set_content(resp.dump(), "application/json");
                    return;
                }
                std::vector<float> out(a.size());
                if (mode == "slerp") {
                    // Normalize both to unit length, SLERP, re-scale to mean L2.
                    double na = 0.0, nb = 0.0;
                    for (size_t i = 0; i < a.size(); ++i) {
                        na += static_cast<double>(a[i]) * a[i];
                        nb += static_cast<double>(b[i]) * b[i];
                    }
                    na = std::sqrt(na); nb = std::sqrt(nb);
                    if (na < 1e-9 || nb < 1e-9) {
                        res.status = 400;
                        resp["error"] = "zero-norm voice";
                        res.set_content(resp.dump(), "application/json");
                        return;
                    }
                    const double avg_n = 0.5 * (na + nb);
                    double dot = 0.0;
                    for (size_t i = 0; i < a.size(); ++i) {
                        dot += (static_cast<double>(a[i]) / na) *
                               (static_cast<double>(b[i]) / nb);
                    }
                    if (dot >  1.0) dot =  1.0;
                    if (dot < -1.0) dot = -1.0;
                    const double omega = std::acos(dot);
                    if (omega < 1e-4) {
                        for (size_t i = 0; i < a.size(); ++i) {
                            double ua = static_cast<double>(a[i]) / na;
                            double ub = static_cast<double>(b[i]) / nb;
                            out[i] = static_cast<float>(((1.0 - t) * ua + t * ub) * avg_n);
                        }
                    } else {
                        const double sin_o = std::sin(omega);
                        const double wa = std::sin((1.0 - t) * omega) / sin_o;
                        const double wb = std::sin(t * omega) / sin_o;
                        for (size_t i = 0; i < a.size(); ++i) {
                            double ua = static_cast<double>(a[i]) / na;
                            double ub = static_cast<double>(b[i]) / nb;
                            out[i] = static_cast<float>((wa * ua + wb * ub) * avg_n);
                        }
                    }
                    resp["omega_rad"]  = omega;
                    resp["omega_deg"]  = omega * 180.0 / M_PI;
                } else {
                    // Linear interpolation.
                    for (size_t i = 0; i < a.size(); ++i)
                        out[i] = (1.0f - t) * a[i] + t * b[i];
                }
                if (!write_voice_to_disk(out_name, out)) {
                    res.status = 500;
                    resp["error"] = "write failed";
                    res.set_content(resp.dump(), "application/json");
                    return;
                }
                resp["ok"] = true;
                resp["name"] = out_name;
                resp["mode"] = mode;
                resp["t"] = t;
                resp["dim"] = out.size();
                res.set_content(resp.dump(), "application/json");
            });

            // POST /v1/voices/shift — apply a steering direction to a voice.
            //   { base, direction, scale, out }
            svr->Post("/v1/voices/shift",
                      [load_voice_by_name, write_voice_to_disk](
                          const httplib::Request& req, httplib::Response& res) {
                json resp;
                json body;
                try { body = json::parse(req.body); }
                catch (...) {
                    res.status = 400;
                    resp["error"] = "invalid json";
                    res.set_content(resp.dump(), "application/json");
                    return;
                }
                std::string base_name = body.value("base", "");
                std::string dir_name  = body.value("direction", "");
                std::string out_name  = body.value("out", "");
                float scale = body.value("scale", 1.0f);
                if (base_name.empty() || dir_name.empty() || out_name.empty()) {
                    res.status = 400;
                    resp["error"] = "requires base, direction, out";
                    res.set_content(resp.dump(), "application/json");
                    return;
                }
                std::vector<float> base_v, dir_v;
                std::string err;
                if (!load_voice_by_name(base_name, base_v, err) ||
                    !load_voice_by_name(dir_name,  dir_v,  err)) {
                    res.status = 404;
                    resp["error"] = err;
                    res.set_content(resp.dump(), "application/json");
                    return;
                }
                if (base_v.size() != dir_v.size()) {
                    res.status = 400;
                    resp["error"] = "dim mismatch";
                    res.set_content(resp.dump(), "application/json");
                    return;
                }
                std::vector<float> out(base_v.size());
                for (size_t i = 0; i < base_v.size(); ++i)
                    out[i] = base_v[i] + scale * dir_v[i];
                if (!write_voice_to_disk(out_name, out)) {
                    res.status = 500;
                    resp["error"] = "write failed";
                    res.set_content(resp.dump(), "application/json");
                    return;
                }
                resp["ok"] = true;
                resp["name"] = out_name;
                resp["scale"] = scale;
                res.set_content(resp.dump(), "application/json");
            });

            // POST /v1/voices/discover_direction — build a steering vector
            //   from two labelled lists of voices.
            //   { positive: ["v1.voice", ...], negative: [...], out: "name.dir" }
            svr->Post("/v1/voices/discover_direction",
                      [load_voice_by_name, write_voice_to_disk](
                          const httplib::Request& req, httplib::Response& res) {
                json resp;
                json body;
                try { body = json::parse(req.body); }
                catch (...) {
                    res.status = 400;
                    resp["error"] = "invalid json";
                    res.set_content(resp.dump(), "application/json");
                    return;
                }
                std::string out_name = body.value("out", "");
                if (out_name.empty() || !body.contains("positive") ||
                    !body.contains("negative")) {
                    res.status = 400;
                    resp["error"] = "requires positive[], negative[], out";
                    res.set_content(resp.dump(), "application/json");
                    return;
                }
                auto mean_of_list = [&](const json& arr, std::vector<float>& mean,
                                         size_t& dim, size_t& count) -> bool {
                    count = 0; dim = 0; mean.clear();
                    std::vector<float> acc;
                    for (const auto& name_j : arr) {
                        if (!name_j.is_string()) continue;
                        std::vector<float> v;
                        std::string err;
                        if (!load_voice_by_name(name_j.get<std::string>(), v, err))
                            continue;
                        if (acc.empty()) {
                            acc.assign(v.size(), 0.0f);
                            dim = v.size();
                        } else if (v.size() != dim) continue;
                        for (size_t i = 0; i < dim; ++i) acc[i] += v[i];
                        ++count;
                    }
                    if (count == 0) return false;
                    mean.assign(dim, 0.0f);
                    for (size_t i = 0; i < dim; ++i)
                        mean[i] = acc[i] / static_cast<float>(count);
                    return true;
                };
                std::vector<float> mean_p, mean_n;
                size_t dim_p, dim_n, cnt_p, cnt_n;
                if (!mean_of_list(body["positive"], mean_p, dim_p, cnt_p)) {
                    res.status = 400;
                    resp["error"] = "no positive voices loaded";
                    res.set_content(resp.dump(), "application/json");
                    return;
                }
                if (!mean_of_list(body["negative"], mean_n, dim_n, cnt_n)) {
                    res.status = 400;
                    resp["error"] = "no negative voices loaded";
                    res.set_content(resp.dump(), "application/json");
                    return;
                }
                if (dim_p != dim_n) {
                    res.status = 400;
                    resp["error"] = "dim mismatch between groups";
                    res.set_content(resp.dump(), "application/json");
                    return;
                }
                std::vector<float> dir_v(dim_p);
                for (size_t i = 0; i < dim_p; ++i)
                    dir_v[i] = mean_p[i] - mean_n[i];
                if (!write_voice_to_disk(out_name, dir_v)) {
                    res.status = 500;
                    resp["error"] = "write failed";
                    res.set_content(resp.dump(), "application/json");
                    return;
                }
                double norm = 0.0;
                for (float x : dir_v) norm += static_cast<double>(x) * x;
                norm = std::sqrt(norm);
                resp["ok"] = true;
                resp["name"] = out_name;
                resp["positive_count"] = cnt_p;
                resp["negative_count"] = cnt_n;
                resp["dim"] = dim_p;
                resp["norm"] = norm;
                res.set_content(resp.dump(), "application/json");
            });
        }
    }

    // ── Embedded webapp assets ────────────────────────────────────────────
#ifdef AUDIOCORE_HAS_WEBAPP
    svr->Get("/", [](const httplib::Request&, httplib::Response& res) {
        auto d = asset_data("index.html");
        res.set_content(std::string(d.data(), d.size()), "text/html; charset=utf-8");
    });
    svr->Get("/style.css", [](const httplib::Request&, httplib::Response& res) {
        auto d = asset_data("style.css");
        res.set_content(std::string(d.data(), d.size()), "text/css; charset=utf-8");
    });
    svr->Get("/app.js", [](const httplib::Request&, httplib::Response& res) {
        auto d = asset_data("app.js");
        res.set_content(std::string(d.data(), d.size()),
                        "application/javascript; charset=utf-8");
    });
#endif

    return svr;
}

}  // namespace audiocore
