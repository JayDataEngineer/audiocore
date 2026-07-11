// qwen3tts_proxy.cpp — Translation layer implementation.
//
// Translates our TtsRequest into the reference qwen_tts server's
// OpenAI-compatible JSON format, POSTs to /v1/audio/speech, and
// returns the WAV bytes.
//
// The reference API flow (synchronous — no job polling):
//   POST /v1/audio/speech { "input":"text","voice":"speaker",... }
//     → 200 OK audio/wav (binary)
//
// Supported JSON fields by the reference server (see qwen_tts_server.c):
//   text | input      — synthesis text (required)
//   speaker | voice   — speaker name or numeric ID
//   language          — language code ("en","zh","ja",...)
//   instruct          — natural-language style instruction (1.7B)
//   temperature       — sampling temperature [0..2]
//   top_p, top_k      — nucleus / top-k sampling
//   max_tokens        — max generation length
//   seed              — random seed (-1 = time-based)
//   volume            — output gain (1.0 = unchanged)
//   rate              — speaking rate (1.0 = unchanged, pitch-preserving)
//   response_format   — "wav" (only WAV supported on this path)

#include "audiocore/server/qwen3tts_proxy.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>

namespace audiocore {

using json = nlohmann::json;

// ── Helpers ────────────────────────────────────────────────────────────

// Parse host:port from a URL like "http://localhost:8086".
static void parse_url(const std::string& url, std::string& host, uint16_t& port) {
    host = url;
    port = 8086;
    size_t scheme = host.find("://");
    if (scheme != std::string::npos) host = host.substr(scheme + 3);
    size_t colon = host.rfind(':');
    if (colon != std::string::npos) {
        try {
            port = (uint16_t)std::stoi(host.substr(colon + 1));
        } catch (...) {}
        host = host.substr(0, colon);
    }
    // Strip any trailing slash / path.
    size_t slash = host.find('/');
    if (slash != std::string::npos) host = host.substr(0, slash);
}

// Resolve a voice field. The webapp sends "Vivian", a numeric ID, a
// .voice file path, or a .qvoice graft. The reference server wants a
// speaker name or numeric ID. If the field looks like a file path
// (contains '/' or ends in .voice/.qvoice), pass the basename without
// extension — the reference server treats unknown speakers as IDs.
static std::string resolve_speaker(const std::string& voice_path,
                                   const std::string& speaker_name) {
    if (!speaker_name.empty()) return speaker_name;
    if (!voice_path.empty()) {
        // Voice file: take basename, drop extension.
        std::string v = voice_path;
        size_t slash = v.find_last_of('/');
        if (slash != std::string::npos) v = v.substr(slash + 1);
        size_t dot = v.find_last_of('.');
        if (dot != std::string::npos) v = v.substr(0, dot);
        return v;
    }
    return "";
}

// ── Public API ─────────────────────────────────────────────────────────

std::vector<uint8_t> qwen3tts_proxy_generate(
        const Qwen3TtsProxyConfig& config,
        const TtsRequest& req,
        std::string* error) {

    std::string host;
    uint16_t port;
    parse_url(config.server_url, host, port);

    httplib::Client cli(host, port);
    cli.set_connection_timeout(30);
    cli.set_read_timeout(config.timeout_seconds);
    cli.set_write_timeout(30);

    fprintf(stderr, "[qwen3tts_proxy] → qwen_tts-server at %s:%d\n",
            host.c_str(), port);

    // Build the request JSON — translate our TtsRequest to the reference schema.
    json j;
    // Prefer the explicit text; if messages are populated, join them with
    // newlines so the reference synthesizes the full dialogue in one pass.
    if (!req.text.empty()) {
        j["input"] = req.text;
    } else if (!req.messages.empty()) {
        std::string joined;
        for (size_t i = 0; i < req.messages.size(); ++i) {
            if (i) joined += "\n";
            joined += req.messages[i].content;
        }
        j["input"] = joined;
    } else {
        if (error) *error = "no text to synthesize";
        return {};
    }

    std::string speaker = resolve_speaker(req.voice_path, req.speaker_name);
    if (!speaker.empty())  j["voice"] = speaker;
    if (!req.language.empty()) j["language"] = req.language;
    if (!req.instruct.empty()) j["instruct"] = req.instruct;

    // Sampling parameters.
    if (req.temperature > 0)   j["temperature"] = req.temperature;
    if (req.top_p > 0)         j["top_p"]       = req.top_p;
    if (req.text_top_k > 0)    j["top_k"]       = req.text_top_k;
    if (req.max_new_tokens > 0) j["max_tokens"] = req.max_new_tokens;
    if (req.seed != 0)         j["seed"]        = req.seed;
    if (std::fabs(req.repetition_penalty - 1.0f) > 0.001f)
        j["rep-penalty"] = req.repetition_penalty;

    // Rate / volume — map our speed → their rate.
    if (std::fabs(req.speed - 1.0f) > 0.001f) j["rate"] = req.speed;

    j["response_format"] = "wav";

    std::string body = j.dump();
    fprintf(stderr, "[qwen3tts_proxy] POST /v1/audio/speech (input='%s...', voice='%s')\n",
            req.text.substr(0, 50).c_str(), speaker.c_str());

    auto res = cli.Post("/v1/audio/speech", body, "application/json");
    if (!res) {
        if (error) *error = "qwen_tts-server connection failed";
        return {};
    }
    if (res->status != 200) {
        if (error) *error = "qwen_tts-server returned status " +
                            std::to_string(res->status) + ": " + res->body;
        return {};
    }

    // The reference server returns audio/wav as the response body.
    std::vector<uint8_t> wav(res->body.begin(), res->body.end());
    fprintf(stderr, "[qwen3tts_proxy] ← %zu bytes WAV\n", wav.size());
    return wav;
}

bool qwen3tts_proxy_health(const Qwen3TtsProxyConfig& config) {
    std::string host;
    uint16_t port;
    parse_url(config.server_url, host, port);
    httplib::Client cli(host, port);
    cli.set_connection_timeout(5);
    cli.set_read_timeout(5);
    auto res = cli.Get("/v1/health");
    return res && res->status == 200;
}

}  // namespace audiocore
