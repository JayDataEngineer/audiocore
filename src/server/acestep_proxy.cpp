// acestep_proxy.cpp — Translation layer implementation.
//
// Translates our MusicRequest into the reference AceRequest format,
// drives the async /lm + /synth job pipeline, and extracts WAV audio
// from the multipart response.
//
// The reference API flow:
//   1. POST /lm  { AceRequest JSON }         → { "id": "jobid" }
//   2. GET  /job?id=jobid                    → { "status": "running"|"done"|"error" }
//   3. GET  /job?id=jobid&result=1           → [ AceRequest (enriched with audio_codes) ]
//   4. POST /synth { enriched AceRequest }   → { "id": "jobid2" }
//   5. GET  /job?id=jobid2                   → { "status": "running"|"done"|"error" }
//   6. GET  /job?id=jobid2&result=1          → multipart/audio-wav (binary)

#include "audiocore/server/acestep_proxy.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>

namespace audiocore {

using json = nlohmann::json;

// ── Helpers ────────────────────────────────────────────────────────────

static json build_ace_request(const AceStepProxyConfig& cfg,
                              const acestep::MusicRequest& req) {
    json j;
    j["caption"]       = req.caption;
    j["lyrics"]        = req.lyrics;
    j["duration"]      = req.duration;
    j["lm_mode"]       = "generate";
    j["output_format"] = "wav16";

    // Musical metadata — when all provided, Phase 1 is skipped.
    if (req.bpm > 0)               j["bpm"]             = req.bpm;
    if (!req.keyscale.empty())     j["keyscale"]        = req.keyscale;
    if (!req.timesignature.empty())j["timesignature"]   = req.timesignature;
    if (!req.vocal_language.empty())j["vocal_language"] = req.vocal_language;

    // DiT variant selection: ace-server names models by FILENAME (e.g.
    // "acestep-v15-turbo-Q8_0.gguf"), not by variant label. When only one DiT
    // is present in models_dir, resolve_name() falls through to bucket[0], so
    // we omit synth_model entirely and let ace-server auto-select. To target a
    // specific DiT, set synth_model to the exact GGUF filename via the
    // "dit_variant" config extra.
    if (!cfg.dit_variant.empty() && cfg.dit_variant.find(".gguf") != std::string::npos)
        j["synth_model"] = cfg.dit_variant;

    // VAE override — ScragVAE or other community VAEs. Must be the exact
    // GGUF filename present in ace-server's models_dir.
    if (!cfg.vae_override.empty())
        j["vae"] = cfg.vae_override;

    // Sampling params (reference defaults: temp=0.85, cfg=2.0, top_p=0.9).
    j["lm_temperature"] = req.temperature;
    j["lm_cfg_scale"]   = req.lm_cfg_scale;
    j["lm_top_p"]       = req.top_p;

    // DiT params.
    if (req.guidance_scale > 0)    j["guidance_scale"]  = req.guidance_scale;
    if (req.n_diffusion_steps > 0) j["inference_steps"] = req.n_diffusion_steps;

    // Seed.
    if (req.seed != 0) {
        j["seed"]    = req.seed;
        j["lm_seed"] = req.seed;
    }

    return j;
}

// Poll a job until done/error/timeout. Returns the final status JSON.
static json wait_for_job(httplib::Client& cli, const std::string& job_id,
                         int timeout_s) {
    auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - t0);
        if (elapsed.count() >= timeout_s)
            return {{"status", "timeout"}};

        auto res = cli.Get("/job?id=" + job_id);
        if (!res || res->status != 200) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
        }
        try {
            auto j = json::parse(res->body);
            std::string status = j.value("status", "");
            if (status == "done" || status == "error" || status == "failed")
                return j;
        } catch (...) {}
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

// Extract WAV data from an ace-server multipart response body.
// The body looks like:
//   --ace-batch-boundary\r\nContent-Type: audio/wav\r\n\r\nRIFF....\r\n--ace-batch-boundary--
// We find "RIFF" and extract from there to the closing boundary.
static std::vector<uint8_t> extract_wav(const std::string& body) {
    // Find RIFF header.
    size_t riff_pos = body.find("RIFF");
    if (riff_pos == std::string::npos) return {};

    // Find closing boundary after RIFF.
    const char boundary[] = "--ace-batch-boundary";
    size_t end = body.find(boundary, riff_pos);
    if (end == std::string::npos) end = body.size();

    // Trim trailing \r\n before boundary.
    while (end > riff_pos &&
           (body[end - 1] == '\r' || body[end - 1] == '\n'))
        --end;

    return std::vector<uint8_t>(body.begin() + riff_pos,
                                body.begin() + end);
}

// ── Public API ─────────────────────────────────────────────────────────

std::vector<uint8_t> acestep_proxy_generate(
        const AceStepProxyConfig& config,
        const acestep::MusicRequest& req,
        std::string* error) {

    // Parse host:port from ace_server_url.
    // Expected format: "http://localhost:8085"
    std::string host = config.ace_server_url;
    uint16_t port = 8085;
    // Strip http:// or https://
    size_t scheme = host.find("://");
    if (scheme != std::string::npos) host = host.substr(scheme + 3);
    // Split host:port
    size_t colon = host.rfind(':');
    if (colon != std::string::npos) {
        port = (uint16_t)std::stoi(host.substr(colon + 1));
        host = host.substr(0, colon);
    }

    httplib::Client cli(host, port);
    cli.set_connection_timeout(30);
    cli.set_read_timeout(config.timeout_seconds);
    cli.set_write_timeout(30);

    fprintf(stderr, "[acestep_proxy] → ace-server at %s:%d\n", host.c_str(), port);

    // ── Step 1: POST /lm ──
    json ace_req = build_ace_request(config, req);
    auto lm_body = ace_req.dump();

    fprintf(stderr, "[acestep_proxy] POST /lm (caption='%s...', dur=%.0f)\n",
            req.caption.substr(0, 50).c_str(), req.duration);

    auto lm_res = cli.Post("/lm", lm_body, "application/json");
    if (!lm_res || lm_res->status != 200) {
        if (error) *error = "ace-server /lm failed: " +
                            (lm_res ? std::to_string(lm_res->status) : std::string("connection"));
        return {};
    }

    std::string lm_job_id;
    try {
        auto j = json::parse(lm_res->body);
        lm_job_id = j.value("id", "");
    } catch (...) {
        if (error) *error = "ace-server /lm returned invalid JSON";
        return {};
    }
    if (lm_job_id.empty()) {
        if (error) *error = "ace-server /lm returned empty job id";
        return {};
    }
    fprintf(stderr, "[acestep_proxy] LM job: %s\n", lm_job_id.c_str());

    // ── Step 2: Poll LM job ──
    auto lm_status = wait_for_job(cli, lm_job_id, config.timeout_seconds);
    if (lm_status.value("status", "") != "done") {
        if (error) *error = "ace-server LM job failed: " + lm_status.dump();
        return {};
    }
    fprintf(stderr, "[acestep_proxy] LM done\n");

    // ── Step 3: GET LM result (enriched AceRequest with audio_codes) ──
    auto lm_result_res = cli.Get("/job?id=" + lm_job_id + "&result=1");
    if (!lm_result_res || lm_result_res->status != 200) {
        if (error) *error = "ace-server LM result fetch failed";
        return {};
    }

    json enriched_req;
    try {
        auto arr = json::parse(lm_result_res->body);
        if (arr.is_array() && !arr.empty()) {
            enriched_req = arr[0];
        } else if (arr.is_object()) {
            enriched_req = arr;
        } else {
            if (error) *error = "ace-server LM result: unexpected format";
            return {};
        }
    } catch (...) {
        if (error) *error = "ace-server LM result: invalid JSON";
        return {};
    }

    // Verify we got codes.
    std::string codes = enriched_req.value("audio_codes", "");
    if (codes.empty()) {
        if (error) *error = "ace-server LM produced no audio codes";
        return {};
    }
    fprintf(stderr, "[acestep_proxy] LM produced %zu chars of codes\n", codes.size());

    // ── Step 4: POST /synth ──
    enriched_req["output_format"] = "wav16";
    // Reiterate synth_model only when targeting a specific GGUF filename
    // (see comment in build_ace_request). Empty = ace-server auto-selects.
    if (!config.dit_variant.empty() && config.dit_variant.find(".gguf") != std::string::npos)
        enriched_req["synth_model"] = config.dit_variant;
    // Reiterate VAE override for the synth step.
    if (!config.vae_override.empty())
        enriched_req["vae"] = config.vae_override;

    auto synth_body = enriched_req.dump();
    fprintf(stderr, "[acestep_proxy] POST /synth\n");

    auto synth_res = cli.Post("/synth", synth_body, "application/json");
    if (!synth_res || synth_res->status != 200) {
        if (error) *error = "ace-server /synth failed: " +
                            (synth_res ? std::to_string(synth_res->status) : std::string("connection"));
        return {};
    }

    std::string synth_job_id;
    try {
        auto j = json::parse(synth_res->body);
        synth_job_id = j.value("id", "");
    } catch (...) {
        if (error) *error = "ace-server /synth returned invalid JSON";
        return {};
    }
    fprintf(stderr, "[acestep_proxy] Synth job: %s\n", synth_job_id.c_str());

    // ── Step 5: Poll synth job ──
    auto synth_status = wait_for_job(cli, synth_job_id, config.timeout_seconds);
    if (synth_status.value("status", "") != "done") {
        if (error) *error = "ace-server synth job failed: " + synth_status.dump();
        return {};
    }
    fprintf(stderr, "[acestep_proxy] Synth done\n");

    // ── Step 6: GET synth result (multipart with WAV) ──
    auto result_res = cli.Get("/job?id=" + synth_job_id + "&result=1");
    if (!result_res || result_res->status != 200) {
        if (error) *error = "ace-server synth result fetch failed";
        return {};
    }

    auto wav = extract_wav(result_res->body);
    if (wav.empty()) {
        if (error) *error = "ace-server synth result: no WAV data found in multipart";
        return {};
    }

    fprintf(stderr, "[acestep_proxy] ← %zu bytes WAV\n", wav.size());
    return wav;
}

bool acestep_proxy_health(const AceStepProxyConfig& config) {
    std::string host = config.ace_server_url;
    uint16_t port = 8085;
    size_t scheme = host.find("://");
    if (scheme != std::string::npos) host = host.substr(scheme + 3);
    size_t colon = host.rfind(':');
    if (colon != std::string::npos) {
        port = (uint16_t)std::stoi(host.substr(colon + 1));
        host = host.substr(0, colon);
    }
    httplib::Client cli(host, port);
    cli.set_connection_timeout(5);
    auto res = cli.Get("/health");
    return res && res->status == 200;
}

}  // namespace audiocore
