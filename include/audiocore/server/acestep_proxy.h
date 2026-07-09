// acestep_proxy.h — Translation layer between our /v1/audio/music API
// and the reference acestep.cpp ace-server (POST /lm + POST /synth).
//
// WHY: Our own ACE-Step reimplementation (session.cpp + dit_runner.cpp +
// vae_runner.cpp = 6606 lines) has a broken LM that collapses to 24/150
// unique codes and a VAE 17x slower than the reference. The reference
// (acestep.cpp by ServeurpersoCom) generates 148/150 unique codes and
// runs DiT in 145ms + VAE in 228ms. This proxy routes music requests to
// the reference server, giving us reference-quality output with zero
// reimplementation risk.
//
// ARCHITECTURE:
//   client → our server (:39517) /v1/audio/music
//          → [proxy] → ace-server (:8085) POST /lm → codes
//                     → ace-server (:8085) POST /synth → WAV
//          ← WAV returned to client
//
// The reference uses async jobs: POST creates a job, GET /job?id=X polls.
// This proxy handles all the polling internally and returns synchronously.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "audiocore/models/ace_step/family.h"

namespace audiocore {

struct AceStepProxyConfig {
    // Base URL of the reference ace-server, e.g. "http://localhost:8085".
    std::string ace_server_url;

    // Which DiT variant to request ("turbo", "xl-turbo", "base", "sft").
    // Empty = let ace-server pick the first DiT in its registry.
    std::string dit_variant;

    // VAE override — exact GGUF filename in ace-server's models_dir.
    // Used for ScragVAE swap-in ("vae-scrag-BF16.gguf").
    // Empty = ace-server uses its default VAE.
    std::string vae_override;

    // Max time to wait for each LM/synth job (seconds).
    int timeout_seconds = 600;
};

// Generate music via the reference ace-server.
// Translates MusicRequest → AceRequest, calls /lm then /synth,
// extracts WAV from the multipart response.
// Returns WAV file bytes, or empty vector on failure (error filled).
std::vector<uint8_t> acestep_proxy_generate(
    const AceStepProxyConfig& config,
    const acestep::MusicRequest& req,
    std::string* error);

// Health check: returns true if ace-server responds to GET /health.
bool acestep_proxy_health(const AceStepProxyConfig& config);

}  // namespace audiocore
