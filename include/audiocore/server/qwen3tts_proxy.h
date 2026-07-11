// qwen3tts_proxy.h — Translation layer between our /v1/audio/speech API
// and the reference qwen_tts server (POST /v1/audio/speech, OpenAI-compatible).
//
// WHY: Our own Qwen3-TTS reimplementation through our llama.cpp fork produces
// corrupted, machine-gun-stuttering audio (VLM verdict: FAIL). The reference
// implementation (gabriele-mastrapasqua/qwen3-tts — pure C, own kernels,
// safetensors-direct) generates clean, natural speech (VLM verdict: PASS).
// This proxy routes TTS requests to the reference server, giving us
// reference-quality output with zero reimplementation risk.
//
// ARCHITECTURE:
//   client → our server (:39517) /v1/audio/speech
//          → [proxy] → qwen_tts-server (:8086) POST /v1/audio/speech
//                     ← audio/wav (synchronous, no async job polling)
//          ← WAV returned to client
//
// The reference server is synchronous and OpenAI-compatible, so the proxy
// is a single HTTP round-trip — far simpler than acestep_proxy's job polling.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "audiocore/framework/runtime/tasks.h"

namespace audiocore {

struct Qwen3TtsProxyConfig {
    // Base URL of the reference qwen_tts server, e.g. "http://localhost:8086".
    std::string server_url;

    // Path to the model directory the reference server was started with.
    // Used only for documentation / health checks.
    std::string model_dir;

    // Max time to wait for the synthesis response (seconds).
    int timeout_seconds = 600;
};

// Generate speech via the reference qwen_tts server.
// Translates TtsRequest → reference JSON, POSTs to /v1/audio/speech,
// returns WAV file bytes, or empty vector on failure (error filled).
std::vector<uint8_t> qwen3tts_proxy_generate(
    const Qwen3TtsProxyConfig& config,
    const TtsRequest& req,
    std::string* error);

// Health check: returns true if the reference qwen_tts server responds
// to GET /v1/health.
bool qwen3tts_proxy_health(const Qwen3TtsProxyConfig& config);

}  // namespace audiocore
