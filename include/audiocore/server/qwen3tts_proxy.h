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

    // ── CLI runner paths (for voice-bearing requests via shell-out) ──
    std::string binary_path;       // path to qwen_tts CLI binary
    std::string customvoice_dir;   // 1.7B-CustomVoice model dir (--load-voice path)
    std::string voicedesign_dir;   // VoiceDesign model dir (export: instruct→voice)
    std::string base_dir;          // 1.7B-Base model dir (export: bake .qvoice)
    std::string voices_dir;        // directory for saved .qvoice files
    bool        use_gpu = false;   // --backend cuda for CLI calls
    int         gpu_device = 0;    // CUDA_VISIBLE_DEVICES
};

// Generate speech via the reference qwen_tts server.
// Translates TtsRequest → reference JSON, POSTs to /v1/audio/speech,
// returns WAV file bytes, or empty vector on failure (error filled).
std::vector<uint8_t> qwen3tts_proxy_generate(
    const Qwen3TtsProxyConfig& config,
    const TtsRequest& req,
    std::string* error);

// Generate speech via CLI shell-out (for voice-bearing requests with .qvoice).
// Runs qwen_tts --load-voice --icl-only --voice-strength --emotion --instruct.
// Returns WAV file bytes, or empty vector on failure (error filled).
std::vector<uint8_t> qwen3tts_cli_generate(
    const Qwen3TtsProxyConfig& config,
    const TtsRequest& req,
    const std::string& voice_qvoice_path,
    std::string* error);

// Export a voice: VoiceDesign generate reference WAV, then bake with Base.
// Saves <voices_dir>/<name>.qvoice. Returns true on success.
// If wdelta=true, creates heavy WDELTA voice (~0.8-3GB) with --target-cv.
bool qwen3tts_cli_export_voice(
    const Qwen3TtsProxyConfig& config,
    const std::string& name,
    const std::string& instruct,
    const std::string& text,
    std::string* error,
    bool wdelta = false);

// List saved .qvoice files in voices_dir.
// Returns vector of {name, path, size_bytes}.
struct VoiceInfo {
    std::string name;
    std::string path;
    int64_t     size_bytes = 0;
};
std::vector<VoiceInfo> qwen3tts_list_voices(
    const Qwen3TtsProxyConfig& config);

// Low-level CLI helpers (exposed for server.cpp direct use).
int qwen3tts_run_cli(const std::string& binary,
                     const std::vector<std::string>& argv,
                     const std::vector<std::string>& envp);
std::vector<std::string> qwen3tts_build_cli_env(const Qwen3TtsProxyConfig& config);
std::vector<uint8_t> qwen3tts_read_file(const std::string& path);

// Health check: returns true if the reference qwen_tts server responds
// to GET /v1/health.
bool qwen3tts_proxy_health(const Qwen3TtsProxyConfig& config);

}  // namespace audiocore
