// tasks.h — unified task I/O types for audiocore.
//
// One TtsRequest / TtsResponse shared by every TTS family (moss_tts,
// qwen3_tts). Families read the subset of fields they care about and ignore
// the rest, so the server's /v1/audio/speech handler parses JSON into
// TtsRequest exactly once regardless of which family serves the request.
//
// Until Stage 6 the moss and qwen3_tts families shipped their own
// request/response structs with overlapping-but-not-identical fields. The
// server held a branch per family to translate JSON → family struct. Both
// are gone now: there is one struct and one code path.
//
// MusicRequest / MusicResponse remain family-specific for now — only
// ace_step serves music and the field set is genuinely different from TTS
// (caption, lyrics, duration, guidance_scale, n_diffusion_steps). They
// still live in ace_step/family.h.

#ifndef AUDIOCORE_FRAMEWORK_RUNTIME_TASKS_H
#define AUDIOCORE_FRAMEWORK_RUNTIME_TASKS_H

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace audiocore {

// ── Chat message (OpenAI-compatible) ──────────────────────────────────────
struct ChatMessage {
    std::string role;     // "system" | "user" | "assistant"
    std::string content;
};

// ── Streaming / progress callbacks ────────────────────────────────────────
struct AudioStreamCallbacks {
    // Called with PCM chunks (mono float32, native sample rate) during
    // generation.  The family emits audio as it's produced — families that
    // decode incrementally (MOSS: during AR loop; ACE-Step: during denoising)
    // call this per-chunk while the generation loop runs.  Families that
    // only produce audio at the end (Qwen3-TTS talker + predictor) batch-
    // decode and call this once at completion.
    //
    // Return true to continue generation, false to abort.
    // Thread-safety: called from the family's generation thread.
    std::function<bool(const float* pcm, size_t n_samples)> on_audio;

    // Called to report inference progress. phase is a human-readable
    // phase name (e.g. "prefill", "ar_decode", "codec_decode"). pct is
    // 0.0–1.0. Called from the generation thread.
    std::function<void(const char* phase, float pct, const char* msg)> on_progress;
};

// ── Voice Activity Detection (silero_vad family) ──────────────────────────
struct VadSegment {
    int64_t start_sample = 0;
    int64_t end_sample   = 0;
    float   start_sec    = 0.0f;
    float   end_sec      = 0.0f;
};

struct VadRequest {
    std::string audio_path;              // input WAV (16 kHz mono expected)
    int         sample_rate          = 16000;
    float       threshold            = 0.5f;   // speech probability threshold
    float       min_speech_dur_sec   = 0.0f;
    float       max_speech_dur_sec   = 30.0f;
    float       min_silence_dur_sec  = 2.0f;
    float       speech_pad_ms        = 30.0f;
    // If true, populate VadResponse.probabilities with per-chunk speech
    // probabilities (debug / visualization). Otherwise left empty.
    bool        emit_probabilities   = false;
};

struct VadResponse {
    std::vector<VadSegment> segments;
    std::vector<float>      probabilities;  // per-chunk, only if requested
    std::string             error;
};

// ── Text-to-Speech ────────────────────────────────────────────────────────
struct TtsRequest {
    // ── Core (every TTS family reads these) ──
    std::string text;                  // text to synthesize
    std::string language;              // "en", "zh", "ja", … empty = auto
    float       temperature    = 0.8f;
    float       top_p          = 0.9f;
    int32_t     max_new_tokens = 0;    // 0 → model default
    int32_t     seed           = 0;    // 0 → nondeterministic

    // ── Multi-turn / dialogue ──
    std::vector<ChatMessage> messages; // replaces `text` when populated

    // ── Voice / cloning ──
    std::string voice_path;            // reference audio file (MOSS clone)
    std::string speaker_name;          // named speaker ("Vivian", "Ryan", …)
    std::string instruct;              // natural-language style instruction
    std::string reference_audio;       // path to reference audio (Qwen3-TTS base)
    std::string reference_text;        // reference text for ICL cloning
    std::vector<float> speaker_embedding;  // pre-computed embedding (float vector, bypasses WAV load)

    // ── Streaming ──
    AudioStreamCallbacks* stream = nullptr;  // non-null → enable streaming

    // ── Output format ──
    std::string response_format = "wav"; // "wav" | "mp3"

    // ── MOSS-specific knobs ──
    std::string mode = "tts";          // MOSS: "tts" | "voice_clone"
    std::string quality;               // MOSS: free-text quality hint (empty = "None")
    int32_t     duration_tokens = 0;   // MOSS: duration hint in codec frames (0 = "None"; 1s ≈ 12.5)

    // ── Diffusion (moss_sfx_v2) knobs ──
    int32_t     n_diffusion_steps = 50;     // number of denoising steps
    float       guidance_scale    = 5.0f;   // CFG guidance scale

    // ── Per-family text-column sampling (MOSS exposes independent text/audio params) ──
    float       text_temperature = 0.0f;  // 0 → use family default (MOSS: 1.5)
    float       text_top_p       = 0.0f;  // 0 → use family default (MOSS: 1.0)
    int32_t     text_top_k       = 0;     // 0 → use family default (MOSS: 50)

    // ── Qwen3-TTS knobs ──
    float       speed = 1.0f;                 // speed multiplier
    float       repetition_penalty = 1.05f;   // repetition penalty (CB0, HuggingFace style)
    float       embedding_strength = 1.0f;    // speaker embedding scale [0.0..2.0+]; 1.0=normal, <1=softer, >1=stronger

    // ── Voice sidecar DSP defaults ──
    // Populated by resolve_voice_field() when a `<name>.voice.json` sidecar
    // exists next to the loaded `<name>.voice` file. These are DEFAULTS —
    // explicit `pitch_shift` / `speed` in the request body still override.
    // The frontend Voice Maker writes the sidecar via PUT /v1/voices/<n>/meta.
    float       voice_pitch_shift = 0.0f;     // semitones, -12..+12
    float       voice_speed       = 1.0f;     // 0.5..2.0
    bool        has_voice_meta    = false;    // true once sidecar loaded
};

struct TtsResponse {
    std::vector<float> pcm_mono;
    int32_t            sampling_rate = 24000;
    std::string        error;          // optional diagnostic
};

}  // namespace audiocore

#endif  // AUDIOCORE_FRAMEWORK_RUNTIME_TASKS_H
