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
#include <string>
#include <vector>

namespace audiocore {

// ── Text-to-Speech ────────────────────────────────────────────────────────
struct TtsRequest {
    // ── Core (every TTS family reads these) ──
    std::string text;                  // text to synthesize
    std::string language;              // "en", "zh", "ja", … empty = auto
    float       temperature    = 0.8f;
    float       top_p          = 0.9f;
    int32_t     max_new_tokens = 0;    // 0 → model default
    int32_t     seed           = 0;    // 0 → nondeterministic

    // ── Voice / cloning ──
    std::string voice_path;            // reference audio file (MOSS clone)
    std::string speaker_name;          // named speaker ("Vivian", "Ryan", …)
    std::string instruct;              // natural-language style instruction
    std::string reference_audio;       // path to reference audio (Qwen3-TTS base)
    std::string reference_text;        // reference text for ICL cloning
    std::string speaker_embedding;     // pre-computed embedding (opaque blob)

    // ── Family-specific knobs ──
    std::string mode = "tts";          // MOSS: "tts" | "sfx" | "voice_clone"
    float       speed = 1.0f;          // Qwen3-TTS: speed multiplier
};

struct TtsResponse {
    std::vector<float> pcm_mono;
    int32_t            sampling_rate = 24000;
    std::string        error;          // optional diagnostic
};

}  // namespace audiocore

#endif  // AUDIOCORE_FRAMEWORK_RUNTIME_TASKS_H
