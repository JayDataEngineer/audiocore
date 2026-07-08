// delay_state.h — MOSS-TTS delay-pattern autoregressive state machine.
//
// The delay pattern introduces a diagonal time-shift across codebooks:
//   Head 0 predicts text at t, Head k predicts codebook k-1 at t-(k-1).
// Ported from upstream moss_tts_delay/llama_cpp/delay_state.py.

#ifndef AUDIOCORE_MODELS_MOSS_TTS_DELAY_STATE_H
#define AUDIOCORE_MODELS_MOSS_TTS_DELAY_STATE_H

#include <cstdint>
#include <limits>
#include <vector>

#include "audiocore/framework/sampling/sampler.h"

namespace audiocore::moss {

// ── MOSS token IDs (match upstream _constants.py) ─────────────────────────
inline constexpr int32_t PAD_TOKEN_ID = 151643;
inline constexpr int32_t IM_START_TOKEN_ID = 151644;
inline constexpr int32_t IM_END_TOKEN_ID = 151645;
inline constexpr int32_t AUDIO_START_TOKEN_ID = 151652;
inline constexpr int32_t AUDIO_END_TOKEN_ID = 151653;
inline constexpr int32_t AUDIO_USER_SLOT_TOKEN_ID = 151654;
inline constexpr int32_t AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID = 151656;
inline constexpr int32_t AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID = 151662;
inline constexpr int32_t AUDIO_PAD_CODE = 1024;
inline constexpr int32_t AUDIO_VOCAB_SIZE = 1024;  // 0-1023, pad=1024

// ── Sampling configuration ────────────────────────────────────────────────
struct SamplingConfig {
    float text_temperature = 1.5f;
    float text_top_p = 1.0f;
    int   text_top_k = 50;
    float audio_temperature = 1.7f;
    float audio_top_p = 0.8f;
    int   audio_top_k = 25;
    float audio_repetition_penalty = 1.0f;
    audiocore::sampler::PhiloxRng rng{0};        // seeded in run_tts from req->seed
};

// ── Delay state (batch_size=1) ────────────────────────────────────────────
struct DelayState {
    int32_t audio_length = 0;
    int32_t delayed_length = std::numeric_limits<int32_t>::max();
    bool    is_audio = false;
    bool    is_stopping = false;
    int32_t time_step = 0;
    int32_t n_vq = 32;
    std::vector<int32_t> text_history;

    // Audio history: each entry is n_vq int32 codes.
    std::vector<std::vector<int32_t>> audio_buf;
};

// Initialize delay state from prompt multi-channel tokens.
// prompt: (S, 1+n_vq) — channel 0 = text tokens, channels 1..n_vq = audio codes.
DelayState init_delay_state(const std::vector<std::vector<int32_t>>& prompt,
                            int32_t n_vq);

// Execute one autoregressive step. Returns (1+n_vq) next token IDs.
//   text_logits:  (vocab_size,) — llama text logits at last position
//   audio_logits: (n_vq, audio_vocab_size+1) — per-stream audio logits
// config is taken by non-const reference because the rng inside it advances.
std::vector<int32_t> delay_step(DelayState& state,
                                const float* text_logits, int32_t text_vocab,
                                const float* audio_logits, int32_t audio_vocab,
                                SamplingConfig& config);

// Remove delay pattern from generated audio codes.
// delay_codes: (T_delayed, n_vq) — the audio channels from generation
// Returns: (T, n_vq) — de-delayed codes, T = T_delayed - n_vq + 1
std::vector<std::vector<int32_t>> apply_de_delay_pattern(
    const std::vector<std::vector<int32_t>>& delay_codes,
    int32_t n_vq);

// Extract non-padding audio segments after de-delaying.
std::vector<std::vector<int32_t>> extract_audio_segments(
    const std::vector<std::vector<int32_t>>& codes,
    int32_t n_vq);

}  // namespace audiocore::moss

#endif  // AUDIOCORE_MODELS_MOSS_TTS_DELAY_STATE_H
