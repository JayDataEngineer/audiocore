// sampler.h — MOSS-TTS sampling functions (top-k, top-p, repetition penalty).
// Ported from upstream moss_tts_delay/llama_cpp/sampling.py.

#ifndef AUDIOCORE_MODELS_MOSS_TTS_SAMPLER_H
#define AUDIOCORE_MODELS_MOSS_TTS_SAMPLER_H

#include <cstdint>
#include <random>
#include <vector>

namespace audiocore::moss {

// Sample a single token from logits with optional filtering.
//   logits: (vocab_size,) row-major
//   prev_tokens: optional pointer to previously sampled tokens (for rep penalty)
//   n_prev: number of prev tokens
//   returns: sampled token id (int32)
int32_t sample_token(const float* logits, int32_t vocab_size,
                     const int32_t* prev_tokens = nullptr, int32_t n_prev = 0,
                     float repetition_penalty = 1.0f,
                     float top_p = 1.0f, int top_k = 0,
                     bool do_sample = true,
                     std::mt19937* rng = nullptr);

// Sample from per-stream audio logits. Convenience wrapper for the 2D case.
//   logits: (n_streams, vocab_size) row-major
//   prev_tokens: (n_prev, n_streams) or nullptr
//   returns: (n_streams,) sampled token ids
void sample_audio_tokens(const float* logits, int32_t n_streams, int32_t vocab_size,
                         int32_t* out, const int32_t* prev_tokens = nullptr,
                         int32_t n_prev = 0,
                         float repetition_penalty = 1.0f,
                         float top_p = 1.0f, int top_k = 0,
                         bool do_sample = true);

}  // namespace audiocore::moss

#endif  // AUDIOCORE_MODELS_MOSS_TTS_SAMPLER_H
