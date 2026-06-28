// sampler.h — unified token sampler for audiocore.
//
// One implementation of top-k / top-p / temperature / repetition-penalty
// sampling shared by every family (moss_tts, qwen3_tts, ace_step) and by
// the qwen3::Runner MTP predictor. Before this header existed, three
// near-duplicate samplers lived in:
//   • src/models/moss_tts/sampler.cpp        (top-k + top-p + rep penalty)
//   • src/framework/models/qwen3/runner.cpp  (temp + top-p, "sample_token_basic")
//   • src/models/qwen3_tts/session.cpp       (temp + top-p, local static)
// All three now delegate to sampler::sample_token.

#ifndef AUDIOCORE_FRAMEWORK_SAMPLING_SAMPLER_H
#define AUDIOCORE_FRAMEWORK_SAMPLING_SAMPLER_H

#include <cstdint>
#include <random>

namespace audiocore::sampler {

// All knobs the unified sampler supports. Defaults give plain multinomial
// sampling with no filtering — equivalent to a single softmax draw.
struct Params {
    float temperature        = 1.0f;   // 1.0 = no scaling (use logits as-is)
    float top_p              = 1.0f;   // 1.0 = no nucleus filtering
    int   top_k              = 0;      // 0 = no top-k filtering
    float repetition_penalty = 1.0f;   // 1.0 = no penalty
    bool  do_sample          = true;   // false = greedy argmax
};

// Sample one token id from logits[0..vocab_size).
//
// `prev_tokens` (optional) is the history used for repetition penalty; it
// is only consulted when Params::repetition_penalty != 1.0f.
//
// `rng` (optional) selects the RNG used for the multinomial draw. Pass
// nullptr to use a thread-local std::mt19937 seeded from random_device.
// Ignored when Params::do_sample == false (argmax path is deterministic).
//
// Returns a token id in [0, vocab_size). On degenerate input
// (vocab_size <= 0 or all -inf logits) returns 0.
int32_t sample_token(const float* logits, int32_t vocab_size,
                     const Params& params,
                     const int32_t* prev_tokens = nullptr,
                     int32_t n_prev = 0,
                     std::mt19937* rng = nullptr);

}  // namespace audiocore::sampler

#endif  // AUDIOCORE_FRAMEWORK_SAMPLING_SAMPLER_H
