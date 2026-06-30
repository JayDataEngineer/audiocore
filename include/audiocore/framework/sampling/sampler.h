#ifndef AUDIOCORE_FRAMEWORK_SAMPLING_SAMPLER_H
#define AUDIOCORE_FRAMEWORK_SAMPLING_SAMPLER_H

#include <cstdint>

namespace audiocore::sampler {

// Counter-based PRNG (Philox4x32-10) — deterministic, seedable, thread-safe.
// Replace mt19937 with this for reproducible sampling.
struct PhiloxRng {
    int64_t seed;
    int64_t subseq = 0;
};

// All knobs the unified sampler supports. Defaults give plain multinomial
// sampling with no filtering — equivalent to a single softmax draw.
struct Params {
    float temperature        = 1.0f;
    float top_p              = 1.0f;
    int   top_k              = 0;
    float repetition_penalty = 1.0f;
    bool  do_sample          = true;
};

// Sample one token id from logits[0..vocab_size).
//
// `prev_tokens` (optional) is the history used for repetition penalty; it
// is only consulted when Params::repetition_penalty != 1.0f.
//
// `rng` (optional) selects the RNG used for the multinomial draw. Pass
// nullptr to use a thread-local default. Ignored when do_sample == false.
//
// Returns a token id in [0, vocab_size). On degenerate input
// (vocab_size <= 0 or all -inf logits) returns 0.
int32_t sample_token(const float* logits, int32_t vocab_size,
                     const Params& params,
                     const int32_t* prev_tokens = nullptr,
                     int32_t n_prev = 0,
                     PhiloxRng* rng = nullptr);

}  // namespace audiocore::sampler

#endif  // AUDIOCORE_FRAMEWORK_SAMPLING_SAMPLER_H
