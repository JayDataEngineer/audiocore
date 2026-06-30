#include "audiocore/models/moss_tts/sampler.h"

#include "audiocore/framework/sampling/sampler.h"

namespace audiocore::moss {

int32_t sample_token(const float* logits, int32_t vocab_size,
                     const int32_t* prev_tokens, int32_t n_prev,
                     float repetition_penalty,
                     float top_p, int top_k,
                     bool do_sample,
                     PhiloxRng* rng) {
    sampler::Params p;
    p.temperature        = 1.0f;
    p.top_p              = top_p;
    p.top_k              = top_k;
    p.repetition_penalty = repetition_penalty;
    p.do_sample          = do_sample;
    return sampler::sample_token(logits, vocab_size, p,
                                 prev_tokens, n_prev, rng);
}

void sample_audio_tokens(const float* logits, int32_t n_streams, int32_t vocab_size,
                         int32_t* out, const int32_t* prev_tokens,
                         int32_t n_prev,
                         float repetition_penalty,
                         float top_p, int top_k,
                         bool do_sample) {
    for (int32_t s = 0; s < n_streams; ++s) {
        const float* row = logits + static_cast<size_t>(s) * vocab_size;
        const int32_t* prev = prev_tokens
            ? prev_tokens + static_cast<size_t>(s)
            : nullptr;
        const int32_t n_prev_s = (prev_tokens && n_prev > 0) ? n_prev : 0;
        out[s] = sample_token(row, vocab_size,
                              prev, n_prev_s,
                              repetition_penalty,
                              top_p, top_k,
                              do_sample);
    }
}

}  // namespace audiocore::moss
