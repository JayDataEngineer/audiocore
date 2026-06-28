// sampler.cpp — MOSS-TTS sampling functions.
// Ported from moss_tts_delay/llama_cpp/sampling.py.

#include "audiocore/models/moss_tts/sampler.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>

namespace audiocore::moss {

namespace {

// Thread-local RNG seeded once so each sample call gets independent draws.
std::mt19937& rng() {
    static thread_local std::mt19937 gen(std::random_device{}());
    return gen;
}

// Stable softmax over the last axis. Input length = n.
void softmax_inplace(float* logits, int32_t n) {
    float maxv = logits[0];
    for (int32_t i = 1; i < n; ++i)
        if (logits[i] > maxv) maxv = logits[i];
    float sum = 0.0f;
    for (int32_t i = 0; i < n; ++i) {
        logits[i] = std::exp(logits[i] - maxv);
        sum += logits[i];
    }
    float inv_sum = 1.0f / (sum + 1e-9f);
    for (int32_t i = 0; i < n; ++i)
        logits[i] *= inv_sum;
}

// argmax
int32_t argmax(const float* logits, int32_t n) {
    int32_t best = 0;
    float bestv = logits[0];
    for (int32_t i = 1; i < n; ++i) {
        if (logits[i] > bestv) {
            bestv = logits[i];
            best = i;
        }
    }
    return best;
}

} // anonymous namespace

int32_t sample_token(const float* logits, int32_t vocab_size,
                     const int32_t* prev_tokens, int32_t n_prev,
                     float repetition_penalty,
                     float top_p, int top_k,
                     bool do_sample) {
    // Copy logits to scratch buffer for manipulation.
    std::vector<float> buf(logits, logits + vocab_size);

    // Repetition penalty
    if (prev_tokens && repetition_penalty != 1.0f) {
        for (int32_t i = 0; i < n_prev; ++i) {
            int32_t tok = prev_tokens[i];
            if (tok < 0 || tok >= vocab_size) continue;
            if (buf[tok] > 0)
                buf[tok] /= repetition_penalty;
            else
                buf[tok] *= repetition_penalty;
        }
    }

    if (!do_sample)
        return argmax(buf.data(), vocab_size);

    // Top-K: keep only top-k values
    if (top_k > 0 && top_k < vocab_size) {
        // Use a COPY of the original logits for nth_element to find the threshold
        // WITHOUT rearranging the main buf (which would destroy token-ID-to-position mapping).
        std::vector<float> sorted(buf);
        std::nth_element(sorted.begin(), sorted.begin() + (vocab_size - top_k),
                         sorted.end());
        float threshold = sorted[vocab_size - top_k];
        for (int32_t i = 0; i < vocab_size; ++i)
            if (buf[i] < threshold) buf[i] = -std::numeric_limits<float>::infinity();
    }

    // Top-P (nucleus) filtering
    if (top_p < 1.0f && top_p > 0.0f) {
        // Build index-value pairs for sorting by logit descending
        std::vector<std::pair<float, int32_t>> sorted;
        sorted.reserve(vocab_size);
        for (int32_t i = 0; i < vocab_size; ++i)
            if (buf[i] > -std::numeric_limits<float>::infinity())
                sorted.emplace_back(buf[i], i);
        std::sort(sorted.begin(), sorted.end(),
                  [](auto& a, auto& b) { return a.first > b.first; });

        // Softmax on the valid entries
        float maxv = sorted.empty() ? 0.0f : sorted[0].first;
        for (auto& p : sorted) p.first = std::exp(p.first - maxv);
        float sum = 0.0f;
        for (auto& p : sorted) sum += p.first;
        float inv_sum = 1.0f / (sum + 1e-9f);
        for (auto& p : sorted) p.first *= inv_sum;

        // Cumulative sum, zero out beyond top_p
        float cum = 0.0f;
        for (size_t i = 0; i < sorted.size(); ++i) {
            cum += sorted[i].first;
            if (cum > top_p && i > 0) {
                for (size_t j = i; j < sorted.size(); ++j)
                    buf[sorted[j].second] = -std::numeric_limits<float>::infinity();
                break;
            }
        }
    }

    // Softmax for proper distribution
    softmax_inplace(buf.data(), vocab_size);

    // Multinomial sample
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng());
    float cum = 0.0f;
    for (int32_t i = 0; i < vocab_size; ++i) {
        cum += buf[i];
        if (r <= cum) return i;
    }
    return vocab_size - 1; // fallback
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
            ? prev_tokens + static_cast<size_t>(s)  // interleaved: [stream0_pos1, stream0_pos2, ...]
            : nullptr;

        // For repetition penalty, collect unique previous tokens for this stream
        // This is a simplified version — prev_tokens is per-stream history
        int32_t n_prev_s = (prev_tokens && n_prev > 0) ? n_prev : 0;

        out[s] = sample_token(row, vocab_size,
                              prev, n_prev_s,
                              repetition_penalty,
                              top_p, top_k,
                              do_sample);
    }
}

}  // namespace audiocore::moss
