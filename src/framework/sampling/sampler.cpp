// sampler.cpp — unified token sampler implementation.
//
// Subsumes the former moss_tts/sampler.cpp core, the qwen3::Runner
// sample_token_basic helper, and the qwen3_tts/session.cpp local sampler.
// Behavior preserves the moss implementation (stable softmax + sort-by-logit
// nucleus filtering with -inf masking) extended with temperature scaling
// from the qwen3 variants.

#include "audiocore/framework/sampling/sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace audiocore::sampler {

namespace {

std::mt19937& default_rng() {
    static thread_local std::mt19937 gen(std::random_device{}());
    return gen;
}

int32_t argmax(const float* v, int32_t n) {
    int32_t best = 0;
    float bestv = v[0];
    for (int32_t i = 1; i < n; ++i) {
        if (v[i] > bestv) {
            bestv = v[i];
            best = i;
        }
    }
    return best;
}

}  // namespace

int32_t sample_token(const float* logits, int32_t vocab_size,
                     const Params& p,
                     const int32_t* prev_tokens,
                     int32_t n_prev,
                     std::mt19937* rng) {
    if (vocab_size <= 0) return 0;

    // Working buffer. Starts as a temperature-scaled copy of the logits.
    std::vector<float> buf(static_cast<size_t>(vocab_size));
    const float inv_temp = 1.0f / std::max(p.temperature, 1e-6f);
    for (int32_t i = 0; i < vocab_size; ++i)
        buf[static_cast<size_t>(i)] = logits[i] * inv_temp;

    // Repetition penalty (moss semantics): positive logits are divided,
    // non-positive are multiplied. Applied before any filtering.
    if (prev_tokens && p.repetition_penalty != 1.0f) {
        for (int32_t i = 0; i < n_prev; ++i) {
            const int32_t tok = prev_tokens[i];
            if (tok < 0 || tok >= vocab_size) continue;
            const size_t idx = static_cast<size_t>(tok);
            if (buf[idx] > 0.0f) buf[idx] /= p.repetition_penalty;
            else                  buf[idx] *= p.repetition_penalty;
        }
    }

    if (!p.do_sample)
        return argmax(buf.data(), vocab_size);

    // Top-K: keep only the k highest logits, mask the rest to -inf.
    if (p.top_k > 0 && p.top_k < vocab_size) {
        std::vector<float> sorted(buf);
        std::nth_element(sorted.begin(),
                         sorted.begin() + (vocab_size - p.top_k),
                         sorted.end());
        const float threshold = sorted[static_cast<size_t>(vocab_size - p.top_k)];
        for (int32_t i = 0; i < vocab_size; ++i)
            if (buf[static_cast<size_t>(i)] < threshold)
                buf[static_cast<size_t>(i)] = -std::numeric_limits<float>::infinity();
    }

    // Top-P (nucleus): sort surviving logits descending, softmax, drop the
    // tail once cumulative probability exceeds top_p.
    if (p.top_p < 1.0f && p.top_p > 0.0f) {
        std::vector<std::pair<float, int32_t>> sorted;
        sorted.reserve(static_cast<size_t>(vocab_size));
        for (int32_t i = 0; i < vocab_size; ++i) {
            if (buf[static_cast<size_t>(i)] > -std::numeric_limits<float>::infinity())
                sorted.emplace_back(buf[static_cast<size_t>(i)], i);
        }
        std::sort(sorted.begin(), sorted.end(),
                  [](const std::pair<float, int32_t>& a,
                     const std::pair<float, int32_t>& b) {
                      return a.first > b.first;
                  });

        // Softmax over the surviving entries.
        const float maxv = sorted.empty() ? 0.0f : sorted[0].first;
        float sum = 0.0f;
        for (auto& pr : sorted) {
            pr.first = std::exp(pr.first - maxv);
            sum += pr.first;
        }
        const float inv_sum = 1.0f / (sum + 1e-9f);
        for (auto& pr : sorted) pr.first *= inv_sum;

        // Mask tail entries beyond top_p.
        float cum = 0.0f;
        for (size_t i = 0; i < sorted.size(); ++i) {
            cum += sorted[i].first;
            if (cum > p.top_p && i > 0) {
                for (size_t j = i; j < sorted.size(); ++j)
                    buf[static_cast<size_t>(sorted[j].second)] =
                        -std::numeric_limits<float>::infinity();
                break;
            }
        }
    }

    // Final softmax → multinomial draw.
    float maxv = -std::numeric_limits<float>::infinity();
    for (int32_t i = 0; i < vocab_size; ++i)
        if (buf[static_cast<size_t>(i)] > maxv) maxv = buf[static_cast<size_t>(i)];
    float sum = 0.0f;
    for (int32_t i = 0; i < vocab_size; ++i) {
        const float e = (buf[static_cast<size_t>(i)] == -std::numeric_limits<float>::infinity())
                            ? 0.0f
                            : std::exp(buf[static_cast<size_t>(i)] - maxv);
        buf[static_cast<size_t>(i)] = e;
        sum += e;
    }
    if (sum <= 0.0f) return 0;

    std::mt19937& g = rng ? *rng : default_rng();
    std::uniform_real_distribution<float> dist(0.0f, sum);
    const float r = dist(g);
    float cum = 0.0f;
    for (int32_t i = 0; i < vocab_size; ++i) {
        cum += buf[static_cast<size_t>(i)];
        if (r <= cum) return i;
    }
    return vocab_size - 1;
}

}  // namespace audiocore::sampler
