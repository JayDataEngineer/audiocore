#include "audiocore/framework/sampling/sampler.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace audiocore::sampler {

// ── Philox4x32-10 counter-based PRNG ──────────────────────────────────────

namespace {

struct philox4 {
    uint32_t x, y, z, w;
};

inline void mulhilo32(uint32_t a, uint32_t b, uint32_t* hi, uint32_t* lo) {
    const uint64_t prod = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    *lo = static_cast<uint32_t>(prod);
    *hi = static_cast<uint32_t>(prod >> 32);
}

inline philox4 philox_round(philox4 ctr, uint32_t k0, uint32_t k1) {
    uint32_t hi0, lo0, hi1, lo1;
    mulhilo32(0xD2511F53u, ctr.x, &hi0, &lo0);
    mulhilo32(0xCD9E8D57u, ctr.z, &hi1, &lo1);
    return {hi1 ^ ctr.y ^ k0, lo1, hi0 ^ ctr.w ^ k1, lo0};
}

inline philox4 philox4x32_10(philox4 ctr, uint32_t seed_lo, uint32_t seed_hi) {
    uint32_t k0 = seed_lo;
    uint32_t k1 = seed_hi;
    for (int i = 0; i < 10; ++i) {
        ctr = philox_round(ctr, k0, k1);
        if (i != 9) {
            k0 += 0x9E3779B9u;
            k1 += 0xBB67AE85u;
        }
    }
    return ctr;
}

float philox_uniform(PhiloxRng& rng) {
    const uint32_t seed_lo = static_cast<uint32_t>(rng.seed);
    const uint32_t seed_hi = static_cast<uint32_t>(static_cast<uint64_t>(rng.seed) >> 32);
    const uint64_t s = static_cast<uint64_t>(rng.subseq++);
    const philox4 ctr = {0u, 0u, static_cast<uint32_t>(s), static_cast<uint32_t>(s >> 32)};
    const philox4 r = philox4x32_10(ctr, seed_lo, seed_hi);
    return (static_cast<float>(r.x) + 0.5f) * 2.3283064365386963e-10f;
}

PhiloxRng& default_rng() {
    static thread_local PhiloxRng rng{0, 0};
    return rng;
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
                     PhiloxRng* rng) {
    if (vocab_size <= 0) return 0;

    std::vector<float> buf(static_cast<size_t>(vocab_size));
    const float inv_temp = 1.0f / std::max(p.temperature, 1e-6f);
    for (int32_t i = 0; i < vocab_size; ++i)
        buf[static_cast<size_t>(i)] = logits[i] * inv_temp;

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

    // Top-K
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

    // Top-P (nucleus)
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

        const float maxv = sorted.empty() ? 0.0f : sorted[0].first;
        float sum = 0.0f;
        for (auto& pr : sorted) {
            pr.first = std::exp(pr.first - maxv);
            sum += pr.first;
        }
        const float inv_sum = 1.0f / (sum + 1e-9f);
        for (auto& pr : sorted) pr.first *= inv_sum;

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

    // Final softmax → multinomial draw
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

    PhiloxRng& g = rng ? *rng : default_rng();
    const float r = philox_uniform(g);
    float cum = 0.0f;
    for (int32_t i = 0; i < vocab_size; ++i) {
        cum += buf[static_cast<size_t>(i)];
        if (r <= cum) return i;
    }
    return vocab_size - 1;
}

}  // namespace audiocore::sampler
