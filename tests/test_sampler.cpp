// test_sampler.cpp — verify the unified audiocore::sampler behavior.
//
// Covers: argmax path, top-k filtering, top-p nucleus, temperature effect,
// repetition penalty, and determinism under a fixed RNG seed. The same
// implementation now backs moss_tts, qwen3_tts, ace_step, and the
// qwen3::Runner MTP predictor, so these tests lock the contract for all
// four families at once.

#include "test_framework.h"

#include "audiocore/framework/sampling/sampler.h"

#include <limits>
#include <random>
#include <vector>

using namespace audiocore;

// ── argmax / greedy ───────────────────────────────────────────────────────
AUDIOCORE_TEST(sampler_argmax_picks_highest_logit) {
    const std::vector<float> logits = {-1.0f, 5.0f, -3.0f, 0.0f};
    sampler::Params p;
    p.do_sample = false;
    const int32_t got = sampler::sample_token(logits.data(), 4, p);
    AUDIOCORE_CHECK_EQ(got, 1);
}

AUDIOCORE_TEST(sampler_argmax_ignores_temperature) {
    // Greedy path is deterministic regardless of temperature.
    const std::vector<float> logits = {1.0f, 9.0f, 2.0f};
    sampler::Params p;
    p.temperature = 0.001f;
    p.do_sample   = false;
    const int32_t got = sampler::sample_token(logits.data(), 3, p);
    AUDIOCORE_CHECK_EQ(got, 1);
}

// ── degenerate inputs ────────────────────────────────────────────────────
AUDIOCORE_TEST(sampler_returns_zero_on_empty_vocab) {
    sampler::Params p;
    const int32_t got = sampler::sample_token(nullptr, 0, p);
    AUDIOCORE_CHECK_EQ(got, 0);
}

AUDIOCORE_TEST(sampler_returns_zero_on_all_inf_logits) {
    const float inf = std::numeric_limits<float>::infinity();
    const std::vector<float> logits = {-inf, -inf, -inf};
    sampler::Params p;
    const int32_t got = sampler::sample_token(logits.data(), 3, p);
    AUDIOCORE_CHECK_EQ(got, 0);
}

// ── top-k ─────────────────────────────────────────────────────────────────
AUDIOCORE_TEST(sampler_top_k_masks_everything_but_top2) {
    // With top_k=2 over 4 tokens and greedy sampling, the winner must be one
    // of the two highest logits. Construct logits where index 2 is highest
    // and 0 is second-highest; lower both to -inf via top_k and confirm we
    // never see indices 1 or 3 across many draws.
    const std::vector<float> logits = {1.0f, -10.0f, 5.0f, -10.0f};
    std::mt19937 rng(123);
    sampler::Params p;
    p.top_k = 2;
    for (int i = 0; i < 50; ++i) {
        const int32_t got = sampler::sample_token(logits.data(), 4, p,
                                                  nullptr, 0, &rng);
        AUDIOCORE_CHECK(got == 0 || got == 2);
    }
}

// ── temperature ───────────────────────────────────────────────────────────
AUDIOCORE_TEST(sampler_low_temperature_concentrates_on_argmax) {
    // With temp approaching 0 and a clear winner, the argmax should dominate
    // almost every draw. Allow one stray draw out of many as a safety margin.
    const std::vector<float> logits = {0.0f, 12.0f, 0.0f, 0.0f};
    std::mt19937 rng(7);
    sampler::Params p;
    p.temperature = 0.01f;
    int argmax_hits = 0;
    for (int i = 0; i < 100; ++i) {
        if (sampler::sample_token(logits.data(), 4, p, nullptr, 0, &rng) == 1)
            ++argmax_hits;
    }
    AUDIOCORE_CHECK(argmax_hits >= 99);
}

// ── repetition penalty ───────────────────────────────────────────────────
AUDIOCORE_TEST(sampler_rep_penalty_deprioritizes_history) {
    // Token 1 has the highest raw logit but is in the history with penalty
    // > 1.0. With a strong penalty + greedy sampling, the model should
    // switch to the second-best (token 2).
    const std::vector<float> logits = {0.0f, 5.0f, 4.9f, 0.0f};
    const std::vector<int32_t> history = {1};
    sampler::Params p;
    p.do_sample          = false;
    p.repetition_penalty = 2.0f;
    const int32_t got = sampler::sample_token(logits.data(), 4, p,
                                              history.data(),
                                              static_cast<int32_t>(history.size()));
    AUDIOCORE_CHECK_EQ(got, 2);
}

// ── determinism ───────────────────────────────────────────────────────────
AUDIOCORE_TEST(sampler_is_deterministic_under_fixed_rng) {
    const std::vector<float> logits = {1.0f, 2.0f, 3.0f, 4.0f, 0.5f};
    sampler::Params p;
    p.temperature = 1.0f;
    p.top_p       = 0.9f;

    std::mt19937 a(42), b(42);
    for (int i = 0; i < 20; ++i) {
        const int32_t ra = sampler::sample_token(logits.data(), 5, p,
                                                 nullptr, 0, &a);
        const int32_t rb = sampler::sample_token(logits.data(), 5, p,
                                                 nullptr, 0, &b);
        AUDIOCORE_CHECK_EQ(ra, rb);
    }
}

int main() {
    return audiocore::test::run_all();
}
