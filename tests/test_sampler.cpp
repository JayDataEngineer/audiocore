#include "test_framework.h"

#include "audiocore/framework/sampling/sampler.h"

#include <limits>
#include <vector>

using namespace audiocore;

AUDIOCORE_TEST(sampler_argmax_picks_highest_logit) {
    const std::vector<float> logits = {-1.0f, 5.0f, -3.0f, 0.0f};
    sampler::Params p;
    p.do_sample = false;
    const int32_t got = sampler::sample_token(logits.data(), 4, p);
    AUDIOCORE_CHECK_EQ(got, 1);
}

AUDIOCORE_TEST(sampler_argmax_ignores_temperature) {
    const std::vector<float> logits = {1.0f, 9.0f, 2.0f};
    sampler::Params p;
    p.temperature = 0.001f;
    p.do_sample   = false;
    const int32_t got = sampler::sample_token(logits.data(), 3, p);
    AUDIOCORE_CHECK_EQ(got, 1);
}

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

AUDIOCORE_TEST(sampler_top_k_masks_everything_but_top2) {
    const std::vector<float> logits = {1.0f, -10.0f, 5.0f, -10.0f};
    sampler::PhiloxRng rng{123};
    sampler::Params p;
    p.top_k = 2;
    for (int i = 0; i < 50; ++i) {
        const int32_t got = sampler::sample_token(logits.data(), 4, p,
                                                  nullptr, 0, &rng);
        AUDIOCORE_CHECK(got == 0 || got == 2);
    }
}

AUDIOCORE_TEST(sampler_low_temperature_concentrates_on_argmax) {
    const std::vector<float> logits = {0.0f, 12.0f, 0.0f, 0.0f};
    sampler::PhiloxRng rng{7};
    sampler::Params p;
    p.temperature = 0.01f;
    int argmax_hits = 0;
    for (int i = 0; i < 100; ++i) {
        if (sampler::sample_token(logits.data(), 4, p, nullptr, 0, &rng) == 1)
            ++argmax_hits;
    }
    AUDIOCORE_CHECK(argmax_hits >= 99);
}

AUDIOCORE_TEST(sampler_rep_penalty_deprioritizes_history) {
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

AUDIOCORE_TEST(sampler_is_deterministic_under_fixed_rng) {
    const std::vector<float> logits = {1.0f, 2.0f, 3.0f, 4.0f, 0.5f};
    sampler::Params p;
    p.temperature = 1.0f;
    p.top_p       = 0.9f;

    sampler::PhiloxRng a{42}, b{42};
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
