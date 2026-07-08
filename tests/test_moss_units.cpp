// test_moss_units.cpp — Unit tests for MOSS-TTS components.
//
// Tests delay state and codec components with synthetic inputs.
// These tests do NOT require model files.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "audiocore/models/moss_tts/delay_state.h"

using namespace audiocore::moss;

// Local constant for test dimension (the default 32-codebook layout).
static constexpr int32_t N_VQ = 32;

// ── Test macros (void-safe, accumulate failures) ────────────────────────────
static int g_n_pass = 0;
static int g_n_fail = 0;

#define CHECK(cond, msg) do {                                          \
    if (!(cond)) {                                                     \
        std::fprintf(stderr, "  [FAIL] %s:%d: %s\n",                   \
                     __FILE__, __LINE__, msg);                         \
        g_n_fail++;                                                    \
    } else {                                                           \
        g_n_pass++;                                                    \
    }                                                                  \
} while(0)

#define CHECK_EQUAL(a, b, msg) CHECK((a) == (b), msg)

// =========================================================================
// delay_state tests
// =========================================================================

static void test_init_delay_state_empty() {
    auto s = init_delay_state({}, N_VQ);
    CHECK(s.audio_buf.empty(), "empty prompt → empty audio_buf");
    CHECK(s.text_history.empty(), "empty prompt → empty text_history");
    CHECK(!s.is_audio, "empty prompt → is_audio=false");
    CHECK(!s.is_stopping, "empty prompt → is_stopping=false");
}

static void test_init_delay_state_text_only() {
    std::vector<std::vector<int32_t>> prompt(5, std::vector<int32_t>(1 + N_VQ, 0));
    for (int i = 0; i < 5; ++i) {
        prompt[i][0] = 100 + i;
        for (int s = 0; s < N_VQ; ++s)
            prompt[i][1 + s] = AUDIO_PAD_CODE;
    }
    auto s = init_delay_state(prompt, N_VQ);
    CHECK_EQUAL(s.text_history.size(), 5, "5 text history entries");
    CHECK_EQUAL(s.audio_buf.size(), 5, "5 audio buf entries");
    CHECK_EQUAL(s.audio_length, 0, "audio_length=0 with no audio start");
    CHECK(!s.is_audio, "no audio start token");
}

static void test_init_delay_state_with_audio() {
    std::vector<std::vector<int32_t>> prompt(5, std::vector<int32_t>(1 + N_VQ, 0));
    for (int i = 0; i < 4; ++i) prompt[i][0] = 100 + i;
    prompt[4][0] = AUDIO_START_TOKEN_ID;
    auto s = init_delay_state(prompt, N_VQ);
    CHECK_EQUAL(s.audio_length, 1, "audio_length=1 with audio start at end");
    CHECK(s.is_audio, "is_audio=true with audio start");
}

static void test_delay_step_stopping() {
    DelayState state;
    state.is_stopping = true;
    SamplingConfig cfg;
    std::vector<float> text_logits(1000, 0.0f);
    std::vector<float> audio_logits(static_cast<size_t>(N_VQ) * 1025, 0.0f);
    text_logits[50] = 10.0f;

    auto result = delay_step(state, text_logits.data(), 1000,
                              audio_logits.data(), 1025, cfg);
    CHECK_EQUAL(result[0], PAD_TOKEN_ID, "stopping → text=PAD");
    for (int s = 0; s < N_VQ; ++s)
        CHECK_EQUAL(result[1 + s], AUDIO_PAD_CODE, "stopping → audio=PAD");
}

static void test_delay_step_greedy_text() {
    DelayState state;
    state.is_audio = false;
    SamplingConfig cfg;
    cfg.text_temperature = 0.0f;
    cfg.text_top_k = 0;

    std::vector<float> text_logits(1000, 0.0f);
    text_logits[42] = 10.0f;  // dominating logit
    std::vector<float> audio_logits(static_cast<size_t>(N_VQ) * 1025, 0.0f);

    auto result = delay_step(state, text_logits.data(), 1000,
                              audio_logits.data(), 1025, cfg);
    CHECK(result[0] >= 0 && result[0] < 1000, "text token in valid range");
    CHECK(result[0] != PAD_TOKEN_ID, "not a pad token");
}

static void test_apply_de_delay_basic() {
    const int T_delayed = 5 + N_VQ;  // 37
    std::vector<std::vector<int32_t>> input(T_delayed,
                                             std::vector<int32_t>(N_VQ, 0));
    // Fill with time-varying values: input[t][s] = t (all streams same)
    for (int t = 0; t < T_delayed; ++t)
        for (int s = 0; s < N_VQ; ++s)
            input[t][s] = t;

    auto output = apply_de_delay_pattern(input, N_VQ);
    // output size = T_delayed - N_VQ + 1 = 37 - 32 + 1 = 6
    CHECK_EQUAL((int)output.size(), 6, "output has T_delayed - N_VQ + 1 frames");
    // result[t][s] = input[t+s][s] = t+s
    CHECK_EQUAL(output[0][0], 0, "de-delay result[0][0] = 0");
    CHECK_EQUAL(output[0][5], 5, "de-delay result[0][5] = 5");
    CHECK_EQUAL(output[5][0], 5, "de-delay result[5][0] = 5");
    CHECK_EQUAL(output[5][31], 36, "de-delay result[5][31] = 36");
}

static void test_apply_de_delay_empty() {
    auto result = apply_de_delay_pattern({}, N_VQ);
    CHECK(result.empty(), "empty input → empty output");
}

static void test_apply_de_delay_short() {
    std::vector<std::vector<int32_t>> input(10, std::vector<int32_t>(N_VQ, 0));
    auto result = apply_de_delay_pattern(input, N_VQ);
    CHECK(result.empty(), "< N_VQ frames → empty");
}

static void test_extract_audio_segments() {
    // Create input where all rows except the first N_VQ have non-pad codes.
    // After de-delay, output[t][s] = input[t+s][s]. With input[t][s] = t for all s,
    // output[t][s] = t+s. Since t+s ranges from 0..36 (for t=0..5, s=0..31),
    // none of these equal AUDIO_PAD_CODE=1024.
    const int T_delayed = 5 + N_VQ;
    std::vector<std::vector<int32_t>> input(T_delayed,
                                             std::vector<int32_t>(N_VQ, AUDIO_PAD_CODE));
    for (int t = 0; t < T_delayed; ++t)
        for (int s = 0; s < N_VQ; ++s)
            input[t][s] = t;  // small ints, never AUDIO_PAD_CODE

    auto result = extract_audio_segments(input, N_VQ);
    CHECK(!result.empty(), "non-pad frames → non-empty");
    // After de-delay: result[t][s] = t+s, all != 1024, so frames should be kept
    // Output size should be T_delayed - N_VQ + 1 = 6
    CHECK_EQUAL((int)result.size(), 6, "output has T_delayed - N_VQ + 1 frames");
}

static void test_extract_audio_segments_all_pad() {
    std::vector<std::vector<int32_t>> input(33, std::vector<int32_t>(N_VQ, AUDIO_PAD_CODE));
    auto result = extract_audio_segments(input, N_VQ);
    CHECK(result.empty(), "all pad → empty");
}

static void test_extract_audio_segments_empty() {
    auto result = extract_audio_segments({}, N_VQ);
    CHECK(result.empty(), "empty → empty");
}

// =========================================================================
// main
// =========================================================================

int main() {
    std::fprintf(stderr, "=== MOSS-TTS Unit Tests ===\n");

    struct { const char* name; void (*fn)(); } tests[] = {
        {"init_delay_state_empty",               test_init_delay_state_empty},
        {"init_delay_state_text_only",           test_init_delay_state_text_only},
        {"init_delay_state_with_audio",          test_init_delay_state_with_audio},
        {"delay_step_stopping",                  test_delay_step_stopping},
        {"delay_step_greedy_text",               test_delay_step_greedy_text},
        {"apply_de_delay_basic",                 test_apply_de_delay_basic},
        {"apply_de_delay_empty",                 test_apply_de_delay_empty},
        {"apply_de_delay_short",                 test_apply_de_delay_short},
        {"extract_audio_segments",               test_extract_audio_segments},
        {"extract_audio_segments_all_pad",        test_extract_audio_segments_all_pad},
        {"extract_audio_segments_empty",          test_extract_audio_segments_empty},
    };

    for (auto& t : tests) {
        std::fprintf(stderr, "  %s ... ", t.name);
        int before = g_n_fail;
        t.fn();
        std::fprintf(stderr, "%s\n", (g_n_fail == before) ? "PASS" : "FAIL");
    }

    std::fprintf(stderr, "\n  %d/%d checks passed\n", g_n_pass, g_n_pass + g_n_fail);
    return (g_n_fail == 0) ? 0 : 1;
}
