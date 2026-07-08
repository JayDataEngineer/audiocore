// delay_state.cpp — MOSS-TTS delay-pattern autoregressive state machine.
//
// Ported from moss_tts_delay/llama_cpp/delay_state.py.
// The delay pattern introduces a diagonal time-shift across codebooks:
//   Head 0 predicts text at t, Head k predicts codebook k-1 at t-(k-1).
// During the "delay slot" flush after audio ends, the model emits n_vq
// additional steps to drain the staircase.

#include "audiocore/models/moss_tts/delay_state.h"
#include "audiocore/models/moss_tts/sampler.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace audiocore::moss {

// ── Pre-excluded IDs for text token sampling ──────────────────────────────
// These tokens are never sampled when is_audio=false.
static const int32_t PRE_EXCLUDE_IDS[] = {
    PAD_TOKEN_ID,
    AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID,
    AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID,
    AUDIO_END_TOKEN_ID,
};
static constexpr int32_t N_PRE_EXCLUDE = 4;

// When is_audio=true, only these tokens are allowed.
static const int32_t AUDIO_ALLOWED_IDS[] = {
    AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID,
    AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID,
};
static constexpr int32_t N_AUDIO_ALLOWED = 2;

// ── Utils ─────────────────────────────────────────────────────────────────
static int32_t find_last_equal(const std::vector<int32_t>& arr, int32_t value) {
    for (int i = static_cast<int>(arr.size()) - 1; i >= 0; --i)
        if (arr[i] == value) return i;
    return -1;
}

static bool contains(const int32_t* arr, int32_t n, int32_t val) {
    for (int32_t i = 0; i < n; ++i)
        if (arr[i] == val) return true;
    return false;
}

// ── init_delay_state ──────────────────────────────────────────────────────
DelayState init_delay_state(const std::vector<std::vector<int32_t>>& prompt,
                            int32_t n_vq) {
    if (prompt.empty()) {
        DelayState s;
        s.n_vq = n_vq;
        s.audio_buf.reserve(256);
        return s;
    }

    int32_t seq_len = static_cast<int32_t>(prompt.size());
    DelayState state;
    state.n_vq = n_vq;

    // Extract text channel (column 0)
    std::vector<int32_t> text_channel(seq_len);
    for (int32_t i = 0; i < seq_len; ++i)
        text_channel[i] = prompt[i][0];

    int32_t last_text = text_channel.back();
    bool is_continuation = (last_text == AUDIO_START_TOKEN_ID ||
                            last_text == AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID);

    if (is_continuation) {
        int32_t audio_start_idx = find_last_equal(text_channel, AUDIO_START_TOKEN_ID);
        if (audio_start_idx >= 0) {
            state.audio_length = seq_len - audio_start_idx;
            state.is_audio = true;
        }
    }

    state.text_history = text_channel;

    // Copy audio channels from prompt
    state.audio_buf.reserve(std::max(seq_len + 1024, 256));
    for (int32_t i = 0; i < seq_len; ++i) {
        std::vector<int32_t> frame(static_cast<size_t>(n_vq));
        for (int s = 0; s < n_vq; ++s)
            frame[static_cast<size_t>(s)] = prompt[i][1 + s];
        state.audio_buf.push_back(std::move(frame));
    }

    return state;
}

// ── delay_step ────────────────────────────────────────────────────────────
std::vector<int32_t> delay_step(DelayState& state,
                                const float* text_logits, int32_t text_vocab,
                                const float* audio_logits, int32_t audio_vocab,
                                SamplingConfig& config) {
    std::vector<int32_t> result(1 + state.n_vq);

    if (state.is_stopping) {
        std::fill(result.begin(), result.end(), AUDIO_PAD_CODE);
        result[0] = PAD_TOKEN_ID;
        return result;
    }

    // ── Text token decision ───────────────────────────────────────────────
    int32_t next_text;

    if (state.delayed_length < static_cast<int32_t>(state.n_vq)) {
        next_text = AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID;
    } else if (state.delayed_length == static_cast<int32_t>(state.n_vq)) {
        next_text = AUDIO_END_TOKEN_ID;
        state.is_audio = false;
    } else {
        float text_temp = config.text_temperature > 0.0f
            ? config.text_temperature : 1.0f;
        bool do_sample = config.text_temperature > 0.0f;

        // Build scratch copy of text logits for manipulation
        std::vector<float> scaled(text_vocab);
        float inv_temp = 1.0f / text_temp;
        for (int32_t i = 0; i < text_vocab; ++i)
            scaled[i] = text_logits[i] * inv_temp;

        // Apply masks
        if (!state.is_audio) {
            for (int32_t i = 0; i < N_PRE_EXCLUDE; ++i)
                if (PRE_EXCLUDE_IDS[i] < text_vocab)
                    scaled[PRE_EXCLUDE_IDS[i]] = -std::numeric_limits<float>::infinity();
        } else {
            // Only allow audio slot tokens
            for (int32_t i = 0; i < text_vocab; ++i) {
                if (!contains(AUDIO_ALLOWED_IDS, N_AUDIO_ALLOWED, i))
                    scaled[i] = -std::numeric_limits<float>::infinity();
            }
        }

        if (state.time_step == 0 && AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID < text_vocab)
            scaled[AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID] = -std::numeric_limits<float>::infinity();

        if (state.time_step <= state.n_vq && IM_END_TOKEN_ID < text_vocab)
            scaled[IM_END_TOKEN_ID] = -std::numeric_limits<float>::infinity();

        // Sample
        next_text = sample_token(scaled.data(), text_vocab,
                                 nullptr, 0,  // no rep penalty for text
                                 1.0f, config.text_top_p, config.text_top_k,
                                 do_sample, &config.rng);
    }

    if (next_text == AUDIO_START_TOKEN_ID)
        state.is_audio = true;
    if (next_text == IM_END_TOKEN_ID)
        state.is_stopping = true;

    // ── Audio token decision ──────────────────────────────────────────────
    std::vector<int32_t> next_audio(state.n_vq, AUDIO_PAD_CODE);

    // pre_audio_mask: streams with index < state.audio_length
    // post_audio_mask: streams with index > (delayed_length - 1) or all if delayed_length == MAX
    bool post_all = (state.delayed_length == std::numeric_limits<int32_t>::max());

    std::vector<int> sampling_streams;
    for (int s = 0; s < state.n_vq; ++s) {
        bool pre = (s < state.audio_length);
        bool post = post_all || (s > state.delayed_length - 1);
        if (pre && post)
            sampling_streams.push_back(s);
    }

    if (!sampling_streams.empty()) {
        float audio_temp = config.audio_temperature > 0.0f
            ? config.audio_temperature : 1.0f;
        bool do_sample = config.audio_temperature > 0.0f;
        float inv_audio_temp = 1.0f / audio_temp;

        // Get audio history for repetition penalty
        // prev_audio = audio_buf[:-1, :] — all previously generated frames
        // (not including the current step which hasn't been appended yet)
        int n_prev_frames = static_cast<int>(state.audio_buf.size()) - 1;

        for (size_t si = 0; si < sampling_streams.size(); ++si) {
            int s = sampling_streams[si];

            // Scale logits for this stream
            std::vector<float> scaled(audio_vocab);
            const float* stream_logits = audio_logits + static_cast<size_t>(s) * audio_vocab;
            for (int32_t i = 0; i < audio_vocab; ++i)
                scaled[i] = stream_logits[i] * inv_audio_temp;

            // Exclude pad code from sampling
            if (AUDIO_PAD_CODE < audio_vocab)
                scaled[AUDIO_PAD_CODE] = -std::numeric_limits<float>::infinity();

            // Collect per-stream previous tokens for repetition penalty
            std::vector<int32_t> prev_for_stream;
            if (config.audio_repetition_penalty != 1.0f && n_prev_frames > 0) {
                prev_for_stream.reserve(n_prev_frames);
                for (int t = 0; t < n_prev_frames; ++t) {
                    int32_t code = state.audio_buf[t][s];
                    if (code >= 0) // pad codes (AUDIO_PAD_CODE=1024, which is >= 0)
                        prev_for_stream.push_back(code);
                }
            }

            next_audio[s] = sample_token(scaled.data(), audio_vocab,
                                         prev_for_stream.data(),
                                         static_cast<int32_t>(prev_for_stream.size()),
                                         config.audio_repetition_penalty,
                                         config.audio_top_p, config.audio_top_k,
                                         do_sample, &config.rng);
        }
    }

    // ── State updates ─────────────────────────────────────────────────────
    if (next_text == AUDIO_START_TOKEN_ID ||
        next_text == AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID ||
        next_text == AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID) {
        state.audio_length++;
    }
    if (next_text == AUDIO_END_TOKEN_ID) {
        state.audio_length = 0;
    }

    if (state.delayed_length == std::numeric_limits<int32_t>::max() &&
        next_text == AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID) {
        state.delayed_length = 0;
    }
    if (state.delayed_length != std::numeric_limits<int32_t>::max()) {
        state.delayed_length++;
    }
    if (state.delayed_length > static_cast<int32_t>(state.n_vq)) {
        state.delayed_length = std::numeric_limits<int32_t>::max();
    }

    state.time_step++;
    state.text_history.push_back(next_text);
    state.audio_buf.push_back(next_audio);

    result[0] = next_text;
    for (int s = 0; s < state.n_vq; ++s)
        result[1 + s] = next_audio[s];

    return result;
}

// ── apply_de_delay_pattern ────────────────────────────────────────────────
std::vector<std::vector<int32_t>> apply_de_delay_pattern(
    const std::vector<std::vector<int32_t>>& delay_codes,
    int32_t n_vq) {

    if (delay_codes.empty()) return {};

    int32_t total_len = static_cast<int32_t>(delay_codes.size());
    int32_t T = total_len - n_vq + 1;
    if (T <= 0) return {};

    std::vector<std::vector<int32_t>> result(T, std::vector<int32_t>(n_vq, 0));
    for (int s = 0; s < n_vq; ++s) {
        for (int t = 0; t < T; ++t) {
            if (t + s < total_len)
                result[t][s] = delay_codes[t + s][s];
        }
    }
    return result;
}

// ── extract_audio_segments ────────────────────────────────────────────────
std::vector<std::vector<int32_t>> extract_audio_segments(
    const std::vector<std::vector<int32_t>>& codes,
    int32_t n_vq) {

    if (codes.empty()) return {};

    // De-delay first
    auto delayed = apply_de_delay_pattern(codes, n_vq);
    if (delayed.empty()) return {};

    // Find non-padding rows
    std::vector<int> non_pad_indices;
    for (int32_t t = 0; t < static_cast<int32_t>(delayed.size()); ++t) {
        bool all_pad = true;
        for (int s = 0; s < n_vq; ++s) {
            if (delayed[t][s] != AUDIO_PAD_CODE) {
                all_pad = false;
                break;
            }
        }
        if (!all_pad)
            non_pad_indices.push_back(t);
    }

    if (non_pad_indices.empty()) return {};

    // Group into consecutive segments
    std::vector<std::vector<std::vector<int32_t>>> segments;
    int start = non_pad_indices[0];
    int prev = non_pad_indices[0];
    for (size_t i = 1; i < non_pad_indices.size(); ++i) {
        int cur = non_pad_indices[i];
        if (cur != prev + 1) {
            // Break segment
            std::vector<std::vector<int32_t>> seg;
            for (int t = start; t <= prev; ++t)
                seg.push_back(delayed[t]);
            segments.push_back(std::move(seg));
            start = cur;
        }
        prev = cur;
    }
    // Last segment
    std::vector<std::vector<int32_t>> last_seg;
    for (int t = start; t <= prev; ++t)
        last_seg.push_back(delayed[t]);
    segments.push_back(std::move(last_seg));

    // Concatenate all segments
    std::vector<std::vector<int32_t>> result;
    for (auto& seg : segments) {
        for (auto& frame : seg)
            result.push_back(std::move(frame));
    }
    return result;
}

}  // namespace audiocore::moss
