// dsp.cpp — Time-domain audio DSP for post-processing TTS output.
//
// WSOLA pitch-shift + time-stretch. Pure C++14, no external deps. Designed
// for moderate-quality speech manipulation (±5 semitones, 0.5–2.0× speed).
// Artifact level is well below what most listeners perceive as "processed"
// for speech; for music, prefer a phase vocoder.

#include "audiocore/framework/audio/dsp.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace audiocore {

// ── Linear resample ─────────────────────────────────────────────────────
// Pure ratio change — pitch and duration shift together. Uses linear
// interpolation; sufficient because WSOLA will reconstruct smoothness.
std::vector<float> resample_linear(const float* x, size_t n, double ratio) {
    if (n == 0 || ratio <= 0.0) return {};
    size_t out_n = static_cast<size_t>(std::llround(static_cast<double>(n) * ratio));
    if (out_n == 0) return {};
    std::vector<float> y(out_n);
    if (ratio == 1.0) {
        std::memcpy(y.data(), x, n * sizeof(float));
        return y;
    }
    for (size_t i = 0; i < out_n; ++i) {
        // Input read position (in source samples).
        double pos = static_cast<double>(i) / ratio;
        size_t idx = static_cast<size_t>(pos);
        double frac = pos - static_cast<double>(idx);
        if (idx + 1 < n) {
            y[i] = static_cast<float>(
                (1.0 - frac) * static_cast<double>(x[idx]) +
                frac * static_cast<double>(x[idx + 1]));
        } else if (idx < n) {
            y[i] = x[idx];
        } else {
            y[i] = 0.0f;
        }
    }
    return y;
}

namespace {

// Cross-correlation between two windows of the same length.
// Returns the normalized similarity in [-1, 1].
double cross_corr(const float* a, const float* b, size_t n) {
    double dot = 0.0, ea = 0.0, eb = 0.0;
    for (size_t i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        ea  += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        eb  += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    const double denom = std::sqrt(ea * eb) + 1e-12;
    return dot / denom;
}

// Hann window of length N (symmetric, used for overlap-add).
void hann_window(float* w, int N) {
    for (int i = 0; i < N; ++i) {
        // Symmetric Hann (periodic would be N-1 in denominator, used for
        // COLA-perfect reconstruction with non-integer overlap; symmetric
        // is fine here because we normalize by the window sum on output).
        w[i] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) *
                                         static_cast<float>(i) /
                                         static_cast<float>(N - 1)));
    }
}

}  // namespace

// ── WSOLA time-stretch ──────────────────────────────────────────────────
//
// Reference: W. Verhelst and M. Roelands, "An overlap-add technique based
// on waveform similarity (WSOLA) for high-quality time-scale modification
// of speech," ICASSP 1993.
//
// Algorithm:
//   1. Output is built as a sequence of overlapping Hann-windowed frames.
//   2. The next input frame is placed at input_ptr + natural_advance.
//   3. We search ±search_samples around that anchor for the offset that
//      maximizes cross-correlation with the tail of the previous output
//      frame — this preserves local phase continuity.
//   4. Overlap-add the best-matching frame with the running output buffer.
//
std::vector<float> wsola_stretch(const float* x, size_t n, double rate,
                                  int sample_rate,
                                  double frame_ms,
                                  double hop_ms,
                                  double search_ms) {
    if (n == 0 || rate <= 0.0 || sample_rate <= 0) return {};
    if (rate == 1.0) {
        return std::vector<float>(x, x + n);
    }

    const int frame   = std::max(32, static_cast<int>(frame_ms   * 1e-3 * sample_rate));
    const int hop     = std::max(8,  static_cast<int>(hop_ms     * 1e-3 * sample_rate));
    const int search  = std::max(0,  static_cast<int>(search_ms  * 1e-3 * sample_rate));
    const int overlap = frame - hop;
    if (overlap <= 0) return std::vector<float>(x, x + n);

    // Output length scales with 1/rate (rate>1 = compress = shorter/faster).
    const size_t out_n_est = static_cast<size_t>(
        std::llround(static_cast<double>(n) / rate)) + static_cast<size_t>(frame) * 2;
    std::vector<float> y(out_n_est, 0.0f);
    std::vector<float> winsum(out_n_est, 0.0f);
    std::vector<float> win(frame);
    hann_window(win.data(), frame);

    // Natural advance in input per output hop:
    //   output advances by `hop`; input advances by hop * rate.
    // The "tolerance" search window lets us correct small misalignments.
    const double in_advance = static_cast<double>(hop) * rate;

    // Previous-frame tail (last `overlap` samples of the last written output)
    // used as the cross-correlation reference.
    std::vector<float> prev_tail(overlap, 0.0f);

    // Initialize prev_tail with the first `overlap` samples of input
    // (so the first cross-correlation has something meaningful to match).
    if (n >= static_cast<size_t>(overlap)) {
        for (int i = 0; i < overlap; ++i) prev_tail[i] = x[i];
    }

    size_t out_ptr = 0;
    double in_ptr_d = 0.0;

    while (true) {
        int in_idx = static_cast<int>(in_ptr_d);
        // Anchor: ideal natural position.
        int best_off = 0;
        double best_score = -2.0;

        // Search ±search around in_idx for best cross-correlation.
        for (int off = -search; off <= search; ++off) {
            int cand = in_idx + off;
            if (cand < 0) continue;
            if (static_cast<size_t>(cand) + frame > n) break;
            double score = cross_corr(prev_tail.data(),
                                       x + cand,
                                       static_cast<size_t>(overlap));
            if (score > best_score) {
                best_score = score;
                best_off = off;
            }
        }

        int src = in_idx + best_off;
        if (src < 0) src = 0;
        if (static_cast<size_t>(src) + frame > n) break;

        // Overlap-add windowed frame into output.
        for (int i = 0; i < frame; ++i) {
            size_t oi = out_ptr + static_cast<size_t>(i);
            if (oi >= y.size()) {
                y.resize(y.size() * 2, 0.0f);
                winsum.resize(winsum.size() * 2, 0.0f);
            }
            y[oi]      += x[src + i] * win[i];
            winsum[oi] += win[i];
        }

        // Update prev_tail: last `overlap` samples of what we just wrote.
        size_t tail_start = out_ptr + static_cast<size_t>(frame - overlap);
        for (int i = 0; i < overlap; ++i) {
            size_t ti = tail_start + static_cast<size_t>(i);
            prev_tail[i] = (ti < y.size()) ? y[ti] : 0.0f;
        }

        out_ptr += static_cast<size_t>(hop);
        in_ptr_d += in_advance;
        if (out_ptr + static_cast<size_t>(frame) > y.size()) break;
        if (in_idx + frame >= static_cast<int>(n)) break;
    }

    // Normalize by window sum.
    for (size_t i = 0; i < out_ptr + static_cast<size_t>(frame) && i < y.size(); ++i) {
        if (winsum[i] > 1e-6f) y[i] /= winsum[i];
    }

    // Trim trailing zeros / overlap region.
    if (out_ptr + frame < y.size()) y.resize(out_ptr + frame);
    return y;
}

// ── Pitch shift ─────────────────────────────────────────────────────────
// 1. Resample by 2^(-semitones/12): shifts pitch up + speeds up (or vice).
// 2. WSOLA-stretch by the inverse to restore original duration.
std::vector<float> pitch_shift(const float* x, size_t n, double semitones,
                                int sample_rate) {
    if (n == 0 || semitones == 0.0) return std::vector<float>(x, x + n);
    // Clamp to safe range.
    if (semitones < -12.0) semitones = -12.0;
    if (semitones >  12.0) semitones =  12.0;

    // Step 1: resample — shifts pitch + duration together.
    // If semitones > 0 (higher pitch): we want a SHORTER input (fewer samples)
    //   so that after stretching back, the pitch is higher.
    //   Equivalent to "play faster" → ratio = 2^(-semitones/12).
    double ps_ratio = std::pow(2.0, -semitones / 12.0);
    auto shifted = resample_linear(x, n, ps_ratio);

    // Step 2: WSOLA-stretch back to original length.
    //   shifted.size() * rate ≈ n
    //   rate = n / shifted.size() ≈ 1/ps_ratio
    if (shifted.empty()) return shifted;
    double stretch_rate = static_cast<double>(shifted.size()) / static_cast<double>(n);
    if (stretch_rate <= 0.0) return shifted;

    return wsola_stretch(shifted.data(), shifted.size(), stretch_rate,
                          sample_rate);
}

// ── Speed change (no pitch change) ──────────────────────────────────────
std::vector<float> change_speed(const float* x, size_t n, double rate,
                                 int sample_rate) {
    if (n == 0 || rate == 1.0) return std::vector<float>(x, x + n);
    if (rate <= 0.0) return {};
    // rate > 1 = faster playback = shorter output.
    // wsola_stretch's actual behavior is "rate>1 = shorter output"
    // (out_n = n/rate), so we pass rate directly. The previous
    // implementation inverted to 1/rate which made speed=1.5 produce
    // 1.5x LONGER audio (i.e. slower) — the opposite of what callers
    // and the docstring describe.
    return wsola_stretch(x, n, rate, sample_rate);
}

}  // namespace audiocore
