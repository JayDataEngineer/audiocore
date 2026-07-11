// dsp.cpp — Time-domain audio DSP for post-processing TTS output.
//
// WSOLA pitch-shift + time-stretch. Pure C++14, no external deps. Designed
// for moderate-quality speech manipulation (±5 semitones, 0.5–2.0× speed).
// Artifact level is well below what most listeners perceive as "processed"
// for speech; for music, prefer a phase vocoder.

#include "audiocore/framework/audio/dsp.h"

#include <cmath>
#include <cstring>
#include <cstdint>
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

// =========================================================================
//  Voice-enhance: biquad EQ cascade + breathiness noise excitation
// =========================================================================
namespace {

// RBJ Audio EQ Cookbook biquad (Direct Form II transposed). a0 normalized.
struct Biquad {
    double b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0;
    double s1 = 0, s2 = 0;

    inline float process(float in) {
        double x = static_cast<double>(in);
        double y = b0 * x + s1;
        s1 = b1 * x - a1 * y + s2;
        s2 = b2 * x - a2 * y;
        return static_cast<float>(y);
    }

    static Biquad peaking(double f0, double gain_db, double Q, int sr) {
        Biquad bq;
        double A  = std::pow(10.0, gain_db / 40.0);
        double w0 = 2.0 * M_PI * f0 / sr;
        double cw = std::cos(w0), sw = std::sin(w0);
        double al = sw / (2.0 * Q);
        double a0 = 1.0 + al / A;
        bq.b0 = (1.0 + al * A) / a0;
        bq.b1 = (-2.0 * cw) / a0;
        bq.b2 = (1.0 - al * A) / a0;
        bq.a1 = (-2.0 * cw) / a0;
        bq.a2 = (1.0 - al / A) / a0;
        return bq;
    }
    static Biquad low_shelf(double f0, double gain_db, double S, int sr) {
        Biquad bq;
        double A  = std::pow(10.0, gain_db / 40.0);
        double w0 = 2.0 * M_PI * f0 / sr;
        double cw = std::cos(w0), sw = std::sin(w0);
        double al = sw / 2.0 * std::sqrt((A + 1.0 / A) * (1.0 / S - 1.0) + 2.0);
        double sq = 2.0 * std::sqrt(A) * al;
        double a0 = (A + 1.0) + (A - 1.0) * cw + sq;
        bq.b0 = (A * ((A + 1.0) - (A - 1.0) * cw + sq)) / a0;
        bq.b1 = (2.0 * A * ((A - 1.0) - (A + 1.0) * cw)) / a0;
        bq.b2 = (A * ((A + 1.0) - (A - 1.0) * cw - sq)) / a0;
        bq.a1 = (-2.0 * ((A - 1.0) + (A + 1.0) * cw)) / a0;
        bq.a2 = ((A + 1.0) + (A - 1.0) * cw - sq) / a0;
        return bq;
    }
    static Biquad high_shelf(double f0, double gain_db, double S, int sr) {
        Biquad bq;
        double A  = std::pow(10.0, gain_db / 40.0);
        double w0 = 2.0 * M_PI * f0 / sr;
        double cw = std::cos(w0), sw = std::sin(w0);
        double al = sw / 2.0 * std::sqrt((A + 1.0 / A) * (1.0 / S - 1.0) + 2.0);
        double sq = 2.0 * std::sqrt(A) * al;
        double a0 = (A + 1.0) - (A - 1.0) * cw + sq;
        bq.b0 = (A * ((A + 1.0) + (A - 1.0) * cw + sq)) / a0;
        bq.b1 = (-2.0 * A * ((A - 1.0) + (A + 1.0) * cw)) / a0;
        bq.b2 = (A * ((A + 1.0) + (A - 1.0) * cw - sq)) / a0;
        bq.a1 = (2.0 * ((A - 1.0) - (A + 1.0) * cw)) / a0;
        bq.a2 = ((A + 1.0) - (A - 1.0) * cw - sq) / a0;
        return bq;
    }
    static Biquad one_pole_lp(double f0, int sr) {
        Biquad bq;
        double a = std::exp(-2.0 * M_PI * f0 / sr);
        bq.b0 = 1.0 - a; bq.b1 = 0.0; bq.b2 = 0.0;
        bq.a1 = -a;      bq.a2 = 0.0;
        return bq;
    }
};

struct NoiseGen {
    uint32_t s;
    explicit NoiseGen(uint32_t seed = 0x9E3779B9u) : s(seed ? seed : 1u) {}
    inline float next() {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        return static_cast<float>(static_cast<int32_t>(s)) / 2147483648.0f;
    }
};

}  // namespace

std::vector<float> voice_enhance(const float* x, size_t n, int sample_rate,
                                  const VoiceEnhanceParams& p) {
    if (n == 0 || sample_rate <= 0) return {};
    const bool any_eq = std::fabs(p.warmth) > 0.001f ||
                        std::fabs(p.formant) > 0.001f ||
                        std::fabs(p.brightness) > 0.001f ||
                        std::fabs(p.airiness) > 0.001f;
    if (!any_eq && p.breathiness <= 0.001f)
        return std::vector<float>(x, x + n);

    std::vector<float> y(x, x + n);

    if (any_eq) {
        std::vector<Biquad> chain;
        if (std::fabs(p.warmth) > 0.001f)
            chain.push_back(Biquad::low_shelf(180.0, p.warmth * 6.0, 0.7, sample_rate));
        if (std::fabs(p.formant) > 0.001f) {
            chain.push_back(Biquad::peaking(700.0,  p.formant * 4.0, 1.0, sample_rate));
            chain.push_back(Biquad::peaking(1700.0, p.formant * 5.0, 1.1, sample_rate));
        }
        if (std::fabs(p.brightness) > 0.001f)
            chain.push_back(Biquad::peaking(3000.0, p.brightness * 5.0, 1.0, sample_rate));
        if (std::fabs(p.airiness) > 0.001f)
            chain.push_back(Biquad::high_shelf(6000.0, p.airiness * 8.0, 0.7, sample_rate));
        for (size_t i = 0; i < n; ++i) {
            float s = y[i];
            for (auto& bq : chain) s = bq.process(s);
            y[i] = s;
        }
    }

    if (p.breathiness > 0.001f) {
        NoiseGen noise;
        Biquad lp = Biquad::one_pole_lp(5000.0, sample_rate);
        double env = 0.0;
        const double att = std::exp(-1.0 / (0.005 * sample_rate));
        const double rel = std::exp(-1.0 / (0.060 * sample_rate));
        const float mix = std::min(1.0f, p.breathiness) * 0.08f;
        for (size_t i = 0; i < n; ++i) {
            double ax = std::fabs(static_cast<double>(x[i]));
            env = (ax > env) ? att * env + (1.0 - att) * ax
                             : rel * env + (1.0 - rel) * ax;
            float nz = lp.process(noise.next());
            float gate = static_cast<float>(std::min(1.0, env / 0.08));
            y[i] += nz * gate * mix;
        }
    }

    for (size_t i = 0; i < n; ++i) {
        if (y[i] > 0.99f) y[i] = 0.99f;
        else if (y[i] < -0.99f) y[i] = -0.99f;
    }
    return y;
}

// =========================================================================
//  Prosody: breath insertion at detected pauses
// =========================================================================
std::vector<float> insert_breaths(const float* x, size_t n, int sample_rate,
                                   float intensity, float silence_db,
                                   float min_gap_ms, float breath_ms) {
    if (n == 0 || intensity <= 0.001f || sample_rate <= 0)
        return std::vector<float>(x, x + n);
    intensity = std::min(1.0f, std::max(0.0f, intensity));
    const double gap_ms = min_gap_ms * (1.0 - 0.4 * intensity);
    const size_t gap_n  = static_cast<size_t>(gap_ms * 1e-3 * sample_rate);
    const size_t br_n   = static_cast<size_t>(breath_ms * 1e-3 * sample_rate);
    if (gap_n < 32 || br_n < 16) return std::vector<float>(x, x + n);

    const double thresh = std::pow(10.0, -silence_db / 20.0);
    std::vector<uint8_t> silent(n, 0);
    {
        double env = 0.0;
        const double rel = std::exp(-1.0 / (0.020 * sample_rate));
        for (size_t i = 0; i < n; ++i) {
            double ax = std::fabs(static_cast<double>(x[i]));
            env = (ax > env) ? ax : rel * env + (1.0 - rel) * ax;
            silent[i] = (env < thresh) ? 1 : 0;
        }
    }

    std::vector<float> y(x, x + n);
    size_t run_start = 0;
    for (size_t i = 0; i <= n; ++i) {
        bool in_sil = (i < n) && silent[i];
        bool prev_sil = (i > 0) && silent[i - 1];
        if (in_sil && !prev_sil) run_start = i;
        if (!in_sil && prev_sil) {
            size_t run_len = i - run_start;
            if (run_len >= gap_n) {
                size_t mid = run_start + run_len / 2;
                size_t start = (mid >= br_n / 2) ? mid - br_n / 2 : 0;
                if (start + br_n > n) start = (n >= br_n) ? n - br_n : 0;
                NoiseGen noise(static_cast<uint32_t>(mid + 1));
                Biquad lp = Biquad::one_pole_lp(4000.0, sample_rate);
                const float amp = 0.10f * intensity;
                for (size_t k = 0; k < br_n && start + k < n; ++k) {
                    double w = 0.5 * (1.0 - std::cos(2.0 * M_PI * k /
                                    static_cast<double>(br_n - 1)));
                    float nz = lp.process(noise.next());
                    y[start + k] = static_cast<float>(
                        static_cast<double>(y[start + k]) * (1.0 - 0.7 * w)
                        + static_cast<double>(nz) * w * amp);
                }
            }
        }
    }
    return y;
}

// =========================================================================
//  Prosody: pitch contour shaping
// =========================================================================
std::vector<float> pitch_contour(const float* x, size_t n, int sample_rate,
                                  const std::string& shape, float depth) {
    if (n == 0 || depth <= 0.001f || sample_rate <= 0 || shape == "flat" || shape.empty())
        return std::vector<float>(x, x + n);
    depth = std::min(6.0f, std::max(0.0f, depth));

    auto contour = [&](double t) -> double {
        if (shape == "rise")  return depth * (2.0 * t - 1.0);
        if (shape == "fall")  return depth * (1.0 - 2.0 * t);
        if (shape == "dip")   return -depth * (1.0 - 4.0 * (t - 0.5) * (t - 0.5));
        if (shape == "wave")  return depth * std::sin(2.0 * M_PI * t);
        return 0.0;
    };

    const size_t seg   = static_cast<size_t>(0.160 * sample_rate);
    const size_t cross = static_cast<size_t>(0.040 * sample_rate);
    if (n < seg + cross) return std::vector<float>(x, x + n);

    std::vector<float> y(n, 0.0f);
    std::vector<float> wsum(n, 0.0f);
    size_t pos = 0;
    while (pos < n) {
        size_t end = std::min(pos + seg, n);
        size_t len = end - pos;
        double t = (static_cast<double>(pos) + len / 2.0) / static_cast<double>(n);
        double semi = contour(t);
        const double denom = static_cast<double>(len > 1 ? len - 1 : 1);
        if (std::fabs(semi) > 0.05) {
            auto shifted = audiocore::pitch_shift(x + pos, len, semi, sample_rate);
            if (shifted.size() < len) shifted.resize(len, 0.0f);
            if (shifted.size() > len) shifted.resize(len);
            for (size_t i = 0; i < len; ++i) {
                double w = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / denom));
                y[pos + i]    += static_cast<float>(shifted[i] * w);
                wsum[pos + i] += static_cast<float>(w);
            }
        } else {
            for (size_t i = 0; i < len; ++i) {
                double w = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / denom));
                y[pos + i]    += x[pos + i] * static_cast<float>(w);
                wsum[pos + i] += static_cast<float>(w);
            }
        }
        if (end >= n) break;
        pos += (seg - cross);
    }
    for (size_t i = 0; i < n; ++i)
        if (wsum[i] > 1e-6f) y[i] /= wsum[i];
    return y;
}

// ── Silence trimming ────────────────────────────────────────────────────
std::vector<float> trim_silence(const float* x, size_t n,
                                int sample_rate,
                                float threshold_db,
                                float window_ms,
                                float margin_ms) {
    if (n == 0) return {};
    const double thresh = std::pow(10.0, -threshold_db / 20.0);
    const size_t win = std::max((size_t)1, (size_t)(window_ms * 1e-3 * sample_rate));
    const size_t margin = (size_t)(margin_ms * 1e-3 * sample_rate);

    // Compute sliding-window RMS envelope.
    // For each sample, RMS over [i, i+win).
    std::vector<float> env(n, 0.0f);
    double run_sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        run_sum += (double)x[i] * x[i];
        if (i >= win) run_sum -= (double)x[i - win] * x[i - win];
        size_t cnt = std::min(i + 1, win);
        env[i] = (float)std::sqrt(run_sum / cnt);
    }

    // Find first sample above threshold.
    size_t start = n;  // if never found, entire signal is silent
    for (size_t i = 0; i < n; ++i) {
        if (env[i] > thresh) { start = i; break; }
    }
    if (start == n) {
        // Entire signal below threshold — return unchanged.
        return std::vector<float>(x, x + n);
    }

    // Find last sample above threshold.
    size_t end = start;
    for (size_t i = n; i-- > 0;) {
        if (env[i] > thresh) { end = i; break; }
    }

    // Add margins, clamped to [0, n).
    size_t out_start = (start >= margin) ? start - margin : 0;
    size_t out_end = std::min(end + margin, n);

    return std::vector<float>(x + out_start, x + out_end);
}

}  // namespace audiocore
