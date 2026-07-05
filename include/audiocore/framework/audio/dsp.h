#ifndef AUDIOCORE_FRAMEWORK_AUDIO_DSP_H
#define AUDIOCORE_FRAMEWORK_AUDIO_DSP_H

#include <vector>
#include <cstddef>

namespace audiocore {

// ── Time-domain audio DSP utilities for post-processing TTS output. ──────
//
// All routines operate on mono float32 PCM in [-1, 1]. They are
// independent of any model — pure signal processing.

// Linear-resample the signal to a new length (changes BOTH pitch and
// duration, like a tape-speed change). Used as a building block for
// pitch_shift.
std::vector<float> resample_linear(const float* x, size_t n, double ratio);

// WSOLA (Waveform Similarity Overlap-Add) time-stretch. Changes duration
// by `rate` (rate=1.5 → output is 1.5× longer, pitch unchanged).
// `frame_ms` and `hop_ms` default to 30 / 10 ms for speech.
std::vector<float> wsola_stretch(const float* x, size_t n, double rate,
                                  int sample_rate,
                                  double frame_ms = 30.0,
                                  double hop_ms   = 10.0,
                                  double search_ms = 8.0);

// Pitch-shift by `semitones` (12 = up one octave, -12 = down). Keeps
// duration unchanged by combining linear resample with WSOLA compensation.
// A simple, robust choice for speech; preserves formants imperfectly,
// but is fast and artifact-light for moderate shifts (±5 semitones).
std::vector<float> pitch_shift(const float* x, size_t n, double semitones,
                                int sample_rate);

// Time-stretch by `rate` (rate=2.0 → 2× faster playback, pitch unchanged).
// Equivalent to wsola_stretch(x, n, 1/rate, ...).
std::vector<float> change_speed(const float* x, size_t n, double rate,
                                 int sample_rate);

}  // namespace audiocore

#endif  // AUDIOCORE_FRAMEWORK_AUDIO_DSP_H
