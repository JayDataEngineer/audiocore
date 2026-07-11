#ifndef AUDIOCORE_FRAMEWORK_AUDIO_DSP_H
#define AUDIOCORE_FRAMEWORK_AUDIO_DSP_H

#include <vector>
#include <cstddef>
#include <string>

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

// ── Voice-enhance chain: formant shaper + airiness + breathiness ────────
//
// A cascaded biquad EQ (RBJ Audio EQ Cookbook) plus optional breathiness
// noise excitation. Every band is centred at a perceptually meaningful
// frequency for speech; each axis ranges -1..+1 where 0 = flat (no change).
//
//   warmth     : low-shelf @ 180 Hz   (+6 dB → chesty/full, -6 dB → thin)
//   formant    : peaking  @ 700/1700 Hz (F1/F2 emphasis → vowel colour)
//   brightness : peaking  @ 3000 Hz   (presence/crispness)
//   airiness   : high-shelf @ 6000 Hz (+8 dB → breathy/airy sheen)
//   breathiness: shaped white-noise excitation, mixed in proportionally to
//                the signal envelope (0 = off, 1 = strong breathy texture)
//
struct VoiceEnhanceParams {
    float warmth     = 0.0f;  // -1..1
    float formant    = 0.0f;  // -1..1
    float brightness = 0.0f;  // -1..1
    float airiness   = 0.0f;  // -1..1
    float breathiness = 0.0f; // 0..1
};
std::vector<float> voice_enhance(const float* x, size_t n,
                                  int sample_rate,
                                  const VoiceEnhanceParams& p);

// ── Prosody: automatic breath insertion at pauses ───────────────────────
//
// Scans for silence gaps (envelope below -silence_db for >= min_gap_ms) and,
// in each gap, lays down a short shaped noise "breath" of breath_ms.
// `intensity` 0..1 scales breath loudness and relaxes the gap threshold
// (higher intensity → breaths in shorter gaps). Returns a signal that is
// never shorter than the input.
//
std::vector<float> insert_breaths(const float* x, size_t n, int sample_rate,
                                   float intensity = 0.5f,
                                   float silence_db = 32.0f,
                                   float min_gap_ms = 120.0f,
                                   float breath_ms  = 90.0f);

// ── Prosody: pitch contour shaping ──────────────────────────────────────
//
// Applies a time-varying pitch shift across the utterance. `shape` selects
// the contour: "flat", "rise", "fall", "dip", "wave". `depth` is the peak
// deviation in semitones (0..6). The signal is split into overlapping
// segments; each is pitch-shifted by the contour value at its midpoint and
// crossfaded with its neighbours, so the result is the same length as the
// input but with a deliberate intonation movement.
//
std::vector<float> pitch_contour(const float* x, size_t n, int sample_rate,
                                  const std::string& shape,
                                  float depth);

// ── Silence trimming ────────────────────────────────────────────────────
//
// Removes leading and trailing silence from a PCM signal. A sliding-window
// RMS envelope is computed (window_ms, default 10 ms); the first and last
// windows whose RMS exceeds -threshold_db define the speech region. A
// margin of margin_ms silence is retained at each end for natural onset/
// decay. If the entire signal is below threshold, the original is returned
// unchanged (don't destroy intentionally-silent output).
//
std::vector<float> trim_silence(const float* x, size_t n,
                                int sample_rate = 24000,
                                float threshold_db = 40.0f,
                                float window_ms = 10.0f,
                                float margin_ms = 100.0f);

}  // namespace audiocore

#endif  // AUDIOCORE_FRAMEWORK_AUDIO_DSP_H
