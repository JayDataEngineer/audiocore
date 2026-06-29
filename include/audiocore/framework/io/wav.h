// wav.h — Minimal RIFF/WAVE I/O, mono float32 ↔ 16-bit PCM.
//
// Ported verbatim from pwilkin/openmoss (Apache-2.0) src/wav.cpp into
// audiocore's framework/io layer so every family that needs to read or write
// a WAV shares one implementation. Supports PCM (8/16/24/32-bit), IEEE-float
// (32-bit), WAVE_FORMAT_EXTENSIBLE, multi-channel (downmixed to mono), and
// linear-interpolation resampling to a target sample rate — everything the
// upstream MOSS pipeline expects from `read_wav_mono` / `decode_wav_mono`.
//
// Previously each family hand-rolled its own loader:
//   • MOSS voice_clone read a .codes binary (skipping WAV entirely)
//   • Qwen3TtsSpeakerEncoder::load_wav hard-required 24 kHz 16/32-bit only
// Both are deleted/replaced by callers moving to these helpers.

#ifndef AUDIOCORE_FRAMEWORK_IO_WAV_H
#define AUDIOCORE_FRAMEWORK_IO_WAV_H

#include <cstdint>
#include <string>
#include <vector>

namespace audiocore::io {

// Read a mono WAV file. Resamples via linear interpolation if the file is not
// at `target_sr`. Returns f32 PCM in [-1, 1]. Throws on error.
std::vector<float> read_wav_mono(const std::string& path, int32_t target_sr);

// Decode a WAV byte buffer (in-memory equivalent of read_wav_mono). Resamples
// linearly to `target_sr` if the embedded sample rate differs. Throws on error.
std::vector<float> decode_wav_mono(const uint8_t* data, size_t n_bytes,
                                    int32_t target_sr);

// Write mono f32 PCM as 16-bit little-endian WAV at `sample_rate`.
void write_wav_mono(const std::string& path,
                    const float* pcm, int64_t n_samples,
                    int32_t sample_rate);

// In-memory WAV encoder: 16-bit mono PCM. Useful for serving over HTTP.
std::vector<uint8_t> encode_wav_mono(const float* pcm, int64_t n_samples,
                                      int32_t sample_rate);

}  // namespace audiocore::io

#endif  // AUDIOCORE_FRAMEWORK_IO_WAV_H
