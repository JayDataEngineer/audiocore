#ifndef AUDIOCORE_FRAMEWORK_IO_MP3_ENCODER_H
#define AUDIOCORE_FRAMEWORK_IO_MP3_ENCODER_H

#include <cstdint>
#include <string>
#include <vector>

namespace audiocore {

// Encode mono float32 PCM to MP3. Values in [-1, 1].
// Returns MP3 bytes on success, empty vector on error.
std::vector<uint8_t> pcm_mono_to_mp3(const float* pcm, size_t n_samples,
                                      int32_t sample_rate);

// Encode stereo interleaved float32 PCM to MP3.
std::vector<uint8_t> pcm_stereo_to_mp3(const float* pcm, size_t n_samples,
                                        int32_t sample_rate);

}  // namespace audiocore

#endif  // AUDIOCORE_FRAMEWORK_IO_MP3_ENCODER_H
