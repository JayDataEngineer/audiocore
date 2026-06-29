#include "audiocore/framework/io/mp3_encoder.h"

#include <lame/lame.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <vector>

namespace audiocore {

static std::vector<uint8_t> pcm_to_mp3_impl(const float* pcm,
                                              size_t n_samples,
                                              int32_t sample_rate,
                                              int channels) {
    if (!pcm || n_samples == 0) return {};

    lame_global_flags* gfp = lame_init();
    if (!gfp) {
        std::fprintf(stderr, "mp3_encoder: lame_init failed\n");
        return {};
    }

    lame_set_num_channels(gfp, channels);
    lame_set_in_samplerate(gfp, static_cast<int>(sample_rate));
    lame_set_brate(gfp, 192);
    lame_set_quality(gfp, 2);
    lame_set_mode(gfp, channels == 1 ? MONO : STEREO);
    lame_set_VBR(gfp, vbr_default);

    if (lame_init_params(gfp) < 0) {
        std::fprintf(stderr, "mp3_encoder: lame_init_params failed\n");
        lame_close(gfp);
        return {};
    }

    int mp3_size = static_cast<int>(1.25 * n_samples + 7200);
    std::vector<uint8_t> mp3_buf(static_cast<size_t>(mp3_size));
    int n_encoded = 0;

    if (channels == 1) {
        n_encoded = lame_encode_buffer_ieee_float(
            gfp, const_cast<float*>(pcm), const_cast<float*>(pcm),
            static_cast<int>(n_samples), mp3_buf.data(), mp3_size);
    } else {
        const size_t n_frames = n_samples / 2;
        std::vector<float> left(n_frames);
        std::vector<float> right(n_frames);
        for (size_t i = 0; i < n_frames; i++) {
            left[i]  = pcm[i * 2];
            right[i] = pcm[i * 2 + 1];
        }
        n_encoded = lame_encode_buffer_ieee_float(
            gfp, left.data(), right.data(),
            static_cast<int>(n_frames), mp3_buf.data(), mp3_size);
    }

    if (n_encoded < 0) {
        std::fprintf(stderr, "mp3_encoder: encode returned %d\n", n_encoded);
        lame_close(gfp);
        return {};
    }

    int n_flush = lame_encode_flush(gfp, mp3_buf.data() + n_encoded,
                                     mp3_size - n_encoded);
    if (n_flush > 0) n_encoded += n_flush;

    lame_close(gfp);
    mp3_buf.resize(static_cast<size_t>(n_encoded));
    return mp3_buf;
}

std::vector<uint8_t> pcm_mono_to_mp3(const float* pcm, size_t n_samples,
                                      int32_t sample_rate) {
    return pcm_to_mp3_impl(pcm, n_samples, sample_rate, 1);
}

std::vector<uint8_t> pcm_stereo_to_mp3(const float* pcm, size_t n_samples,
                                        int32_t sample_rate) {
    return pcm_to_mp3_impl(pcm, n_samples, sample_rate, 2);
}

}  // namespace audiocore
