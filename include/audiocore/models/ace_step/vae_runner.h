// vae_runner.h — VAE decoder graph builder + forward pass.
//
// The VAE decoder takes 25Hz latent frames [T, 64] and produces 48kHz stereo
// PCM audio. Total upsampling factor: 1920× across 5 decoder blocks with
// Snake activations, ConvTranspose1d layers, and ResUnits.
//
// Architecture reference: ServeurpersoCom/acestep.cpp — vae.h / vae_ggml_decode

#ifndef AUDIOCORE_MODELS_ACE_STEP_VAE_RUNNER_H
#define AUDIOCORE_MODELS_ACE_STEP_VAE_RUNNER_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;

namespace audiocore::acestep {

class VAERunner {
public:
    VAERunner(ggml_context* ext_ctx);
    ~VAERunner();

    // Decode latent frames [T, 64] → stereo PCM at 48kHz.
    //   latents:  [T * 64] float32 — time-major latent frames
    //   n_frames: T — number of 25Hz latent frames
    //   pcm:      output — interleaved stereo float32 samples
    bool decode(const float* latents, int32_t n_frames,
                std::vector<float>* pcm, std::string* error);

    // Encode stereo PCM at 48kHz → latent frames [T, 64].
    //   pcm_stereo: [n_samples * 2] float32 — interleaved L,R,L,R,…
    //   n_samples:  number of stereo samples (must be multiple of 1920)
    //   latents:    output — [T_latent * 64] float32 time-major
    //   T_latent:   n_samples / 1920
    bool encode(const float* pcm_stereo, int32_t n_samples,
                std::vector<float>* latents, std::string* error);

    // Look up a VAE weight tensor (bound with vae. prefix).
    ggml_tensor* weight(const char* name) const;

private:
    ggml_context* ext_ctx_;   // weight tensors (no_alloc=true, mmap'd)
};

}  // namespace audiocore::acestep

#endif  // AUDIOCORE_MODELS_ACE_STEP_VAE_RUNNER_H
