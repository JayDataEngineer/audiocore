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

    // Look up a VAE weight tensor (bound with vae. prefix).
    ggml_tensor* weight(const char* name) const;

private:
    ggml_context* ext_ctx_;   // weight tensors (no_alloc=true, mmap'd)
};

}  // namespace audiocore::acestep

#endif  // AUDIOCORE_MODELS_ACE_STEP_VAE_RUNNER_H
