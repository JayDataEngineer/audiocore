// vae_runner.h — VAE encoder/decoder graph builder + forward pass.
//
// The VAE decoder takes 25Hz latent frames [T, 64] and produces 48kHz stereo
// PCM audio. Total upsampling factor: 1920× across 5 decoder blocks with
// Snake activations, ConvTranspose1d layers, and ResUnits.
// The encoder mirrors this in reverse, striding 1920× down to 25Hz latent.
//
// All convolutional weights are WSConv (weight_g + weight_v in bf16 in GGUF).
// precompute_weights() composes them into f32 at construction time.
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
struct ggml_backend;
struct ggml_backend_sched;
using ggml_backend_t       = ggml_backend*;
using ggml_backend_sched_t = ggml_backend_sched*;

namespace audiocore::ggml_utils {
struct BackendPair;
}  // namespace audiocore::ggml_utils

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

    // Lazily initialize GPU backend + scheduler on first call. Idempotent.
    bool ensure_backend();

public:  // weight structs are public so graph builders can reference them
    // ── Per-layer pre-computed weight structs ────────────────────────────
    struct ResUnitWeights {
        std::vector<float> s1a_;   // snake1 alpha [C]
        std::vector<float> s1b_;   // snake1 beta  [C]
        std::vector<float> c1w_;   // conv1 WSConv [K, C, C] f32
        int32_t            c1K_ = 7;
        std::vector<float> c1b_;   // conv1 bias [C] (empty if none)
        std::vector<float> s2a_;   // snake2 alpha [C]
        std::vector<float> s2b_;   // snake2 beta  [C]
        std::vector<float> c2w_;   // conv2 WSConv [1, C, C] f32
        int32_t            c2K_ = 1;
        std::vector<float> c2b_;   // conv2 bias [C] (empty if none)
    };

    struct BlockWeights {
        // Block-level snake (pre-stride for decoder, post-ResUnit for encoder)
        std::vector<float> snake_a_;     // [in_ch]
        std::vector<float> snake_b_;     // [in_ch]
        // Strided conv (conv_t1d for decoder, conv1d for encoder)
        std::vector<float> ct1_w_;       // decoder conv_t permuted 2D [IC, K*OC]
        int32_t            ct1_K_ = 0;
        std::vector<float> ct1_b_;       // [OC]
        std::vector<float> conv_w_;      // encoder conv1 WSConv [K, OC, IC] f32
        int32_t            conv_K_ = 0;
        std::vector<float> conv_b_;      // [OC]
        ResUnitWeights     res_[3];      // 3 ResUnits
    };

    // Pre-compute all weights from GGUF bf16/WSConv format
    void precompute_weights(ggml_context* ext_ctx);

    // Per-block sub-graph decode (fast CUDA path). Builds 7 sub-graphs
    // (conv1, 5 decoder blocks, final snake+conv2) instead of 73 per-op
    // calls, keeping data on the GPU within each block.
    bool decode_blocks(const float* latents, int32_t n_frames,
                       std::vector<float>* pcm, std::string* error);

    ggml_context* ext_ctx_;

    // ── Decoder weights ──────────────────────────────────────────────────
    std::vector<float> dec_conv1_w_;    // [7, 2048, 64] WSConv f32
    int32_t            dec_conv1_K_ = 7;
    std::vector<float> dec_conv1_b_;    // [2048]
    std::vector<float> dec_conv2_w_;    // [7, 2, 128] WSConv f32
    int32_t            dec_conv2_K_ = 7;
    std::vector<float> dec_fn_exp_a_;   // final snake alpha [128]
    std::vector<float> dec_fn_inv_b_;   // final snake beta  [128]
    BlockWeights       dec_blk_[5];     // 5 decoder blocks

    // ── Encoder weights ──────────────────────────────────────────────────
    std::vector<float> enc_conv1_w_;    // [7, 128, 2] WSConv f32
    int32_t            enc_conv1_K_ = 7;
    std::vector<float> enc_conv1_b_;    // [128]
    std::vector<float> enc_conv2_w_;    // [3, 128, 2048] WSConv f32
    int32_t            enc_conv2_K_ = 3;
    std::vector<float> enc_conv2_b_;    // [128]
    std::vector<float> enc_fn_exp_a_;   // final encoder snake alpha [2048]
    std::vector<float> enc_fn_inv_b_;   // final encoder snake beta  [2048]
    BlockWeights       enc_blk_[5];     // 5 encoder blocks

    // Backend state — lazily initialized on first decode/encode call.
    // We bypass the scheduler entirely (it forces INPUT-flagged tensors to
    // the last backend = CPU, preventing CUDA offload) and compute directly
    // on the CUDA backend via ggml_backend_alloc_ctx_tensors + graph_compute.
    std::unique_ptr<ggml_utils::BackendPair> backend_pair_;
    ggml_backend_t                           cuda_backend_  = nullptr;
    bool                                     backend_ready_ = false;
    // Set true when decode_blocks() fails within the current tiled decode
    // (e.g. block4 CUDA OOM on a large tile). Subsequent TILES in the same
    // decode skip the fast-path and go straight to per-op fallback, saving
    // ~200ms/tile of wasted alloc. Reset to false at the start of each new
    // tiled decode() so transient OOM (e.g. right after a model swap) doesn't
    // permanently cripple the fast path.
    bool                                     blocks_failed_once_ = false;
};

}  // namespace audiocore::acestep

#endif  // AUDIOCORE_MODELS_ACE_STEP_VAE_RUNNER_H
