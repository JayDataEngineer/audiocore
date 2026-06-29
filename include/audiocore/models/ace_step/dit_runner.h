// dit_runner.h — DiT (diffusion transformer) graph builder + forward pass.
//
// The DiT is a 24-layer transformer using AdaLN, GQA with sliding-window
// self-attention + cross-attention, and SwiGLU MLP. It implements flow-
// matching denoising with classifier-free guidance.
//
// Weights are pre-bound in ext_ctx_ by loader.cpp. This runner builds ggml
// computation graphs for a single DiT forward pass, called repeatedly by
// the denoising loop in session.cpp.
//
// Architecture reference: ServeurpersoCom/acestep.cpp — dit.h / dit_ggml_generate

#ifndef AUDIOCORE_MODELS_ACE_STEP_DIT_RUNNER_H
#define AUDIOCORE_MODELS_ACE_STEP_DIT_RUNNER_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;

namespace audiocore::acestep {

struct DitConfig {
    int32_t hidden_size          = 0;
    int32_t intermediate_size    = 0;
    int32_t n_heads              = 0;
    int32_t n_kv_heads           = 0;
    int32_t head_dim             = 0;
    int32_t n_layers             = 0;
    int32_t sliding_window       = 0;
    int32_t in_channels          = 0;   // acestep.in_channels  (192)
    int32_t out_channels         = 0;   // acestep.audio_acoustic_hidden_dim (64)
    int32_t patch_size           = 0;
    float   rope_theta           = 10000.0f;
    float   rms_norm_eps         = 1e-6f;
};

class DiTRunner {
public:
    DiTRunner(ggml_context* ext_ctx, const DitConfig& cfg);
    ~DiTRunner();

    // One DiT forward pass: given noise latent x_t and timestep t, predict v.
    //   x_t:    [T_patches, hidden_size] float32 — current noisy latent
    //   t:      scalar timestep (0..1)
    //   cond:   [T_cond, encoder_hidden] float32 — text encoder output
    //   cond_nc:[T_cond, encoder_hidden] float32 — null-text encoder (CFG uncond)
    //   output: [T_patches, hidden_size] float32 — predicted velocity v
    bool forward(const float* x_t, float t,
                 const float* cond, int32_t T_cond, int32_t cond_hidden,
                 const float* cond_nc, int32_t T_cond_nc,
                 float guidance_scale, int32_t n_patches,
                 float* output, std::string* error);

    const DitConfig& config() const { return cfg_; }

    // Shortcuts to weight tensors in ext_ctx_
    ggml_tensor* weight(const char* name) const;

private:
    ggml_context* ext_ctx_;   // weight tensors only (no_alloc=true, mmap'd)
    DitConfig cfg_;

    // Pre-converted per-layer and global scale_shift_table (bf16 → f32).
    // Applied as a learned bias to time_mod per layer (per-layer) and as
    // final output scale/shift (global).
    std::vector<std::vector<float>> ss_table_f32_;   // [n_layers][H * 6]
    std::vector<float>              global_ss_f32_;  // [H * 2]

    // Cached max graph size — allocate once, reuse across steps
    size_t graph_mem_size_ = 0;
    char*  graph_mem_buf_  = nullptr;
};

}  // namespace audiocore::acestep

#endif  // AUDIOCORE_MODELS_ACE_STEP_DIT_RUNNER_H
