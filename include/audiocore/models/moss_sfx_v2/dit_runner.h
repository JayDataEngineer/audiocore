#ifndef AUDIOCORE_MODELS_MOSS_SFX_V2_DIT_RUNNER_H
#define AUDIOCORE_MODELS_MOSS_SFX_V2_DIT_RUNNER_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "audiocore/framework/ggml/backend_helper.h"

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;
typedef struct ggml_backend_sched * ggml_backend_sched_t;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
typedef struct ggml_gallocr * ggml_gallocr_t;

namespace audiocore::moss_sfx_v2 {

struct DitConfig {
    int32_t dim          = 1536;
    int32_t ffn_dim      = 8960;
    int32_t n_heads      = 12;
    int32_t head_dim     = 128;   // dim / n_heads
    int32_t n_layers     = 30;
    int32_t in_dim       = 128;   // patch_embedding input channels
    int32_t out_dim      = 128;   // head output channels
    int32_t text_dim     = 2048;  // Qwen3 hidden size
    int32_t freq_dim     = 256;   // sinusoidal embedding dim
    float   eps          = 1e-6f;
    int32_t patch_size   = 1;     // 1D patch size
};

class DiTRunner {
public:
    DiTRunner(ggml_context* ext_ctx, const DitConfig& cfg);
    ~DiTRunner();

    bool forward(const float* x_t, const float* t,
                 const float* context, int32_t T_text,
                 int32_t B, int32_t T_latent,
                 float* output, std::string* error,
                 bool force_rebuild = false);

    bool forward_cfg(const float* x_t, const float* t,
                     const float* context_cond, int32_t T_cond,
                     const float* context_uncond, int32_t T_uncond,
                     float guidance_scale,
                     int32_t B, int32_t T_latent,
                     float* output, std::string* error);

    const DitConfig& config() const { return cfg_; }

private:
    ggml_context* ext_ctx_;
    DitConfig cfg_;

    std::vector<std::vector<float>> modulation_f32_;
    std::vector<float> head_modulation_f32_;

    // Static weights read before GPU migration (time embedding + projection)
    std::vector<float> cpu_te0_w_, cpu_te0_b_;
    std::vector<float> cpu_te2_w_, cpu_te2_b_;
    std::vector<float> cpu_tp_w_, cpu_tp_b_;
    void load_cpu_weights();

    // Compute AdaLN modulation vector on CPU (uses pre-read weights)
    void compute_modulation(const float* temb, int temb_dim, int H,
                            float* mod_buf) const;

    // GPU backend (lazy init on first forward)
    // We bypass the scheduler (forces INPUT-flagged tensors to CPU) and
    // compute directly on the CUDA backend.
    std::unique_ptr<audiocore::ggml_utils::BackendPair> bp_;
    ggml_backend_sched_t   sched_         = nullptr;
    ggml_backend_t         cuda_backend_  = nullptr;
    ggml_backend_buffer_t  migrated_buf_  = nullptr;
    ggml_gallocr_t         gallocr_       = nullptr;
    bool backend_initialized_ = false;

    // Cached graph-building context (reused across forwards)
    char*        cached_ctx_buf_ = nullptr;
    ggml_context* cached_ctx_    = nullptr;
    bool         graph_built_    = false;

    // ── Cached graph for pinned graph reuse (no rebuild on subsequent forwards) ──
    struct CachedGraph {
        ggml_cgraph* gf = nullptr;
        ggml_tensor* x_t = nullptr;
        ggml_tensor* ctx_emb = nullptr;
        ggml_tensor* pos_q = nullptr;
        ggml_tensor* pos_k = nullptr;
        ggml_tensor* output = nullptr;
        std::vector<ggml_tensor*> mod_tensors;  // [layer*6..layer*6+5] for chunks
        int32_t T_latent = 0;
        int32_t ct_len = 0;
    };
    CachedGraph cg_;
    // Fast scratch for position arrays
    std::vector<int32_t> pos_scratch_;

    void update_mod_tensors(const float* mod_buf, int H, int n_lyr);

    bool ensure_backend(std::string* error);
    bool run_one_forward(const float* x_t, int32_t T_latent, int32_t H,
                          const float* mod_buf,
                          const float* cond_data, int ct_len, int cond_hidden,
                          float* result, std::string* error,
                          bool rebuild = true);
    void release_cached();
};

}  // namespace audiocore::moss_sfx_v2

#endif
