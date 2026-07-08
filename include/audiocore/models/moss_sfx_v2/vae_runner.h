#ifndef AUDIOCORE_MODELS_MOSS_SFX_V2_VAE_RUNNER_H
#define AUDIOCORE_MODELS_MOSS_SFX_V2_VAE_RUNNER_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// ggml_fp16_t is uint16_t — forward-declared to avoid pulling in ggml.h here.
using mse2_fp16_t = uint16_t;

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

namespace audiocore::moss_sfx_v2 {

// DAC VAE decoder config — read from GGUF metadata at bind time.
struct VAEConfig {
    int32_t latent_dim     = 128;    // input channels
    int32_t decoder_dim    = 2048;   // first conv output channels
    int32_t hop_length     = 960;    // total down/up sampling factor
    int32_t sample_rate    = 48000;
    bool    continuous     = true;   // continuous-mode VAE: apply post_quant_conv
};

struct Conv1dOp {
    int IC = 0, OC = 0, K = 0, stride = 1, pad = 0, dilation = 1;
    std::vector<float> weight_f32;
    std::vector<float> bias_;
    std::vector<mse2_fp16_t> weight_f16;  // pre-converted F16 cache

    bool run(const float* x, int T_in, std::vector<float>& out,
             std::string* error, ggml_backend_sched_t sched,
             char* scratch, size_t scratch_size) const;
    void cache_f16();
};

struct ConvT1dOp {
    int IC = 0, OC = 0, K = 0, stride = 1, pad = 0, output_padding = 0;
    std::vector<float> weight_2d;
    std::vector<float> bias_;
    std::vector<mse2_fp16_t> weight_f16;  // pre-converted F16 cache
    std::vector<float> weight_f32_t;       // F32 transposed cache [K,OC,IC] col-major
    // Col2im-ordered weight for the mul_mat + col2im_1d graph path.
    // weight_2d has columns indexed k*OC+oc (K-major), but the col2im_1d
    // CUDA kernel reads col[(oc*K+k) + t_in*K_OC] (OC-major).  This
    // pre-transposed copy stores columns as oc*K+k so the graph path
    // matches the kernel's expectation.
    std::vector<float> weight_2d_c2i;      // [IC, OC*K] col-major, col = oc*K+k

    bool run(const float* x, int T_in, std::vector<float>& out,
             std::string* error, ggml_backend_sched_t sched,
             char* scratch, size_t scratch_size) const;
    void cache_f16();
};

struct SnakeOp {
    int C = 0;
    std::vector<float> alpha;
    std::vector<float> inv_alpha;

    bool run(const float* x, int T, std::vector<float>& out,
             std::string* error, ggml_backend_sched_t sched,
             char* scratch, size_t scratch_size) const;
};

struct ResUnitOps {
    SnakeOp snake1;
    Conv1dOp conv1;
    SnakeOp snake2;
    Conv1dOp conv2;
};

struct DecoderBlockOps {
    SnakeOp snake;
    ConvT1dOp conv_t;
    ResUnitOps res_units[3];
};

class VAERunner {
public:
    VAERunner(ggml_context* ext_ctx, const VAEConfig& cfg);
    ~VAERunner();

    bool decode(const float* z, int32_t B, int32_t T_latent,
                float* out, std::string* error);

    // Traced decode — emits intermediate outputs for parity testing.
    // Maps: "post_pqc" → post_quant_conv out, "vae_dec_0".."vae_dec_8" →
    // each Sequential layer, "vae_final" → final tanh output.
    struct Trace {
        std::vector<float> post_pqc;        // [T, latent_dim]
        std::vector<float> vae_dec[9];      // [T_i, C_i] per layer
        std::vector<float> vae_final;       // [T_out, 1]

        // Per-sub-operation traces for DecoderBlock 1 (block index 0):
        std::vector<float> blk1_snake;      // Snake output pre-ConvT1d
        std::vector<float> blk1_convt;      // ConvT1d output pre-ResUnits
        std::vector<float> blk1_res[4];     // ResUnit outputs (0..3, N+1 = block output)
        // ResUnit 0 sub-traces:
        std::vector<float> blk1_res0_s1;    // Snake 1 output
        std::vector<float> blk1_res0_c1;    // Conv 1 output
        std::vector<float> blk1_res0_s2;    // Snake 2 output
        std::vector<float> blk1_res0_c2;    // Conv 2 output (pre-residual)
    };
    bool decode_traced(const float* z, int32_t B, int32_t T_latent,
                       float* out, Trace* trace, std::string* error);

    const VAEConfig& config() const { return cfg_; }

    // Lazily initialize GPU backend + scheduler on first call. Idempotent.
    bool ensure_backend();

private:
    ggml_context* ext_ctx_;
    VAEConfig cfg_;

    Conv1dOp post_quant_conv_;  // 1×1 Conv1d (latent_dim → latent_dim), continuous mode only
    Conv1dOp conv_in_;
    std::vector<float> bias_0_;
    std::vector<DecoderBlockOps> blocks_;
    SnakeOp snake_out_;
    Conv1dOp conv_out_;

    // Backend state — lazily initialized on first decode call.
    std::unique_ptr<ggml_utils::BackendPair> backend_pair_;
    ggml_backend_sched_t                     sched_        = nullptr;
    bool                                     backend_ready_ = false;

    // Shared scratch buffer — allocated once, reused across all ops in a decode.
    // Eliminates the ~95× per-op 1GB allocation that dominated VAE decode time.
    std::vector<char> scratch_;
};

}  // namespace audiocore::moss_sfx_v2

#endif
