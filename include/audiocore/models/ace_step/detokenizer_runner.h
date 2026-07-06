// detokenizer_runner.h — Audio token detokenizer (FSQ codes → DiT src latents)
//
// Bridges the LM (5 Hz discrete codes) and the DiT (25 Hz continuous latents).
//
// Pipeline:
//   LM codes [N, 5Hz] (int32 in [0, 64000))
//     → fsq_decode_one (mixed-radix → 6-D)
//     → tokenizer.quantizer.project_out (Linear 6 → 2048)
//     → detokenizer transformer:
//         · embed_tokens Linear (2048 → 2048)
//         · expand each token to P=5 patches, add learnable special_tokens
//         · 2× Qwen3-style encoder layers (bidirectional within each P=5 group,
//           GQA 16/8 heads, head_dim=128, QK-norm, RoPE θ=1e6)
//         · final RMSNorm
//         · proj_out Linear (2048 → 64)
//     → 25 Hz, 64-D latents fed to the DiT src channels (0..63)
//
// Architecture reference: ACE-Step v1.5 (modeling_acestep_v15_turbo.py)
//   AudioTokenDetokenizer + vector_quantize_pytorch.ResidualFSQ.
//
// All weights are referenced from the model's ext_ctx (no copy); only the FSQ
// codebook path runs on CPU since it's a tiny N×6×2048 matmul.

#ifndef AUDIOCORE_MODELS_ACE_STEP_DETOKENIZER_RUNNER_H
#define AUDIOCORE_MODELS_ACE_STEP_DETOKENIZER_RUNNER_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_context;

namespace audiocore::acestep {

class DetokenizerRunner {
public:
    explicit DetokenizerRunner(ggml_context* ext_ctx);
    ~DetokenizerRunner();

    // Convert LM FSQ codes (5 Hz, one int per frame) → DiT src latents.
    //   codes:    [N] int32, each in [0, 64000)
    //   n_codes:  N — number of 5 Hz frames
    //   latents:  output — [N * 5 * 64] float32, time-major at 25 Hz
    //             layout: latents[t*64 + c] where t ∈ [0, N*5)
    bool decode(const int32_t* codes, int32_t n_codes,
                std::vector<float>* latents, std::string* error);

private:
    ggml_context* ext_ctx_;
};

}  // namespace audiocore::acestep

#endif  // AUDIOCORE_MODELS_ACE_STEP_DETOKENIZER_RUNNER_H
