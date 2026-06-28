// predictor_runner.h — Code Predictor (qwen3tts_cp architecture) via llama.cpp.
//
// Supports two work modes:
//   1. Standard (Lunavox GGUFs): forward_tokens with interleaved logits.
//   2. MTP mode (official Qwen3-TTS, 31 fine codebooks): loads codec_embd.{i}
//      and lm_head.{i} tensors from GGUF, runs the multi-codebook predictor loop.
//
// Official Qwen3-TTS code predictor architecture:
//   At each generation step, the predictor takes the talker's last hidden state
//   and all previously-predicted fine codebook IDs, then autoregressively
//   generates 31 fine codebooks using 31 separate embedding tables + 31 lm_heads.

#ifndef AUDIOCORE_MODELS_QWEN3_TTS_PREDICTOR_RUNNER_H
#define AUDIOCORE_MODELS_QWEN3_TTS_PREDICTOR_RUNNER_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct llama_model;
struct llama_context;

namespace audiocore::qwen3_tts {

struct PredictorConfig {
    int  n_codebooks   = 32;   // total codebooks (1 coarse + 31 fine)
    int  n_fine_books  = 31;   // fine codebooks predicted by this module
    int  n_ctx         = 4096;
    int  n_batch       = 512;
    int  n_threads     = 0;
    int  n_gpu_layers  = -1;
    int  main_gpu      = -1;
    bool flash_attn    = true;
};

class PredictorRunner {
public:
    static std::unique_ptr<PredictorRunner> load(const std::string& gguf_path,
                                                  const PredictorConfig& cfg,
                                                  std::string* error = nullptr);
    ~PredictorRunner();

    PredictorRunner(const PredictorRunner&)            = delete;
    PredictorRunner& operator=(const PredictorRunner&) = delete;

    // ── Standard token-mode (Lunavox GGUFs) ────────────────────────────

    // Token input → logits. [n_tokens × n_vocab] row-major.
    bool forward_tokens(const int32_t* tokens, int32_t n_tokens, int32_t n_pos,
                        float* logits, std::string* error = nullptr);

    // ── MTP mode (official Qwen3-TTS, 31 fine codebooks) ──────────────

    // Returns true if the MTP tensors (codec_embd.*, lm_head.*) are loaded.
    bool has_mtp() const { return has_mtp_; }

    // Run one step of code prediction.
    // talker_hidden: [1, hidden_size] float32 — the talker's last hidden state
    // prev_codes: [32] int32 — the 32 codebook IDs from the previous step (or
    //             all -1/0 for the first step)
    // out_codes: [31] int32 — output fine codebook IDs for this step
    bool predict_one_step(const float* talker_hidden,
                          const int32_t* prev_codes,
                          int32_t* out_codes,
                          std::string* error = nullptr);

    // ── Utilities ──────────────────────────────────────────────────────

    const float* get_logits_ith(int32_t i) const;

    int32_t hidden_size() const { return n_embd_; }
    int32_t vocab_size()  const { return vocab_size_; }
    int     n_codebooks() const { return n_codebooks_; }

    llama_model*   raw_model()   const { return model_; }
    llama_context* raw_context() const { return ctx_;   }

    // Access fine embedding tables and lm heads (for manual sampling).
    const float* fine_embedding(int i) const { return (i >= 0 && i < n_fine_books_) ? fine_embd_[i] : nullptr; }
    const float* fine_head(int i)      const { return (i >= 0 && i < n_fine_books_) ? fine_head_[i] : nullptr; }

private:
    PredictorRunner();
    bool load_mtp_tensors(const std::string& gguf_path, std::string* error);

    llama_model*   model_ = nullptr;
    llama_context* ctx_   = nullptr;

    int32_t n_embd_        = 0;
    int32_t vocab_size_    = 0;
    int32_t n_codebooks_   = 32;
    int32_t n_fine_books_  = 31;

    // MTP-mode tensors loaded directly from GGUF
    bool     has_mtp_     = false;
    float**  fine_embd_   = nullptr;  // [31] each [fine_vocab, n_embd]
    float**  fine_head_   = nullptr;  // [31] each [n_embd, fine_vocab]
    int32_t  fine_vocab_  = 0;
    float*   small_to_mtp_w_ = nullptr;  // [n_embd, n_embd]
    float*   small_to_mtp_b_ = nullptr;  // [n_embd]

    // Workspace for MTP decode
    std::vector<float> scratch_embd_;
    std::vector<float> scratch_logits_;
};

}  // namespace audiocore::qwen3_tts

#endif  // AUDIOCORE_MODELS_QWEN3_TTS_PREDICTOR_RUNNER_H
