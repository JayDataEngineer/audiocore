// talker_runner.h — Talker (qwen3tts architecture) inference via llama.cpp.
//
// Supports two work modes:
//   1. Token mode (forward_tokens / forward_get_hidden) — standard llama.cpp
//      token → embedding → transformer → output. Compatible with Lunavox GGUFs.
//   2. Embedding mode (forward_embeddings) — takes pre-computed [T, n_embd]
//      embeddings directly, skipping the token_embd lookup. Required for the
//      official Qwen3-TTS dual-embedding (text_embedding + codec_embedding sum).
//
// Extra tensors loaded directly from GGUF (for official Qwen3-TTS weights):
//   text_embd.weight     — text embedding table [text_vocab, 2048]
//   text_proj.0.weight   — text projection layer 1 [2048, 2048]
//   text_proj.0.bias     — text projection bias 1 [2048]
//   text_proj.1.weight   — text projection layer 2 [1024, 2048]
//   text_proj.1.bias     — text projection bias 2 [1024]

#ifndef AUDIOCORE_MODELS_QWEN3_TTS_TALKER_RUNNER_H
#define AUDIOCORE_MODELS_QWEN3_TTS_TALKER_RUNNER_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct llama_model;
struct llama_context;

namespace audiocore::qwen3_tts {

struct TalkerConfig {
    int  n_ctx        = 4096;
    int  n_batch      = 512;
    int  n_threads    = 0;
    int  n_gpu_layers = -1;
    int  main_gpu     = -1;
    bool flash_attn   = true;
};

class TalkerRunner {
public:
    static std::unique_ptr<TalkerRunner> load(const std::string& gguf_path,
                                              const TalkerConfig& cfg,
                                              std::string* error = nullptr);
    ~TalkerRunner();

    TalkerRunner(const TalkerRunner&)            = delete;
    TalkerRunner& operator=(const TalkerRunner&) = delete;

    // ── Standard token-mode inference ───────────────────────────────────

    // Token input → logits.
    bool forward_tokens(const int32_t* tokens, int32_t n_tokens, int32_t n_pos,
                        float* logits, std::string* error = nullptr);

    // Token input → hidden states (output of final norm, before lm_head).
    bool forward_get_hidden(const int32_t* tokens, int32_t n_tokens, int32_t n_pos,
                            float* hidden, std::string* error = nullptr);

    // ── Embedding-mode inference (for official dual-embedding model) ──

    // Pre-computed embeddings [T, n_embd_f32] → hidden states.
    // Skips token_embd lookup — the caller must have already summed
    // text_embedding + codec_embedding.
    bool forward_embeddings(const float* embd, int32_t n_tokens, int32_t n_pos,
                            float* hidden, std::string* error = nullptr);

    // ── Text embedding (official Qwen3-TTS) ────────────────────────────

    // Compute text embeddings: tokenize → text_embedding lookup →
    // text_projection → [n_tokens, 1024].
    // Returns true if text_embd/text_proj tensors are available.
    bool has_text_embedding() const { return has_text_embd_; }

    bool compute_text_embedding(const std::string& text,
                                std::vector<float>* out_embd,
                                int32_t* out_n_tokens,
                                std::string* error = nullptr);

    // ── Utilities ──────────────────────────────────────────────────────

    // Tokenizer.
    bool tokenize(const std::string& text, bool add_special, bool parse_special,
                  std::vector<int32_t>* tokens, std::string* error = nullptr) const;

    bool is_eog(int32_t token) const;

    int32_t hidden_size()    const { return hidden_size_; }
    int32_t n_embd()         const { return hidden_size_; }
    int32_t vocab_size()     const { return vocab_size_; }

    const float* get_embeddings_ith(int32_t i) const;
    const float* get_logits_ith(int32_t i) const;

    llama_model*   raw_model()   const { return model_; }
    llama_context* raw_context() const { return ctx_;   }

    // Codec embedding table (token_embd from the model).
    const float* codec_embedding() const { return codec_embd_; }
    int32_t      codec_embd_dim()  const { return codec_embd_dim_; }
    int32_t      codec_vocab()     const { return codec_vocab_; }

    // Codec head (output.weight from the model).
    const float* codec_head() const { return codec_head_; }

private:
    TalkerRunner();
    bool load_extra_tensors(const std::string& gguf_path, std::string* error);

    llama_model*   model_ = nullptr;
    llama_context* ctx_   = nullptr;

    int32_t hidden_size_ = 0;
    int32_t vocab_size_  = 0;

    // Extra tensors loaded directly from GGUF
    bool    has_text_embd_ = false;
    float*  text_embd_         = nullptr;  // [text_vocab, 2048]
    int32_t text_vocab_        = 0;
    int32_t text_embd_dim_     = 0;         // 2048

    float*  text_proj_0_w_     = nullptr;  // [2048, 2048]
    float*  text_proj_0_b_     = nullptr;  // [2048]
    float*  text_proj_1_w_     = nullptr;  // [1024, 2048]
    float*  text_proj_1_b_     = nullptr;  // [1024]

    // codec_embedding (token_embd from the llama model, or loaded raw)
    float*  codec_embd_        = nullptr;
    int32_t codec_embd_dim_    = 0;
    int32_t codec_vocab_       = 0;

    // codec_head (output from the llama model, or loaded raw)
    float*  codec_head_        = nullptr;
};

}  // namespace audiocore::qwen3_tts

#endif  // AUDIOCORE_MODELS_QWEN3_TTS_TALKER_RUNNER_H
