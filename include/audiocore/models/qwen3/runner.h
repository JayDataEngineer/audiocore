// runner.h — libllama-backed Qwen3 transformer execution.
//
// The ONE Qwen3 inference engine in audiocore. Both MOSS (8B backbone) and
// ACE-Step (1.7B 5Hz LM, 0.6B text encoder) use this. Their GGUFs are all
// in llama.cpp's standard Qwen3 layout, so libllama loads them natively —
// no per-family transformer code in audiocore.
//
// ACE-Step ships GGUFs with HF-style tensor names (model.embed_tokens.weight
// etc.) that libllama refuses. Our converter (tools/convert_acestep_gguf.py)
// rewrites them to llama.cpp names (token_embd.weight etc.) once at download
// time so the runner can load them like any other Qwen3.
//
// This is the only place in audiocore that calls llama_* APIs.

#ifndef AUDIOCORE_MODELS_QWEN3_RUNNER_H
#define AUDIOCORE_MODELS_QWEN3_RUNNER_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct llama_model;
struct llama_context;

namespace audiocore::qwen3 {

struct RunnerConfig {
    int  n_ctx        = 8192;   // max sequence length
    int  n_batch      = 512;
    int  n_threads    = 0;       // 0 → libllama default
    int  n_gpu_layers = -1;      // -1 → all on GPU
    int  main_gpu     = -1;      // -1 → auto-pick GPU with most free VRAM
    bool flash_attn   = true;
};

// One Qwen3 model + its inference context. NOT thread-safe; the Session
// that owns it serializes requests.
class Runner {
public:
    static std::unique_ptr<Runner> load(const std::string& gguf_path,
                                        const RunnerConfig& cfg,
                                        std::string* error = nullptr);
    ~Runner();

    Runner(const Runner&)            = delete;
    Runner& operator=(const Runner&) = delete;

    // Embedding-shape input → hidden-state output, single forward pass.
    // Used by MOSS to feed summed text+audio embeddings into the backbone.
    //   embd:    (n_tokens, hidden_size) row-major float32
    //   n_pos:   position offset for RoPE
    //   hidden:  out (n_tokens, hidden_size) row-major float32
    bool forward_embeddings(const float* embd, int32_t n_tokens, int32_t n_pos,
                            float* hidden, std::string* error = nullptr);

    // Token-id input → logits. Used by ACE-Step's 5Hz LM for next-code
    // prediction.
    //   tokens:  (n_tokens,) int32
    //   n_pos:   position offset
    //   logits:  out (n_tokens, vocab_size) row-major float32
    bool forward_tokens(const int32_t* tokens, int32_t n_tokens, int32_t n_pos,
                        float* logits, std::string* error = nullptr);

    // Token-id input → hidden state embeddings. Used by ACE-Step's text encoder
    // (TE / Qwen3-Embedding) to produce per-token hidden states that condition
    // the DiT cross-attention.
    //   tokens:   (n_tokens,) int32
    //   n_pos:    position offset
    //   hidden:   out (n_tokens, hidden_size) row-major float32
    bool forward_get_embeddings(const int32_t* tokens, int32_t n_tokens,
                                int32_t n_pos, float* hidden,
                                std::string* error = nullptr);

    // ---- Tokenizer (libllama-native, no vendored SentencePiece) ---------
    // These delegate to libllama's Qwen3 tokenizer. add_special inserts
    // BOS/EOS per the model config; parse_special decodes <|im_start|>-style
    // control tokens instead of treating them as plaintext.
    //
    // Returns false only on hard failure (empty vocab). A too-small output
    // buffer is reported via the size hint: callers should retry with a
    // larger buffer when `*needed` > tokens.size().
    bool tokenize(const std::string& text, bool add_special, bool parse_special,
                  std::vector<int32_t>* tokens, int32_t* needed = nullptr,
                  std::string* error = nullptr) const;

    // Inverse: token id → byte piece (no null terminator appended).
    bool token_to_piece(int32_t token, std::string* out,
                        std::string* error = nullptr) const;

    // True if `token` is the model's end-of-generation marker.
    bool is_eog(int32_t token) const;

    // Apply the model's chat template to a message list, returning the
    // templated string ready for tokenize(). Each message is {role, content}.
    // libllama supports the Qwen3 chat template natively; we don't need to
    // ship our own jinja interpreter.
    bool apply_chat_template(const std::vector<std::pair<std::string, std::string>>& messages,
                             bool add_assistant_prompt,
                             std::string* out,
                             std::string* error = nullptr) const;

    int32_t hidden_size() const;
    int32_t vocab_size()  const;

    // Low-level access to decoder results. Pointers valid until next decode.
    const float* get_embeddings_ith(int32_t i) const;
    const float* get_logits_ith(int32_t i) const;

    llama_model*   raw_model()    const { return model_; }
    llama_context* raw_context()  const { return ctx_;   }

private:
    Runner();
    llama_model*   model_ = nullptr;
    llama_context* ctx_   = nullptr;
    int32_t        hidden_size_ = 0;
    int32_t        vocab_size_  = 0;
};

}  // namespace audiocore::qwen3

#endif  // AUDIOCORE_MODELS_QWEN3_RUNNER_H
