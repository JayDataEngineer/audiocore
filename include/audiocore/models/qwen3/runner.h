// runner.h — libllama-backed Qwen3 transformer execution.
//
// The ONE Qwen3 inference engine in audiocore. Every family that needs a
// Qwen3-style transformer speaks to this class:
//
//   • MOSS-TTS — 8B backbone
//   • ACE-Step — 1.7B 5Hz LM + 0.6B Qwen3-Embedding text encoder
//   • Qwen3-TTS — Talker (qwen3tts) and Code Predictor (qwen3tts_cp)
//
// All these GGUFs are in llama.cpp's standard Qwen3 layout, so libllama
// loads them natively. The Qwen3-TTS "talker" and "predictor" need a few
// extra tensors (text-embedding MLP, codec embedding/head, MTP tables) that
// libllama doesn't surface; those are loaded from the same GGUF by the
// runner's optional extras path (see load_extras / ExtraKind).
//
// ACE-Step ships GGUFs with HF-style tensor names (model.embed_tokens.weight
// etc.) that libllama refuses. Our converter (tools/convert_acestep.cpp)
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
//
// A Runner serves either as a "stock" Qwen3 backbone (MOSS, ACE-Step LM/TE)
// or as a Qwen3-TTS component (talker or predictor) by calling load_extras
// with the matching ExtraKind. The extras are optional; a stock runner has
// has_text_embedding() == false and has_mtp() == false.
class Runner {
public:
    static std::unique_ptr<Runner> load(const std::string& gguf_path,
                                        const RunnerConfig& cfg,
                                        std::string* error = nullptr);
    ~Runner();

    Runner(const Runner&)            = delete;
    Runner& operator=(const Runner&) = delete;

    // Which set of family-specific extra tensors to pull out of the GGUF
    // after the libllama model is loaded. None = stock Qwen3 backbone.
    enum class ExtraKind {
        None,        // MOSS / ACE-Step LM / ACE-Step TE
        Talker,      // Qwen3-TTS talker: text_embd + text_proj + codec_embd/head
        Predictor,   // Qwen3-TTS code predictor: codec_embd.{i} + lm_head.{i}
                     // + small_to_mtp (MTP mode)
    };

    // Optional second stage of loading. Reads the extra tensors named in
    // docs/GGUF_FORMAT.md → "Qwen3-TTS" directly from the GGUF so the runner
    // can serve as a talker or predictor. Missing tensors are non-fatal —
    // the corresponding has_*() accessor returns false and the family falls
    // back to the Lunavox-style plain-token path.
    //
    // n_fine_books is only consulted for Predictor; it sizes the MTP tables.
    bool load_extras(const std::string& gguf_path, ExtraKind kind,
                     int n_fine_books = 31,
                     std::string* error = nullptr);

    // ── Standard forward paths (every ExtraKind supports these) ───────────

    // Embedding-shape input → hidden-state output, single forward pass.
    // Used by MOSS to feed summed text+audio embeddings into the backbone.
    //   embd:    (n_tokens, hidden_size) row-major float32
    //   n_pos:   position offset for RoPE
    //   hidden:  out (n_tokens, hidden_size) row-major float32
    bool forward_embeddings(const float* embd, int32_t n_tokens, int32_t n_pos,
                            float* hidden, std::string* error = nullptr);

    // Token-id input → logits. Used by ACE-Step's 5Hz LM and Qwen3-TTS's
    // talker/predictor for next-token/code prediction.
    //   tokens:  (n_tokens,) int32
    //   n_pos:   position offset
    //   logits:  out (n_tokens, vocab_size) row-major float32
    bool forward_tokens(const int32_t* tokens, int32_t n_tokens, int32_t n_pos,
                        float* logits, std::string* error = nullptr);

    // Token-id input → hidden state. Used by ACE-Step's text encoder
    // (TE / Qwen3-Embedding) and Qwen3-TTS's talker (Lunavox fallback path).
    //   tokens:   (n_tokens,) int32
    //   n_pos:    position offset
    //   hidden:   out (n_tokens, hidden_size) row-major float32
    bool forward_get_embeddings(const int32_t* tokens, int32_t n_tokens,
                                int32_t n_pos, float* hidden,
                                std::string* error = nullptr);
    // Alias kept so Qwen3-TTS family code reads naturally.
    bool forward_get_hidden(const int32_t* tokens, int32_t n_tokens,
                            int32_t n_pos, float* hidden,
                            std::string* error = nullptr) {
        return forward_get_embeddings(tokens, n_tokens, n_pos, hidden, error);
    }

    // ── Talker extras (load_extras(ExtraKind::Talker)) ────────────────────

    bool has_text_embedding() const { return has_text_embd_; }

    // tokenize → text_embedding lookup → text_projection MLP → [n_tokens, H].
    // Returns false if has_text_embedding() is false.
    bool compute_text_embedding(const std::string& text,
                                std::vector<float>* out_embd,
                                int32_t* out_n_tokens,
                                std::string* error = nullptr);

    // Codec embedding table ([codec_vocab, n_embd]) and codec head
    // ([codec_vocab, n_embd]) — both anchored on the runner. libllama owns
    // its own internal copy of token_embd/output; these pointers alias the
    // raw GGUF data we loaded as extras.
    const float* codec_embedding() const { return codec_embd_; }
    int32_t      codec_embd_dim()  const { return codec_embd_dim_; }
    int32_t      codec_vocab()     const { return codec_vocab_; }
    const float* codec_head()      const { return codec_head_; }

    // ── Predictor extras (load_extras(ExtraKind::Predictor)) ──────────────

    bool has_mtp() const { return has_mtp_; }

    // Run one MTP step: condition on the talker's last hidden state and the
    // previously-sampled codes, autoregressively emit n_fine_books fine
    // codebook IDs. n_fine_books must match the value passed to load_extras.
    bool predict_one_step(const float* talker_hidden,
                          const int32_t* prev_codes,
                          int32_t n_fine_books,
                          int32_t* out_codes,
                          std::string* error = nullptr);

    // Per-codebook embedding table for fine codebook i ∈ [0, n_fine_books).
    const float* fine_embedding(int i) const {
        return (i >= 0 && i < n_fine_books_) ? fine_embd_[i] : nullptr;
    }
    int32_t fine_vocab() const { return fine_vocab_; }

    // ── Tokenizer (libllama-native, no vendored SentencePiece) ---------
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
    int32_t n_embd()      const { return hidden_size(); }   // alias
    int32_t vocab_size()  const;
    int     n_codebooks() const { return n_codebooks_; }

    // Low-level access to decoder results. Pointers valid until next decode.
    const float* get_embeddings_ith(int32_t i) const;
    const float* get_logits_ith(int32_t i) const;

    llama_model*   raw_model()    const { return model_; }
    llama_context* raw_context()  const { return ctx_;   }

private:
    Runner();
    bool load_talker_extras(const std::string& gguf_path, std::string* error);
    bool load_predictor_extras(const std::string& gguf_path,
                                int n_fine_books, std::string* error);

    llama_model*   model_ = nullptr;
    llama_context* ctx_   = nullptr;
    int32_t        hidden_size_ = 0;
    int32_t        vocab_size_  = 0;
    int32_t        n_codebooks_ = 32;       // Qwen3-TTS default

    // ── Talker extras (owned; raw GGUF bytes copied into these buffers) ──
    bool    has_text_embd_ = false;
    float*  text_embd_         = nullptr;   // [text_vocab, 2048]
    int32_t text_vocab_        = 0;
    int32_t text_embd_dim_     = 0;          // 2048
    float*  text_proj_0_w_     = nullptr;   // [2048, 2048]
    float*  text_proj_0_b_     = nullptr;   // [2048]
    float*  text_proj_1_w_     = nullptr;   // [1024, 2048]
    float*  text_proj_1_b_     = nullptr;   // [1024]
    float*  codec_embd_        = nullptr;   // [codec_vocab, n_embd]
    int32_t codec_embd_dim_    = 0;
    int32_t codec_vocab_       = 0;
    float*  codec_head_        = nullptr;   // [codec_vocab, n_embd]

    // ── Predictor extras (MTP mode) ──────────────────────────────────────
    bool     has_mtp_       = false;
    float**  fine_embd_     = nullptr;      // [n_fine_books_] each [fine_vocab, n_embd]
    float**  fine_head_     = nullptr;      // [n_fine_books_] each [n_embd, fine_vocab]
    int32_t  fine_vocab_    = 0;
    int32_t  n_fine_books_  = 0;
    float*   small_to_mtp_w_ = nullptr;     // [n_embd, n_embd]
    float*   small_to_mtp_b_ = nullptr;     // [n_embd]
    // Scratch for predict_one_step; sized on load_extras.
    std::vector<float> scratch_embd_;
    std::vector<float> scratch_logits_;
};

}  // namespace audiocore::qwen3

#endif  // AUDIOCORE_MODELS_QWEN3_RUNNER_H
