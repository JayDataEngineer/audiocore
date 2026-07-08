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

    // Project pre-tokenized IDs through text_embd + text_proj MLP.
    // Skips tokenization — callers that already have token IDs
    // (e.g. qwen3_tts encode_for_tts) use this instead of compute_text_embedding.
    bool project_text_tokens(const int32_t* token_ids, int32_t n_tokens,
                             float* out_embd, std::string* error = nullptr);

    // Project a single token through text_embd + text_proj MLP.
    bool project_single_token(int32_t token_id, float* out_embd,
                              std::string* error = nullptr);

    // Codec embedding table ([codec_vocab, n_embd]) and codec head
    // ([codec_vocab, n_embd]) — both anchored on the runner. libllama owns
    // its own internal copy of token_embd/output; these pointers alias the
    // raw GGUF data we loaded as extras.
    const float* codec_embedding() const { return codec_embd_; }
    int32_t      codec_embd_dim()  const { return codec_embd_dim_; }
    int32_t      codec_vocab()     const { return codec_vocab_; }
    const float* codec_head()      const { return codec_head_; }

    // ── WDELTA: Base→CustomVoice weight patching ──────────────────────────
    //
    // After load_extras(ExtraKind::Talker) has materialized this runner's
    // text_proj_*_ and codec_embd_ buffers from a CustomVoice (CV) talker
    // GGUF, calling apply_wdelta_patch(base_gguf_path) OVERWRITES those
    // buffers with the corresponding tensors from a sibling Base talker GGUF.
    //
    // Why: CV and Base share 99.99% of transformer weights (cosine > 0.9999
    // on all attn/ffn matrices) but differ sharply on:
    //   - text_proj.{0,1}.bias      (CV trained on discrete speaker tokens;
    //                                Base trained on continuous ECAPA vectors)
    //   - token_embd.weight         (9 preset speaker tokens are near-zero in
    //                                Base, identity-bearing rows in CV)
    // After the patch, the CV talker accepts a continuous ECAPA embedding at
    // the speaker slot (Base's embedding pathway) WHILE RETAINING its own
    // transformer norms (which carry the instruct-tuning that makes emotion
    // control strong rather than "soft"). The 4-feature pipeline unlocked:
    //   (a) speaker embedding  +  (b) instruct emotion
    //   (c) text prompt        +  (d) high quality
    // — the gap that the Mastrapasqua `qwen3-tts` engine closes via its
    // `.qvoice WOVR` section, ported here for our GGUF stack. See
    // docs/QWEN3-TTS-GAPS.md §A4 and blog/cross-model-voice-analysis.md.
    //
    // Returns true on success. *error receives a message on failure. Safe
    // to call multiple times — later calls re-overwrite with whatever the
    // base GGUF contains.
    bool apply_wdelta_patch(const std::string& base_gguf_path,
                            std::string* error = nullptr);

    // ── Predictor extras (load_extras(ExtraKind::Predictor)) ──────────────

    bool has_mtp() const { return has_mtp_; }
    int32_t n_fine_books() const { return n_fine_books_; }

    // Run one MTP step: condition on the talker's last hidden state and
    // layer0_embed (the talker's codec_embedding[code_0]) to autoregressively
    // emit n_fine_books fine codebook IDs. prev_codes[0] must be code_0.
    // n_fine_books must match the value passed to load_extras.
    //
    // Sampling params (temperature, top_p, top_k) default to greedy (temp=0)
    // which matches Python's reference behavior when do_sample=False. Without
    // greedy, the predictor emits random fine codes that destabilize the
    // talker's AR feedback loop and collapse it into repetition.
    bool predict_one_step(const float* talker_hidden,
                          const float* layer0_embed,
                          int32_t code_0,
                          int32_t n_fine_books,
                          int32_t* out_codes,
                          std::string* error = nullptr,
                          float temperature = 0.0f,
                          float top_p = 1.0f,
                          int32_t top_k = 1);

    // Per-codebook embedding table for fine codebook i ∈ [0, n_fine_books).
    const float* fine_embedding(int i) const {
        return (i >= 0 && i < n_fine_books_) ? fine_embd_[i] : nullptr;
    }
    int32_t fine_vocab() const { return fine_vocab_; }
    int32_t fine_embd_dim() const { return fine_embd_dim_; }

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

    // Load a vocab-only tokenizer GGUF (produced by convert_qwen3tts
    // write_tokenizer). The sidecar carries the real Qwen3 BPE tokenizer
    // (151 936 tokens, 151 291 merges, pre=qwen2). tokenize() will use it
    // automatically once loaded. Safe to call multiple times.
    bool load_tokenizer(const std::string& gguf_path,
                        std::string* error = nullptr);
    bool has_tokenizer() const { return tokenizer_model_ != nullptr; }

    // Inverse: token id → byte piece (no null terminator appended).
    bool token_to_piece(int32_t token, std::string* out,
                        std::string* error = nullptr) const;

    // True if `token` is the model's end-of-generation marker.
    bool is_eog(int32_t token) const;

    // BOS / EOS token IDs from the loaded vocabulary.
    int32_t bos_token_id() const;
    int32_t eos_token_id() const;

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

    // Raw token-embedding table lookup — NO transformer forward pass.
    // Reads the `token_embd.weight` tensor from the loaded model and extracts
    // the embedding row for each token ID. Used by ACE-Step's lyric encoder,
    // which takes raw embeddings (not post-forward hidden states) as input
    // (matching upstream qwen3_embed_lookup in cond-enc.h).
    //   token_ids: (n_tokens,) int32
    //   output:    (n_tokens, hidden_size) row-major float32
    bool embed_lookup(const int32_t* token_ids, int32_t n_tokens,
                      float* output, std::string* error = nullptr);

    // Clear all KV cache entries. Must be called before starting a new
    // inference sequence when reusing the same Runner across requests.
    // Returns false on failure.
    bool clear_kv_cache(std::string* error = nullptr);

    // ── Secondary context (for CFG / classifier-free guidance) ────────────
    // ACE-Step's LM uses classifier-free guidance (cfg_scale=2.0 by default)
    // which requires running BOTH a conditional and an unconditional forward
    // pass per step. The secondary context shares the model weights (no
    // extra GPU memory for weights) but has its own KV cache.
    // Returns false if the secondary context could not be created.
    bool init_secondary_context(std::string* error = nullptr);
    bool has_secondary_context() const { return ctx_secondary_ != nullptr; }

    // Forward on the secondary context. Same contract as forward_tokens.
    bool forward_tokens_secondary(const int32_t* tokens, int32_t n_tokens,
                                  int32_t n_pos, float* logits,
                                  std::string* error = nullptr);
    bool clear_secondary_kv_cache(std::string* error = nullptr);

    llama_model*   raw_model()    const { return model_; }
    llama_context* raw_context()  const { return ctx_;   }

private:
    Runner();
    bool load_talker_extras(const std::string& gguf_path, std::string* error);
    bool load_predictor_extras(const std::string& gguf_path,
                                int n_fine_books, std::string* error);

    llama_model*   model_ = nullptr;
    llama_context* ctx_   = nullptr;
    llama_context* ctx_secondary_ = nullptr;  // for CFG (shares model_)
    int32_t        hidden_size_ = 0;
    int32_t        vocab_size_  = 0;
    int32_t        n_codebooks_ = 32;       // Qwen3-TTS default
    uint32_t       n_pos_per_embd_ = 1;     // M-RoPE: 4, normal RoPE: 1

    // Tokenizer sidecar (vocab-only llama_model loaded from the text BPE GGUF).
    // When non-null, tokenize() uses this instead of model_'s built-in vocab.
    llama_model*   tokenizer_model_ = nullptr;

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
    int32_t  fine_embd_dim_ = 0;
    int32_t  n_fine_books_  = 0;
    float*   small_to_mtp_w_ = nullptr;     // [small_to_mtp_in_, small_to_mtp_out_]
    float*   small_to_mtp_b_ = nullptr;     // [small_to_mtp_out_]
    // Input/output dims of small_to_mtp. For 0.6B talker+predictor are both
    // 1024, so in_==out_==1024. For 1.7B the talker is 2048-d and the
    // predictor is 1024-d, so in_=2048 and out_=1024 — the projection
    // shrinks the talker hidden down to predictor hidden. The projection
    // loop must iterate `in_` over the talker hidden and use `in_` as the
    // row stride; using `hidden_size_` (predictor dim) for both is the
    // 1.7B bug that produced garbage fine codebooks.
    int32_t  small_to_mtp_in_  = 0;         // talker n_embd (input dim)
    int32_t  small_to_mtp_out_ = 0;         // predictor n_embd (output dim)
    // Scratch for predict_one_step; sized on load_extras.
    std::vector<float> scratch_embd_;
    std::vector<float> scratch_logits_;
};

}  // namespace audiocore::qwen3

#endif  // AUDIOCORE_MODELS_QWEN3_RUNNER_H
