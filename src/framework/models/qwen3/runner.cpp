// runner.cpp — libllama-backed Qwen3 transformer execution.
//
// One Qwen3 inference path for the whole project. Loaded by the MOSS family
// (8B backbone), the ACE-Step family (1.7B 5Hz LM, 0.6B text encoder), and
// any future family that needs a Qwen3 transformer. There is no other
// Qwen3 implementation in audiocore.

#include "audiocore/models/qwen3/runner.h"

#include "llama.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace audiocore::qwen3 {

Runner::Runner() = default;

Runner::~Runner() {
    if (ctx_)   llama_free(ctx_);
    if (model_) llama_model_free(model_);
}

std::unique_ptr<Runner> Runner::load(const std::string& gguf_path,
                                     const RunnerConfig& cfg,
                                     std::string* error) {
    auto self = std::unique_ptr<Runner>(new Runner());

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = cfg.n_gpu_layers;
    mp.main_gpu     = cfg.main_gpu;
    mp.use_mmap     = true;
    mp.use_mlock    = false;

    self->model_ = llama_model_load_from_file(gguf_path.c_str(), mp);
    if (!self->model_) {
        if (error) *error = "llama_model_load_from_file failed: " + gguf_path;
        return nullptr;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = cfg.n_ctx;
    cp.n_batch         = cfg.n_batch;
    cp.n_threads       = cfg.n_threads;
    cp.n_threads_batch = cfg.n_threads;
    cp.flash_attn_type = cfg.flash_attn
        ? LLAMA_FLASH_ATTN_TYPE_ENABLED
        : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.embeddings = true;   // we always need embedding access
    cp.no_perf    = true;

    self->ctx_ = llama_init_from_model(self->model_, cp);
    if (!self->ctx_) {
        if (error) *error = "llama_init_from_model failed";
        llama_model_free(self->model_);
        self->model_ = nullptr;
        return nullptr;
    }

    self->hidden_size_ = llama_model_n_embd(self->model_);
    const llama_vocab* vocab = llama_model_get_vocab(self->model_);
    self->vocab_size_  = vocab ? llama_vocab_n_tokens(vocab) : 0;
    return self;
}

// Build a llama_batch ready for llama_decode with embd OR token input,
// positions starting at n_pos, and EVERY position marked for output. We use
// llama_batch_init (not _get_one) so the logits/output flags are properly
// allocated; this is required to read embeddings/logits for all rows.
//
// `is_embd` switches between embd (float*) and token (llama_token*) input.
static llama_batch make_batch(int32_t n_tokens, int32_t n_pos, bool is_embd,
                              const float* embd, const llama_token* tokens) {
    llama_batch b = llama_batch_init(n_tokens,
                                     /*embd=*/ is_embd ? 1 : 0,
                                     /*n_seq_max=*/ 1);
    b.n_tokens = n_tokens;  // llama_batch_init allocates storage but leaves n_tokens=0
    if (is_embd) {
        b.token = nullptr;
        b.embd  = const_cast<float*>(embd);
    } else {
        b.token = const_cast<llama_token*>(tokens);
        b.embd  = nullptr;
    }
    // Positions: contiguous from n_pos.
    for (int32_t i = 0; i < n_tokens; i++) b.pos[i] = n_pos + i;
    // Mark every position for output so we can read per-row logits/embd.
    for (int32_t i = 0; i < n_tokens; i++) b.logits[i] = 1;
    // Default each token to sequence 0.
    for (int32_t i = 0; i < n_tokens; i++) {
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
    }
    return b;
}

bool Runner::forward_embeddings(const float* embd, int32_t n_tokens, int32_t n_pos,
                                float* hidden, std::string* error) {
    llama_batch b = make_batch(n_tokens, n_pos, /*is_embd=*/true, embd, nullptr);
    const int32_t n_embd = hidden_size_;
    bool ok = (llama_decode(ctx_, b) == 0);
    if (!ok) {
        if (error) *error = "llama_decode failed";
    } else {
        for (int32_t i = 0; i < n_tokens && ok; i++) {
            const float* row = llama_get_embeddings_ith(ctx_, i);
            if (!row) {
                if (error) *error = "llama_get_embeddings_ith failed";
                ok = false;
                break;
            }
            std::memcpy(hidden + static_cast<size_t>(i) * n_embd, row,
                        static_cast<size_t>(n_embd) * sizeof(float));
        }
    }
    // make_batch overrode b.embd (and/or b.token) to point at OUR data.
    // Null them so llama_batch_free doesn't free caller-owned memory.
    b.token = nullptr;
    b.embd  = nullptr;
    llama_batch_free(b);
    return ok;
}

bool Runner::forward_tokens(const int32_t* tokens, int32_t n_tokens, int32_t n_pos,
                            float* logits, std::string* error) {
    llama_batch b = make_batch(n_tokens, n_pos, /*is_embd=*/false,
                               nullptr, reinterpret_cast<const llama_token*>(tokens));
    const int32_t n_vocab = vocab_size_;
    bool ok = (llama_decode(ctx_, b) == 0);
    if (!ok) {
        if (error) *error = "llama_decode failed";
    } else {
        for (int32_t i = 0; i < n_tokens && ok; i++) {
            const float* row = llama_get_logits_ith(ctx_, i);
            if (!row) {
                if (error) *error = "llama_get_logits_ith failed";
                ok = false;
                break;
            }
            std::memcpy(logits + static_cast<size_t>(i) * n_vocab, row,
                        static_cast<size_t>(n_vocab) * sizeof(float));
        }
    }
    // make_batch overrode b.token; null it so batch_free doesn't free caller memory.
    b.token = nullptr;
    b.embd  = nullptr;
    llama_batch_free(b);
    return ok;
}

bool Runner::forward_get_embeddings(const int32_t* tokens, int32_t n_tokens,
                                     int32_t n_pos, float* hidden,
                                     std::string* error) {
    llama_batch b = make_batch(n_tokens, n_pos, /*is_embd=*/false,
                               nullptr, reinterpret_cast<const llama_token*>(tokens));
    const int32_t n_embd = hidden_size_;
    bool ok = (llama_decode(ctx_, b) == 0);
    if (!ok) {
        if (error) *error = "llama_decode failed (forward_get_embeddings)";
    } else {
        for (int32_t i = 0; i < n_tokens && ok; i++) {
            const float* row = llama_get_embeddings_ith(ctx_, i);
            if (!row) {
                if (error) *error = "forward_get_embeddings: llama_get_embeddings_ith failed";
                ok = false;
                break;
            }
            std::memcpy(hidden + static_cast<size_t>(i) * n_embd, row,
                        static_cast<size_t>(n_embd) * sizeof(float));
        }
    }
    b.token = nullptr;
    b.embd  = nullptr;
    llama_batch_free(b);
    return ok;
}

int32_t Runner::hidden_size() const { return hidden_size_; }
int32_t Runner::vocab_size()  const { return vocab_size_;  }

const float* Runner::get_embeddings_ith(int32_t i) const {
    return llama_get_embeddings_ith(ctx_, i);
}

const float* Runner::get_logits_ith(int32_t i) const {
    return llama_get_logits_ith(ctx_, i);
}

bool Runner::tokenize(const std::string& text, bool add_special,
                      bool parse_special, std::vector<int32_t>* tokens,
                      int32_t* needed, std::string* error) const {
    const llama_vocab* vocab = model_ ? llama_model_get_vocab(model_) : nullptr;
    if (!vocab) {
        if (error) *error = "model has no vocab";
        return false;
    }
    // Two-pass: query size, then fill. llama_tokenize returns the required
    // length (negative) when the buffer is too small — same convention as
    // snprintf.
    const int32_t hint = llama_tokenize(vocab, text.c_str(),
                                        static_cast<int32_t>(text.size()),
                                        nullptr, 0,
                                        add_special, parse_special);
    if (hint < 0) {
        // |hint| is the required token count.
        tokens->resize(static_cast<size_t>(-hint));
        const int32_t n = llama_tokenize(vocab, text.c_str(),
                                         static_cast<int32_t>(text.size()),
                                         tokens->data(),
                                         static_cast<int32_t>(tokens->size()),
                                         add_special, parse_special);
        if (n < 0) {
            if (error) *error = "tokenize retry failed unexpectedly";
            return false;
        }
        tokens->resize(static_cast<size_t>(n));
        if (needed) *needed = n;
        return true;
    }
    // hint == 0 only on empty input; nothing to fill.
    tokens->clear();
    if (needed) *needed = 0;
    return true;
}

bool Runner::token_to_piece(int32_t token, std::string* out,
                            std::string* error) const {
    const llama_vocab* vocab = model_ ? llama_model_get_vocab(model_) : nullptr;
    if (!vocab) {
        if (error) *error = "model has no vocab";
        return false;
    }
    char buf[128];
    int32_t n = llama_token_to_piece(vocab, static_cast<llama_token>(token),
                                     buf, sizeof(buf), /*lstrip=*/0,
                                     /*special=*/true);
    if (n < 0) {
        std::vector<char> big(static_cast<size_t>(-n) + 1);
        n = llama_token_to_piece(vocab, static_cast<llama_token>(token),
                                 big.data(), static_cast<int32_t>(big.size()),
                                 0, true);
        if (n < 0) {
            if (error) *error = "token_to_piece retry failed";
            return false;
        }
        out->assign(big.data(), static_cast<size_t>(n));
        return true;
    }
    out->assign(buf, static_cast<size_t>(n));
    return true;
}

bool Runner::is_eog(int32_t token) const {
    const llama_vocab* vocab = model_ ? llama_model_get_vocab(model_) : nullptr;
    if (!vocab) return false;
    return llama_vocab_is_eog(vocab, static_cast<llama_token>(token));
}

bool Runner::apply_chat_template(
        const std::vector<std::pair<std::string, std::string>>& messages,
        bool add_assistant_prompt,
        std::string* out,
        std::string* error) const {
    // libllama's chat-apply takes (a) a Jinja template string — fetched from
    // the model's GGUF metadata via llama_model_chat_template — and (b) a C
    // array of {role, content} structs. No JSON. The template name "qwen2"
    // etc. is detected inside libllama; passing the model's own baked-in
    // template string covers Qwen3 (its metadata carries a chatml.jinja).
    if (!model_) {
        if (error) *error = "no model loaded";
        return false;
    }
    const char* tmpl = llama_model_chat_template(model_, /*name=*/nullptr);

    // Fallback: hardcoded ChatML when the GGUF has no baked-in template.
    // Common for community GGUFs that only supply the weight tensors.
    if (!tmpl || !*tmpl) {
        std::string result;
        for (const auto& [role, content] : messages) {
            result += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
        }
        if (add_assistant_prompt) {
            result += "<|im_start|>assistant\n";
        }
        *out = result;
        return true;
    }

    // Keep the underlying std::strings alive across the call — the C structs
    // hold const char* pointers into them.
    std::vector<std::string>        role_storage;
    std::vector<std::string>        content_storage;
    std::vector<llama_chat_message> chat;
    role_storage.reserve(messages.size());
    content_storage.reserve(messages.size());
    chat.reserve(messages.size());
    for (const auto& [role, content] : messages) {
        role_storage.push_back(role);
        content_storage.push_back(content);
        chat.push_back({role_storage.back().c_str(),
                        content_storage.back().c_str()});
    }

    // Two-pass: query required length (buf=nullptr, length=0), then fill.
    // Returns -1 on unknown template, otherwise the formatted byte count.
    const int32_t hint = llama_chat_apply_template(
        tmpl, chat.data(), chat.size(), add_assistant_prompt,
        nullptr, 0);
    if (hint < 0) {
        if (error) *error = "llama_chat_apply_template: unknown template "
                            "(model metadata may not be chatml-compatible)";
        return false;
    }
    out->resize(static_cast<size_t>(hint));
    const int32_t n = llama_chat_apply_template(
        tmpl, chat.data(), chat.size(), add_assistant_prompt,
        out->data(), static_cast<int32_t>(out->size()));
    if (n < 0) {
        if (error) *error = "llama_chat_apply_template fill call failed";
        return false;
    }
    out->resize(static_cast<size_t>(n));
    return true;
}

}  // namespace audiocore::qwen3
