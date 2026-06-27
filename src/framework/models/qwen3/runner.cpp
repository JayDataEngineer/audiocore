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
    llama_batch_free(b);
    return ok;
}

int32_t Runner::hidden_size() const { return hidden_size_; }
int32_t Runner::vocab_size()  const { return vocab_size_;  }

}  // namespace audiocore::qwen3
