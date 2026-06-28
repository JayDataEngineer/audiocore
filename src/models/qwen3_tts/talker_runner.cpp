// talker_runner.cpp — Talker (qwen3tts) inference via llama.cpp.

#include "audiocore/models/qwen3_tts/talker_runner.h"

#include "llama.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace audiocore::qwen3_tts {

TalkerRunner::TalkerRunner() = default;

TalkerRunner::~TalkerRunner() {
    delete[] text_embd_;
    delete[] text_proj_0_w_;
    delete[] text_proj_0_b_;
    delete[] text_proj_1_w_;
    delete[] text_proj_1_b_;
    delete[] codec_embd_;
    delete[] codec_head_;
    if (ctx_)   llama_free(ctx_);
    if (model_) llama_model_free(model_);
}

std::unique_ptr<TalkerRunner> TalkerRunner::load(const std::string& gguf_path,
                                                  const TalkerConfig& cfg,
                                                  std::string* error) {
    auto self = std::unique_ptr<TalkerRunner>(new TalkerRunner());

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
    cp.embeddings = true;
    cp.no_perf    = true;

    self->ctx_ = llama_init_from_model(self->model_, cp);
    if (!self->ctx_) {
        if (error) *error = "llama_init_from_model failed";
        llama_model_free(self->model_);
        self->model_ = nullptr;
        return nullptr;
    }

    self->hidden_size_ = (int32_t)llama_model_n_embd(self->model_);
    const llama_vocab* vocab = llama_model_get_vocab(self->model_);
    self->vocab_size_  = vocab ? (int32_t)llama_vocab_n_tokens(vocab) : 0;

    // Load extra Qwen3-TTS tensors (text_embd, text_proj, codec_embd, codec_head)
    if (!self->load_extra_tensors(gguf_path, error)) {
        // Non-fatal: these are optional for Lunavox GGUFs
        std::fprintf(stderr, "talker: extra tensor loading (optional): %s\n",
                     error ? error->c_str() : "unknown");
        if (error) error->clear();
    }

    return self;
}

// ── Raw GGUF tensor reading ──────────────────────────────────────────────

struct TensorRef {
    int64_t  tid;       // tensor index in GGUF
    size_t   size;      // tensor data size in bytes
    size_t   offset;    // byte offset in file
    ggml_type type;     // element type
};

// Find a tensor by name in an open GGUF context. Returns {-1,0,0,GGML_TYPE_F32} if not found.
static TensorRef find_tensor_in_gguf(gguf_context* ctx, const char* name) {
    int64_t n = gguf_get_n_tensors(ctx);
    for (int64_t i = 0; i < n; i++) {
        if (strcmp(gguf_get_tensor_name(ctx, i), name) == 0) {
            TensorRef tr;
            tr.tid    = i;
            tr.size   = gguf_get_tensor_size(ctx, i);
            tr.offset = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, i);
            tr.type   = gguf_get_tensor_type(ctx, i);
            return tr;
        }
    }
    TensorRef tr = {-1, 0, 0, GGML_TYPE_F32};
    return tr;
}

// Load a single F32 tensor from a GGUF file by name.
// Returns nullptr if not found.  out_nfloats receives the number of floats read.
static float* load_gguf_tensor_f32(const char* path, const char* name,
                                   size_t* out_nfloats = nullptr) {
    gguf_init_params params = {true, nullptr};
    gguf_context* ctx = gguf_init_from_file(path, params);
    if (!ctx) return nullptr;

    TensorRef tr = find_tensor_in_gguf(ctx, name);
    float* result = nullptr;
    if (tr.tid >= 0 && tr.size > 0) {
        size_t n_floats = tr.size / sizeof(float);
        if (out_nfloats) *out_nfloats = n_floats;
        if (tr.type == GGML_TYPE_F32) {
            result = new float[n_floats];
            FILE* fp = fopen(path, "rb");
            if (fp) {
                fseek(fp, (long)tr.offset, SEEK_SET);
                if (fread(result, 1, tr.size, fp) != tr.size) {
                    delete[] result;
                    result = nullptr;
                }
                fclose(fp);
            } else {
                delete[] result;
                result = nullptr;
            }
        } else {
            // Non-F32 type — read raw and hope caller handles it
            result = new float[n_floats];
            FILE* fp = fopen(path, "rb");
            if (fp) {
                fseek(fp, (long)tr.offset, SEEK_SET);
                if (fread(result, 1, tr.size, fp) != tr.size) {
                    delete[] result;
                    result = nullptr;
                }
                fclose(fp);
            } else {
                delete[] result;
                result = nullptr;
            }
        }
    }

    gguf_free(ctx);
    return result;
}

bool TalkerRunner::load_extra_tensors(const std::string& gguf_path, std::string* error) {
    const char* path = gguf_path.c_str();

    // text_embd.weight [text_vocab, 2048]  — stored as F32
    // In GGUF format: innermost dim is ne[0]. For a 2D [text_vocab, 2048] tensor,
    // ne[0]=2048, ne[1]=text_vocab. So ne[1] gives text_vocab.
    // We read the raw data, determine count from size.
    size_t nf = 0;
    float* te = load_gguf_tensor_f32(path, "text_embd.weight", &nf);
    if (te) {
        text_embd_ = te;
        // shape: [text_vocab, 2048] → nf = text_vocab * 2048
        // text_embd_dim_ = 2048, text_vocab_ = nf / 2048
        text_embd_dim_ = 2048;
        text_vocab_ = (int32_t)(nf / 2048);
    }

    text_proj_0_w_ = load_gguf_tensor_f32(path, "text_proj.0.weight");
    text_proj_0_b_ = load_gguf_tensor_f32(path, "text_proj.0.bias");
    text_proj_1_w_ = load_gguf_tensor_f32(path, "text_proj.1.weight");
    text_proj_1_b_ = load_gguf_tensor_f32(path, "text_proj.1.bias");

    // codec_embedding = token_embd.weight [codec_vocab, 1024]
    if (!codec_embd_) {
        size_t ce_nf = 0;
        float* ce = load_gguf_tensor_f32(path, "token_embd.weight", &ce_nf);
        if (ce) {
            codec_embd_ = ce;
            codec_embd_dim_ = 1024;
            codec_vocab_ = (int32_t)(ce_nf / 1024);
        }
    }

    // codec_head = output.weight [codec_vocab, 1024]
    if (!codec_head_) {
        float* ch = load_gguf_tensor_f32(path, "output.weight");
        if (ch) {
            codec_head_ = ch;
        }
    }

    has_text_embd_ = (text_embd_ != nullptr && text_proj_0_w_ != nullptr);

    return true;
}

// ── Batch builder ──────────────────────────────────────────────────────────

static llama_batch make_batch(int32_t n_tokens, int32_t n_pos,
                              const llama_token* tokens) {
    llama_batch b = llama_batch_init(n_tokens, /*embd=*/0, /*n_seq_max=*/1);
    b.n_tokens = n_tokens;
    b.token = const_cast<llama_token*>(tokens);
    b.embd  = nullptr;
    for (int32_t i = 0; i < n_tokens; i++) {
        b.pos[i] = n_pos + i;
        b.logits[i] = (i == n_tokens - 1) ? 1 : 0;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
    }
    return b;
}

static llama_batch make_embd_batch(const float* embd, int32_t n_tokens,
                                   int32_t n_pos) {
    llama_batch b = llama_batch_init(n_tokens, /*embd=*/n_tokens, /*n_seq_max=*/1);
    b.n_tokens = n_tokens;
    b.token = nullptr;
    b.embd  = const_cast<float*>(embd);
    for (int32_t i = 0; i < n_tokens; i++) {
        b.pos[i] = n_pos + i;
        b.logits[i] = (i == n_tokens - 1) ? 1 : 0;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
    }
    return b;
}

// ── Token-mode forward ─────────────────────────────────────────────────────

bool TalkerRunner::forward_tokens(const int32_t* tokens, int32_t n_tokens,
                                   int32_t n_pos, float* logits,
                                   std::string* error) {
    llama_batch b = make_batch(n_tokens, n_pos,
                               reinterpret_cast<const llama_token*>(tokens));
    const int32_t n_vocab = vocab_size_;
    bool ok = (llama_decode(ctx_, b) == 0);
    if (ok) {
        for (int32_t i = 0; i < n_tokens; i++) {
            const float* row = llama_get_logits_ith(ctx_, i);
            if (!row) { ok = false; break; }
            std::memcpy(logits + (size_t)i * n_vocab, row,
                        (size_t)n_vocab * sizeof(float));
        }
    } else {
        if (error) *error = "talker: llama_decode failed";
    }
    b.token = nullptr;
    llama_batch_free(b);
    return ok;
}

bool TalkerRunner::forward_get_hidden(const int32_t* tokens, int32_t n_tokens,
                                       int32_t n_pos, float* hidden,
                                       std::string* error) {
    llama_batch b = make_batch(n_tokens, n_pos,
                               reinterpret_cast<const llama_token*>(tokens));
    const int32_t n_embd = hidden_size_;
    bool ok = (llama_decode(ctx_, b) == 0);
    if (ok) {
        for (int32_t i = 0; i < n_tokens; i++) {
            const float* row = llama_get_embeddings_ith(ctx_, i);
            if (!row) { ok = false; break; }
            std::memcpy(hidden + (size_t)i * n_embd, row,
                        (size_t)n_embd * sizeof(float));
        }
    } else {
        if (error) *error = "talker: llama_decode (hidden) failed";
    }
    b.token = nullptr;
    llama_batch_free(b);
    return ok;
}

// ── Embedding-mode forward ─────────────────────────────────────────────────

bool TalkerRunner::forward_embeddings(const float* embd, int32_t n_tokens,
                                       int32_t n_pos, float* hidden,
                                       std::string* error) {
    llama_batch b = make_embd_batch(embd, n_tokens, n_pos);
    const int32_t n_embd = hidden_size_;
    bool ok = (llama_decode(ctx_, b) == 0);
    if (ok) {
        for (int32_t i = 0; i < n_tokens; i++) {
            const float* row = llama_get_embeddings_ith(ctx_, i);
            if (!row) { ok = false; break; }
            std::memcpy(hidden + (size_t)i * n_embd, row,
                        (size_t)n_embd * sizeof(float));
        }
    } else {
        if (error) *error = "talker: forward_embeddings failed";
    }
    b.embd = nullptr;
    llama_batch_free(b);
    return ok;
}

// ── Text embedding (official Qwen3-TTS) ──────────────────────────────────

bool TalkerRunner::compute_text_embedding(const std::string& text,
                                           std::vector<float>* out_embd,
                                           int32_t* out_n_tokens,
                                           std::string* error) {
    if (!has_text_embd_) {
        if (error) *error = "talker: text_embedding tensors not loaded";
        return false;
    }

    // Tokenize text using the model's vocab
    std::vector<int32_t> tokens;
    if (!tokenize(text, /*add_special=*/true, /*parse_special=*/false,
                  &tokens, error)) {
        return false;
    }

    const int32_t n_tokens = (int32_t)tokens.size();
    const int32_t text_dim = text_embd_dim_;  // 2048
    const int32_t proj_dim = hidden_size_;     // 1024

    // Temporary buffer for text embeddings (before projection)
    std::vector<float> raw_embd((size_t)n_tokens * text_dim);

    // Lookup each token in text_embd table
    for (int32_t i = 0; i < n_tokens; i++) {
        int32_t id = tokens[(size_t)i];
        if (id >= 0 && id < text_vocab_) {
            const float* row = text_embd_ + (size_t)id * text_dim;
            std::memcpy(&raw_embd[(size_t)i * text_dim], row,
                        (size_t)text_dim * sizeof(float));
        } else {
            // Out-of-range token: zero embedding
            std::memset(&raw_embd[(size_t)i * text_dim], 0,
                        (size_t)text_dim * sizeof(float));
        }
    }

    // Project through text_projection MLP: Linear(2048→2048) + SiLU + Linear(2048→1024)
    out_embd->resize((size_t)n_tokens * proj_dim);

    for (int32_t i = 0; i < n_tokens; i++) {
        const float* x = &raw_embd[(size_t)i * text_dim];

        // layer 0: Linear(2048 → 2048) + SiLU
        float h[2048];
        if (text_proj_0_w_ && text_proj_0_b_) {
            for (int j = 0; j < text_dim; j++) {
                float sum = text_proj_0_b_[j];
                for (int k = 0; k < text_dim; k++) {
                    sum += x[k] * text_proj_0_w_[(size_t)j * text_dim + k];
                }
                h[j] = sum;
            }
        } else {
            std::memcpy(h, x, text_dim * sizeof(float));
        }

        // SiLU activation
        for (int j = 0; j < text_dim; j++) {
            float v = h[j];
            h[j] = v / (1.0f + std::exp(-v));
        }

        // layer 1: Linear(2048 → 1024)
        float* y = &(*out_embd)[(size_t)i * proj_dim];
        if (text_proj_1_w_ && text_proj_1_b_) {
            for (int j = 0; j < proj_dim; j++) {
                float sum = text_proj_1_b_[j];
                for (int k = 0; k < text_dim; k++) {
                    sum += h[k] * text_proj_1_w_[(size_t)j * text_dim + k];
                }
                y[j] = sum;
            }
        } else {
            std::memcpy(y, h, (size_t)proj_dim * sizeof(float));
        }
    }

    *out_n_tokens = n_tokens;
    return true;
}

// ── Tokenizer ──────────────────────────────────────────────────────────────

bool TalkerRunner::tokenize(const std::string& text, bool add_special,
                             bool parse_special, std::vector<int32_t>* tokens,
                             std::string* error) const {
    const llama_vocab* vocab = model_ ? llama_model_get_vocab(model_) : nullptr;
    if (!vocab) {
        if (error) *error = "talker: model has no vocab";
        return false;
    }
    int hint = llama_tokenize(vocab, text.c_str(),
                               (int32_t)text.size(), nullptr, 0,
                               add_special, parse_special);
    if (hint < 0) {
        tokens->resize((size_t)(-hint));
        int n = llama_tokenize(vocab, text.c_str(),
                                (int32_t)text.size(), tokens->data(),
                                (int32_t)tokens->size(),
                                add_special, parse_special);
        if (n < 0) { if (error) *error = "tokenize retry failed"; return false; }
        tokens->resize((size_t)n);
    } else {
        tokens->resize((size_t)hint);
        if (hint > 0) {
            int n = llama_tokenize(vocab, text.c_str(),
                                    (int32_t)text.size(), tokens->data(),
                                    (int32_t)tokens->size(),
                                    add_special, parse_special);
            tokens->resize((size_t)(n > 0 ? n : 0));
        }
    }
    return true;
}

bool TalkerRunner::is_eog(int32_t token) const {
    const llama_vocab* vocab = model_ ? llama_model_get_vocab(model_) : nullptr;
    return vocab && llama_vocab_is_eog(vocab, token);
}

const float* TalkerRunner::get_embeddings_ith(int32_t i) const {
    return llama_get_embeddings_ith(ctx_, i);
}

const float* TalkerRunner::get_logits_ith(int32_t i) const {
    return llama_get_logits_ith(ctx_, i);
}

}  // namespace audiocore::qwen3_tts
