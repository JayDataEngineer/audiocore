// runner.cpp — libllama-backed Qwen3 transformer execution.
//
// One Qwen3 inference path for the whole project. Loaded by the MOSS family
// (8B backbone), the ACE-Step family (1.7B 5Hz LM, 0.6B text encoder), the
// Qwen3-TTS talker, and the Qwen3-TTS code predictor. There is no other
// Qwen3 implementation in audiocore.

#include "audiocore/models/qwen3/runner.h"

#include "llama.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace audiocore::qwen3 {

// ═══════════════════════════════════════════════════════════════════════════
//  Shared GGUF raw-tensor reader (consolidated from talker_runner.cpp and
//  predictor_runner.cpp — both shipped near-identical copies).
// ═══════════════════════════════════════════════════════════════════════════

namespace {

struct TensorRef {
    int64_t  tid;       // tensor index in GGUF
    size_t   size;      // tensor data size in bytes
    size_t   offset;    // byte offset in file
    ggml_type type;     // element type
};

// Find a tensor by name in an open GGUF context.
TensorRef find_tensor_in_gguf(gguf_context* ctx, const char* name) {
    const int64_t n = gguf_get_n_tensors(ctx);
    for (int64_t i = 0; i < n; i++) {
        if (std::strcmp(gguf_get_tensor_name(ctx, i), name) == 0) {
            return {i,
                    gguf_get_tensor_size(ctx, i),
                    gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, i),
                    gguf_get_tensor_type(ctx, i)};
        }
    }
    return {-1, 0, 0, GGML_TYPE_F32};
}

// Load a single F32 tensor from a GGUF file by name. Returns nullptr if not
// found. Caller owns the returned buffer (delete[]).
float* load_gguf_tensor_f32(const char* path, const char* name,
                             size_t* out_nfloats = nullptr) {
    gguf_init_params params = {true, nullptr};
    gguf_context* ctx = gguf_init_from_file(path, params);
    if (!ctx) return nullptr;

    TensorRef tr = find_tensor_in_gguf(ctx, name);
    float* result = nullptr;
    if (tr.tid >= 0 && tr.size > 0) {
        const size_t n_floats = tr.size / sizeof(float);
        if (out_nfloats) *out_nfloats = n_floats;
        result = new float[n_floats];
        std::FILE* fp = std::fopen(path, "rb");
        if (fp) {
            std::fseek(fp, static_cast<long>(tr.offset), SEEK_SET);
            if (std::fread(result, 1, tr.size, fp) != tr.size) {
                delete[] result;
                result = nullptr;
            }
            std::fclose(fp);
        } else {
            delete[] result;
            result = nullptr;
        }
    }
    gguf_free(ctx);
    return result;
}

// Local RNG + top-p/top-t sampling used by predict_one_step. Consolidated
// with the moss_tts sampler in Stage 4 — kept here as a private helper for
// now to avoid pulling model-level headers into the framework library.
int32_t sample_token_basic(const float* logits, int32_t n_vocab,
                            float temp, float top_p, std::mt19937& rng) {
    if (n_vocab <= 0) return 0;
    std::vector<float> probs(static_cast<size_t>(n_vocab));
    float max_l = -1e38f;
    for (int i = 0; i < n_vocab; i++)
        if (logits[i] > max_l) max_l = logits[i];
    float sum = 0;
    for (int i = 0; i < n_vocab; i++) {
        probs[static_cast<size_t>(i)] =
            std::exp((logits[i] - max_l) / std::max(temp, 1e-6f));
        sum += probs[static_cast<size_t>(i)];
    }
    if (top_p < 1.0f && top_p > 0.0f && sum > 0) {
        std::vector<int32_t> idx(static_cast<size_t>(n_vocab));
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b) { return probs[a] > probs[b]; });
        float cum = 0;
        std::vector<bool> keep(static_cast<size_t>(n_vocab), false);
        for (auto ix : idx) {
            cum += probs[static_cast<size_t>(ix)] / sum;
            keep[static_cast<size_t>(ix)] = true;
            if (cum >= top_p) break;
        }
        for (size_t i = 0; i < static_cast<size_t>(n_vocab); i++)
            if (!keep[i]) probs[i] = 0;
        sum = 0;
        for (size_t i = 0; i < static_cast<size_t>(n_vocab); i++) sum += probs[i];
    }
    if (sum <= 0) return 0;
    std::uniform_real_distribution<float> dist(0, sum);
    const float sample = dist(rng);
    float cum = 0;
    for (int i = 0; i < n_vocab; i++) {
        cum += probs[static_cast<size_t>(i)];
        if (sample <= cum) return i;
    }
    return n_vocab - 1;
}

}  // namespace

Runner::Runner() = default;

Runner::~Runner() {
    delete[] text_embd_;
    delete[] text_proj_0_w_;
    delete[] text_proj_0_b_;
    delete[] text_proj_1_w_;
    delete[] text_proj_1_b_;
    delete[] codec_embd_;
    delete[] codec_head_;
    if (fine_embd_) {
        for (int i = 0; i < n_fine_books_; i++) delete[] fine_embd_[i];
        delete[] fine_embd_;
    }
    if (fine_head_) {
        for (int i = 0; i < n_fine_books_; i++) delete[] fine_head_[i];
        delete[] fine_head_;
    }
    delete[] small_to_mtp_w_;
    delete[] small_to_mtp_b_;
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

// ═══════════════════════════════════════════════════════════════════════════
//  Extras loading (talker / predictor) — consolidated verbatim from the
//  former talker_runner.cpp and predictor_runner.cpp so the Qwen3-TTS
//  family can use the same Runner class MOSS and ACE-Step use.
// ═══════════════════════════════════════════════════════════════════════════

bool Runner::load_extras(const std::string& gguf_path, ExtraKind kind,
                          int n_fine_books, std::string* error) {
    switch (kind) {
        case ExtraKind::None:
            return true;
        case ExtraKind::Talker:
            return load_talker_extras(gguf_path, error);
        case ExtraKind::Predictor:
            return load_predictor_extras(gguf_path, n_fine_books, error);
    }
    return true;
}

bool Runner::load_talker_extras(const std::string& gguf_path,
                                 std::string* error) {
    (void)error;
    const char* path = gguf_path.c_str();

    // text_embd.weight [text_vocab, 2048] — F32
    size_t nf = 0;
    if (float* te = load_gguf_tensor_f32(path, "text_embd.weight", &nf)) {
        text_embd_     = te;
        text_embd_dim_ = 2048;
        text_vocab_    = static_cast<int32_t>(nf / 2048);
    }

    text_proj_0_w_ = load_gguf_tensor_f32(path, "text_proj.0.weight");
    text_proj_0_b_ = load_gguf_tensor_f32(path, "text_proj.0.bias");
    text_proj_1_w_ = load_gguf_tensor_f32(path, "text_proj.1.weight");
    text_proj_1_b_ = load_gguf_tensor_f32(path, "text_proj.1.bias");

    // codec_embedding = token_embd.weight [codec_vocab, 1024]
    if (!codec_embd_) {
        size_t ce_nf = 0;
        if (float* ce = load_gguf_tensor_f32(path, "token_embd.weight", &ce_nf)) {
            codec_embd_     = ce;
            codec_embd_dim_ = 1024;
            codec_vocab_    = static_cast<int32_t>(ce_nf / 1024);
        }
    }

    // codec_head = output.weight [codec_vocab, 1024]
    if (!codec_head_) {
        codec_head_ = load_gguf_tensor_f32(path, "output.weight");
    }

    has_text_embd_ = (text_embd_ != nullptr && text_proj_0_w_ != nullptr);
    return true;
}

bool Runner::load_predictor_extras(const std::string& gguf_path,
                                    int n_fine_books, std::string* error) {
    (void)error;
    const char* path = gguf_path.c_str();
    n_fine_books_ = n_fine_books;
    n_codebooks_  = n_fine_books + 1;   // coarse + fine

    // Detect MTP tensors — if none, this is a Lunavox-style predictor that
    // only uses the libllama-loaded weights via forward_tokens.
    int found = 0;
    for (int i = 0; i < n_fine_books_; i++) {
        char name[256];
        std::snprintf(name, sizeof(name), "codec_embd.%d.weight", i);
        float* t = load_gguf_tensor_f32(path, name);
        if (t) { found++; delete[] t; }
    }
    if (found == 0) return true;   // Lunavox-style GGUF; no MTP

    has_mtp_ = true;
    fine_embd_ = new float*[static_cast<size_t>(n_fine_books_)]();
    fine_head_ = new float*[static_cast<size_t>(n_fine_books_)]();

    for (int i = 0; i < n_fine_books_; i++) {
        char name[256];

        // codec_embd.{i}.weight [fine_vocab, 1024]
        std::snprintf(name, sizeof(name), "codec_embd.%d.weight", i);
        fine_embd_[i] = load_gguf_tensor_f32(path, name);

        // lm_head.{i}.weight [1024, fine_vocab]
        std::snprintf(name, sizeof(name), "lm_head.%d.weight", i);
        fine_head_[i] = load_gguf_tensor_f32(path, name);
    }

    // Infer fine_vocab_ from the on-disk tensor size.
    {
        gguf_init_params params = {true, nullptr};
        gguf_context* ctx = gguf_init_from_file(path, params);
        if (ctx) {
            const int64_t n = gguf_get_n_tensors(ctx);
            for (int64_t i = 0; i < n; i++) {
                const char* tn = gguf_get_tensor_name(ctx, i);
                if (tn && std::strcmp(tn, "codec_embd.0.weight") == 0) {
                    const size_t sz = gguf_get_tensor_size(ctx, i);
                    fine_vocab_ = static_cast<int32_t>(sz / (sizeof(float) * 1024));
                    break;
                }
            }
            gguf_free(ctx);
        }
    }

    small_to_mtp_w_ = load_gguf_tensor_f32(path, "small_to_mtp.weight");
    small_to_mtp_b_ = load_gguf_tensor_f32(path, "small_to_mtp.bias");

    // Pre-allocate workspace for predict_one_step.
    scratch_embd_.resize(static_cast<size_t>(8192) *
                         static_cast<size_t>(std::max(1, hidden_size_)));
    scratch_logits_.resize(static_cast<size_t>(n_fine_books_) * 8192);
    return true;
}

// ── Talker text-embedding (compute_text_embedding) ─────────────────────────
//
// Ported verbatim from the former TalkerRunner::compute_text_embedding.
// Linear(2048→2048) + SiLU + Linear(2048→1024) over text_embd.weight rows.

bool Runner::compute_text_embedding(const std::string& text,
                                     std::vector<float>* out_embd,
                                     int32_t* out_n_tokens,
                                     std::string* error) {
    if (!has_text_embd_) {
        if (error) *error = "runner: text_embedding tensors not loaded";
        return false;
    }

    std::vector<int32_t> tokens;
    if (!tokenize(text, /*add_special=*/true, /*parse_special=*/false,
                  &tokens, /*needed=*/nullptr, error)) {
        return false;
    }

    const int32_t n_tokens = static_cast<int32_t>(tokens.size());
    const int32_t text_dim = text_embd_dim_;   // 2048
    const int32_t proj_dim = hidden_size_;     // 1024

    std::vector<float> raw_embd(static_cast<size_t>(n_tokens) * text_dim);
    for (int32_t i = 0; i < n_tokens; i++) {
        const int32_t id = tokens[static_cast<size_t>(i)];
        if (id >= 0 && id < text_vocab_) {
            const float* row = text_embd_ + static_cast<size_t>(id) * text_dim;
            std::memcpy(&raw_embd[static_cast<size_t>(i) * text_dim], row,
                        static_cast<size_t>(text_dim) * sizeof(float));
        } else {
            std::memset(&raw_embd[static_cast<size_t>(i) * text_dim], 0,
                        static_cast<size_t>(text_dim) * sizeof(float));
        }
    }

    out_embd->resize(static_cast<size_t>(n_tokens) * proj_dim);

    for (int32_t i = 0; i < n_tokens; i++) {
        const float* x = &raw_embd[static_cast<size_t>(i) * text_dim];

        // layer 0: Linear(2048 → 2048) + SiLU
        float h[2048];
        if (text_proj_0_w_ && text_proj_0_b_) {
            for (int j = 0; j < text_dim; j++) {
                float sum = text_proj_0_b_[j];
                for (int k = 0; k < text_dim; k++) {
                    sum += x[k] * text_proj_0_w_[static_cast<size_t>(j) * text_dim + k];
                }
                h[j] = sum;
            }
        } else {
            std::memcpy(h, x, static_cast<size_t>(text_dim) * sizeof(float));
        }
        for (int j = 0; j < text_dim; j++) {
            const float v = h[j];
            h[j] = v / (1.0f + std::exp(-v));
        }

        // layer 1: Linear(2048 → 1024)
        float* y = &(*out_embd)[static_cast<size_t>(i) * proj_dim];
        if (text_proj_1_w_ && text_proj_1_b_) {
            for (int j = 0; j < proj_dim; j++) {
                float sum = text_proj_1_b_[j];
                for (int k = 0; k < text_dim; k++) {
                    sum += h[k] * text_proj_1_w_[static_cast<size_t>(j) * text_dim + k];
                }
                y[j] = sum;
            }
        } else {
            std::memcpy(y, h, static_cast<size_t>(proj_dim) * sizeof(float));
        }
    }

    *out_n_tokens = n_tokens;
    return true;
}

// ── Predictor MTP step (predict_one_step) ───────────────────────────────────
//
// Ported verbatim from the former PredictorRunner::predict_one_step. At each
// sub-step k (0..n_fine_books-1), predict fine codebook (k+1) by conditioning
// on the talker's projected hidden state + all previously emitted codes.

bool Runner::predict_one_step(const float* talker_hidden,
                                const int32_t* prev_codes,
                                int32_t n_fine_books,
                                int32_t* out_codes,
                                std::string* error) {
    if (!has_mtp_) {
        if (error) *error = "runner: MTP tensors not loaded";
        return false;
    }
    if (n_fine_books != n_fine_books_) {
        if (error) *error = "runner: predict_one_step n_fine_books mismatch";
        return false;
    }

    const int32_t ne = hidden_size_;
    const int32_t fv = fine_vocab_;
    const int32_t nf = n_fine_books_;

    // Project talker hidden through small_to_mtp.
    std::vector<float> proj(static_cast<size_t>(ne));
    if (small_to_mtp_w_) {
        for (int j = 0; j < ne; j++) {
            float s = small_to_mtp_b_ ? small_to_mtp_b_[j] : 0.0f;
            for (int k = 0; k < ne; k++)
                s += talker_hidden[k] *
                     small_to_mtp_w_[static_cast<size_t>(j) * ne + k];
            proj[static_cast<size_t>(j)] = s;
        }
    } else {
        std::memcpy(proj.data(), talker_hidden, static_cast<size_t>(ne) * sizeof(float));
    }

    std::vector<float> seq;
    seq.reserve(static_cast<size_t>(nf + 2) * ne);

    std::mt19937 rng(42);
    std::vector<int32_t> cur_codes(prev_codes, prev_codes + nf);

    for (int k = 0; k < nf; k++) {
        seq.clear();
        seq.insert(seq.end(), proj.begin(), proj.end());

        for (int i = 0; i <= k; i++) {
            int32_t cid = cur_codes[static_cast<size_t>(i)];
            if (cid < 0) cid = 0;
            if (cid >= fv) cid %= fv;
            const float* row = fine_embd_[i] + static_cast<size_t>(cid) * ne;
            seq.insert(seq.end(), row, row + ne);
        }

        const int32_t seq_len = k + 2;   // proj + (k+1) code embeddings
        llama_batch b = llama_batch_init(seq_len, /*embd=*/seq_len, /*n_seq_max=*/1);
        b.n_tokens = seq_len;
        b.token = nullptr;
        b.embd  = const_cast<float*>(seq.data());
        for (int32_t i = 0; i < seq_len; i++) {
            b.pos[i] = i;
            b.logits[i] = (i == seq_len - 1) ? 1 : 0;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
        }
        const bool ok = (llama_decode(ctx_, b) == 0);
        b.embd = nullptr;
        llama_batch_free(b);
        if (!ok) {
            if (error) *error = "runner: MTP sub-step decode failed";
            return false;
        }

        const float* last_h = llama_get_embeddings_ith(ctx_, seq_len - 1);
        if (!last_h) {
            if (error) *error = "runner: MTP embeddings unavailable";
            return false;
        }

        std::vector<float> logits(static_cast<size_t>(fv));
        const float* head_w = fine_head_[k];
        for (int j = 0; j < fv; j++) {
            float s = 0.0f;
            for (int d = 0; d < ne; d++)
                s += last_h[d] * head_w[static_cast<size_t>(j) * ne + d];
            logits[static_cast<size_t>(j)] = s;
        }

        const int32_t sampled = sample_token_basic(logits.data(), fv,
                                                    0.7f, 0.9f, rng);
        out_codes[k] = sampled;
        cur_codes[static_cast<size_t>(k)] = sampled;
    }

    return true;
}

}  // namespace audiocore::qwen3
