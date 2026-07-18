// runner.cpp — libllama-backed Qwen3 transformer execution.
//
// One Qwen3 inference path for the whole project. Loaded by the MOSS family
// (8B backbone), the ACE-Step family (1.7B 5Hz LM, 0.6B text encoder), the
// Qwen3-TTS talker, and the Qwen3-TTS code predictor. There is no other
// Qwen3 implementation in audiocore.

#include "audiocore/models/qwen3/runner.h"

#include "llama.h"
#include "llama-model.h"  // internal: for tok_embd access in embed_lookup

#include "audiocore/framework/io/gguf_reader.h"
#include "audiocore/framework/sampling/sampler.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace audiocore::qwen3 {

using audiocore::GgufReader;
using audiocore::TensorStorage;
using audiocore::sampler::Params;
using audiocore::sampler::sample_token;
using audiocore::sampler::PhiloxRng;

// ═══════════════════════════════════════════════════════════════════════════
//  Extras tensor materialization via WeightLoader.
//
//  The talker/predictor extras (text_embd, codec_embd, per-codebook tables,
//  lm_heads, MTP projection) live in the same GGUF libllama loaded but are
//  not surfaced by any llama_* API. We pull them out through the same
//  WeightLoader interface the moss_tts and ace_step families use — opening
//  the GGUF exactly once per extras pass instead of once per tensor.
// ═══════════════════════════════════════════════════════════════════════════

namespace {

// Look up `name` in `reader` and materialize it into a freshly-allocated
// float[] that the caller owns (delete[]). Returns nullptr if the tensor is
// absent or the read fails. *out_nfloats (if non-null) receives the element
// count.
//
// Handles both F32 and quantized on-disk tensors: F32 is memcpy'd directly;
// quantized types (Q5_K, Q8_0, ...) are dequantized via ggml's type-traits
// to_float converter. This is needed because the Lunavox GGUFs store
// output.weight / token_embd.weight as Q5_K while the converter-based GGUFs
// store the extras as F32 — the function adapts transparently.
float* materialize_f32(const GgufReader& reader, const char* name,
                       size_t* out_nfloats = nullptr) {
    const TensorStorage* t = reader.find(name);
    if (!t) return nullptr;
    const size_t n_floats = static_cast<size_t>(t->nelements());
    float* buf = new float[n_floats];

    if (t->type == GGML_TYPE_F32) {
        // Fast path: tensor is already F32 on disk — memcpy directly.
        std::string err;
        if (!reader.materialize(*t, buf, &err)) {
            std::fprintf(stderr,
                         "qwen3::Runner: materialize('%s') failed: %s\n",
                         name, err.c_str());
            delete[] buf;
            return nullptr;
        }
    } else {
        // Dequantization path: read raw quantized bytes, then dequantize
        // to F32 via ggml's type-traits to_float converter.
        const size_t raw_bytes = static_cast<size_t>(t->nbytes());
        std::vector<uint8_t> raw(raw_bytes);
        std::string err;
        if (!reader.materialize(*t, raw.data(), &err)) {
            std::fprintf(stderr,
                         "qwen3::Runner: materialize('%s') failed: %s\n",
                         name, err.c_str());
            delete[] buf;
            return nullptr;
        }
        const auto* traits = ggml_get_type_traits(t->type);
        if (!traits || !traits->to_float) {
            std::fprintf(stderr,
                         "qwen3::Runner: no to_float converter for type %d "
                         "('%s')\n", (int)t->type, name);
            delete[] buf;
            return nullptr;
        }
        traits->to_float(raw.data(), buf, static_cast<int64_t>(n_floats));
    }

    if (out_nfloats) *out_nfloats = n_floats;
    return buf;
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
    if (ctx_secondary_) llama_free(ctx_secondary_);
    if (model_) llama_model_free(model_);
    if (tokenizer_model_) llama_model_free(tokenizer_model_);
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
    // RoPE type: both talker and predictor use NEOX (1 pos per embd) in TTS-only
    // mode. The talker's config says rope_scaling.interleaved=true with
    // mrope_section=[24,20,20], but the actual rotation is rotate_half (NEOX
    // pairs (i, i+n/2)) after a cos/sin reorder. In TTS-only mode where all 3
    // multimodal axes share the same position id, IMROPE and NEOX produce
    // identical results. We force rope_type=NEOX in llama-model.cpp and use
    // standard 1D positions here.
    {
        const auto rt = llama_model_rope_type(self->model_);
        self->n_pos_per_embd_ = (rt == LLAMA_ROPE_TYPE_MROPE ||
                                  rt == LLAMA_ROPE_TYPE_IMROPE) ? 4 : 1;
    }
    return self;
}

// Build a llama_batch ready for llama_decode with embd OR token input,
// positions starting at n_pos. By default every position is marked for
// output so callers can read per-row logits/embeddings. Pass
// `last_only=true` to mark only the final position for output — used for
// intermediate prefill chunks where intermediate hidden states are not needed.
//
// n_pos_per_embd: from llama_model_rope_type(). M-RoPE/IM-RoPE → 4, else 1.
// For embedding input with M-RoPE, the pos array must hold n_pos_per_embd
// contiguous copies of the position sequence (one per RoPE section).
static llama_batch make_batch(int32_t n_tokens, int32_t n_pos, bool is_embd,
                              const float* embd, const llama_token* tokens,
                              bool last_only, uint32_t n_pos_per_embd) {
    llama_batch b = llama_batch_init(n_tokens,
                                     /*embd=*/ is_embd ? 1 : 0,
                                     /*n_seq_max=*/ 1);
    b.n_tokens = n_tokens;
    if (is_embd) {
        b.token = nullptr;
        b.embd  = const_cast<float*>(embd);
        // For M-RoPE (n_pos_per_embd > 1) with embedding input, llama.cpp's
        // batch allocator reads batch.pos[j * n_tokens + i] for each RoPE
        // section j. The default pos allocation (n_tokens entries) is too
        // small — realloc to n_pos_per_embd * n_tokens and fill all sections
        // with the same monotonically increasing positions.
        if (n_pos_per_embd > 1) {
            b.pos = (llama_pos*) realloc(b.pos,
                        sizeof(llama_pos) * n_tokens * n_pos_per_embd);
            for (uint32_t j = 0; j < n_pos_per_embd; j++) {
                for (int32_t i = 0; i < n_tokens; i++)
                    b.pos[j * n_tokens + i] = n_pos + i;
            }
        } else {
            for (int32_t i = 0; i < n_tokens; i++) b.pos[i] = n_pos + i;
        }
    } else {
        b.token = const_cast<llama_token*>(tokens);
        b.embd  = nullptr;
        // Token input: batch.token is non-null, so ubatch_add broadcasts
        // pos[i] across all RoPE sections — no extended array needed.
        for (int32_t i = 0; i < n_tokens; i++) b.pos[i] = n_pos + i;
    }
    // Mark output positions. last_only=true: only the final row is live,
    // which avoids allocating/computing output for prefill chunks.
    for (int32_t i = 0; i < n_tokens; i++)
        b.logits[i] = (!last_only || i == n_tokens - 1) ? 1 : 0;
    for (int32_t i = 0; i < n_tokens; i++) {
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
    }
    return b;
}

bool Runner::forward_embeddings(const float* embd, int32_t n_tokens, int32_t n_pos,
                                float* hidden, std::string* error) {
    // Chunk the prefill when n_tokens exceeds n_batch. The KV cache is
    // accumulated across chunks; only the last position's hidden state is
    // returned (all callers only use hidden[(n_tokens-1)*n_embd]).
    // Intermediate chunks use last_only=true to avoid allocating output
    // buffers for positions whose hidden states are never read.
    // Use n_ubatch (physical micro-batch) as the chunk ceiling: that's what
    // sizes the output buffer that llama_get_embeddings_ith indexes into.
    // n_batch (logical) may be much larger (e.g. == n_ctx) but submitting a
    // chunk larger than n_ubatch would push output indices out of range.
    const int32_t n_batch = static_cast<int32_t>(llama_n_ubatch(ctx_));
    const int32_t n_embd  = hidden_size_;

    for (int32_t chunk_start = 0; chunk_start < n_tokens; ) {
        const int32_t chunk_size = std::min(n_batch, n_tokens - chunk_start);
        const bool    is_last    = (chunk_start + chunk_size == n_tokens);
        const float*  chunk_embd = embd + static_cast<size_t>(chunk_start) * n_embd;

        llama_batch b = make_batch(chunk_size, n_pos + chunk_start,
                                   /*is_embd=*/true, chunk_embd, nullptr,
                                   /*last_only=*/true, n_pos_per_embd_);
        const bool ok = (llama_decode(ctx_, b) == 0);
        b.token = nullptr;
        b.embd  = nullptr;
        if (!ok) {
            llama_batch_free(b);
            if (error) *error = "llama_decode failed (prefill chunk)";
            return false;
        }

        // Only copy the last position's hidden state, and only from the
        // final chunk (that's the only slot callers actually read).
        // last_only=true marks only batch position (chunk_size - 1) for
        // output, so output_ids[chunk_size - 1] resolves to the only
        // logits-enabled row regardless of chunk_size.
        if (is_last && hidden) {
            const float* row = llama_get_embeddings_ith(ctx_, chunk_size - 1);
            if (!row) {
                llama_batch_free(b);
                if (error) *error = "forward_embeddings: llama_get_embeddings_ith failed (prefill chunk)";
                return false;
            }
            std::memcpy(hidden + static_cast<size_t>(n_tokens - 1) * n_embd,
                        row, static_cast<size_t>(n_embd) * sizeof(float));
        }

        llama_batch_free(b);
        chunk_start += chunk_size;
    }
    return true;
}

bool Runner::forward_tokens(const int32_t* tokens, int32_t n_tokens, int32_t n_pos,
                            float* logits, std::string* error) {
    llama_batch b = make_batch(n_tokens, n_pos, /*is_embd=*/false,
                               nullptr, reinterpret_cast<const llama_token*>(tokens),
                               /*last_only=*/false, n_pos_per_embd_);
    const int32_t n_vocab = vocab_size_;
    bool ok = (llama_decode(ctx_, b) == 0);
    if (!ok) {
        if (error) *error = "llama_decode failed";
    } else if (logits) {
        // Copy logits per token (skip when caller passes nullptr — only
        // needs KV cache populated, e.g. LM prefill in ACE-Step).
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
                               nullptr, reinterpret_cast<const llama_token*>(tokens),
                               /*last_only=*/false, n_pos_per_embd_);
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

bool Runner::embed_lookup(const int32_t* token_ids, int32_t n_tokens,
                          float* output, std::string* error) {
    // Raw token-embedding lookup — NO transformer forward pass.
    // Accesses the model's token_embd.weight tensor directly and extracts
    // the embedding row for each token ID. Matches the reference
    // qwen3_embed_lookup in cond-enc.h.
    if (!model_) {
        if (error) *error = "embed_lookup: model not loaded";
        return false;
    }
    if (n_tokens <= 0) return true;

    // Access the model's token embedding tensor via internal header.
    // llama_model::tok_embd is a ggml_tensor* holding the full vocab table.
    const auto* internal_model = reinterpret_cast<const llama_model*>(model_);
    const ggml_tensor* tok_embd = internal_model->tok_embd;
    if (!tok_embd) {
        if (error) *error = "embed_lookup: model has no tok_embd tensor";
        return false;
    }

    const int32_t n_embd = (int32_t) tok_embd->ne[0];  // hidden_size
    const int32_t n_vocab = (int32_t) tok_embd->ne[1]; // vocab size

    // Read the embedding table from the backend buffer.
    // For quantized models, dequantize each row as we read it.
    // Allocate a temporary buffer for one row at a time to save memory.
    std::vector<uint8_t> row_bytes;
    const size_t row_size = ggml_row_size(tok_embd->type, n_embd);

    // If the tensor is f32, we can read directly; otherwise dequantize.
    if (tok_embd->type == GGML_TYPE_F32) {
        // Direct read — each row is n_embd * sizeof(float) bytes
        for (int32_t i = 0; i < n_tokens; i++) {
            int32_t tid = token_ids[i];
            if (tid < 0 || tid >= n_vocab) {
                if (error) *error = "embed_lookup: token id " + std::to_string(tid) + " out of range";
                return false;
            }
            ggml_backend_tensor_get(tok_embd, output + (size_t)i * n_embd,
                                    (size_t)tid * n_embd * sizeof(float),
                                    n_embd * sizeof(float));
        }
    } else {
        // Quantized — read full tensor to f32 then pick rows.
        // For efficiency, cache the dequantized table on first call.
        static thread_local std::vector<float> embd_table_f32;
        static thread_local int32_t cached_n_vocab = 0;
        static thread_local int32_t cached_n_embd = 0;

        if (cached_n_vocab != n_vocab || cached_n_embd != n_embd || embd_table_f32.empty()) {
            embd_table_f32.resize((size_t)n_vocab * n_embd);
            // Dequantize using a ggml graph on the CPU backend.
            size_t ctx_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
            struct ggml_init_params gp = {ctx_size, nullptr, true};
            struct ggml_context* ctx = ggml_init(gp);

            ggml_tensor* dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);
            ggml_set_name(dst, "token_embd_f32");

            ggml_backend_t backend = ggml_backend_cpu_init();
            if (!ggml_backend_alloc_ctx_tensors(ctx, backend)) {
                if (error) *error = "embed_lookup: failed to alloc ctx tensors";
                ggml_free(ctx);
                return false;
            }

            // Build a copy graph: dst = tok_embd (with type conversion)
            struct ggml_cgraph* gf = ggml_new_graph(ctx);
            ggml_tensor* cvt = ggml_cpy(ctx, const_cast<ggml_tensor*>(tok_embd), dst);
            ggml_build_forward_expand(gf, cvt);
            ggml_backend_graph_compute(backend, gf);
            ggml_backend_tensor_get(dst, embd_table_f32.data(), 0,
                                    (size_t)n_vocab * n_embd * sizeof(float));
            ggml_free(ctx);
            ggml_backend_free(backend);

            cached_n_vocab = n_vocab;
            cached_n_embd = n_embd;

            fprintf(stderr, "[Runner] embed_lookup: cached %d×%d embedding table (%.1f MB)\n",
                    n_vocab, n_embd, (double)(n_vocab * n_embd * sizeof(float)) / (1024*1024));
        }

        for (int32_t i = 0; i < n_tokens; i++) {
            int32_t tid = token_ids[i];
            if (tid < 0 || tid >= n_vocab) {
                if (error) *error = "embed_lookup: token id " + std::to_string(tid) + " out of range";
                return false;
            }
            memcpy(output + (size_t)i * n_embd,
                   embd_table_f32.data() + (size_t)tid * n_embd,
                   n_embd * sizeof(float));
        }
    }

    return true;
}

bool Runner::tokenize(const std::string& text, bool add_special,
                      bool parse_special, std::vector<int32_t>* tokens,
                      int32_t* needed, std::string* error) const {
    // Prefer the tokenizer sidecar (real Qwen3 BPE) when loaded; fall back
    // to the talker model's built-in tokenizer (the dummy codec-vocab stub).
    llama_model* tok_model = tokenizer_model_ ? tokenizer_model_ : model_;
    const llama_vocab* vocab = tok_model ? llama_model_get_vocab(tok_model) : nullptr;
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

bool Runner::load_tokenizer(const std::string& gguf_path,
                            std::string* error) {
    if (tokenizer_model_) {
        llama_model_free(tokenizer_model_);
        tokenizer_model_ = nullptr;
    }
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only  = true;   // no weights, just the tokenizer
    mp.use_mmap    = true;
    mp.n_gpu_layers = 0;

    tokenizer_model_ = llama_model_load_from_file(gguf_path.c_str(), mp);
    if (!tokenizer_model_) {
        if (error) *error = "load_tokenizer: llama_model_load_from_file failed: " + gguf_path;
        return false;
    }
    // Quick sanity: verify the vocab size matches what we expect.
    const llama_vocab* vocab = llama_model_get_vocab(tokenizer_model_);
    int32_t n = vocab ? llama_vocab_n_tokens(vocab) : 0;
    if (n < 1000) {
        if (error) *error = "load_tokenizer: vocab too small (" + std::to_string(n) + " tokens)";
        llama_model_free(tokenizer_model_);
        tokenizer_model_ = nullptr;
        return false;
    }
    std::fprintf(stderr, "qwen3_tts: tokenizer sidecar loaded (%d tokens)\n", n);
    return true;
}

bool Runner::token_to_piece(int32_t token, std::string* out,
                            std::string* error) const {
    llama_model* tok_model = tokenizer_model_ ? tokenizer_model_ : model_;
    const llama_vocab* vocab = tok_model ? llama_model_get_vocab(tok_model) : nullptr;
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
    llama_model* tok_model = tokenizer_model_ ? tokenizer_model_ : model_;
    const llama_vocab* vocab = tok_model ? llama_model_get_vocab(tok_model) : nullptr;
    if (!vocab) return false;
    return llama_vocab_is_eog(vocab, static_cast<llama_token>(token));
}

int32_t Runner::bos_token_id() const {
    llama_model* tok_model = tokenizer_model_ ? tokenizer_model_ : model_;
    const llama_vocab* vocab = tok_model ? llama_model_get_vocab(tok_model) : nullptr;
    if (!vocab) return 151644;
    return llama_vocab_bos(vocab);
}

int32_t Runner::eos_token_id() const {
    llama_model* tok_model = tokenizer_model_ ? tokenizer_model_ : model_;
    const llama_vocab* vocab = tok_model ? llama_model_get_vocab(tok_model) : nullptr;
    if (!vocab) return 151645;
    return llama_vocab_eos(vocab);
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
    // Open the GGUF once via WeightLoader. The talker extras are typically
    // 7 tensors — going through GgufReader turns 7 file opens into 1.
    GgufReader reader;
    std::string load_err;
    if (!reader.load(gguf_path, &load_err)) {
        if (error) *error = "talker extras: " + load_err;
        return false;
    }

    // text_embd.weight [text_vocab, 2048] — F32
    size_t nf = 0;
    if (float* te = materialize_f32(reader, "text_embd.weight", &nf)) {
        text_embd_     = te;
        text_embd_dim_ = 2048;
        text_vocab_    = static_cast<int32_t>(nf / 2048);
    }

    text_proj_0_w_ = materialize_f32(reader, "text_proj.0.weight");
    text_proj_0_b_ = materialize_f32(reader, "text_proj.0.bias");
    text_proj_1_w_ = materialize_f32(reader, "text_proj.1.weight");
    text_proj_1_b_ = materialize_f32(reader, "text_proj.1.bias");

    // codec_embedding = token_embd.weight [codec_vocab, codec_embd_dim]
    // GGML stores ne[0]=innermost (n_embd), ne[1]=outermost (vocab).
    if (!codec_embd_) {
        const TensorStorage* te = reader.find("token_embd.weight");
        if (te) {
            size_t ce_nf = 0;
            if (float* ce = materialize_f32(reader, "token_embd.weight", &ce_nf)) {
                codec_embd_ = ce;
                codec_embd_dim_ = (te->n_dims >= 2)
                    ? static_cast<int32_t>(te->ne[0])
                    : 0;
                codec_vocab_ = static_cast<int32_t>(ce_nf / codec_embd_dim_);
            }
        }
    }

    // codec_head = output.weight [codec_vocab, codec_embd_dim]
    if (!codec_head_) {
        codec_head_ = materialize_f32(reader, "output.weight");
    }

    has_text_embd_ = (text_embd_ != nullptr && text_proj_0_w_ != nullptr);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
//  WDELTA: Base → CustomVoice weight patching
//
//  See header docstring for the full rationale. Summary: load_extras(Talker)
//  on a CV talker produces CV-flavored text_proj biases and codec_embd rows
//  (the 9 preset speaker tokens). For the CV talker to accept a continuous
//  ECAPA embedding at the speaker slot — while keeping its instruct-tuned
//  transformer norms — we overwrite just the embedding-pathway buffers with
//  Base's versions. Other tensors (attn/ffn/norm weights in libllama) are
//  left untouched; those carry CV's instruction-following behavior.
// ═══════════════════════════════════════════════════════════════════════════
bool Runner::apply_wdelta_patch(const std::string& base_gguf_path,
                                 std::string* error) {
    if (!has_text_embd_) {
        if (error) *error = "apply_wdelta_patch: talker extras not loaded";
        return false;
    }

    GgufReader reader;
    std::string load_err;
    if (!reader.load(base_gguf_path, &load_err)) {
        if (error) *error = "wdelta: failed to open base GGUF '" +
                            base_gguf_path + "': " + load_err;
        return false;
    }

    // Helper: materialize a tensor from Base, validate its element count
    // matches the destination buffer, and memcpy over it.
    auto swap_f32 = [&](const char* name, float* dst, size_t dst_n,
                        const char* what) -> bool {
        if (!dst) {
            if (error) *error = std::string("wdelta: destination ") + what +
                                " is null (load_extras did not allocate it)";
            return false;
        }
        size_t got_n = 0;
        float* src = materialize_f32(reader, name, &got_n);
        if (!src) {
            if (error) *error = std::string("wdelta: base GGUF missing '") +
                                name + "' for " + what;
            return false;
        }
        if (got_n != dst_n) {
            if (error) *error = std::string("wdelta: ") + what +
                                " dimension mismatch (base=" +
                                std::to_string(got_n) +
                                " vs cv=" + std::to_string(dst_n) + ")";
            delete[] src;
            return false;
        }
        std::memcpy(dst, src, dst_n * sizeof(float));
        delete[] src;
        return true;
    };

    // Validate embedding pathway dim first — Base must produce the same
    // hidden size as the CV talker we already loaded. Without this guard,
    // a 0.6B Base patch over a 1.7B CV (or vice versa) would corrupt memory.
    const TensorStorage* tok_embd = reader.find("token_embd.weight");
    if (!tok_embd || tok_embd->n_dims < 2) {
        if (error) *error = "wdelta: base GGUF has no token_embd.weight";
        return false;
    }
    const int32_t base_embd_dim = static_cast<int32_t>(tok_embd->ne[0]);
    if (base_embd_dim != codec_embd_dim_) {
        if (error) *error = "wdelta: hidden size mismatch (base n_embd=" +
                            std::to_string(base_embd_dim) + " vs cv=" +
                            std::to_string(codec_embd_dim_) +
                            "). Pick a Base GGUF that matches the CV size.";
        return false;
    }

    // The 5-tensor swap. Order matches load_talker_extras so a partial
    // failure partway through leaves us closer to the CV state than to a
    // half-patched chimera (text_proj overwritten but codec_embd untouched
    // is more recoverable than the reverse, since text_proj biases are the
    // smaller delta).
    //
    // 1) text_proj.0.weight [2048, 2048]
    // 2) text_proj.0.bias   [2048]
    // 3) text_proj.1.weight [1024 or 2048, 2048]
    // 4) text_proj.1.bias   [1024 or 2048]
    // 5) token_embd.weight  (codec_embedding) [codec_vocab, n_embd]
    //
    // Compute dst sizes from the SAME shapes load_talker_extras used so we
    // don't drift if the converter changes layout later.
    const int32_t text_dim_in  = text_embd_dim_;   // 2048 (always)
    const int32_t proj_dim_out = hidden_size_;     // 1024 (0.6B) or 2048 (1.7B)

    if (text_proj_0_w_) {
        if (!swap_f32("text_proj.0.weight", text_proj_0_w_,
                       static_cast<size_t>(text_dim_in) * text_dim_in,
                       "text_proj.0.weight")) return false;
    }
    if (text_proj_0_b_) {
        if (!swap_f32("text_proj.0.bias", text_proj_0_b_,
                       static_cast<size_t>(text_dim_in),
                       "text_proj.0.bias")) return false;
    }
    if (text_proj_1_w_) {
        if (!swap_f32("text_proj.1.weight", text_proj_1_w_,
                       static_cast<size_t>(text_dim_in) * proj_dim_out,
                       "text_proj.1.weight")) return false;
    }
    if (text_proj_1_b_) {
        if (!swap_f32("text_proj.1.bias", text_proj_1_b_,
                       static_cast<size_t>(proj_dim_out),
                       "text_proj.1.bias")) return false;
    }
    if (codec_embd_) {
        // codec_vocab may differ between CV (3072) and Base (3072) — same
        // for all 0.6B/1.7B variants we ship. The dimension check in
        // swap_f32 will catch a mismatch if a future model deviates.
        const size_t codec_n = static_cast<size_t>(codec_vocab_) *
                               static_cast<size_t>(codec_embd_dim_);
        if (!swap_f32("token_embd.weight", codec_embd_, codec_n,
                       "token_embd (codec_embedding)")) return false;
    }

    std::fprintf(stderr,
        "qwen3::Runner: WDELTA patch applied from %s\n"
        "  (text_proj biases + codec_embedding overwritten with Base;\n"
        "   transformer attn/ffn/norm weights retain CV's instruct tuning)\n",
        base_gguf_path.c_str());
    return true;
}

bool Runner::load_predictor_extras(const std::string& gguf_path,
                                    int n_fine_books, std::string* error) {
    n_fine_books_ = n_fine_books;
    n_codebooks_  = n_fine_books + 1;   // coarse + fine

    // Open the GGUF once via WeightLoader. The predictor extras for a
    // 31-codebook MTP model are ~65 tensors — the former code re-opened
    // the GGUF for every single one.
    GgufReader reader;
    std::string load_err;
    if (!reader.load(gguf_path, &load_err)) {
        if (error) *error = "predictor extras: " + load_err;
        return false;
    }

    // Detect MTP tensors — if none, this is a Lunavox-style predictor that
    // only uses the libllama-loaded weights via forward_tokens.
    // Count the ACTUAL number of codec_embd.{i}.weight tensors in the GGUF,
    // which may be less than the requested n_fine_books_ (e.g. 15 vs 31).
    int detected_books = 0;
    for (int i = 0; i < n_fine_books_; i++) {
        char name[256];
        std::snprintf(name, sizeof(name), "codec_embd.%d.weight", i);
        if (reader.find(name)) ++detected_books;
        else break;   // tensors are contiguous; stop at first gap
    }
    if (detected_books == 0) return true;   // Lunavox-style GGUF; no MTP

    // Update n_fine_books_ to the actual count from the GGUF.
    if (detected_books < n_fine_books_) {
        n_fine_books_  = detected_books;
        n_codebooks_   = n_fine_books_ + 1;
    }
    has_mtp_ = true;
    fine_embd_ = new float*[static_cast<size_t>(n_fine_books_)]();
    fine_head_ = new float*[static_cast<size_t>(n_fine_books_)]();

    // Infer fine_vocab_ and fine_embd_dim_ from codec_embd.0.weight's shape.
    // GGML: ne[0]=innermost (n_embd), ne[1]=outermost (fine_vocab).
    if (const TensorStorage* t0 = reader.find("codec_embd.0.weight")) {
        fine_embd_dim_ = (t0->n_dims >= 2)
            ? static_cast<int32_t>(t0->ne[0])
            : 1024;
        fine_vocab_ = static_cast<int32_t>(t0->nelements() / fine_embd_dim_);
    } else {
        fine_embd_dim_ = 1024;
    }

    for (int i = 0; i < n_fine_books_; i++) {
        char name[256];

        // codec_embd.{i}.weight [fine_vocab, fine_embd_dim]
        std::snprintf(name, sizeof(name), "codec_embd.%d.weight", i);
        fine_embd_[i] = materialize_f32(reader, name);

        // lm_head.{i}.weight [fine_embd_dim, fine_vocab]
        std::snprintf(name, sizeof(name), "lm_head.%d.weight", i);
        fine_head_[i] = materialize_f32(reader, name);
    }

    small_to_mtp_w_ = materialize_f32(reader, "small_to_mtp.weight");
    small_to_mtp_b_ = materialize_f32(reader, "small_to_mtp.bias");
    // Capture the small_to_mtp weight shape so the projection loop knows
    // the talker hidden dim (input). Without this, the loop would use
    // `hidden_size_` (predictor dim) for both dims, which only works on
    // 0.6B where talker==predictor==1024. On 1.7B (talker=2048,
    // predictor=1024) the mismatch drops half the input and uses the wrong
    // row stride → garbage fine codebooks → unintelligible audio.
    if (small_to_mtp_w_) {
        const TensorStorage* ts = reader.find("small_to_mtp.weight");
        if (ts && ts->n_dims >= 2) {
            // The converter (convert_qwen3tts.cpp add_tensor_2d) stores
            // PyTorch nn.Linear weights in their native [out, in] row-major
            // layout. In ggml ne convention that maps to ne[0]=in_features
            // (innermost, contiguous) and ne[1]=out_features.
            //
            // For 1.7B: weight shape is [out=1024, in=2048] in PyTorch →
            //   ne[0] = in_features  = 2048
            //   ne[1] = out_features = 1024
            // which matches the bias length (1024).
            //
            // (Earlier code had these swapped, projecting 1024 → 2048
            //  instead of 2048 → 1024 and producing garbage fine codebooks
            //  on 1.7B. Verified correct via bias length = 1024.)
            small_to_mtp_in_  = static_cast<int32_t>(ts->ne[0]);  // PyTorch in_features
            small_to_mtp_out_ = static_cast<int32_t>(ts->ne[1]);  // PyTorch out_features
        }
        // Safety fallback: if shape lookup failed, assume square (0.6B case).
        if (small_to_mtp_in_ == 0 || small_to_mtp_out_ == 0) {
            small_to_mtp_in_  = hidden_size_;
            small_to_mtp_out_ = hidden_size_;
        }
        std::fprintf(stderr,
            "qwen3::Runner: small_to_mtp shape = [%d in → %d out] "
            "(predictor hidden=%d)\n",
            small_to_mtp_in_, small_to_mtp_out_, hidden_size_);
    }

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

bool Runner::project_text_tokens(const int32_t* token_ids, int32_t n_tokens,
                                  float* out_embd, std::string* error) {
    if (!has_text_embd_) {
        if (error) *error = "runner: text_embedding tensors not loaded";
        return false;
    }
    if (n_tokens <= 0) return true;

    const int32_t text_dim = text_embd_dim_;
    const int32_t proj_dim = hidden_size_;

    std::vector<float> raw_embd(static_cast<size_t>(n_tokens) * text_dim);
    for (int32_t i = 0; i < n_tokens; i++) {
        const int32_t id = token_ids[static_cast<size_t>(i)];
        if (id >= 0 && id < text_vocab_) {
            const float* row = text_embd_ + static_cast<size_t>(id) * text_dim;
            std::memcpy(&raw_embd[static_cast<size_t>(i) * text_dim], row,
                        static_cast<size_t>(text_dim) * sizeof(float));
        } else {
            std::memset(&raw_embd[static_cast<size_t>(i) * text_dim], 0,
                        static_cast<size_t>(text_dim) * sizeof(float));
        }
    }

    for (int32_t i = 0; i < n_tokens; i++) {
        const float* x = &raw_embd[static_cast<size_t>(i) * text_dim];
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

        float* y = out_embd + static_cast<size_t>(i) * proj_dim;
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
    return true;
}

bool Runner::project_single_token(int32_t token_id, float* out_embd,
                                   std::string* error) {
    return project_text_tokens(&token_id, 1, out_embd, error);
}

// ── Predictor MTP step (predict_one_step) ───────────────────────────────────
//
// Ported verbatim from the former PredictorRunner::predict_one_step. At each
// sub-step k (0..n_fine_books-1), predict fine codebook (k+1) by conditioning
// on the talker's projected hidden state + all previously emitted codes.

bool Runner::predict_one_step(const float* talker_hidden,
                                const float* layer0_embed,
                                int32_t code_0,
                                int32_t n_fine_books,
                                int32_t* out_codes,
                                std::string* error,
                                float temperature,
                                float top_p,
                                int32_t top_k) {
    if (!has_mtp_) {
        if (error) *error = "runner: MTP tensors not loaded";
        return false;
    }
    if (n_fine_books != n_fine_books_) {
        if (error) *error = "runner: predict_one_step n_fine_books mismatch";
        return false;
    }

    const int32_t ne = hidden_size_;   // predictor hidden_size
    const int32_t fv = fine_vocab_;    // 2048
    const int32_t nf = n_fine_books_;  // 15 (num_code_groups - 1)

    // Project talker hidden through small_to_mtp → position 0.
    // The weight is stored as [in_features=talker_n_embd, out_features=ne]
    // in ggml's ne[0]=innermost convention — so the flat row-major layout
    // is `out_features` rows of `in_features` elements each. For 1.7B that
    // is [2048 in → 1024 out]; for 0.6B it is [1024 → 1024]. Using `ne` for
    // both dims is the historic 1.7B bug.
    std::vector<float> proj(static_cast<size_t>(ne));
    if (small_to_mtp_w_) {
        const int32_t in_dim  = small_to_mtp_in_;   // talker hidden
        const int32_t out_dim = small_to_mtp_out_;  // predictor hidden (== ne)
        for (int j = 0; j < out_dim; j++) {
            float s = small_to_mtp_b_ ? small_to_mtp_b_[j] : 0.0f;
            const float* w_row = small_to_mtp_w_ +
                                 static_cast<size_t>(j) * in_dim;
            for (int k = 0; k < in_dim; k++)
                s += talker_hidden[k] * w_row[k];
            proj[static_cast<size_t>(j)] = s;
        }
    } else {
        std::memcpy(proj.data(), talker_hidden, static_cast<size_t>(ne) * sizeof(float));
    }

    std::vector<float> seq;
    seq.reserve(static_cast<size_t>(nf + 2) * ne);

    PhiloxRng rng{42};

    // cur_codes stores fine codes generated WITHIN this MTP step.
    // cur_codes[k] = fine[k], the k-th fine codebook sampled at sub-step k.
    // The predictor is purely within-step autoregressive: it does NOT use
    // previous AR steps' fine codes. Only code_0 (from the talker) conditions
    // the sequence via layer0_embed at position 1.
    std::vector<int32_t> cur_codes(static_cast<size_t>(nf), 0);

    static const bool dbg = std::getenv("QWEN3TTS_DEBUG");

    // Helper: project a talker-hidden vector through small_to_mtp down to the
    // predictor's hidden size (ne). For 0.6B talker and predictor hidden are
    // both 1024 and small_to_mtp is absent, so this is identity. For 1.7B the
    // talker is 2048-d and the predictor is 1024-d; without this projection
    // every layer0_embed / fine_embd row is fed at the wrong dimension and
    // the predictor emits garbage fine codes that destabilize the talker AR
    // loop. Matches the canonical vllm-omni CodePredictorWrapper.forward:
    //   proj_buf[:,0]   = projection(last_talker_hidden)
    //   proj_buf[:,1]   = projection(layer0_embed)
    //   proj_buf[:,s+1] = projection(codec_embeds[s-1](fine[s-1]))
    auto project_down = [&](const float* src, std::vector<float>* out) {
        out->assign(static_cast<size_t>(ne), 0.0f);
        if (small_to_mtp_w_) {
            const int32_t in_dim = small_to_mtp_in_;   // talker hidden
            for (int j = 0; j < ne; j++) {
                float s = small_to_mtp_b_ ? small_to_mtp_b_[j] : 0.0f;
                const float* w_row = small_to_mtp_w_ +
                                     static_cast<size_t>(j) * in_dim;
                for (int k2 = 0; k2 < in_dim; k2++)
                    s += src[k2] * w_row[k2];
                (*out)[static_cast<size_t>(j)] = s;
            }
        } else {
            // No projection needed (talker and predictor share hidden size).
            for (int j = 0; j < ne; j++)
                (*out)[static_cast<size_t>(j)] = src[j];
        }
    };

    // layer0_embed is constant across sub-steps — project once outside the
    // k-loop. Its true dimension is the talker's codec_embd_dim (== talker
    // hidden), NOT the predictor hidden. Copying only `ne` floats without
    // projecting (the old behaviour) fed a truncated, unprojected vector.
    std::vector<float> layer0_proj;
    project_down(layer0_embed, &layer0_proj);

    // fine_embd rows are at fine_embd_dim_ (talker hidden for 1.7B), NOT at
    // `ne`. Use fine_embd_dim_ as the row stride, then project before append.
    const int32_t fe_dim = fine_embd_dim_;

    for (int k = 0; k < nf; k++) {
        // Build the predictor input sequence (all rows at predictor hidden):
        //   Position 0: proj(talker_hidden)
        //   Position 1: proj(layer0_embed)
        //   Position 2..k+1: proj(fine_embd_[i](fine[i]))  for i in [0,k)
        seq.clear();
        seq.insert(seq.end(), proj.begin(), proj.end());              // pos 0
        seq.insert(seq.end(), layer0_proj.begin(), layer0_proj.end());// pos 1
        for (int i = 0; i < k; i++) {                                  // pos 2..k+1
            int32_t cid = cur_codes[static_cast<size_t>(i)];
            if (cid < 0) cid = 0;
            if (cid >= fv) cid %= fv;
            const float* row = fine_embd_[static_cast<size_t>(i)] +
                               static_cast<size_t>(cid) * fe_dim;
            std::vector<float> row_proj;
            project_down(row, &row_proj);
            seq.insert(seq.end(), row_proj.begin(), row_proj.end());
        }

        const int32_t seq_len = k + 2;   // proj + layer0 + k fine codes

        if (dbg) std::fprintf(stderr, "  mtp[%d/%d]: seq_len=%d, clearing KV...\n", k, nf, seq_len);
        // Clear KV cache for each sub-step — the MTP loop rebuilds the full
        // sequence from position 0 each time, so stale KV entries from the
        // previous sub-step would conflict.
        {
            llama_memory_t mem = llama_get_memory(ctx_);
            if (mem) llama_memory_clear(mem, /*partial=*/false);
        }

        llama_batch b = make_batch(seq_len, /*n_pos=*/0,
                                   /*is_embd=*/true, seq.data(), nullptr,
                                   /*last_only=*/true, n_pos_per_embd_);
        const bool ok = (llama_decode(ctx_, b) == 0);
        b.embd = nullptr;  // caller-owned
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

        // Debug: dump prefill output (k=0) for comparison with Python
        if (std::getenv("QWEN3TTS_PRED_DUMP") && k == 0) {
            std::fprintf(stderr, "  [predict k=0] in_pos0[0:6]=[");
            for (int i = 0; i < 6; i++) std::fprintf(stderr, "%s%.4f", i?",":"", seq[i]);
            std::fprintf(stderr, "] in_pos1[0:6]=[");
            for (int i = 0; i < 6; i++) std::fprintf(stderr, "%s%.4f", i?",":"", seq[ne + i]);
            std::fprintf(stderr, "] out_last_h[0:6]=[");
            for (int i = 0; i < 6; i++) std::fprintf(stderr, "%s%.4f", i?",":"", last_h[i]);
            std::fprintf(stderr, "]\n");
        }

        // Predict fine[k] via fine_head_[k]
        std::vector<float> logits(static_cast<size_t>(fv));
        const float* head_w = fine_head_[static_cast<size_t>(k)];
        for (int j = 0; j < fv; j++) {
            float s = 0.0f;
            for (int d = 0; d < ne; d++)
                s += last_h[d] * head_w[static_cast<size_t>(j) * ne + d];
            logits[static_cast<size_t>(j)] = s;
        }

        if (dbg) {
            // Find top-5 logits for diagnostics
            std::vector<std::pair<float,int32_t>> sorted;
            for (int j = 0; j < fv; j++)
                sorted.emplace_back(logits[static_cast<size_t>(j)], j);
            std::partial_sort(sorted.begin(), sorted.begin()+5, sorted.end(),
                              [](auto& a, auto& b){ return a.first > b.first; });
            std::fprintf(stderr, "  mtp[%d/%d]: top-5:", k, nf);
            for (int kk = 0; kk < 5; kk++)
                std::fprintf(stderr, " [%d]=%.3f", sorted[kk].second, sorted[kk].first);
            std::fprintf(stderr, "\n");
        }

        // Sample: use caller-supplied params (default greedy). The reference
        // Python code_predictor.generate defaults to do_sample=True with
        // temperature=0.9, top_k=50, top_p=1.0, but those random fine codes
        // destabilize the talker's AR feedback loop in C++ inference and
        // collapse it into cb0 repetition. Greedy (temp=0) matches Python's
        // do_sample=False and produces stable, intelligible output.
        const int32_t sampled = [&] {
            Params sp;
            sp.temperature = temperature;
            sp.top_k       = top_k;
            sp.top_p       = top_p;
            return sample_token(logits.data(), fv, sp,
                                /*prev_tokens=*/nullptr, /*n_prev=*/0, &rng);
        }();
        out_codes[static_cast<size_t>(k)] = sampled;
        cur_codes[static_cast<size_t>(k)] = sampled;
        // Debug: dump first frame's per-step code
        if (std::getenv("QWEN3TTS_CODES") && k < 5) {
            std::fprintf(stderr, "  [predict k=%d] c_%d=%d\n", k, k+1, sampled);
        }
    }

    return true;
}

bool Runner::clear_kv_cache(std::string* error) {
    llama_memory_t mem = llama_get_memory(ctx_);
    if (!mem) {
        if (error) *error = "clear_kv_cache: llama_get_memory returned null";
        return false;
    }
    // Remove all tokens of all sequences (seq_id < 0 = any sequence,
    // p0 = 0, p1 = -1 = infinity) to effectively clear the KV cache.
    return llama_memory_seq_rm(mem, -1, 0, -1);
}

// ── Secondary context (for CFG) ────────────────────────────────────────────
//
// ACE-Step's LM uses classifier-free guidance by default (lm_cfg_scale=2.0).
// Each code step requires BOTH a conditional and unconditional forward pass.
// The secondary context shares the model weights (no extra weight memory)
// but has its own independent KV cache.

bool Runner::init_secondary_context(std::string* error) {
    if (ctx_secondary_) return true;  // already initialized
    if (!model_) {
        if (error) *error = "init_secondary_context: no model loaded";
        return false;
    }
    // Clone the context params from the primary context.
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = llama_n_ctx(ctx_);
    cp.n_batch         = llama_n_batch(ctx_);
    // Match threading of primary (single-thread is fine for batch=1 decode).
    cp.n_threads       = 1;
    cp.n_threads_batch = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.embeddings      = true;
    cp.no_perf         = true;

    ctx_secondary_ = llama_init_from_model(model_, cp);
    if (!ctx_secondary_) {
        if (error) *error = "init_secondary_context: llama_init_from_model failed";
        return false;
    }
    return true;
}

bool Runner::forward_tokens_secondary(const int32_t* tokens, int32_t n_tokens,
                                      int32_t n_pos, float* logits,
                                      std::string* error) {
    if (!ctx_secondary_) {
        if (error) *error = "forward_tokens_secondary: no secondary context "
                            "(call init_secondary_context first)";
        return false;
    }
    llama_batch b = make_batch(n_tokens, n_pos, /*is_embd=*/false,
                               nullptr, reinterpret_cast<const llama_token*>(tokens),
                               /*last_only=*/false, n_pos_per_embd_);
    const int32_t n_vocab = vocab_size_;
    bool ok = (llama_decode(ctx_secondary_, b) == 0);
    if (!ok) {
        if (error) *error = "llama_decode (secondary) failed";
    } else if (logits) {
        for (int32_t i = 0; i < n_tokens && ok; i++) {
            const float* row = llama_get_logits_ith(ctx_secondary_, i);
            if (!row) {
                if (error) *error = "llama_get_logits_ith (secondary) failed";
                ok = false;
                break;
            }
            std::memcpy(logits + static_cast<size_t>(i) * n_vocab, row,
                        static_cast<size_t>(n_vocab) * sizeof(float));
        }
    }
    b.token = nullptr;
    b.embd  = nullptr;
    llama_batch_free(b);
    return ok;
}

bool Runner::clear_secondary_kv_cache(std::string* error) {
    if (!ctx_secondary_) {
        if (error) *error = "clear_secondary_kv_cache: no secondary context";
        return false;
    }
    llama_memory_t mem = llama_get_memory(ctx_secondary_);
    if (!mem) {
        if (error) *error = "clear_secondary_kv_cache: llama_get_memory null";
        return false;
    }
    return llama_memory_seq_rm(mem, -1, 0, -1);
}

}  // namespace audiocore::qwen3
