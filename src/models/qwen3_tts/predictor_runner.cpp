// predictor_runner.cpp — Code Predictor (qwen3tts_cp) inference via llama.cpp.

#include "audiocore/models/qwen3_tts/predictor_runner.h"

#include "llama.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace audiocore::qwen3_tts {

PredictorRunner::PredictorRunner() = default;

PredictorRunner::~PredictorRunner() {
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

std::unique_ptr<PredictorRunner> PredictorRunner::load(const std::string& gguf_path,
                                                        const PredictorConfig& cfg,
                                                        std::string* error) {
    auto self = std::unique_ptr<PredictorRunner>(new PredictorRunner());

    self->n_codebooks_  = cfg.n_codebooks;
    self->n_fine_books_ = cfg.n_fine_books;

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = cfg.n_gpu_layers;
    mp.main_gpu     = cfg.main_gpu;
    mp.use_mmap     = true;
    mp.use_mlock    = false;

    self->model_ = llama_model_load_from_file(gguf_path.c_str(), mp);
    if (!self->model_) {
        if (error) *error = "predictor: llama_model_load_from_file failed: " + gguf_path;
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
    cp.embeddings = true;  // we need hidden states for manual lm_head
    cp.no_perf    = true;

    self->ctx_ = llama_init_from_model(self->model_, cp);
    if (!self->ctx_) {
        if (error) *error = "predictor: llama_init_from_model failed";
        llama_model_free(self->model_);
        self->model_ = nullptr;
        return nullptr;
    }

    self->n_embd_     = (int32_t)llama_model_n_embd(self->model_);
    const llama_vocab* vocab = llama_model_get_vocab(self->model_);
    self->vocab_size_ = vocab ? (int32_t)llama_vocab_n_tokens(vocab) : 0;

    // Try loading MTP tensors (codec_embd.*, lm_head.*)
    if (!self->load_mtp_tensors(gguf_path, error)) {
        std::fprintf(stderr, "predictor: MTP tensors not found, using standard mode\n");
        if (error) error->clear();
    }

    // Pre-allocate workspace
    self->scratch_embd_.resize((size_t)cfg.n_ctx * (size_t)self->n_embd_);
    self->scratch_logits_.resize((size_t)self->n_fine_books_ * 8192);  // max fine_vocab ~2048

    return self;
}

// ── Raw GGUF tensor loading ────────────────────────────────────────────────

struct TensorRef {
    int64_t  tid;
    size_t   size;
    size_t   offset;
    ggml_type type;
};

static TensorRef find_tensor(gguf_context* ctx, const char* name) {
    int64_t n = gguf_get_n_tensors(ctx);
    for (int64_t i = 0; i < n; i++) {
        if (strcmp(gguf_get_tensor_name(ctx, i), name) == 0) {
            return {i, gguf_get_tensor_size(ctx, i),
                    gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, i),
                    gguf_get_tensor_type(ctx, i)};
        }
    }
    return {-1, 0, 0, GGML_TYPE_F32};
}

static float* gguf_load_f32(const char* path, const char* name) {
    gguf_init_params params = {true, nullptr};
    gguf_context* ctx = gguf_init_from_file(path, params);
    if (!ctx) return nullptr;

    TensorRef tr = find_tensor(ctx, name);
    float* result = nullptr;
    if (tr.tid >= 0 && tr.size > 0) {
        int nf = (int)(tr.size / sizeof(float));
        result = new float[nf];
        FILE* fp = fopen(path, "rb");
        if (fp) {
            fseek(fp, (long)tr.offset, SEEK_SET);
            if (fread(result, 1, tr.size, fp) != tr.size) { delete[] result; result = nullptr; }
            fclose(fp);
        } else { delete[] result; result = nullptr; }
    }
    gguf_free(ctx);
    return result;
}

bool PredictorRunner::load_mtp_tensors(const std::string& gguf_path, std::string* error) {
    const char* path = gguf_path.c_str();

    // Count how many codec_embd.{i}.weight tensors exist
    int found = 0;
    for (int i = 0; i < n_fine_books_; i++) {
        char name[256];
        std::snprintf(name, sizeof(name), "codec_embd.%d.weight", i);
        float* t = gguf_load_f32(path, name);
        if (t) {
            found++;
            delete[] t;
        }
    }

    if (found == 0) {
        // No MTP tensors — this is a Lunavox-style GGUF
        return true;
    }

    has_mtp_ = true;
    fine_embd_ = new float*[(size_t)n_fine_books_]();
    fine_head_ = new float*[(size_t)n_fine_books_]();

    for (int i = 0; i < n_fine_books_; i++) {
        char name[256];

        // codec_embd.{i}.weight [fine_vocab, 1024] in GGUF: ne[0]=1024, ne[1]=fine_vocab
        std::snprintf(name, sizeof(name), "codec_embd.%d.weight", i);
        fine_embd_[i] = gguf_load_f32(path, name);
        if (i == 0 && fine_embd_[0]) {
            // Infer fine_vocab_ from known embedding dim
            // We read it as a flat float[], total count = n_embd * fine_vocab
            // but we don't track count directly — compute from a known size probe
            // (see probe below)
        }

        // lm_head.{i}.weight [1024, fine_vocab] in GGUF: ne[0]=1024 (wait — actually
        // for lm_head: shape [1024, fine_vocab], so ne[0]=fine_vocab, ne[1]=1024)
        std::snprintf(name, sizeof(name), "lm_head.%d.weight", i);
        fine_head_[i] = gguf_load_f32(path, name);
    }

    // Infer fine_vocab_ from tensor metadata using the GGUF file directly.
    // We re-open the file briefly to read the tensor dimensions.
    {
        gguf_init_params params = {true, nullptr};
        gguf_context* ctx = gguf_init_from_file(path, params);
        if (ctx) {
            int64_t n = gguf_get_n_tensors(ctx);
            for (int64_t i = 0; i < n; i++) {
                const char* tn = gguf_get_tensor_name(ctx, i);
                if (tn && strcmp(tn, "codec_embd.0.weight") == 0) {
                    size_t sz = gguf_get_tensor_size(ctx, i);
                    // shape: [fine_vocab, 1024], ne[0]=1024, ne[1]=fine_vocab
                    // total elems = sz / sizeof(float) = 1024 * fine_vocab
                    fine_vocab_ = (int32_t)(sz / (sizeof(float) * 1024));
                    break;
                }
            }
            gguf_free(ctx);
        }
    }

    // small_to_mtp.weight [+ bias]
    small_to_mtp_w_ = gguf_load_f32(path, "small_to_mtp.weight");
    small_to_mtp_b_ = gguf_load_f32(path, "small_to_mtp.bias");

    return true;
}

// ── Batch builders ─────────────────────────────────────────────────────────

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

static llama_batch make_token_batch(const int32_t* tokens, int32_t n_tokens,
                                    int32_t n_pos) {
    llama_batch b = llama_batch_init(n_tokens, /*embd=*/0, /*n_seq_max=*/1);
    b.n_tokens = n_tokens;
    b.token = const_cast<int32_t*>(tokens);
    b.embd  = nullptr;
    for (int32_t i = 0; i < n_tokens; i++) {
        b.pos[i] = n_pos + i;
        b.logits[i] = (i == n_tokens - 1) ? 1 : 0;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
    }
    return b;
}

// ── Sampling ───────────────────────────────────────────────────────────────

static int32_t sample_token(const float* logits, int32_t n_vocab,
                            float temp, float top_p, std::mt19937& rng) {
    if (n_vocab <= 0) return 0;
    std::vector<float> probs((size_t)n_vocab);
    float max_l = -1e38f;
    for (int i = 0; i < n_vocab; i++) if (logits[i] > max_l) max_l = logits[i];
    float sum = 0;
    for (int i = 0; i < n_vocab; i++) {
        probs[(size_t)i] = std::exp((logits[i] - max_l) / std::max(temp, 1e-6f));
        sum += probs[(size_t)i];
    }

    if (top_p < 1.0f && top_p > 0.0f && sum > 0) {
        std::vector<int32_t> idx((size_t)n_vocab);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b) { return probs[(size_t)a] > probs[(size_t)b]; });
        float cum = 0;
        std::vector<bool> keep((size_t)n_vocab, false);
        for (auto ix : idx) {
            cum += probs[(size_t)ix] / sum;
            keep[(size_t)ix] = true;
            if (cum >= top_p) break;
        }
        for (size_t i = 0; i < (size_t)n_vocab; i++) if (!keep[i]) probs[i] = 0;
        sum = 0;
        for (size_t i = 0; i < (size_t)n_vocab; i++) sum += probs[i];
    }

    if (sum <= 0) return 0;
    std::uniform_real_distribution<float> dist(0, sum);
    float sample = dist(rng);
    float cum = 0;
    for (int i = 0; i < n_vocab; i++) {
        cum += probs[(size_t)i];
        if (sample <= cum) return i;
    }
    return n_vocab - 1;
}

// ── Standard token-mode forward ────────────────────────────────────────────

bool PredictorRunner::forward_tokens(const int32_t* tokens, int32_t n_tokens,
                                      int32_t n_pos, float* logits,
                                      std::string* error) {
    llama_batch b = make_token_batch(tokens, n_tokens, n_pos);
    bool ok = (llama_decode(ctx_, b) == 0);
    if (ok) {
        for (int32_t i = 0; i < n_tokens; i++) {
            const float* row = llama_get_logits_ith(ctx_, i);
            if (!row) { ok = false; break; }
            std::memcpy(logits + (size_t)i * vocab_size_, row,
                        (size_t)vocab_size_ * sizeof(float));
        }
    } else {
        if (error) *error = "predictor: llama_decode failed";
    }
    b.token = nullptr;
    llama_batch_free(b);
    return ok;
}

// ── MTP prediction loop (official Qwen3-TTS) ──────────────────────────────
//
// At each generation step, the code predictor generates 31 fine codebooks
// autoregressively. Sub-step k (k=0..30) predicts codebook (k+1):
//
//   Input sequence: [talker_hidden_proj, code_0_embd, code_1_embd, ..., code_k_embd]
//   where code_i_embd = fine_embd_[i](sampled_code_i)
//   → transformer → last_hidden → lm_head[k] → logits → sample → code_{k+1}
//
// Each sub-step rebuilds the full sequence (up to 33 tokens) so embedding
// changes between sub-steps are correctly reflected. For such short sequences
// the overhead is negligible.

bool PredictorRunner::predict_one_step(const float* talker_hidden,
                                        const int32_t* prev_codes,
                                        int32_t* out_codes,
                                        std::string* error) {
    if (!has_mtp_) {
        if (error) *error = "predictor: MTP tensors not loaded";
        return false;
    }

    const int32_t ne = n_embd_;
    const int32_t fv = fine_vocab_;
    const int32_t nf = n_fine_books_;

    // Project talker hidden through small_to_mtp
    std::vector<float> proj((size_t)ne);
    if (small_to_mtp_w_) {
        for (int j = 0; j < ne; j++) {
            float s = small_to_mtp_b_ ? small_to_mtp_b_[j] : 0.0f;
            for (int k = 0; k < ne; k++) s += talker_hidden[k] * small_to_mtp_w_[(size_t)j * ne + k];
            proj[(size_t)j] = s;
        }
    } else {
        std::memcpy(proj.data(), talker_hidden, (size_t)ne * sizeof(float));
    }

    // Temporary sequence buffer: [talker_hidden_proj, code0_embd, code1_embd, ...]
    std::vector<float> seq;
    seq.reserve((size_t)(nf + 2) * ne);

    std::mt19937 rng(42);

    // Track the sampled fine codes; initially use prev_codes for coarse + fine
    std::vector<int32_t> cur_codes(prev_codes, prev_codes + (size_t)nf);

    for (int k = 0; k < nf; k++) {
        // Build full sequence:
        // [proj, embd_0(cur_codes[0]), embd_0(cur_codes[0]), ..., embd_k(cur_codes[k])]
        seq.clear();
        seq.insert(seq.end(), proj.begin(), proj.end());

        // Attention: For k=0, we embed prev_codes[0] (the coarse code from talker)
        // using fine_embd_[0]. The official model uses codec_embedding from the
        // talker for codebook 0, not fine_embd_[0]. But fine_embd_[0] is a
        // separate embedding for the first fine codebook that may differ from
        // the talker's codec_embedding. For now we use fine_embd_[0] with
        // prev_codes[0].
        for (int i = 0; i <= k; i++) {
            int32_t cid = cur_codes[(size_t)i];
            if (cid < 0) cid = 0;
            if (cid >= fv) cid = cid % fv;
            const float* row = fine_embd_[i] + (size_t)cid * ne;
            seq.insert(seq.end(), row, row + ne);
        }

        // Run transformer
        int32_t seq_len = k + 2;  // proj + k+1 code embeddings
        llama_batch b = make_embd_batch(seq.data(), seq_len, 0);
        bool ok = (llama_decode(ctx_, b) == 0);
        b.embd = nullptr;
        llama_batch_free(b);

        if (!ok) {
            if (error) *error = "predictor: MTP sub-step decode failed";
            return false;
        }

        // Get last token's hidden state
        const float* last_h = llama_get_embeddings_ith(ctx_, seq_len - 1);
        if (!last_h) {
            if (error) *error = "predictor: MTP embeddings unavailable";
            return false;
        }

        // Apply lm_head[k]: hidden [ne] → logits [fv]
        std::vector<float> logits((size_t)fv);
        const float* head_w = fine_head_[k];
        for (int j = 0; j < fv; j++) {
            float s = 0.0f;
            for (int d = 0; d < ne; d++) s += last_h[d] * head_w[(size_t)j * ne + d];
            logits[(size_t)j] = s;
        }

        // Sample
        int32_t sampled = sample_token(logits.data(), fv, 0.7f, 0.9f, rng);
        out_codes[k] = sampled;

        // Update cur_codes for the next sub-step
        cur_codes[(size_t)k] = sampled;
    }

    return true;
}

const float* PredictorRunner::get_logits_ith(int32_t i) const {
    return llama_get_logits_ith(ctx_, i);
}

}  // namespace audiocore::qwen3_tts
