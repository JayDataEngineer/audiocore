// session.cpp — ACE-Step run_music full pipeline.
//
// Two-step pipeline (ServeurpersoCom/acestep.cpp):
//   Step 1 — run_lm:  caption+lyrics → TE embeddings → 5Hz LM → music codes
//   Step 2 — run_dit_and_vae: FSQ → noise → DiT flow matching → VAE → PCM
//
// Both Qwen3 transformers run through the unified qwen3::Runner (libllama).
// DiT and VAE are ggml_cgraph builds via DiTRunner and VAERunner.

#include "audiocore/models/ace_step/family.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace audiocore::acestep {

// ═════════════════════════════════════════════════════════════════════════════
//  FSQ: Finite Scalar Quantization — decode single code → 6D vector
// ═════════════════════════════════════════════════════════════════════════════
//
// Levels match the upstream: [8, 8, 8, 5, 5, 5] = 64 000 codes.
// Mixed-radix decomposition:
//   code = c0 + c1·8 + c2·64 + c3·512 + c4·2560 + c5·12800
// where cumulative products are L0, L0·L1, L0·L1·L2, … = 1, 8, 64, 512, 2560, 12800.
// Each ci ∈ [0, level_i) → mapped to [-1, 1] as  2·ci/(level_i−1) − 1.

static void fsq_decode_one(int32_t code, float out[6]) {
    static const int levels[6] = {8, 8, 8, 5, 5, 5};
    int tmp = code;
    for (int i = 0; i < 6; i++) {
        int ci = tmp % levels[i];
        tmp /= levels[i];
        out[i] = 2.0f * static_cast<float>(ci) / (levels[i] - 1) - 1.0f;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Flow-matching Euler schedule
// ═════════════════════════════════════════════════════════════════════════════
//
// Returns the schedule as a vector of timesteps (descending) plus the step
// count.  The caller integrates  x_{t-dt} = x_t + dt · v(x_t, t)  with
// dt = t[i] − t[i+1] (or t[i] − 0 for the last step).

struct FlowSchedule {
    std::vector<float> timesteps;   // descending, [1.0 … ~0.0]
    int                n_steps;     // convenience alias
};

static FlowSchedule build_schedule(const std::string& variant, int override_steps) {
    FlowSchedule s;

    // ── Turbo: shifted-cosine, 8 steps, shift=3.0 ──────────────────────────
    if ((variant == "turbo" || variant.empty()) &&
        (override_steps <= 0 || override_steps == 8)) {
        // Verified against upstream acestep.cpp (turbo mode)
        static const float tbl[] = {
            1.0f, 0.955f, 0.9f, 0.833f, 0.75f, 0.643f, 0.5f, 0.3f
        };
        s.timesteps.assign(tbl, tbl + 8);
        s.n_steps = 8;
        return s;
    }

    // ── SFT: linear schedule, 50 steps, shift=1.0 ──────────────────────────
    if (variant == "sft" && (override_steps <= 0 || override_steps == 50)) {
        s.timesteps.reserve(50);
        for (int i = 0; i < 50; i++) {
            s.timesteps.push_back(0.98f - i * (0.96f / 49.0f));
        }
        s.n_steps = 50;
        return s;
    }

    // ── Custom: uniform spacing ────────────────────────────────────────────
    const int steps = (override_steps > 0) ? override_steps : 8;
    s.timesteps.reserve(steps);
    for (int i = 0; i < steps; i++) {
        s.timesteps.push_back(1.0f - static_cast<float>(i) / steps);
    }
    s.n_steps = steps;
    return s;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Helper: manual MatMul + bias for a 2D weight tensor in ggml layout
// ═════════════════════════════════════════════════════════════════════════════
//
// ggml stores weights column-major: ne[0] = out_features, ne[1] = in_features.
// This computes  y[t, o] = sum_i x[t, i] · W[o, i] + b[o].

static void manual_linear(const float* x, int32_t T, int32_t in_dim,
                          const ggml_tensor* w, const float* b,
                          int32_t out_dim, float* y) {
    if (!w) {
        // Identity-like: copy first min(in_dim, out_dim) channels
        const int32_t copy = std::min(in_dim, out_dim);
        for (int32_t t = 0; t < T; t++) {
            std::memcpy(&y[static_cast<size_t>(t) * out_dim],
                        &x[static_cast<size_t>(t) * in_dim],
                        static_cast<size_t>(copy) * sizeof(float));
            // Zero the rest
            if (out_dim > copy) {
                std::memset(&y[static_cast<size_t>(t) * out_dim + copy],
                            0, static_cast<size_t>(out_dim - copy) * sizeof(float));
            }
        }
        return;
    }

    const float* wd = static_cast<const float*>(w->data);
    const int32_t w_out = static_cast<int32_t>(w->ne[0]);  // ne[0] = out_features
    const int32_t w_in  = static_cast<int32_t>(w->ne[1]);  // ne[1] = in_features
    const int32_t eff_in  = std::min(in_dim, w_in);
    const int32_t eff_out = std::min(out_dim, w_out);

    for (int32_t t = 0; t < T; t++) {
        for (int32_t o = 0; o < eff_out; o++) {
            float sum = 0.0f;
            for (int32_t i = 0; i < eff_in; i++) {
                // column-major: W[o, i] = wd[o + i * w_out]
                sum += x[static_cast<size_t>(t) * in_dim + i] *
                       wd[static_cast<size_t>(o) + static_cast<size_t>(i) * w_out];
            }
            y[static_cast<size_t>(t) * out_dim + o] = sum + (b ? b[o] : 0.0f);
        }
        // Zero remaining output dims
        if (eff_out < out_dim) {
            std::memset(&y[static_cast<size_t>(t) * out_dim + eff_out],
                        0, static_cast<size_t>(out_dim - eff_out) * sizeof(float));
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  run_lm — LM pipeline (step 1 of 2)
// ═════════════════════════════════════════════════════════════════════════════
//
// 1. Tokenize caption (+ optional lyrics).
// 2. TE encode → hidden states cached for DiT conditioning.
// 3. LM prefill with text tokens.
// 4. LM autoregressive decode at 5 Hz (no CFG on LM during step 1 — quality
//    comes from the DiT-side CFG in step 2).  Each LM token in the code
//    vocabulary range [vocab_size − 64000, vocab_size) is a music code.

bool AceStepSession::run_lm(const MusicRequest& req,
                            std::vector<int32_t>* music_codes,
                            std::string* error) {
    // (1) Assemble the text prompt
    std::string prompt = req.caption;
    if (!req.lyrics.empty()) {
        prompt += "\nLyrics: " + req.lyrics;
    }

    // (2) Tokenize for TE and LM
    std::vector<int32_t> te_tokens;
    if (!te_->tokenize(prompt, /*add_special=*/true, /*parse_special=*/true,
                       &te_tokens, nullptr, error))
        return false;

    std::vector<int32_t> lm_prompt;
    if (!lm_->tokenize(prompt, /*add_special=*/true, /*parse_special=*/true,
                       &lm_prompt, nullptr, error))
        return false;

    if (te_tokens.empty() || lm_prompt.empty()) {
        if (error) *error = "ACE-Step: tokenization produced empty result";
        return false;
    }

    // (3) TE encode → hidden states cached for run_dit_and_vae
    const int32_t te_hs = cfg_.encoder_hidden_size;
    te_cond_len_ = static_cast<int32_t>(te_tokens.size());
    te_cond_.resize(static_cast<size_t>(te_cond_len_) * te_hs);
    if (!te_->forward_get_embeddings(te_tokens.data(), te_cond_len_, 0,
                                     te_cond_.data(), error))
        return false;

    // (4) Null-text TE embedding for CFG (DiT side)
    {
        std::vector<int32_t> null_tok;
        if (!te_->tokenize("", true, true, &null_tok, nullptr, error))
            return false;
        const int32_t n_null = static_cast<int32_t>(null_tok.size());
        if (n_null > 0) {
            te_uncond_.resize(static_cast<size_t>(n_null) * te_hs);
            if (!te_->forward_get_embeddings(null_tok.data(), n_null, 0,
                                             te_uncond_.data(), error))
                return false;
        } else {
            te_uncond_.clear();  // triggers null_condition_emb fallback in DiT
        }
    }

    // (5) LM prefill with text tokens (populates KV cache for positions 0..N-1)
    const int32_t prompt_len = static_cast<int32_t>(lm_prompt.size());
    if (!lm_->forward_tokens(lm_prompt.data(), prompt_len, 0, nullptr, error))
        return false;

    // (6) Decode loop: generate N codes at 5 Hz
    const int32_t n_codes = std::max(1, static_cast<int32_t>(req.duration * 5.0f + 0.5f));
    music_codes->clear();
    music_codes->reserve(static_cast<size_t>(n_codes));

    const int32_t code_vocab_size = 64000;
    const int32_t code_start = std::max(0, lm_->vocab_size() - code_vocab_size);

    // Start token: the last prompt token (typically EOS/BOS).  The LM will
    // predict the first code token from this context.
    int32_t prev_token = lm_prompt.back();

    // RNG for stochastic sampling (seeded for reproducibility)
    std::mt19937 rng(static_cast<unsigned>(req.seed != 0 ? req.seed : 42));

    for (int32_t s = 0; s < n_codes; s++) {
        const int32_t n_pos = prompt_len + s;

        const int32_t vs = lm_->vocab_size();
        std::vector<float> logits(static_cast<size_t>(vs));

        if (!lm_->forward_tokens(&prev_token, 1, n_pos, logits.data(), error))
            return false;

        const float* code_logits = logits.data() + code_start;

        // Sample within the code vocabulary
        int32_t chosen = 0;
        if (req.temperature <= 0.0f) {
            // Argmax (deterministic)
            float best_val = -std::numeric_limits<float>::infinity();
            for (int32_t i = 0; i < code_vocab_size; i++) {
                if (code_logits[i] > best_val) {
                    best_val = code_logits[i];
                    chosen = i;
                }
            }
        } else {
            // Temperature-scaling + top-p nucleus sampling
            std::vector<float> scaled(static_cast<size_t>(code_vocab_size));
            float inv_temp = 1.0f / req.temperature;
            float max_l = -std::numeric_limits<float>::infinity();
            for (int32_t i = 0; i < code_vocab_size; i++) {
                float v = code_logits[i] * inv_temp;
                scaled[static_cast<size_t>(i)] = v;
                if (v > max_l) max_l = v;
            }
            // Stable softmax
            float sum_exp = 0.0f;
            for (int32_t i = 0; i < code_vocab_size; i++) {
                scaled[static_cast<size_t>(i)] = std::exp(scaled[static_cast<size_t>(i)] - max_l);
                sum_exp += scaled[static_cast<size_t>(i)];
            }
            float inv_sum = 1.0f / (sum_exp + 1e-8f);
            for (int32_t i = 0; i < code_vocab_size; i++)
                scaled[static_cast<size_t>(i)] *= inv_sum;

            // Top-p: sort descending, accumulate until cumulative >= top_p
            if (req.top_p < 1.0f) {
                std::vector<std::pair<float, int32_t>> sorted;
                sorted.reserve(static_cast<size_t>(code_vocab_size));
                for (int32_t i = 0; i < code_vocab_size; i++)
                    sorted.emplace_back(scaled[static_cast<size_t>(i)], i);
                std::sort(sorted.begin(), sorted.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });
                float top_cum = 0.0f;
                std::vector<bool> keep(static_cast<size_t>(code_vocab_size), false);
                for (auto& p : sorted) {
                    top_cum += p.first;
                    keep[static_cast<size_t>(p.second)] = true;
                    if (top_cum >= req.top_p) break;
                }
                for (int32_t i = 0; i < code_vocab_size; i++)
                    if (!keep[static_cast<size_t>(i)])
                        scaled[static_cast<size_t>(i)] = 0.0f;
                // Renormalize
                sum_exp = 0.0f;
                for (int32_t i = 0; i < code_vocab_size; i++)
                    sum_exp += scaled[static_cast<size_t>(i)];
                inv_sum = 1.0f / (sum_exp + 1e-8f);
                for (int32_t i = 0; i < code_vocab_size; i++)
                    scaled[static_cast<size_t>(i)] *= inv_sum;
            }

            // Multinomial sample
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            float r = dist(rng);
            float mn_cum = 0.0f;
            chosen = code_vocab_size - 1;
            for (int32_t i = 0; i < code_vocab_size; i++) {
                mn_cum += scaled[static_cast<size_t>(i)];
                if (r < mn_cum) { chosen = i; break; }
            }
        }

        music_codes->push_back(chosen);
        prev_token = code_start + chosen;
    }

    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
//  run_dit_and_vae — DiT flow matching + VAE decode (step 2 of 2)
// ═════════════════════════════════════════════════════════════════════════════
//
// 1. FSQ detokenize codes → 6D → upsampled latent at 25 Hz.
// 2. Initialize Gaussian noise in hidden_size space.
// 3. Euler flow-matching loop:
//      x = x + dt · DiT(x, t, TE_cond, TE_uncond, guidance_scale)
// 4. proj_out: [T, H] → [T, out_channels=64]
// 5. VAE decode → 48 kHz stereo PCM.

bool AceStepSession::run_dit_and_vae(const MusicRequest& req,
                                     const std::vector<int32_t>& music_codes,
                                     std::vector<float>* pcm_stereo,
                                     std::string* error) {
    // ── Dimensionality ───────────────────────────────────────────────────────
    const int32_t n_codes_5  = static_cast<int32_t>(music_codes.size());
    const int32_t n_frames   = n_codes_5 * 5;        // 5 Hz → 25 Hz
    const int32_t in_ch      = cfg_.dit.in_channels;  //  192
    const int32_t H          = cfg_.dit.hidden_size;  // 1024 (1.5B DiT)
    const int32_t out_ch     = cfg_.dit.out_channels; //   64 (acoustic latent)
    const int32_t enc_hs     = cfg_.encoder_hidden_size;  // 1024

    if (H <= 0 || in_ch <= 0 || out_ch <= 0) {
        if (error)
            *error = "ACE-Step: DitConfig not populated (H=" + std::to_string(H) +
                     ", in_ch=" + std::to_string(in_ch) +
                     ", out_ch=" + std::to_string(out_ch) + ")";
        return false;
    }

    // ── 1. FSQ decode: codes → 6D → upsample 5→25 Hz ──────────────────────
    // The upstream projects the 6D FSQ vectors through a learned MLP
    // (6 → 2048 → SiLU → LayerNorm → 64).  If the MLP weights exist in
    // ext_ctx_ (as vae.fsq_*), use them; otherwise fall back to identity.
    //
    // We store the result at 25 Hz as [n_frames, out_ch].
    std::vector<float> fsq_latent(static_cast<size_t>(n_frames) * out_ch, 0.0f);

    // Try learned FSQ projection weights (in ext_ctx_, bound as vae.fsq_*
    // or decoder.fsq_* — checked both conventions).
    ggml_tensor* fsq_proj_w = ggml_get_tensor(ext_ctx_, "decoder.fsq_proj.0.weight");
    if (!fsq_proj_w) fsq_proj_w = ggml_get_tensor(ext_ctx_, "vae.fsq_proj.0.weight");
    ggml_tensor* fsq_out_w  = ggml_get_tensor(ext_ctx_, "decoder.fsq_proj.2.weight");
    if (!fsq_out_w) fsq_out_w = ggml_get_tensor(ext_ctx_, "vae.fsq_proj.2.weight");
    ggml_tensor* fsq_norm_w = ggml_get_tensor(ext_ctx_, "decoder.fsq_proj.1.weight");
    if (!fsq_norm_w) fsq_norm_w = ggml_get_tensor(ext_ctx_, "vae.fsq_proj.1.weight");

    const bool has_fsq_mlp = (fsq_proj_w && fsq_out_w);

    // Per 5-Hz frame: decode → project/unpad → repeat 5× at 25 Hz
    for (int32_t i = 0; i < n_codes_5; i++) {
        float f6[6];
        fsq_decode_one(music_codes[static_cast<size_t>(i)], f6);

        float projected[64] = {0};

        if (has_fsq_mlp) {
            // Learned MLP: 6 → 2048 → SiLU → LayerNorm → 64
            const float* w1 = static_cast<const float*>(fsq_proj_w->data);
            const int32_t w1_dim = static_cast<int32_t>(fsq_proj_w->ne[0]);

            // Linear 6 → w1_dim (usually 2048)
            float h2048[2048] = {0};
            const int32_t mlp_hidden = std::min(2048, w1_dim);
            for (int j = 0; j < mlp_hidden; j++) {
                float s = 0.0f;
                for (int k = 0; k < 6; k++) {
                    s += f6[k] * w1[static_cast<size_t>(j) + static_cast<size_t>(k) * w1_dim];
                }
                // Bias if present
                ggml_tensor* fsq_proj_b = ggml_get_tensor(ext_ctx_, "decoder.fsq_proj.0.bias");
                if (!fsq_proj_b) fsq_proj_b = ggml_get_tensor(ext_ctx_, "vae.fsq_proj.0.bias");
                s += fsq_proj_b ? static_cast<const float*>(fsq_proj_b->data)[j] : 0.0f;
                h2048[static_cast<size_t>(j)] = s;
            }

            // SiLU activation
            for (int j = 0; j < mlp_hidden; j++) {
                float v = h2048[static_cast<size_t>(j)];
                h2048[static_cast<size_t>(j)] = v / (1.0f + std::exp(-v));
            }

            // LayerNorm (if weight available)
            if (fsq_norm_w) {
                const float* nw = static_cast<const float*>(fsq_norm_w->data);
                float mean = 0.0f, var = 0.0f;
                for (int j = 0; j < mlp_hidden; j++) mean += h2048[static_cast<size_t>(j)];
                mean /= mlp_hidden;
                for (int j = 0; j < mlp_hidden; j++) {
                    float d = h2048[static_cast<size_t>(j)] - mean;
                    var += d * d;
                }
                var /= mlp_hidden;
                const float inv_std = 1.0f / std::sqrt(var + 1e-5f);
                for (int j = 0; j < mlp_hidden; j++) {
                    h2048[static_cast<size_t>(j)] = (h2048[static_cast<size_t>(j)] - mean) * inv_std * nw[j];
                }
                // Bias
                ggml_tensor* fsq_norm_b = ggml_get_tensor(ext_ctx_, "decoder.fsq_proj.1.bias");
                if (!fsq_norm_b) fsq_norm_b = ggml_get_tensor(ext_ctx_, "vae.fsq_proj.1.bias");
                if (fsq_norm_b) {
                    const float* nb = static_cast<const float*>(fsq_norm_b->data);
                    for (int j = 0; j < mlp_hidden; j++)
                        h2048[static_cast<size_t>(j)] += nb[j];
                }
            }

            // Output projection: mlp_hidden → 64
            const float* w2 = static_cast<const float*>(fsq_out_w->data);
            const int32_t w2_out = std::min(64, static_cast<int32_t>(fsq_out_w->ne[0]));
            for (int j = 0; j < w2_out; j++) {
                float s = 0.0f;
                for (int k = 0; k < mlp_hidden; k++) {
                    s += h2048[static_cast<size_t>(k)] *
                         w2[static_cast<size_t>(j) + static_cast<size_t>(k) * static_cast<size_t>(fsq_out_w->ne[0])];
                }
                projected[static_cast<size_t>(j)] = s;
            }
            // Bias
            ggml_tensor* fsq_out_b = ggml_get_tensor(ext_ctx_, "decoder.fsq_proj.2.bias");
            if (!fsq_out_b) fsq_out_b = ggml_get_tensor(ext_ctx_, "vae.fsq_proj.2.bias");
            if (fsq_out_b) {
                const float* ob = static_cast<const float*>(fsq_out_b->data);
                for (int j = 0; j < w2_out; j++)
                    projected[static_cast<size_t>(j)] += ob[j];
            }
        } else {
            // No learned weights: simple pad (6 → 64) with the FSQ values
            for (int d = 0; d < 6 && d < out_ch; d++)
                projected[static_cast<size_t>(d)] = f6[d];
        }

        // Repeat each 5 Hz frame 5× to get 25 Hz
        for (int j = 0; j < 5; j++) {
            std::memcpy(&fsq_latent[static_cast<size_t>(i * 5 + j) * out_ch],
                        projected, static_cast<size_t>(out_ch) * sizeof(float));
        }
    }

    // ── 2. Prepare conditioning for DiT ─────────────────────────────────────
    if (te_cond_.empty() || te_cond_len_ <= 0) {
        if (error) *error = "ACE-Step: run_lm must execute before run_dit_and_vae";
        return false;
    }

    // ── 3. Initialize noise at [T, in_ch] and project to [T, H] ─────────────
    std::mt19937_64 noise_rng(static_cast<uint64_t>(
        req.seed != 0 ? req.seed : 42));
    std::normal_distribution<float> ndist(0.0f, 1.0f);

    // Raw noise: [n_frames, in_channels]
    std::vector<float> noise_raw(static_cast<size_t>(n_frames) * in_ch);
    for (auto& v : noise_raw) v = ndist(noise_rng);

    // proj_in: [T, in_ch] → [T, H]
    std::vector<float> x(static_cast<size_t>(n_frames) * H);
    {
        ggml_tensor* pi_b = ggml_get_tensor(ext_ctx_, "decoder.proj_in.1.bias");
        manual_linear(noise_raw.data(), n_frames, in_ch,
                      dit_proj_in_,
                      pi_b ? static_cast<const float*>(pi_b->data) : nullptr,
                      H, x.data());
    }

    // ── 4. DiT flow-matching Euler loop ─────────────────────────────────────
    const float* cond_ptr   = te_cond_.data();
    const int32_t T_cond    = te_cond_len_;
    const float* uncond_ptr = te_uncond_.empty() ? nullptr : te_uncond_.data();
    const int32_t T_uncond  = te_uncond_.empty() ? 0 :
                              static_cast<int32_t>(te_uncond_.size() / enc_hs);

    const FlowSchedule sched = build_schedule(cfg_.variant, req.n_diffusion_steps);
    std::vector<float> dit_out(static_cast<size_t>(n_frames) * H);

    for (int step = 0; step < sched.n_steps; step++) {
        const float t  = sched.timesteps[static_cast<size_t>(step)];
        const float dt = (step + 1 < sched.n_steps)
                             ? t - sched.timesteps[static_cast<size_t>(step + 1)]
                             : t;  // last step → go to 0

        if (!dit_runner_->forward(x.data(), t,
                                  cond_ptr, T_cond, enc_hs,
                                  uncond_ptr, T_uncond,
                                  req.guidance_scale,
                                  n_frames,  // n_patches = n_frames
                                  dit_out.data(), error))
            return false;

        // Euler step: x_{t-dt} = x_t + dt · v(x_t, t)
        for (size_t i = 0; i < x.size(); i++) {
            x[i] += dt * dit_out[i];
        }
    }

    // ── 5. proj_out: [T, H] → [T, out_ch] ──────────────────────────────────
    std::vector<float> latents(static_cast<size_t>(n_frames) * out_ch);
    {
        ggml_tensor* po_b = ggml_get_tensor(ext_ctx_, "decoder.proj_out.1.bias");
        manual_linear(x.data(), n_frames, H,
                      dit_proj_out_,
                      po_b ? static_cast<const float*>(po_b->data) : nullptr,
                      out_ch, latents.data());
    }

    // ── 6. VAE decode: [T, 64] → PCM stereo ────────────────────────────────
    if (!vae_runner_->decode(latents.data(), n_frames, pcm_stereo, error))
        return false;

    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
//  run_music — entry point
// ═════════════════════════════════════════════════════════════════════════════

bool AceStepSession::run_music(const void* request, void* response,
                               std::string* error) {
    if (!loaded_) {
        if (error) *error = "AceStepSession not loaded";
        return false;
    }
    const auto* req = static_cast<const MusicRequest*>(request);
    auto*       res = static_cast<MusicResponse*>(response);
    if (!req || !res) {
        if (error) *error = "null request/response";
        return false;
    }
    res->sampling_rate = 48000;
    res->channels      = 2;

    std::vector<int32_t> music_codes;
    if (!run_lm(*req, &music_codes, error)) return false;
    if (!run_dit_and_vae(*req, music_codes, &res->pcm_stereo, error)) return false;
    return true;
}

// Defined here so ggml.h stays out of family.h.
AceStepSession::~AceStepSession() {
    dit_runner_.reset();
    vae_runner_.reset();
    if (owns_ext_ctx_ && ext_ctx_) {
        ggml_free(ext_ctx_);
        ext_ctx_ = nullptr;
    }
}

}  // namespace audiocore::acestep
