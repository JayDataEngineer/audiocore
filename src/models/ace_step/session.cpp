// session.cpp — ACE-Step run_music full pipeline.
//
// Two-step pipeline (ServeurpersoCom/acestep.cpp):
//   Step 1 — run_lm:  caption+lyrics → TE embeddings → 5Hz LM → music codes
//   Step 2 — run_dit_and_vae: FSQ → noise → DiT flow matching → VAE → PCM
//
// Both Qwen3 transformers run through the unified qwen3::Runner (libllama).
// DiT and VAE are ggml_cgraph builds via DiTRunner and VAERunner.

#include "audiocore/models/ace_step/family.h"

#include "audiocore/framework/sampling/sampler.h"

#include "ggml.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include <cstdio>   // stderr diagnostic

// Debug: crash handler for SIGSEGV
#include <csignal>
#include <execinfo.h>
static void sigsegv_handler(int sig) {
    (void)sig;
    std::fprintf(stderr, "\n[ace_step] SIGSEGV! Backtrace:\n");
    void* bt[32];
    int n = backtrace(bt, 32);
    backtrace_symbols_fd(bt, n, 2); // stderr
    std::_Exit(139);
}
struct SigSegvInstaller { SigSegvInstaller() { std::signal(SIGSEGV, sigsegv_handler); } } sigsegv_installer_;

namespace audiocore::acestep {

using audiocore::sampler::Params;
using audiocore::sampler::PhiloxRng;
using audiocore::sampler::sample_token;

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

    // ── SFT / Base: linear schedule, 50 steps, shift=1.0 ────────────────────
    // Base (the pretrained root, not instruction-tuned for the 8-step turbo
    // shortcut) uses the same linear 50-step schedule as SFT.
    if ((variant == "sft" || variant == "base") &&
        (override_steps <= 0 || override_steps == 50)) {
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
//  Conv1D(k=2, s=1, p=0) — temporal mixing of 2 adjacent frames
// ═════════════════════════════════════════════════════════════════════════════
//
// Used by proj_in (192 → 2048) and proj_out (2048 → 64).  The weight is stored
// as a 3D tensor ne=[K=2, OC, IC] in row-major memory (k varies fastest).
// Each output frame blends two adjacent input frames:
//
//   y[t, o] = sum_i (x[t, i] · w[0, o, i] + x[t+1, i] · w[1, o, i]) + b[o]
//
// Output has T_out = T_in − 1 frames (no padding).

static void conv1d_k2_s1_p0(const float* x, int32_t T, int32_t IC, int32_t OC,
                             const float* w, const float* bias, float* y) {
    // w in ggml ne=[2, OC, IC] order → w[k, o, i] at offset k + o·2 + i·2·OC
    const int32_t T_out = T - 1;
    for (int32_t t = 0; t < T_out; t++) {
        const float* x0 = x + static_cast<size_t>(t)     * IC;
        const float* x1 = x + static_cast<size_t>(t + 1) * IC;
        float*       yt = y + static_cast<size_t>(t)     * OC;
        for (int32_t o = 0; o < OC; o++) {
            float sum = bias ? bias[o] : 0.0f;
            for (int32_t i = 0; i < IC; i++) {
                size_t w_base = static_cast<size_t>(o + i * OC) * 2;
                sum += x0[i] * w[w_base] + x1[i] * w[w_base + 1];
            }
            yt[o] = sum;
        }
    }
}

// ── Patchify proj_in: stride=P non-overlapping patch + linear ──────────
// Ported from HOT-Step engine/src/dit-graph.h:583-592.
// Input  x: [T, IC]     raw context (192 channels per frame)
// Weight w: ne=[P, IC, H], element [p, ic, h] at offset p + ic*P + h*P*IC
// Output y: [T/P, H]    hidden state patches
// T must be divisible by P.
static void patchify_proj_in(const float* x, int32_t T, int32_t IC, int32_t P,
                              int32_t H, const float* w, const float* bias,
                              float* y) {
    const int32_t S = T / P;
    const int32_t PIC = P * IC;  // patch input dim (e.g. 384)
    for (int32_t s = 0; s < S; s++) {
        float* ys = y + static_cast<size_t>(s) * H;
        for (int32_t h = 0; h < H; h++) ys[h] = bias ? bias[h] : 0.0f;
        for (int32_t p = 0; p < P; p++) {
            const float* xp = x + static_cast<size_t>(s * P + p) * IC;
            for (int32_t ic = 0; ic < IC; ic++) {
                float xv = xp[ic];
                // w[p, ic, h] at offset p + ic*P + h*P*IC
                const float* wp = w + static_cast<size_t>(p + ic * P);
                for (int32_t h = 0; h < H; h++) {
                    ys[h] += xv * wp[static_cast<size_t>(h) * PIC];
                }
            }
        }
    }
}

// ── Un-patchify proj_out: linear + reshape (ConvTranspose1d equivalent) ─
// Ported from HOT-Step engine/src/dit-graph.h:637-638.
// Input  x: [S, H]      hidden state patches (S = T/P)
// Weight w: ne=[P, OC, H], element [p, oc, h] at offset p + oc*P + h*P*OC
// Output y: [S*P, OC]   latent frames (un-patchified to T = S*P)
static void unpatchify_proj_out(const float* x, int32_t S, int32_t H, int32_t P,
                                 int32_t OC, const float* w, const float* bias,
                                 float* y) {
    const int32_t T = S * P;
    const int32_t POC = P * OC;  // 128
    for (int32_t t = 0; t < T; t++) {
        const int32_t s = t / P;
        const int32_t p = t % P;
        const float* xs = x + static_cast<size_t>(s) * H;
        float* yt = y + static_cast<size_t>(t) * OC;
        for (int32_t oc = 0; oc < OC; oc++) {
            float sum = bias ? bias[oc] : 0.0f;
            // w[p, oc, h] at offset p + oc*P + h*P*OC
            const float* wp = w + static_cast<size_t>(p + oc * P);
            for (int32_t h = 0; h < H; h++) {
                sum += xs[h] * wp[static_cast<size_t>(h) * POC];
            }
            yt[oc] = sum;
        }
    }
}

// ── Repaint injection: replace preserved regions with noised source ──────
// Verbatim port of HOT-Step engine/src/sampler-repaint.h:sampler_repaint_inject.
// For frames OUTSIDE [repaint_t0, repaint_t1): xt = t_next*noise + (1-t_next)*src
// Only runs for the first `injection_ratio * num_steps` steps.
static void sampler_repaint_inject(
    float* xt, const float* noise, const float* repaint_src,
    int N, int T, int Oc,
    int repaint_t0, int repaint_t1,
    float repaint_injection_ratio,
    int step, int num_steps, float t_next)
{
    if (!repaint_src || repaint_t1 <= repaint_t0) return;
    int injection_cutoff = static_cast<int>(repaint_injection_ratio *
                                            static_cast<float>(num_steps) + 0.5f);
    if (step >= injection_cutoff) return;
    const int n_per = T * Oc;
    for (int b = 0; b < N; b++) {
        for (int t = 0; t < T; t++) {
            if (t < repaint_t0 || t >= repaint_t1) {
                for (int ch = 0; ch < Oc; ch++) {
                    int idx = b * n_per + t * Oc + ch;
                    xt[idx] = t_next * noise[idx] +
                              (1.0f - t_next) * repaint_src[t * Oc + ch];
                }
            }
        }
    }
}

// ── Post-loop boundary blend: smooth repaint zone edges ─────────────────
// Verbatim port of HOT-Step engine/src/sampler-repaint.h:sampler_repaint_blend.
static void sampler_repaint_blend(
    float* output, const float* repaint_src,
    int N, int T, int Oc,
    int repaint_t0, int repaint_t1,
    int repaint_crossfade_frames)
{
    if (!repaint_src || repaint_t1 <= repaint_t0 || repaint_crossfade_frames <= 0) return;
    const int n_per = T * Oc;
    int cf = repaint_crossfade_frames;
    int fade_start = repaint_t0 - cf > 0 ? repaint_t0 - cf : 0;
    int fade_end = repaint_t1 + cf < T ? repaint_t1 + cf : T;
    for (int t = fade_start; t < fade_end; t++) {
        if (t >= repaint_t0 && t < repaint_t1) continue;
        float m;
        if (t < repaint_t0) {
            int rl = repaint_t0 - fade_start;
            m = static_cast<float>(t - fade_start + 1) / static_cast<float>(rl + 1);
        } else {
            int rl = fade_end - repaint_t1;
            m = static_cast<float>(fade_end - t) / static_cast<float>(rl + 1);
        }
        for (int b = 0; b < N; b++) {
            for (int ch = 0; ch < Oc; ch++) {
                int idx = b * n_per + t * Oc + ch;
                output[idx] = m * output[idx] + (1.0f - m) * repaint_src[t * Oc + ch];
            }
        }
    }
}

// ── BF16 → f32 buffer (bit-level, no vae_runner dependency) ────────────
static void bf16_to_f32_buf_local(const void* src, float* dst, int32_t n) {
    const uint16_t* s = static_cast<const uint16_t*>(src);
    for (int32_t i = 0; i < n; i++) {
        // BF16 is F32 truncated to top 16 bits — just zero-extend.
        uint32_t f32_bits = static_cast<uint32_t>(s[i]) << 16;
        float f;
        std::memcpy(&f, &f32_bits, sizeof(f));
        dst[i] = f;
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
    // (0) Reset LM + TE KV caches so this run is independent of prior calls.
    //     Without this, the second call's prefill at position 0 collides
    //     with the cached positions from the first call and llama_decode
    //     fails with "inconsistent sequence positions".
    if (lm_) lm_->clear_kv_cache();
    if (te_) te_->clear_kv_cache();

    // (1) Assemble the text prompt
    std::string prompt = req.caption;
    if (!req.lyrics.empty()) {
        prompt += "\nLyrics: " + req.lyrics;
    }

    fprintf(stderr, "[ace_step] run_lm: tokenizing...\n");
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
    fprintf(stderr, "[ace_step] run_lm: TE forward_get_embeddings (n_tok=%zu)...\n", te_tokens.size());
    const int32_t te_hs = cfg_.encoder_hidden_size;
    te_cond_len_ = static_cast<int32_t>(te_tokens.size());
    te_cond_.resize(static_cast<size_t>(te_cond_len_) * te_hs);
    if (!te_->forward_get_embeddings(te_tokens.data(), te_cond_len_, 0,
                                     te_cond_.data(), error))
        return false;
    fprintf(stderr, "[ace_step] run_lm: TE forward_get_embeddings done\n");

    // (4) Null-text TE embedding for CFG (DiT side)
    {
        fprintf(stderr, "[ace_step] run_lm: null-text TE...\n");
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
        fprintf(stderr, "[ace_step] run_lm: null-text TE done\n");
    }

    // (5) LM prefill with text tokens (populates KV cache for positions 0..N-1)
    fprintf(stderr, "[ace_step] run_lm: LM forward_tokens prefill (n_tok=%zu)...\n", lm_prompt.size());
    const int32_t prompt_len = static_cast<int32_t>(lm_prompt.size());
    if (!lm_->forward_tokens(lm_prompt.data(), prompt_len, 0, nullptr, error))
        return false;
    fprintf(stderr, "[ace_step] run_lm: LM prefill done\n");

    // (6) Decode loop: generate N codes at 5 Hz
    const int32_t n_codes = std::max(1, static_cast<int32_t>(req.duration * 5.0f + 0.5f));
    music_codes->clear();
    music_codes->reserve(static_cast<size_t>(n_codes));

    const int32_t code_vocab_size = 64000;
    const int32_t code_start = std::max(0, lm_->vocab_size() - code_vocab_size);

    // Start token: the last prompt token (typically EOS/BOS).  The LM will
    // predict the first code token from this context.
    int32_t prev_token = lm_prompt.back();

    fprintf(stderr, "[ace_step] run_lm: starting decode loop (n_codes=%d, vs=%d, code_start=%d)...\n",
            n_codes, lm_->vocab_size(), code_start);

    // RNG for stochastic sampling (seeded for reproducibility)
    PhiloxRng rng{req.seed != 0 ? req.seed : 42};

    for (int32_t s = 0; s < n_codes; s++) {
        const int32_t n_pos = prompt_len + s;

        const int32_t vs = lm_->vocab_size();
        std::vector<float> logits(static_cast<size_t>(vs));

        if (!lm_->forward_tokens(&prev_token, 1, n_pos, logits.data(), error))
            return false;

        const float* code_logits = logits.data() + code_start;

        // Sample within the code vocabulary via the unified audiocore::sampler.
        // temperature <= 0 selects the greedy argmax path; otherwise temp +
        // top-p nucleus filtering + multinomial draw.
        Params sp;
        sp.temperature = req.temperature;
        sp.top_p       = req.top_p;
        sp.do_sample   = req.temperature > 0.0f;
        const int32_t chosen = sample_token(code_logits, code_vocab_size, sp,
                                            /*prev_tokens=*/nullptr, /*n_prev=*/0,
                                            &rng);

        music_codes->push_back(chosen);
        prev_token = code_start + chosen;
        if (s < 3 || s == n_codes - 1) {
            fprintf(stderr, "[ace_step] run_lm: code[%d/%d]=%d (tok=%d)\n",
                    s, n_codes, chosen, prev_token);
        }
    }
    fprintf(stderr, "[ace_step] run_lm: decode loop done (%zu codes)\n", music_codes->size());

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
    fprintf(stderr, "[ace_step] run_dit_and_vae: enter (codes=%zu)\n", music_codes.size());
    // ── Cover mode: use cover_latent_cond_ to determine frame count ────────
    const bool is_cover_mode = !cover_latent_cond_.empty();
    const int32_t out_ch     = cfg_.dit.out_channels; //   64 (acoustic latent)
    const int32_t in_ch      = cfg_.dit.in_channels;  //  192
    const int32_t H          = cfg_.dit.hidden_size;  // 1024 (1.5B DiT)

    int32_t n_frames, n_codes_5;
    if (is_cover_mode) {
        n_frames = static_cast<int32_t>(cover_latent_cond_.size() / out_ch);
        n_codes_5 = n_frames / 5;
    } else {
        n_codes_5 = static_cast<int32_t>(music_codes.size());
        n_frames  = n_codes_5 * 5;
    }

    const int32_t enc_hs     = cfg_.encoder_hidden_size;  // 1024

    if (H <= 0 || in_ch <= 0 || out_ch <= 0) {
        if (error)
            *error = "ACE-Step: DitConfig not populated (H=" + std::to_string(H) +
                     ", in_ch=" + std::to_string(in_ch) +
                     ", out_ch=" + std::to_string(out_ch) + ")";
        return false;
    }

    // ── 1. Source latent for DiT context (src channels 0..63) ──────────────
    //
    // Upstream ACE-Step prepare_src_latents (pipeline_ace_step.py:626):
    //   • cover from audio_codes  → detokenize residual-FSQ codes
    //   • cover from src_audio    → VAE-encode (then tokenize+detokenize for cover)
    //   • text_to_music (no src)  → silence_latent  ← THIS IS OUR CASE
    //
    // For text_to_music, src_latents = silence_latent (the learned "no music"
    // reference), NOT the detokenized LM codes. The DiT learns: "given silence
    // as src + text conditioning, generate music." The LM codes drive the
    // generation only indirectly — they were already consumed by the LM to
    // produce a musical token sequence that conditions the DiT via the text
    // encoder path (not via src).
    //
    // The detokenizer_runner is kept wired (detokenizer_runner_ member) for
    // future cover-mode support (residual-FSQ decode path), but is NOT invoked
    // for text_to_music. See docs/ACE-STEP-DETOKENIZER-GAP.md.
    //
    // For cover mode the src is cover_latent_cond_ which already holds the
    // VAE-encoded + temporally-smoothed source latents.

    // ── 2. Prepare conditioning for DiT ─────────────────────────────────────
    if (!is_cover_mode && (te_cond_.empty() || te_cond_len_ <= 0)) {
        if (error) *error = "ACE-Step: run_lm must execute before run_dit_and_vae";
        return false;
    }
    // For cover mode, use empty text conditioning (TE is not needed since
    // source audio provides the conditioning through the cover latent). The
    // DiT still needs a valid cond_ptr — use a single EOS/BOS token's
    // embedding as a minimal placeholder.
    const float* cond_ptr;
    int32_t T_cond;
    const float* uncond_ptr;
    int32_t T_uncond;
    std::vector<float> dummy_cond(static_cast<size_t>(enc_hs), 0.0f);
    if (is_cover_mode) {
        cond_ptr   = dummy_cond.data();
        T_cond     = 1;
        uncond_ptr = nullptr;
        T_uncond   = 0;
    } else {
        cond_ptr   = te_cond_.data();
        T_cond     = te_cond_len_;
        uncond_ptr = te_uncond_.empty() ? nullptr : te_uncond_.data();
        T_uncond   = te_uncond_.empty() ? 0 :
                     static_cast<int32_t>(te_uncond_.size() / enc_hs);
    }

    // ════════════════════════════════════════════════════════════════════════
    //  3. Build context buffer + initialize noise
    //
    //  Ported from HOT-Step engine/src/pipeline-synth-ops.cpp:ops_build_context
    //  and engine/src/hot-step-sampler.h.
    //
    //  Channel layout (verified from dit-graph.h:462 + ops_build_context):
    //    channels   0..63  = src   (source latent)
    //    channels  64..127 = mask  (1.0 = generate fresh, 0.0 = keep source)
    //    channels 128..191 = xt    (current noisy latent)
    //
    //  proj_in is stride=P=2 non-overlapping patchify (NOT stride=1 Conv1D).
    //  The DiT graph includes proj_in internally; we re-run proj_in + DiT +
    //  proj_out every step so the Euler step happens in latent space and
    //  repaint injection can operate on xt directly.
    // ════════════════════════════════════════════════════════════════════════
    std::mt19937_64 noise_rng(static_cast<uint64_t>(
        req.seed != 0 ? req.seed : 42));
    std::normal_distribution<float> ndist(0.0f, 1.0f);
    const int32_t P = (cfg_.dit.patch_size > 0) ? cfg_.dit.patch_size : 2;
    const bool is_repaint_mode = !repaint_latent_cond_.empty();

    // Pad T_latent to be divisible by P (patchify requirement)
    int32_t T_latent = n_frames;
    if (T_latent % P != 0)
        T_latent = (T_latent / P + 1) * P;
    const int32_t S_patches = T_latent / P;
    fprintf(stderr, "[ace_step] dit: T_latent=%d S_patches=%d P=%d H=%d "
            "in_ch=%d out_ch=%d\n",
            T_latent, S_patches, P, H, in_ch, out_ch);

    // Load silence latent (used as src for text2music and inside repaint zone)
    const float* silence_ptr = nullptr;
    int32_t silence_T = 0;
    {
        ggml_tensor* st = ggml_get_tensor(ext_ctx_, "silence_latent");
        if (st) {
            silence_ptr = static_cast<const float*>(st->data);
            silence_T = static_cast<int32_t>(st->ne[1]);
        }
    }

    // proj_in / proj_out biases
    ggml_tensor* pi_b = ggml_get_tensor(ext_ctx_, "decoder.proj_in.1.bias");
    ggml_tensor* po_b = ggml_get_tensor(ext_ctx_, "decoder.proj_out.1.bias");
    const float* proj_in_bias  = pi_b ? static_cast<const float*>(pi_b->data) : nullptr;
    const float* proj_out_bias = po_b ? static_cast<const float*>(po_b->data) : nullptr;

    // Repaint zone in latent frames (mask_start/mask_end are normalized [0,1])
    int32_t repaint_t0 = 0, repaint_t1 = 0;
    if (is_repaint_mode) {
        repaint_t0 = static_cast<int32_t>(req.mask_start * T_latent);
        repaint_t1 = static_cast<int32_t>(req.mask_end * T_latent);
        repaint_t0 = std::max(0, std::min(repaint_t0, T_latent));
        repaint_t1 = std::max(0, std::min(repaint_t1, T_latent));
        fprintf(stderr, "[ace_step] repaint zone: [%d, %d) / %d\n",
                repaint_t0, repaint_t1, T_latent);
    }

    // ── Build context: [T_latent, 192] = src(64) | mask(64) | xt(64) ──
    std::vector<float> context(static_cast<size_t>(T_latent) * in_ch, 0.0f);

    // Initialize noise latent [T_latent, out_ch]
    std::vector<float> noise_latent(static_cast<size_t>(T_latent) * out_ch);
    for (auto& v : noise_latent) v = ndist(noise_rng);

    const int32_t T_cover =
        is_cover_mode
            ? std::min(T_latent, static_cast<int32_t>(cover_latent_cond_.size() / out_ch))
            : 0;
    const int32_t T_repaint_src =
        is_repaint_mode
            ? std::min(T_latent, static_cast<int32_t>(repaint_latent_cond_.size() / out_ch))
            : 0;

    for (int32_t t = 0; t < T_latent; t++) {
        float* row = &context[static_cast<size_t>(t) * in_ch];

        // ── src (channels 0–63) ──
        // Matches ops_build_context (pipeline-synth-ops.cpp:965–974):
        //   text2music: silence
        //   cover:      cover_latent (or silence if past T_cover)
        //   repaint:    silence inside zone, actual latent outside
        if (is_repaint_mode) {
            bool in_region = (t >= repaint_t0 && t < repaint_t1);
            if (in_region) {
                if (silence_ptr && t < silence_T)
                    std::memcpy(row, silence_ptr + static_cast<size_t>(t) * out_ch,
                                static_cast<size_t>(out_ch) * sizeof(float));
            } else {
                if (t < T_repaint_src)
                    std::memcpy(row,
                                &repaint_latent_cond_[static_cast<size_t>(t) * out_ch],
                                static_cast<size_t>(out_ch) * sizeof(float));
                else if (silence_ptr && t < silence_T)
                    std::memcpy(row, silence_ptr + static_cast<size_t>(t) * out_ch,
                                static_cast<size_t>(out_ch) * sizeof(float));
            }
        } else if (is_cover_mode) {
            if (t < T_cover)
                std::memcpy(row,
                            &cover_latent_cond_[static_cast<size_t>(t) * out_ch],
                            static_cast<size_t>(out_ch) * sizeof(float));
            else if (silence_ptr && t < silence_T)
                std::memcpy(row, silence_ptr + static_cast<size_t>(t) * out_ch,
                            static_cast<size_t>(out_ch) * sizeof(float));
        } else {
            // text_to_music: src = silence_latent (the learned "no music"
            // reference). Matches upstream prepare_src_latents for the no-source
            // case. The silence_latent tensor (shape [64, 15000]) is cropped
            // or tiled to the target latent length.
            if (silence_ptr && t < silence_T)
                std::memcpy(row, silence_ptr + static_cast<size_t>(t) * out_ch,
                            static_cast<size_t>(out_ch) * sizeof(float));
        }

        // ── mask (channels 64–127) ──
        // Matches ops_build_context:976–981.
        //   repaint: 1.0 inside zone, 0.0 outside
        //   else:    1.0 everywhere (training distribution)
        float mask_val = 1.0f;
        if (is_repaint_mode) {
            bool in_region = (t >= repaint_t0 && t < repaint_t1);
            mask_val = in_region ? 1.0f : 0.0f;
        }
        for (int32_t c = 0; c < out_ch; c++) row[64 + c] = mask_val;

        // ── xt (channels 128–191) = noise ──
        std::memcpy(row + 128,
                    &noise_latent[static_cast<size_t>(t) * out_ch],
                    static_cast<size_t>(out_ch) * sizeof(float));
    }

    // xt_current tracks the denoised latent across steps [T_latent, out_ch]
    std::vector<float> xt_current = noise_latent;

    // ── 4. DiT flow-matching Euler loop ─────────────────────────────────────
    //  Ported from HOT-Step engine/src/hot-step-sampler.h.
    //  Per step:
    //    1. Write xt into context channels 128–191
    //    2. proj_in: context [T, 192] → hidden [S, H]  (patchify stride=P)
    //    3. DiT forward: hidden → v_hidden
    //    4. proj_out: v_hidden → v_latent [T, 64]  (un-patchify)
    //    5. Euler: xt += dt · v_latent  (latent space)
    //    6. Repaint inject on xt (outside zone → noised source)
    const FlowSchedule sched = build_schedule(cfg_.variant, req.n_diffusion_steps);
    fprintf(stderr, "[ace_step] dit: schedule n_steps=%d variant='%s'\n",
            sched.n_steps, cfg_.variant.c_str());

    std::vector<float> hidden(static_cast<size_t>(S_patches) * H);
    std::vector<float> v_hidden(static_cast<size_t>(S_patches) * H);
    std::vector<float> v_latent(static_cast<size_t>(T_latent) * out_ch);

    for (int step = 0; step < sched.n_steps; step++) {
        const float t  = sched.timesteps[static_cast<size_t>(step)];
        const float dt = (step + 1 < sched.n_steps)
                             ? t - sched.timesteps[static_cast<size_t>(step + 1)]
                             : t;  // last step → go to 0
        const float t_next = (step + 1 < sched.n_steps)
                                 ? sched.timesteps[static_cast<size_t>(step + 1)]
                                 : 0.0f;

        fprintf(stderr, "[ace_step] dit: step %d/%d t=%.3f dt=%.3f\n",
                step, sched.n_steps, t, dt);

        // Write current xt into context channels 128–191
        for (int32_t i = 0; i < T_latent; i++) {
            std::memcpy(&context[static_cast<size_t>(i) * in_ch + 128],
                        &xt_current[static_cast<size_t>(i) * out_ch],
                        static_cast<size_t>(out_ch) * sizeof(float));
        }

        // proj_in: context [T, 192] → hidden [S, H]  (patchify stride=P)
        patchify_proj_in(context.data(), T_latent, in_ch, P, H,
                         proj_in_w_f32_.data(), proj_in_bias,
                         hidden.data());

        // DiT forward: hidden [S, H] → v_hidden [S, H]
        // Pass silence_latent slice for timbre conditioning.
        // Upstream uses timbre_fix_frame = ceil(30 * 25) = 750 frames.
        const int32_t T_refer = silence_ptr ?
            std::min<int32_t>(750, silence_T) : 0;
        const float* refer_audio = silence_ptr;
        if (!dit_runner_->forward(hidden.data(), t,
                                  cond_ptr, T_cond, enc_hs,
                                  uncond_ptr, T_uncond,
                                  refer_audio, T_refer,
                                  req.guidance_scale,
                                  S_patches,
                                  v_hidden.data(), error))
            return false;
        fprintf(stderr, "[ace_step] dit: step %d forward done\n", step);

        // proj_out: v_hidden [S, H] → v_latent [T, 64]  (un-patchify)
        unpatchify_proj_out(v_hidden.data(), S_patches, H, P, out_ch,
                            proj_out_w_f32_.data(), proj_out_bias,
                            v_latent.data());

        // NaN check on v_latent (warn-only)
        {
            int nan_cnt = 0;
            for (size_t i = 0; i < v_latent.size(); i++)
                if (std::isnan(v_latent[i])) nan_cnt++;
            if (nan_cnt > 0) {
                fprintf(stderr, "[ace_step] WARNING: step %d v_latent NaN=%d/%zu\n",
                        step, nan_cnt, v_latent.size());
            }
        }

        // Euler step in LATENT space: xt += dt · v_latent
        for (size_t i = 0; i < xt_current.size(); i++) {
            xt_current[i] += dt * v_latent[i];
        }

        // Repaint injection (outside zone → noised source)
        if (is_repaint_mode) {
            sampler_repaint_inject(
                xt_current.data(),
                noise_latent.data(),
                repaint_latent_cond_.data(),
                /*N=*/1, T_latent, out_ch,
                repaint_t0, repaint_t1,
                /*repaint_injection_ratio=*/0.5f,  // HOT-Step default
                step, sched.n_steps, t_next);
        }
    }

    // Post-loop repaint boundary blend (HOT-Step default: crossfade=0, no-op)
    if (is_repaint_mode) {
        sampler_repaint_blend(
            xt_current.data(),
            repaint_latent_cond_.data(),
            /*N=*/1, T_latent, out_ch,
            repaint_t0, repaint_t1,
            /*repaint_crossfade_frames=*/0);  // HOT-Step default
    }

    // ── 5. VAE decode: xt_current [T, 64] → PCM stereo ─────────────────────
    fprintf(stderr, "[ace_step] dit: all steps done, VAE decode (T=%d)\n", T_latent);
    {
        int nan_l = 0;
        for (size_t i = 0; i < xt_current.size(); i++)
            if (std::isnan(xt_current[i])) nan_l++;
        if (nan_l > 0) {
            fprintf(stderr, "[ace_step] WARNING: %d/%zu NaN in final latent\n",
                    nan_l, xt_current.size());
        }
    }
    if (!vae_runner_->decode(xt_current.data(), T_latent, pcm_stereo, error))
        return false;
    {
        int nan_pcm = 0; float mx = -1e30f, mn = 1e30f;
        for (size_t i = 0; i < pcm_stereo->size(); i++) {
            if (std::isnan((*pcm_stereo)[i])) nan_pcm++;
            else { if ((*pcm_stereo)[i] > mx) mx = (*pcm_stereo)[i]; if ((*pcm_stereo)[i] < mn) mn = (*pcm_stereo)[i]; }
        }
        fprintf(stderr, "[ace_step] dit: VAE decode done (pcm=%zu NaN=%d range=[%.6f,%.6f])\n",
                pcm_stereo->size(), nan_pcm, mn, mx);
    }

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

    // ── Mode routing ──────────────────────────────────────────────────────
    //
    //   text_to_music (default) — full pipeline from text only
    //   repaint  — VAE-encode input_audio, blend known region during denoising
    //   completion — same as repaint but mask covers the end
    //   cover    — needs style encoder (fail-fast)
    //   stem     — separate model entirely (fail-fast)
    //   lego     — separate stem-assembler (fail-fast)
    //
    auto lower = [](std::string s) {
        for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    };
    const std::string mode = lower(req->mode);
    const bool is_repaint = (mode == "repaint");
    const bool is_completion = (mode == "completion");
    const bool is_cover = (mode == "cover" || mode == "cover_nofsq");

    if (!is_repaint && !is_completion && !is_cover &&
        mode != "text_to_music" && mode != "text-to-music" && !mode.empty()) {
        static const char* kHelp =
            "See GAPS.md §3.2 for the per-mode roadmap. ";
        if (mode == "stem" || mode == "lego") {
            if (error) *error = std::string(
                "ace_step mode='") + req->mode + "' is a separate model family "
                "(Demucs-class separator / stem-assembler), not an ACE-Step "
                "DiT mode. " + kHelp;
        } else {
            if (error) *error = std::string(
                "ace_step: unknown mode '") + req->mode +
                "'. Supported: 'text_to_music', 'repaint', 'completion', "
                "'cover'. " + kHelp;
        }
        return false;
    }

    res->sampling_rate = 48000;
    res->channels      = 2;

    // ── Repaint / completion: VAE-encode input audio ──────────────────────
    repaint_latent_cond_.clear();
    if (is_repaint || is_completion) {
        if (req->input_audio.empty()) {
            if (error) *error = "ace_step mode='" + req->mode +
                "' requires non-empty input_audio (stereo PCM at 48kHz)";
            return false;
        }
        const int32_t n_audio = static_cast<int32_t>(req->input_audio.size() / 2);
        if (n_audio < 1920) {
            if (error) *error = "ace_step: input_audio too short (min ~1920 samples = 40ms)";
            return false;
        }
        const int32_t remainder = n_audio % 1920;
        std::vector<float> padded = req->input_audio;
        if (remainder != 0) {
            size_t pad = static_cast<size_t>(1920 - remainder) * 2;
            padded.resize(padded.size() + pad, 0.0f);
        }
        const int32_t n_padded = static_cast<int32_t>(padded.size() / 2);
        if (!vae_runner_->encode(padded.data(), n_padded,
                                  &repaint_latent_cond_, error))
            return false;
        (void)n_audio;
        (void)remainder;
    }

    // ── Cover: VAE-encode input audio + temporal smoothing ────────────────
    cover_latent_cond_.clear();
    if (is_cover) {
        if (req->input_audio.empty()) {
            if (error) *error = "ace_step mode='cover' requires non-empty "
                "input_audio (stereo PCM at 48kHz)";
            return false;
        }
        const int32_t n_audio = static_cast<int32_t>(req->input_audio.size() / 2);
        if (n_audio < 1920) {
            if (error) *error = "ace_step: input_audio too short (min ~1920 samples = 40ms)";
            return false;
        }
        const int32_t remainder = n_audio % 1920;
        std::vector<float> padded = req->input_audio;
        if (remainder != 0) {
            size_t pad = static_cast<size_t>(1920 - remainder) * 2;
            padded.resize(padded.size() + pad, 0.0f);
        }
        const int32_t n_padded = static_cast<int32_t>(padded.size() / 2);
        // VAE encode → source latents at 25 Hz [T_latent, 64]
        std::vector<float> src_latent;
        if (!vae_runner_->encode(padded.data(), n_padded, &src_latent, error))
            return false;
        // FSQ roundtrip proxy: average pool 5:1, then repeat 5×
        // (degrades micro-timings without requiring FSQ encoder weights)
        const int32_t T_lat = static_cast<int32_t>(src_latent.size() / 64);
        const int32_t T_pooled = T_lat / 5;  // 25 Hz → 5 Hz
        if (T_pooled == 0) {
            if (error) *error = "ace_step: input_audio too short for cover mode";
            return false;
        }
        cover_latent_cond_.resize(static_cast<size_t>(T_pooled * 5) * 64, 0.0f);
        for (int32_t i = 0; i < T_pooled; i++) {
            // Average pool 5 frames
            float avg[64] = {0};
            for (int j = 0; j < 5; j++) {
                for (int k = 0; k < 64; k++) {
                    avg[k] += src_latent[static_cast<size_t>(i * 5 + j) * 64 + k];
                }
            }
            for (int k = 0; k < 64; k++) avg[k] /= 5.0f;
            // Repeat 5× at 25 Hz
            for (int j = 0; j < 5; j++) {
                std::memcpy(&cover_latent_cond_[
                    static_cast<size_t>(i * 5 + j) * 64],
                    avg, 64 * sizeof(float));
            }
        }
    }

    // ── Run pipeline ──────────────────────────────────────────────────────
    std::vector<int32_t> music_codes;
    if (!is_cover) {
        // Normal mode: run LM to generate FSQ codes from text
        if (!run_lm(*req, &music_codes, error)) return false;
    }
    // For cover mode, music_codes stays empty — run_dit_and_vae will use
    // cover_latent_cond_ for n_frames and skip the FSQ decode step.
    if (!run_dit_and_vae(*req, music_codes, &res->pcm_stereo, error)) return false;

    // ── Post-processing: DC offset removal + gentle peak normalization ──
    // The ACE-Step VAE decoder tends to emit a small constant DC offset
    // (typically -0.02 to -0.04) with the actual audio content oscillating
    // around it at very low amplitude. Left untreated this produces
    // near-silent, click-prone WAV files. We:
    //   1. Subtract the per-channel mean (DC block).
    //   2. Peak-normalize the AC component to 0.9 (only if the peak is
    //      below 0.5 — we never attenuate a loud signal, only boost quiet
    //      ones to use the available headroom).
    // This mirrors what every production mastering chain does anyway, and
    // is essential here because the VAE was trained on data normalized to
    // ~[-1, 1] but our decode produces a tiny, biased output range.
    {
        auto& pcm = res->pcm_stereo;
        const size_t n_total = pcm.size();
        if (n_total >= 4) {
            // Per-channel mean (interleaved L,R,L,R).
            const size_t n_per_ch = n_total / 2;
            double sum_l = 0.0, sum_r = 0.0;
            for (size_t i = 0; i < n_per_ch; i++) {
                sum_l += pcm[i * 2];
                sum_r += pcm[i * 2 + 1];
            }
            const float dc_l = static_cast<float>(sum_l / static_cast<double>(n_per_ch));
            const float dc_r = static_cast<float>(sum_r / static_cast<double>(n_per_ch));
            // Subtract DC offset; track peak.
            float peak = 0.0f;
            for (size_t i = 0; i < n_per_ch; i++) {
                float l = pcm[i * 2]     - dc_l;
                float r = pcm[i * 2 + 1] - dc_r;
                pcm[i * 2]     = l;
                pcm[i * 2 + 1] = r;
                if (std::fabs(l) > peak) peak = std::fabs(l);
                if (std::fabs(r) > peak) peak = std::fabs(r);
            }
            // Boost only if very quiet — leaves louder signals untouched.
            if (peak > 1e-6f && peak < 0.5f) {
                const float gain = 0.9f / peak;
                for (size_t i = 0; i < n_total; i++) pcm[i] *= gain;
            }
            fprintf(stderr,
                "[ace_step] post: DC_offset=[%.5f,%.5f]  peak_after=%.5f  %s\n",
                dc_l, dc_r, peak,
                (peak > 1e-6f && peak < 0.5f) ? "(normalized)" : "(untouched)");
        }
    }
    return true;
}

// Defined here so ggml.h stays out of family.h.
AceStepSession::~AceStepSession() {
    dit_runner_.reset();
    vae_runner_.reset();
    detokenizer_runner_.reset();
    if (owns_ext_ctx_ && ext_ctx_) {
        ggml_free(ext_ctx_);
        ext_ctx_ = nullptr;
    }
}

}  // namespace audiocore::acestep
