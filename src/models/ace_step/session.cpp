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
#include "audiocore/framework/sampling/philox.h"

#include "ggml.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <unordered_map>
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

    // ── Turbo: shifted-cosine, 8 steps by default, shift=3.0 ───────────────
    // Matches upstream pipeline_ace_step.py:_get_timestep_schedule:
    //   t = linspace(1, 0, N+1)
    //   t = shift * t / (1 + (shift - 1) * t)   # shift=3 for turbo
    //   return t[:-1]                            # drop terminal t=0
    // For N=8 shift=3 this reproduces the hardcoded table
    //   [1.0, 0.9545, 0.9, 0.8333, 0.75, 0.6429, 0.5, 0.3]
    // exactly. Apply the same formula for any N so 4/16/20-step turbo runs
    // match what upstream would produce.
    if (variant == "turbo" || variant.empty()) {
        const int steps = (override_steps > 0) ? override_steps : 8;
        const float shift = 3.0f;
        s.timesteps.reserve(steps);
        for (int i = 0; i < steps; i++) {
            float t = 1.0f - static_cast<float>(i) / steps;   // N+1 grid, drop last
            float ts = (shift * t) / (1.0f + (shift - 1.0f) * t);
            s.timesteps.push_back(ts);
        }
        s.n_steps = steps;
        return s;
    }

    // ── SFT / Base: linear schedule, 50 steps, shift=1.0 ────────────────────
    // Base (the pretrained root, not instruction-tuned for the 8-step turbo
    // shortcut) uses the same linear 50-step schedule as SFT.
    if (variant == "sft" || variant == "base") {
        const int steps = (override_steps > 0) ? override_steps : 50;
        s.timesteps.reserve(steps);
        for (int i = 0; i < steps; i++) {
            float t = 1.0f - static_cast<float>(i) / steps;
            s.timesteps.push_back(t);
        }
        s.n_steps = steps;
        return s;
    }

    // ── Fallback: uniform spacing, shift=1 ─────────────────────────────────
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
// Matches PyTorch nn.Conv1d(in_channels=IC, out_channels=H, kernel_size=P,
// stride=P, padding=0). The GGUF stores the weight in PyTorch's natural
// C-order of shape (out=H, in=IC, K=P), so weight[oc, ic, k] lives at
// flat[oc*IC*K + ic*K + k] in the raw buffer. The ggml tensor's ne array is
// the reversed PyTorch shape [K, IC, H], but the byte layout matches PyTorch.
// Verified numerically against torch.nn.functional.conv1d.
static void patchify_proj_in(const float* x, int32_t T, int32_t IC, int32_t P,
                              int32_t H, const float* w, const float* bias,
                              float* y) {
    const int32_t S = T / P;
    const int32_t K = P;
    const int32_t IC_K = IC * K;   // stride between output channels in flat weight
    for (int32_t s = 0; s < S; s++) {
        float* ys = y + static_cast<size_t>(s) * H;
        for (int32_t h = 0; h < H; h++) ys[h] = bias ? bias[h] : 0.0f;
        for (int32_t p = 0; p < P; p++) {
            const float* xp = x + static_cast<size_t>(s * P + p) * IC;
            for (int32_t ic = 0; ic < IC; ic++) {
                const float xv = xp[ic];
                // weight[h, ic, p] at flat[h*IC*K + ic*K + p]
                const float* wbase = w + static_cast<size_t>(ic) * K + p;
                for (int32_t h = 0; h < H; h++) {
                    ys[h] += xv * wbase[static_cast<size_t>(h) * IC_K];
                }
            }
        }
    }
}

// ── Un-patchify proj_out: linear + reshape (ConvTranspose1d equivalent) ─
// Matches PyTorch nn.ConvTranspose1d(in_channels=H, out_channels=OC,
// kernel_size=P, stride=P, padding=0). The GGUF stores the weight in
// PyTorch's natural C-order of shape (in=H, out=OC, K=P), so weight[ic, oc, k]
// lives at flat[ic*OC*K + oc*K + k]. Verified against
// torch.nn.functional.conv_transpose1d.
static void unpatchify_proj_out(const float* x, int32_t S, int32_t H, int32_t P,
                                 int32_t OC, const float* w, const float* bias,
                                 float* y) {
    const int32_t T = S * P;
    const int32_t K = P;
    const int32_t OC_K = OC * K;  // stride between input channels in flat weight
    for (int32_t t = 0; t < T; t++) {
        const int32_t s = t / P;
        const int32_t p = t % P;
        const float* xs = x + static_cast<size_t>(s) * H;
        float* yt = y + static_cast<size_t>(t) * OC;
        for (int32_t oc = 0; oc < OC; oc++) {
            float sum = bias ? bias[oc] : 0.0f;
            // weight[h, oc, p] at flat[h*OC*K + oc*K + p]
            const float* wbase = w + static_cast<size_t>(oc) * K + p;
            for (int32_t h = 0; h < H; h++) {
                sum += xs[h] * wbase[static_cast<size_t>(h) * OC_K];
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
// 3. LM prefill with chat-template prompt (Qwen3 <|im_start|> format).
// 4. LM autoregressive decode at 5 Hz.  Audio code tokens are in
//    [AUDIO_CODE_BASE, AUDIO_CODE_BASE + FSQ_CODE_COUNT). All non-code
//    logits (except EOS = TOKEN_IM_END) are masked to -inf, matching
//    reference pipeline-lm.cpp:373-382.

// ── Qwen3 special token IDs (ACE-Step LM vocabulary) ──────────────────────
// These are FIXED IDs in the Qwen3 tokenizer, not computed from vocab_size.
// Matches ref-acestep/src/prompt.h:16-21.
static constexpr int32_t TOKEN_IM_START  = 151644;
static constexpr int32_t TOKEN_IM_END    = 151645;   // EOS / turn end
static constexpr int32_t TOKEN_THINK     = 151667;
static constexpr int32_t TOKEN_THINK_END = 151668;
static constexpr int32_t AUDIO_CODE_BASE = 151669;   // first audio code token ID
static constexpr int32_t FSQ_CODE_COUNT  = 64000;     // valid FSQ codes (levels [8,8,8,5,5,5])

// LM_INSTRUCTION — matches ref-acestep/src/task-types.h:51
static constexpr const char* LM_INSTRUCTION =
    "Generate audio semantic tokens based on the given conditions:";

// DIT_INSTR_COVER — matches ref-acestep/src/task-types.h:73
// This is the instruction the TE encodes for the DiT cross-attention when
// audio codes are present (text2music, cover, etc). The reference always
// uses COVER when the LM has generated codes (pipeline-synth.cpp:324-325):
//   s.use_source_context = !reqs[0].audio_codes.empty();  ← LM codes present
//   s.instruction_str = use_source_context ? DIT_INSTR_COVER : DIT_INSTR_TEXT2MUSIC;
// Since text2music ALWAYS generates codes via the LM, use_source_context is
// ALWAYS true, so the instruction is ALWAYS DIT_INSTR_COVER.
// Using TEXT2MUSIC ("Fill the audio semantic mask") instead of COVER
// ("Generate audio semantic tokens") tells the DiT the wrong task — it
// treats the src latent as a mask to fill rather than musical content to
// render, producing ambient/drone output instead of the caption-driven genre.
static constexpr const char* DIT_INSTR_COVER =
    "Generate audio semantic tokens based on the given conditions:";
static constexpr const char* DIT_INSTR_TEXT2MUSIC =
    "Fill the audio semantic mask based on the given conditions:";

// Text-encoder prompt (for TE/DiT cross-attention). This is the
// "instruction + caption + metas" format used by the text encoder.
// Matches ref-acestep/src/pipeline-synth-ops.cpp:406-425 (build_prompt_strings).
// Uses enriched metadata (bpm, key, timesig) from Phase 1 — NOT hardcoded
// "N/A". The reference populates these from the LM-enriched AceRequest.
// Seeing "N/A" for all metadata fields is OOD and degrades DiT conditioning.
static std::string format_te_prompt(const std::string& caption,
                                    const std::string& lyrics,
                                    float audio_duration,
                                    int bpm = 0,
                                    const std::string& keyscale = "",
                                    const std::string& timesignature = "") {
    const std::string& instruction = DIT_INSTR_COVER;
    int dur_int = (audio_duration > 0.0f) ? static_cast<int>(audio_duration) : 30;
    std::string bpm_str = (bpm > 0) ? std::to_string(bpm) : "N/A";
    std::string ts_str  = timesignature.empty() ? "N/A" : timesignature;
    std::string ks_str  = keyscale.empty() ? "N/A" : keyscale;
    std::string metas = "- bpm: " + bpm_str + "\n"
                        "- timesignature: " + ts_str + "\n"
                        "- keyscale: " + ks_str + "\n"
                        "- duration: " + std::to_string(dur_int) + " seconds\n";
    return std::string("# Instruction\n") + instruction + "\n\n"
           "# Caption\n" + caption + "\n\n"
           "# Metas\n" + metas + "<|endoftext|>\n";
}

// LM prompt in Qwen3 chat template format.
// Matches ref-acestep/src/prompt.h:270-296 (build_lm_prompt_with_cot).
//
//   <|im_start|>system\n# Instruction\n{LM_INSTRUCTION}\n\n<|im_end|>\n
//   <|im_start|>user\n# Caption\n{caption}\n\n# Lyric\n{lyrics}\n<|im_end|>\n
//   <|im_start|>assistant\n<think>\n{cot_yaml}</think>\n\n
//
// The CoT YAML is a simple metadata block. The reference generates this in
// Phase 1 (caption enrichment) but for text2music we derive it directly from
// the request. The assistant turn ends with </think> so the LM immediately
// enters codes_phase — matching reference Phase 2 (pipeline-lm.cpp:343).
//
// CRITICAL: We build the token IDs DIRECTLY (pushing TOKEN_IM_START,
// TOKEN_IM_END, TOKEN_THINK, TOKEN_THINK_END as raw IDs) instead of relying
// on parse_special=true. The Qwen3 GGUF may not register <think>/</think>
// as parseable special tokens, causing them to be BPE-tokenized as regular
// text. This breaks the codes_phase transition. The reference does the same
// (prompt.h:276-295 — ids.push_back(TOKEN_*)).

static std::vector<int32_t> build_lm_prompt_tokens(const qwen3::Runner& lm,
                                                   const std::string& caption,
                                                   const std::string& lyrics,
                                                   float audio_duration,
                                                   std::string* error) {
    std::vector<int32_t> ids;

    // Helper: tokenize a text segment and append to ids (no special tokens).
    auto append_text = [&](const std::string& text) -> bool {
        std::vector<int32_t> toks;
        if (!lm.tokenize(text, /*add_special=*/false, /*parse_special=*/false,
                         &toks, nullptr, error))
            return false;
        ids.insert(ids.end(), toks.begin(), toks.end());
        return true;
    };

    // System turn
    ids.push_back(TOKEN_IM_START);
    if (!append_text("system\n# Instruction\n" + std::string(LM_INSTRUCTION) + "\n\n"))
        return {};
    ids.push_back(TOKEN_IM_END);
    if (!append_text("\n"))
        return {};

    // User turn
    ids.push_back(TOKEN_IM_START);
    if (!append_text("user\n# Caption\n" + caption + "\n\n# Lyric\n" + lyrics + "\n"))
        return {};
    ids.push_back(TOKEN_IM_END);
    if (!append_text("\n"))
        return {};

    // Assistant turn — OPEN (no <think> block injected).
    // The model generates its OWN <think> reasoning block (genre analysis,
    // BPM, key, etc.), then </think>, then audio codes. This matches
    // ref-acestep/src/prompt.h:171-187 (build_lm_prompt — Phase 1 prompt).
    // Previously we injected a hardcoded CoT and immediately entered
    // codes_phase, which skipped the model's reasoning and produced
    // generic ambient output regardless of caption.
    ids.push_back(TOKEN_IM_START);
    if (!append_text("assistant\n"))
        return {};

    return ids;
}

// LM UNCONDITIONAL prompt — for classifier-free guidance (CFG).
// Matches ref-acestep/src/prompt.h:301-327 (build_lm_prompt_uncond_with_cot).
//
//   <|im_start|>system\n# Instruction\n{LM_INSTRUCTION}\n\n<|im_end|>\n
//   <|im_start|>user\n{negative_prompt or ""}<|im_end|>\n
//   <|im_start|>assistant\n<think>\n\n</think>\n\n
//
// Differences from cond prompt:
//   • User turn has no "# Caption\n" / "# Lyric\n" headers — just the
//     negative_prompt (or empty). Matches training CFG dropout.
//   • Assistant turn has EMPTY reasoning: <think>\n\n</think>. The model
//     has "decided" there's nothing to reason about.
//   • Ends with </think>\n\n — ready to emit codes immediately (no Phase A
//     reasoning loop needed for uncond).
static std::vector<int32_t> build_lm_uncond_prompt_tokens(
        const qwen3::Runner& lm,
        const std::string& negative_prompt,
        std::string* error) {
    std::vector<int32_t> ids;

    auto append_text = [&](const std::string& text) -> bool {
        std::vector<int32_t> toks;
        if (!lm.tokenize(text, /*add_special=*/false, /*parse_special=*/false,
                         &toks, nullptr, error))
            return false;
        ids.insert(ids.end(), toks.begin(), toks.end());
        return true;
    };

    // System turn — same instruction as cond.
    ids.push_back(TOKEN_IM_START);
    if (!append_text("system\n# Instruction\n" + std::string(LM_INSTRUCTION) + "\n\n"))
        return {};
    ids.push_back(TOKEN_IM_END);
    if (!append_text("\n"))
        return {};

    // User turn — bare negative prompt (no Caption/Lyric wrappers).
    // Empty by default (pure unconditional).
    ids.push_back(TOKEN_IM_START);
    if (!append_text("user\n" + negative_prompt))
        return {};
    ids.push_back(TOKEN_IM_END);
    if (!append_text("\n"))
        return {};

    // Assistant turn — EMPTY CoT: <think>\n\n</think>\n\n
    // The model has already "finished reasoning" and is ready for codes.
    ids.push_back(TOKEN_IM_START);
    if (!append_text("assistant\n"))
        return {};
    ids.push_back(TOKEN_THINK);
    if (!append_text("\n\n"))
        return {};
    ids.push_back(TOKEN_THINK_END);
    if (!append_text("\n\n"))
        return {};

    return ids;
}

// Lyrics formatting for the DiT lyric encoder. Matches upstream
// _format_lyrics: "# Languages\n{lang}\n\n# Lyric\n{lyrics}<|endoftext|>"
static std::string format_lyrics(const std::string& lyrics,
                                 const std::string& lang = "en") {
    return "# Languages\n" + lang + "\n\n# Lyric\n" + lyrics + "<|endoftext|>";
}

// ═════════════════════════════════════════════════════════════════════════════
//  Two-Phase LM: metadata struct, CoT parser, YAML builder, Phase 2 prompt
// ═════════════════════════════════════════════════════════════════════════════
//
// Ported from ref-acestep/src/prompt.h. The reference splits LM generation
// into Phase 1 (free reasoning → parse metadata) and Phase 2 (deterministic
// CoT YAML injection → code generation). The model was trained where the
// code-generation prompt ALWAYS contains clean YAML between <think> and
// </think>. Generating codes after free-form (non-YAML) reasoning produces
// out-of-distribution input → code collapse → generic ambient audio.

// Metadata struct — mirrors ref-acestep/src/prompt.h:AcePrompt
struct AceMetadata {
    std::string caption;
    std::string lyrics;
    float       duration     = 0.0f;
    int         bpm          = 0;
    std::string keyscale;
    std::string timesignature;
    std::string vocal_language;
};

// Parse CoT reasoning text → metadata fields.
// Ported from ref-acestep/src/prompt.h:parse_cot_and_lyrics.
static bool parse_cot_text(const std::string& text, AceMetadata* out) {
    size_t ts = text.find("<think>");
    size_t te = text.find("</think>");

    std::string cot;
    std::string lyrics_after;

    if (ts != std::string::npos && te != std::string::npos) {
        cot          = text.substr(ts + 7, te - ts - 7);
        lyrics_after = text.substr(te + 8);
    } else if (te != std::string::npos) {
        cot          = text.substr(0, te);
        lyrics_after = text.substr(te + 8);
    } else {
        cot = text;
    }

    auto get_field = [&](const std::string& key) -> std::string {
        std::string needle = key + ":";
        size_t      p      = cot.find(needle);
        if (p == std::string::npos) return "";
        p += needle.size();
        while (p < cot.size() && (cot[p] == ' ' || cot[p] == '\'')) p++;
        size_t end = cot.find('\n', p);
        if (end == std::string::npos) end = cot.size();
        std::string val = cot.substr(p, end - p);
        while (!val.empty() && (val.back() == ' ' || val.back() == '\'' || val.back() == '\r'))
            val.pop_back();
        return val;
    };

    std::string bpm_s = get_field("bpm");
    if (!bpm_s.empty()) out->bpm = atoi(bpm_s.c_str());

    std::string dur_s = get_field("duration");
    if (!dur_s.empty()) out->duration = (float)atof(dur_s.c_str());

    out->keyscale       = get_field("keyscale");
    out->timesignature  = get_field("timesignature");
    out->vocal_language = get_field("language");

    // Caption: may span multiple lines (YAML word-wrap).
    {
        std::string cap = get_field("caption");
        if (!cap.empty()) {
            size_t cp = cot.find("caption:");
            if (cp != std::string::npos) {
                cp += 8;
                size_t end = cot.find("\nduration:", cp);
                if (end == std::string::npos) end = cot.find("\nkeyscale:", cp);
                if (end == std::string::npos) end = cot.size();
                std::string full_cap = cot.substr(cp, end - cp);
                std::string cleaned;
                bool in_space = true;
                for (char ch : full_cap) {
                    if (ch == '\n' || ch == '\r') ch = ' ';
                    if (ch == ' ') {
                        if (!in_space) cleaned += ' ';
                        in_space = true;
                    } else {
                        cleaned += ch;
                        in_space = false;
                    }
                }
                while (!cleaned.empty() && cleaned.back() == ' ') cleaned.pop_back();
                while (!cleaned.empty() && cleaned.front() == ' ') cleaned.erase(cleaned.begin());
                if (!cleaned.empty()) out->caption = cleaned;
            }
        }
    }

    // Lyrics after </think>
    if (!lyrics_after.empty()) {
        size_t s = lyrics_after.find_first_not_of(" \t\n\r");
        if (s != std::string::npos)
            lyrics_after = lyrics_after.substr(s);
        size_t lp = lyrics_after.find("# Lyric\n");
        if (lp != std::string::npos && lp < 64)
            lyrics_after = lyrics_after.substr(lp + 8);
        while (!lyrics_after.empty() &&
               (lyrics_after.back() == ' ' || lyrics_after.back() == '\n' || lyrics_after.back() == '\r'))
            lyrics_after.pop_back();
        if (!lyrics_after.empty())
            out->lyrics = lyrics_after;
    }

    return (out->bpm > 0 || out->duration > 0);
}

// Build deterministic CoT YAML from metadata.
// Ported from ref-acestep/src/prompt.h:build_cot_yaml.
// Fields are alphabetically sorted (matching Python yaml.dump sort_keys=True).
static std::string build_cot_yaml(const AceMetadata& m) {
    auto yaml_wrap = [](const std::string& key, const std::string& val) -> std::string {
        std::string result = key + ":";
        int         col    = (int)(key.size() + 1);
        size_t      i      = 0;
        while (i < val.size()) {
            size_t end = val.find(' ', i);
            if (end == std::string::npos) end = val.size();
            std::string word = val.substr(i, end - i);
            if (col > 80) {
                result += "\n  ";
                col = 2;
            } else {
                result += " ";
                col += 1;
            }
            result += word;
            col += (int)word.size();
            i = (end < val.size()) ? end + 1 : val.size();
        }
        result += "\n";
        return result;
    };

    std::string yaml;
    if (m.bpm > 0)
        yaml += "bpm: " + std::to_string(m.bpm) + "\n";
    if (!m.caption.empty())
        yaml += yaml_wrap("caption", m.caption);
    if (m.duration > 0)
        yaml += "duration: " + std::to_string((int)m.duration) + "\n";
    if (!m.keyscale.empty())
        yaml += "keyscale: " + m.keyscale + "\n";
    if (!m.vocal_language.empty())
        yaml += "language: " + m.vocal_language + "\n";
    if (!m.timesignature.empty())
        yaml += "timesignature: " + m.timesignature + "\n";
    return yaml;
}

// Normalize timesignature: "4/4" → "4", "6/8" → "6".
static std::string normalize_timesig(const std::string& ts) {
    size_t slash = ts.find('/');
    if (slash != std::string::npos) return ts.substr(0, slash);
    return ts;
}

// Build Phase 2 prompt: ...assistant\n<think>\n{cot_yaml}</think>\n\n
// Ported from ref-acestep/src/prompt.h:build_lm_prompt_with_cot.
static std::vector<int32_t> build_phase2_prompt_tokens(
    const qwen3::Runner& lm,
    const AceMetadata&   m,
    const std::string&   cot_yaml,
    std::string*         error) {

    std::vector<int32_t> ids;

    auto append_text = [&](const std::string& text) -> bool {
        std::vector<int32_t> toks;
        if (!lm.tokenize(text, /*add_special=*/false, /*parse_special=*/false,
                         &toks, nullptr, error))
            return false;
        ids.insert(ids.end(), toks.begin(), toks.end());
        return true;
    };

    // System turn
    ids.push_back(TOKEN_IM_START);
    if (!append_text("system\n# Instruction\n" + std::string(LM_INSTRUCTION) + "\n\n"))
        return {};
    ids.push_back(TOKEN_IM_END);
    if (!append_text("\n"))
        return {};

    // User turn
    ids.push_back(TOKEN_IM_START);
    if (!append_text("user\n# Caption\n" + m.caption + "\n\n# Lyric\n" + m.lyrics + "\n"))
        return {};
    ids.push_back(TOKEN_IM_END);
    if (!append_text("\n"))
        return {};

    // Assistant turn with injected CoT YAML
    ids.push_back(TOKEN_IM_START);
    if (!append_text("assistant\n"))
        return {};
    ids.push_back(TOKEN_THINK);                        // raw ID: 151667
    if (!append_text("\n" + cot_yaml))
        return {};
    ids.push_back(TOKEN_THINK_END);                     // raw ID: 151668
    if (!append_text("\n\n"))
        return {};

    return ids;
}

bool AceStepSession::run_lm(const MusicRequest& req,
                            std::vector<int32_t>* music_codes,
                            std::string* error) {
    // (0) Reset LM + TE KV caches so this run is independent of prior calls.
    //     Without this, the second call's prefill at position 0 collides
    //     with the cached positions from the first call and llama_decode
    //     fails with "inconsistent sequence positions".
    if (lm_) lm_->clear_kv_cache();
    if (te_) te_->clear_kv_cache();

    // (1) Build LM prompt (TE encoding is deferred to after Phase 1 so it
    //     can use the enriched metadata — bpm, key, timesig from the LM's
    //     reasoning. The reference builds the TE prompt from the LM-enriched
    //     AceRequest, not the raw user request.)
    fprintf(stderr, "[ace_step] run_lm: tokenizing (caption='%s', dur=%.1f)...\n",
            req.caption.c_str(), req.duration);

    // Build the LM prompt as RAW TOKEN IDs (not string→tokenize).
    // The chat-template special tokens (<|im_start|>, <|im_end|>, <think>,
    // </think>) must be exact single-token IDs. The Qwen3 GGUF tokenizer
    // may not parse <think>/</think> from text, so we push raw IDs directly.
    // Matches ref-acestep/src/prompt.h:270-296.
    std::vector<int32_t> lm_prompt = build_lm_prompt_tokens(
        *lm_, req.caption, req.lyrics, req.duration, error);
    if (lm_prompt.empty()) {
        if (error && error->empty())
            *error = "ACE-Step: failed to build LM prompt tokens";
        return false;
    }

    fprintf(stderr, "[ace_step] run_lm: LM prompt (%zu tokens), first 3: "
                    "%d %d %d, last 5:",
            lm_prompt.size(),
            lm_prompt.size() > 0 ? lm_prompt[0] : -1,
            lm_prompt.size() > 1 ? lm_prompt[1] : -1,
            lm_prompt.size() > 2 ? lm_prompt[2] : -1);
    for (int i = std::max(0, (int)lm_prompt.size() - 5); i < (int)lm_prompt.size(); i++)
        fprintf(stderr, " %d", lm_prompt[i]);
    fprintf(stderr, "\n");

    // Verify the first token is TOKEN_IM_START (sanity check).
    if (lm_prompt[0] != TOKEN_IM_START) {
        fprintf(stderr, "[ace_step] WARNING: LM prompt first token=%d, expected "
                        "%d (TOKEN_IM_START)\n", lm_prompt[0], TOKEN_IM_START);
    }

    if (lm_prompt.empty()) {
        if (error) *error = "ACE-Step: tokenization produced empty result";
        return false;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  TWO-PHASE LM: Phase 1 (reasoning) → parse → Phase 2 (codes)
    // ═══════════════════════════════════════════════════════════════════════════
    //
    // The reference (ref-acestep/src/pipeline-lm.cpp) splits LM generation:
    //   Phase 1: free reasoning → parse metadata (bpm, key, duration, etc.)
    //   Phase 2: deterministic CoT YAML injection → code generation
    //
    // The model was trained where the code-generation prompt ALWAYS contains
    // clean YAML between <think> and </think>. Phase 2 rebuilds this YAML
    // deterministically from parsed metadata, ensuring in-distribution input.
    // Generating codes after free-form (non-YAML) reasoning produces OOD input
    // → code collapse → generic ambient audio regardless of caption.
    //
    // Optimization: if the user provides ALL metadata fields, Phase 1 is
    // skipped entirely — we go straight to Phase 2 with the user's values.

    const int32_t vs = lm_->vocab_size();

    // ── Build base metadata from user request ──
    AceMetadata meta;
    meta.caption        = req.caption;
    meta.lyrics         = req.lyrics;
    meta.duration       = req.duration;
    meta.bpm            = req.bpm;
    meta.keyscale       = req.keyscale;
    meta.timesignature  = req.timesignature.empty() ? std::string()
                                                    : normalize_timesig(req.timesignature);
    meta.vocal_language = req.vocal_language;

    // Can we skip Phase 1? (all metadata provided by user)
    const bool skip_phase1 = (meta.bpm > 0 &&
                              !meta.keyscale.empty() &&
                              !meta.timesignature.empty() &&
                              !meta.vocal_language.empty() &&
                              meta.duration > 0 &&
                              !meta.caption.empty());

    // ── Constants & helpers ──
    PhiloxRng rng{req.seed != 0 ? req.seed : 42};

    Params sp_reason;
    sp_reason.temperature = req.temperature;
    sp_reason.top_p       = req.top_p;
    sp_reason.do_sample   = req.temperature > 0.0f;
    Params sp_codes = sp_reason;

    auto mask_to_codes = [](float* logits, int32_t vocab_size) {
        for (int32_t v = 0; v < AUDIO_CODE_BASE && v < vocab_size; v++) {
            if (v != TOKEN_IM_END)
                logits[v] = -1e9f;
        }
        for (int32_t v = AUDIO_CODE_BASE + FSQ_CODE_COUNT; v < vocab_size; v++) {
            logits[v] = -1e9f;
        }
    };

    auto mask_codes_out = [](float* logits, int32_t vocab_size) {
        for (int32_t v = AUDIO_CODE_BASE; v < AUDIO_CODE_BASE + FSQ_CODE_COUNT && v < vocab_size; v++) {
            logits[v] = -1e9f;
        }
    };

    auto sample_free = [&](float* logits, int32_t vocab_size,
                           const Params& sp, PhiloxRng* prng) -> int32_t {
        return sample_token(logits, vocab_size, sp, nullptr, 0, prng);
    };

    auto sample_code = [&](float* logits, int32_t vocab_size,
                           const Params& sp, PhiloxRng* prng) -> int32_t {
        const int32_t compact_V = FSQ_CODE_COUNT + 1;
        std::vector<float> compact(compact_V);
        compact[0] = logits[TOKEN_IM_END];
        for (int32_t c = 0; c < FSQ_CODE_COUNT; c++)
            compact[c + 1] = logits[AUDIO_CODE_BASE + c];
        const int32_t compact_tok = sample_token(compact.data(), compact_V, sp,
                                                  nullptr, 0, prng);
        if (compact_tok == 0)
            return -1;  // EOS
        return compact_tok - 1;  // FSQ code index
    };

    // ════════════════════════════════════════════════════════════════════════
    //  Phase 1: Free Reasoning (infer missing metadata)
    // ════════════════════════════════════════════════════════════════════════
    // The model generates <think>, reasoning text (genre analysis, BPM, key,
    // etc.), then </think>. We collect the tokens, detokenize, and parse the
    // YAML-like metadata fields. User-provided values take priority — Phase 1
    // only gap-fills fields the user left empty.
    if (!skip_phase1) {
        fprintf(stderr, "[ace_step] run_lm: Phase 1 — free reasoning (inferring "
                        "missing metadata)...\n");
        const int32_t p1_len = static_cast<int32_t>(lm_prompt.size());
        std::vector<float> p1_logits(static_cast<size_t>(p1_len) * vs, 0.0f);
        if (!lm_->forward_tokens(lm_prompt.data(), p1_len, 0,
                                 p1_logits.data(), error))
            return false;
        fprintf(stderr, "[ace_step] run_lm: Phase 1 prefill done (%d tokens)\n", p1_len);

        // Free reasoning until </think> or safety limit
        std::vector<int32_t> reasoning_tokens;
        const int32_t max_reasoning_tokens = 2048;

        // First token from prefill logits (last row)
        float* last_row = &p1_logits[static_cast<size_t>(p1_len - 1) * vs];
        mask_codes_out(last_row, vs);
        int32_t prev = sample_free(last_row, vs, sp_reason, &rng);
        reasoning_tokens.push_back(prev);
        fprintf(stderr, "[ace_step] run_lm: reasoning tok[0]=%d", prev);
        if (prev == TOKEN_THINK)     fprintf(stderr, " (<think>)");
        if (prev == TOKEN_THINK_END) fprintf(stderr, " (</think>)");
        fprintf(stderr, "\n");

        bool got_think_end = (prev == TOKEN_THINK_END || prev == TOKEN_IM_END);
        if (prev == TOKEN_IM_END) {
            fprintf(stderr, "[ace_step] run_lm: WARNING — LM emitted EOS before "
                            "reasoning, using default metadata\n");
            got_think_end = true;
        }

        int32_t n_pos = p1_len;
        while (!got_think_end) {
            if (n_pos - p1_len >= max_reasoning_tokens) {
                fprintf(stderr, "[ace_step] run_lm: reasoning exceeded %d tokens, "
                                "stopping\n", max_reasoning_tokens);
                break;
            }

            std::vector<float> logits(static_cast<size_t>(vs));
            if (!lm_->forward_tokens(&prev, 1, n_pos, logits.data(), error))
                return false;
            n_pos++;

            mask_codes_out(logits.data(), vs);
            int32_t tok = sample_free(logits.data(), vs, sp_reason, &rng);
            reasoning_tokens.push_back(tok);

            if (tok == TOKEN_THINK_END || tok == TOKEN_IM_END) {
                got_think_end = true;
                fprintf(stderr, "[ace_step] run_lm: </think> at pos %d (reasoning "
                                "took %zu tokens)\n", n_pos, reasoning_tokens.size());
            }
            prev = tok;
        }

        // Detokenize reasoning → text
        std::string reasoning_text;
        for (auto tok : reasoning_tokens) {
            std::string piece;
            if (lm_->token_to_piece(tok, &piece))
                reasoning_text += piece;
        }
        fprintf(stderr, "[ace_step] run_lm: Phase 1 reasoning text:\n%s\n",
                reasoning_text.c_str());

        // Parse reasoning → metadata, gap-fill (user values take priority)
        AceMetadata parsed;
        parse_cot_text(reasoning_text, &parsed);
        if (meta.bpm == 0 && parsed.bpm > 0)
            meta.bpm = parsed.bpm;
        if (meta.duration <= 0 && parsed.duration > 0)
            meta.duration = parsed.duration;
        if (meta.keyscale.empty() && !parsed.keyscale.empty())
            meta.keyscale = parsed.keyscale;
        if (meta.timesignature.empty() && !parsed.timesignature.empty())
            meta.timesignature = normalize_timesig(parsed.timesignature);
        if ((meta.vocal_language.empty() || meta.vocal_language == "unknown") &&
            !parsed.vocal_language.empty())
            meta.vocal_language = parsed.vocal_language;
        // Use parsed lyrics if user provided none
        if (meta.lyrics.empty() && !parsed.lyrics.empty())
            meta.lyrics = parsed.lyrics;

        // ── Caption enrichment (use_cot_caption=true, reference default) ──
        // The reference (request.cpp:36) defaults use_cot_caption=true, meaning
        // the LM's Phase 1 enriched caption REPLACES the user's original for
        // Phase 2 code generation and the TE prompt. This is critical for
        // short captions like "ROCK" — the Phase 1 reasoning expands it to a
        // detailed description ("An energetic and raw instrumental rock track
        // driven by a dual electric guitar attack...") which the LM uses to
        // generate genre-appropriate codes. Without this, "ROCK" produces
        // generic/ambient codes because the short caption lacks the detail
        // the model needs to map to specific audio patterns.
        //
        // The reference's caption preservation (pipeline-lm.cpp:779-785):
        // the LM may enrich the caption but never silently delete it. If
        // the enriched caption is empty, keep the original.
        if (!parsed.caption.empty()) {
            fprintf(stderr, "[ace_step] run_lm: caption enrichment:\n"
                            "  original: '%s'\n"
                            "  enriched: '%s'\n",
                    meta.caption.c_str(), parsed.caption.c_str());
            meta.caption = parsed.caption;
        }

        fprintf(stderr, "[ace_step] run_lm: Phase 1 parsed: bpm=%d duration=%.0f "
                        "keyscale='%s' timesig='%s' lang='%s'\n",
                parsed.bpm, parsed.duration, parsed.keyscale.c_str(),
                parsed.timesignature.c_str(), parsed.vocal_language.c_str());
    } else {
        fprintf(stderr, "[ace_step] run_lm: skipping Phase 1 (all metadata provided)\n");
    }

    // Apply defaults for any remaining missing fields
    if (meta.duration <= 0) meta.duration = 120.0f;
    if (meta.duration > 600) meta.duration = 600.0f;
    if (meta.vocal_language.empty()) meta.vocal_language = "unknown";

    fprintf(stderr, "[ace_step] run_lm: merged metadata: bpm=%d duration=%.0f "
                    "keyscale='%s' timesig='%s' lang='%s'\n",
            meta.bpm, meta.duration, meta.keyscale.c_str(),
            meta.timesignature.c_str(), meta.vocal_language.c_str());

    // ════════════════════════════════════════════════════════════════════════
    //  TE Encoding (DEFERRED from above — now uses enriched metadata)
    // ════════════════════════════════════════════════════════════════════════
    // The TE prompt must use the Phase 1 enriched metadata (bpm, key, timesig),
    // NOT hardcoded "N/A". The reference builds the TE prompt from the
    // LM-enriched AceRequest (pipeline-synth-ops.cpp:470). Seeing "N/A" for
    // all metadata fields is OOD and degrades DiT cross-attention conditioning.
    {
        const int32_t te_hs = cfg_.encoder_hidden_size;
        std::string te_prompt = format_te_prompt(
            meta.caption, meta.lyrics, meta.duration,
            meta.bpm, meta.keyscale, meta.timesignature);
        fprintf(stderr, "[ace_step] run_lm: TE prompt (enriched):\n%s\n", te_prompt.c_str());

        std::vector<int32_t> te_tokens;
        if (!te_->tokenize(te_prompt, /*add_special=*/true, /*parse_special=*/true,
                           &te_tokens, nullptr, error))
            return false;

        // TE text encoding → hidden states cached for run_dit_and_vae
        fprintf(stderr, "[ace_step] run_lm: TE forward_get_embeddings (n_tok=%zu)...\n", te_tokens.size());
        te_cond_len_ = static_cast<int32_t>(te_tokens.size());
        te_cond_.resize(static_cast<size_t>(te_cond_len_) * te_hs);
        if (!te_->forward_get_embeddings(te_tokens.data(), te_cond_len_, 0,
                                         te_cond_.data(), error))
            return false;
        fprintf(stderr, "[ace_step] run_lm: TE forward_get_embeddings done\n");

        // Lyric encoding: tokenize the lyric string and get raw token embeddings.
        // Uses enriched lyrics/language from Phase 1, not the raw request.
        std::string lyrics_str = format_lyrics(meta.lyrics, meta.vocal_language);
        fprintf(stderr, "[ace_step] run_lm: lyric tokenize+embed_lookup...\n");
        std::vector<int32_t> lyric_tok;
        if (!te_->tokenize(lyrics_str, /*add_special=*/true, /*parse_special=*/true,
                           &lyric_tok, nullptr, error))
            return false;
        lyric_cond_len_ = static_cast<int32_t>(lyric_tok.size());
        lyric_cond_.resize(static_cast<size_t>(lyric_cond_len_) * te_hs);
        if (lyric_cond_len_ > 0) {
            if (!te_->embed_lookup(lyric_tok.data(), lyric_cond_len_,
                                   lyric_cond_.data(), error))
                return false;
        }
        fprintf(stderr, "[ace_step] run_lm: lyric embed_lookup done (n_tok=%d)\n",
                lyric_cond_len_);

        // Null lyric for CFG
        std::vector<int32_t> null_lyric_tok;
        if (!te_->tokenize("", true, true, &null_lyric_tok, nullptr, error))
            return false;
        const int32_t n_null_lyric = static_cast<int32_t>(null_lyric_tok.size());
        if (n_null_lyric > 0) {
            lyric_uncond_.resize(static_cast<size_t>(n_null_lyric) * te_hs);
            if (!te_->embed_lookup(null_lyric_tok.data(), n_null_lyric,
                                   lyric_uncond_.data(), error))
                return false;
        } else {
            lyric_uncond_.clear();
        }

        // Null-text TE embedding for CFG (DiT side)
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
            te_uncond_.clear();
        }
        fprintf(stderr, "[ace_step] run_lm: null-text TE done\n");
    }

    // ════════════════════════════════════════════════════════════════════════
    //  Phase 2: Code Generation with Injected CoT
    // ════════════════════════════════════════════════════════════════════════
    // Build deterministic CoT YAML from merged metadata. This YAML is injected
    // between <think> and </think> in the Phase 2 prompt, exactly matching the
    // format the model was trained on. The model starts generating codes
    // immediately after </think>\n\n — codes_phase = true from the first token.
    std::string cot_yaml = build_cot_yaml(meta);
    fprintf(stderr, "[ace_step] run_lm: Phase 2 CoT YAML:\n%s", cot_yaml.c_str());

    std::vector<int32_t> phase2_prompt = build_phase2_prompt_tokens(
        *lm_, meta, cot_yaml, error);
    if (phase2_prompt.empty()) {
        if (error && error->empty())
            *error = "ACE-Step: failed to build Phase 2 prompt";
        return false;
    }

    // RESET KV cache — Phase 2 is a completely new forward pass.
    // The model never sees its own free-form reasoning during code generation;
    // it only sees the clean, deterministic YAML.
    lm_->clear_kv_cache();

    // CFG: prepare unconditional Phase 2 prompt on secondary context.
    // The uncond prompt has empty CoT (<think>\n\n</think>) and bare user
    // content (no Caption/Lyric wrappers) — matches training CFG dropout.
    const bool use_cfg = req.lm_cfg_scale > 1.0f + 1e-6f;
    int32_t uncond_p2_len = 0;
    std::vector<float> uncond_p2_logits;
    if (use_cfg) {
        if (!lm_->has_secondary_context()) {
            if (!lm_->init_secondary_context(error)) return false;
            fprintf(stderr, "[ace_step] run_lm: initialized secondary LM context for CFG\n");
        }
        lm_->clear_secondary_kv_cache();
        std::vector<int32_t> uncond_prompt = build_lm_uncond_prompt_tokens(
            *lm_, /*negative_prompt=*/"", error);
        if (uncond_prompt.empty()) {
            if (error && error->empty())
                *error = "ACE-Step: failed to build uncond LM prompt";
            return false;
        }
        uncond_p2_len = static_cast<int32_t>(uncond_prompt.size());
        uncond_p2_logits.assign(
            static_cast<size_t>(uncond_p2_len) * vs, 0.0f);
        if (!lm_->forward_tokens_secondary(uncond_prompt.data(),
                                           uncond_p2_len, 0,
                                           uncond_p2_logits.data(), error))
            return false;
        fprintf(stderr, "[ace_step] run_lm: uncond Phase 2 prefill done "
                        "(len=%d, cfg_scale=%.2f)\n", uncond_p2_len, req.lm_cfg_scale);
    }

    // Prefill Phase 2 on cond context
    const int32_t p2_len = static_cast<int32_t>(phase2_prompt.size());
    std::vector<float> p2_logits(static_cast<size_t>(p2_len) * vs, 0.0f);
    if (!lm_->forward_tokens(phase2_prompt.data(), p2_len, 0,
                             p2_logits.data(), error))
        return false;
    fprintf(stderr, "[ace_step] run_lm: Phase 2 prefill done (%d tokens)\n", p2_len);

    // ── Sample codes (codes_phase = true from the very first token) ──
    const int32_t n_codes = std::max(1, static_cast<int32_t>(meta.duration * 5.0f + 0.5f));
    music_codes->clear();
    music_codes->reserve(static_cast<size_t>(n_codes));
    fprintf(stderr, "[ace_step] run_lm: Phase 2 code generation (n_codes=%d, "
                    "CFG=%s)\n", n_codes, use_cfg ? "ON" : "OFF");

    int32_t prev_token;
    int32_t n_pos = p2_len;
    int32_t uncond_n_pos = uncond_p2_len;

    // First code from Phase 2 prefill logits (last row)
    {
        float* last_row = &p2_logits[static_cast<size_t>(p2_len - 1) * vs];

        // CFG combine for code[0]
        if (use_cfg) {
            const float* uncond_last = &uncond_p2_logits[
                static_cast<size_t>(uncond_p2_len - 1) * vs];
            const float scale = req.lm_cfg_scale;
            for (int32_t v = 0; v < vs; v++)
                last_row[v] = uncond_last[v] + scale * (last_row[v] - uncond_last[v]);
        }

        // Diagnostic: top-5 codes for first token
        {
            std::vector<std::pair<float, int32_t>> code_logits;
            code_logits.reserve(FSQ_CODE_COUNT);
            for (int32_t c = 0; c < FSQ_CODE_COUNT; c++)
                code_logits.push_back({last_row[AUDIO_CODE_BASE + c], c});
            std::partial_sort(code_logits.begin(), code_logits.begin() + 5,
                              code_logits.end(), [](auto& a, auto& b) {
                                  return a.first > b.first;
                              });
            fprintf(stderr, "[ace_step] logits[code0]: EOS=%.3f top5=",
                    last_row[TOKEN_IM_END]);
            for (int k = 0; k < 5; k++)
                fprintf(stderr, "[c%d=%.2f]", code_logits[k].second,
                        code_logits[k].first);
            fprintf(stderr, "\n");
        }

        mask_to_codes(last_row, vs);
        int32_t chosen = sample_code(last_row, vs, sp_codes, &rng);
        if (chosen < 0) {
            if (error) *error = "ACE-Step: LM emitted EOS before generating "
                                "any audio codes";
            return false;
        }
        music_codes->push_back(chosen);
        prev_token = AUDIO_CODE_BASE + chosen;
        fprintf(stderr, "[ace_step] run_lm: code[0/%d]=%d (tok=%d)\n",
                n_codes, chosen, prev_token);
    }

    // Generate remaining codes
    static const bool ACE_DEBUG = std::getenv("ACE_STEP_DEBUG");
    while (static_cast<int32_t>(music_codes->size()) < n_codes) {
        std::vector<float> logits(static_cast<size_t>(vs));
        if (!lm_->forward_tokens(&prev_token, 1, n_pos, logits.data(), error))
            return false;
        n_pos++;

        // CFG: forward on secondary (uncond) context and combine logits
        if (use_cfg) {
            std::vector<float> uncond_logits(static_cast<size_t>(vs), 0.0f);
            if (!lm_->forward_tokens_secondary(&prev_token, 1, uncond_n_pos,
                                               uncond_logits.data(), error))
                return false;
            uncond_n_pos++;

            // Diagnostic: cond vs uncond for code 452 and top code (verbose)
            if (ACE_DEBUG && music_codes->size() < 16) {
                float cond_452 = logits[AUDIO_CODE_BASE + 452];
                float uncond_452 = uncond_logits[AUDIO_CODE_BASE + 452];
                float cond_eos = logits[TOKEN_IM_END];
                float uncond_eos = uncond_logits[TOKEN_IM_END];
                // Find top cond code
                int top_c = 0; float top_l = logits[AUDIO_CODE_BASE];
                for (int32_t c = 1; c < FSQ_CODE_COUNT; c++)
                    if (logits[AUDIO_CODE_BASE + c] > top_l) { top_l = logits[AUDIO_CODE_BASE + c]; top_c = c; }
                fprintf(stderr, "[ace_step] cfg[%zu]: cond452=%.2f uncond452=%.2f | "
                                "condEOS=%.2f uncondEOS=%.2f | topC=%d(%.2f)\n",
                        music_codes->size(), cond_452, uncond_452,
                        cond_eos, uncond_eos, top_c, top_l);
            }

            const float scale = req.lm_cfg_scale;
            for (int32_t v = 0; v < vs; v++)
                logits[v] = uncond_logits[v] + scale * (logits[v] - uncond_logits[v]);
        }

        // Diagnostic: top-5 code logits for first 12 codes (verbose)
        size_t cur_idx = music_codes->size();
        if (ACE_DEBUG && cur_idx < 12) {
            std::vector<std::pair<float, int32_t>> code_logits;
            code_logits.reserve(FSQ_CODE_COUNT);
            for (int32_t c = 0; c < FSQ_CODE_COUNT; c++)
                code_logits.push_back({logits[AUDIO_CODE_BASE + c], c});
            std::partial_sort(code_logits.begin(), code_logits.begin() + 5,
                              code_logits.end(), [](auto& a, auto& b) {
                                  return a.first > b.first;
                              });
            fprintf(stderr, "[ace_step] logits[%zu]: EOS=%.3f top5=", cur_idx,
                    logits[TOKEN_IM_END]);
            for (int k = 0; k < 5; k++) {
                fprintf(stderr, "[c%d=%.2f]", code_logits[k].second,
                        code_logits[k].first);
            }
            fprintf(stderr, "\n");
        }

        // ── Anti-collapse: count-based repetition penalty ───────────────
        // The Q8_0 quantized LM has a tendency to collapse to a single
        // "default" code after ~10 steps (verified: seeds 42/12345/777 all
        // collapse to one code at 60-72% frequency). Standard divisor-based
        // penalty (logit /= 1.1) is too aggressive at our logit scale (60-70):
        // it drops the logit by ~6, which after softmax is a total ban.
        //
        // Instead, use COUNT-BASED FIXED SUBTRACTION. For each code in the
        // last 8 positions, count how many times it appeared, then subtract
        // 2.0 * count. This ramps up exponentially:
        //   1 occurrence → subtract 2.0  (exp(2/0.85) ≈ 10x reduction)
        //   2 occurrences → subtract 4.0 (exp(4/0.85) ≈ 107x reduction)
        //   3 occurrences → subtract 6.0 (effectively banned)
        // This prevents runaway collapse (verified: fixed 3.0 penalty still
        // collapsed to code 35847 × 9) while allowing musical patterns where
        // a code legitimately appears 1-2 times in the window.
        {
            const int32_t n_codes_so_far = static_cast<int32_t>(music_codes->size());
            const int32_t window = std::min(8, n_codes_so_far);
            // Count occurrences of each code in the window
            std::unordered_map<int32_t, int> code_count;
            for (int32_t k = 0; k < window; k++) {
                int32_t recent_code = (*music_codes)[n_codes_so_far - 1 - k];
                code_count[recent_code]++;
            }
            // Apply penalty proportional to occurrence count
            for (auto& kv : code_count) {
                int32_t idx = AUDIO_CODE_BASE + kv.first;
                float penalty = 2.0f * static_cast<float>(kv.second);
                logits[idx] -= penalty;
            }
        }

        mask_to_codes(logits.data(), vs);
        int32_t chosen = sample_code(logits.data(), vs, sp_codes, &rng);

        if (chosen < 0) {
            fprintf(stderr, "[ace_step] run_lm: LM emitted EOS at code %zu/%d\n",
                    music_codes->size(), n_codes);
            break;
        }

        music_codes->push_back(chosen);
        prev_token = AUDIO_CODE_BASE + chosen;
        size_t idx = music_codes->size() - 1;
        if (idx < 3 || idx == static_cast<size_t>(n_codes - 1)) {
            fprintf(stderr, "[ace_step] run_lm: code[%zu/%d]=%d (tok=%d)\n",
                    idx, n_codes, chosen, prev_token);
        }
    }
    fprintf(stderr, "[ace_step] run_lm: decode loop done (%zu codes)\n", music_codes->size());

    // ── Diagnostic: code distribution + dump to file ──
    // Most codes should be different (high entropy). If 90% of codes are the
    // same value, the LM has collapsed and audio will be a drone regardless
    // of caption.
    {
        std::unordered_map<int32_t, int> hist;
        for (int32_t c : *music_codes) hist[c]++;
        int max_count = 0; int32_t max_code = -1;
        for (auto& kv : hist) {
            if (kv.second > max_count) { max_count = kv.second; max_code = kv.first; }
        }
        fprintf(stderr, "[ace_step] run_lm: code histogram: %zu unique codes, "
                        "most-frequent code=%d (%.1f%% = %d/%zu)\n",
                hist.size(), max_code,
                100.0 * max_count / music_codes->size(),
                max_count, music_codes->size());
        // Dump full code sequence for debugging collapse patterns (verbose)
        if (ACE_DEBUG) {
            fprintf(stderr, "[ace_step] code_seq:");
            for (size_t i = 0; i < music_codes->size(); i++)
                fprintf(stderr, " %d", (*music_codes)[i]);
            fprintf(stderr, "\n");
        }
        // Dump all codes for offline analysis
        if (const char* p = std::getenv("ACE_STEP_DUMP_CODES")) {
            FILE* f = std::fopen(p, "wb");
            if (f) {
                uint32_t n = (uint32_t)music_codes->size();
                std::fwrite(&n, sizeof(n), 1, f);
                std::fwrite(music_codes->data(), sizeof(int32_t), n, f);
                std::fclose(f);
                fprintf(stderr, "[ace_step] dumped %u codes to %s\n", n, p);
            }
        }
    }

    // HARD ERROR: need at least 1 code to produce any audio.
    if (music_codes->empty()) {
        if (error) *error = "ACE-Step: LM produced zero audio codes";
        return false;
    }

    // (7) FSQ detokenize the LM codes → 25 Hz source latent for the DiT.
    //     The reference pipeline (pipeline-synth.cpp:321-324,
    //     pipeline-synth-ops.cpp:662-704) ALWAYS runs this path when audio_codes
    //     are present. The detokenized latent fills the DiT's src channels
    //     (0–63), providing the primary musical conditioning from the caption.
    //     Without it the DiT receives only silence as source and the caption
    //     has near-zero effect on the output.
    if (!detokenizer_runner_) {
        if (error) *error = "ACE-Step: detokenizer_runner not initialized — "
                            "cannot produce DiT source latent from LM codes";
        return false;
    }
    fprintf(stderr, "[ace_step] run_lm: FSQ detokenize %zu codes → 25 Hz latent...\n",
            music_codes->size());
    lm_src_T_ = 0;
    lm_src_latents_.clear();
    if (!detokenizer_runner_->decode(music_codes->data(),
                                     static_cast<int32_t>(music_codes->size()),
                                     &lm_src_latents_, error)) {
        if (error) *error = std::string("ACE-Step: FSQ detokenize FAILED — ") +
                            (error ? *error : "(no detail)");
        return false;
    }
    lm_src_T_ = static_cast<int32_t>(music_codes->size()) * 5;
    fprintf(stderr, "[ace_step] run_lm: detokenize done (T_25Hz=%d, %zu floats)\n",
            lm_src_T_, lm_src_latents_.size());

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
    // The reference pipeline (pipeline-synth-ops.cpp:662-724) ALWAYS
    // detokenizes the LM audio codes through the FSQ detokenizer and uses the
    // result as the DiT's source context. The detokenized latent provides the
    // primary musical conditioning — the caption's influence flows through the
    // LM → codes → detokenize → DiT src path. Frames beyond the detokenized
    // length are padded with silence_latent.
    //
    // Previously we used silence_latent as src for text_to_music, which
    // completely bypassed the LM codes. This caused the caption to have
    // near-zero effect on the output — the DiT had only the weak cross-attention
    // text signal, with no musical content from the LM.
    //
    // HARD ERROR: if lm_src_latents_ is empty, the detokenizer did not run.
    // This is a fatal error — no silent fallback to silence.

    // ── 2. Prepare conditioning for DiT ─────────────────────────────────────
    if (!is_cover_mode && (te_cond_.empty() || te_cond_len_ <= 0)) {
        if (error) *error = "ACE-Step: run_lm must execute before run_dit_and_vae";
        return false;
    }
    // HARD ERROR: detokenized LM source latent must be present for text2music.
    // Without it the DiT runs on silence and the caption is ignored.
    if (!is_cover_mode && repaint_latent_cond_.empty() && lm_src_latents_.empty()) {
        if (error) *error = "ACE-Step: lm_src_latents_ is empty — the FSQ detokenizer "
                            "did not run. Cannot generate music without LM source latent.";
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
    // ── RNG ──
    // Upstream uses Philox4x32-10 + Box-Muller with BF16 round-trip to match
    // torch.randn(..., generator=cuda_manual_seed(seed), dtype=bfloat16).
    // std::mt19937 + std::normal_distribution gives a *different* sequence
    // for the same seed, which prevents layer-by-layer numerical comparison
    // against the upstream reference. Use Philox to match exactly.
    const int64_t noise_seed = static_cast<int64_t>(req.seed != 0 ? req.seed : 42);
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
    // CRITICAL: DiTRunner::ensure_backend() migrates ALL ext_ctx_ tensors to
    // CUDA. After migration, tensor->data is a GPU pointer that CANNOT be
    // dereferenced from CPU. We cache silence_latent + biases on first access
    // (before migration) and reuse the CPU copies on all subsequent requests.
    const float* silence_ptr = nullptr;
    int32_t silence_T = 0;
    const float* proj_in_bias  = nullptr;
    const float* proj_out_bias = nullptr;

    if (!cpu_caches_ready_) {
        // First request: tensors are still on CPU (migration happens later
        // during DiT forward → ensure_backend). Read and cache them now.
        ggml_tensor* st = ggml_get_tensor(ext_ctx_, "silence_latent");
        if (st) {
            silence_T_cache_ = static_cast<int32_t>(st->ne[1]);
            const int32_t D = static_cast<int32_t>(st->ne[0]);
            const size_t N = static_cast<size_t>(silence_T_cache_) * D;
            silence_latent_cache_.resize(N);
            std::memcpy(silence_latent_cache_.data(), st->data, N * sizeof(float));

            // Sanity stats — silence_latent should be ~N(0, 0.5)-ish.
            double sum = 0.0, sq = 0.0; float mn = 1e30f, mx = -1e30f;
            for (size_t i = 0; i < N; i++) {
                float v = silence_latent_cache_[i];
                sum += v; sq += (double)v * v;
                if (v < mn) mn = v; if (v > mx) mx = v;
            }
            fprintf(stderr, "[ace_step] silence_latent [D=%d, T=%d] flat-time-major "
                            "mean=%.4f RMS=%.4f range=[%.4f,%.4f] (cached to CPU)\n",
                    D, silence_T_cache_, sum / N, std::sqrt(sq / N), mn, mx);
        }

        // Cache proj_in / proj_out biases
        ggml_tensor* pi_b = ggml_get_tensor(ext_ctx_, "decoder.proj_in.1.bias");
        ggml_tensor* po_b = ggml_get_tensor(ext_ctx_, "decoder.proj_out.1.bias");
        if (pi_b) {
            size_t n = static_cast<size_t>(pi_b->ne[0]);
            proj_in_bias_cache_.resize(n);
            std::memcpy(proj_in_bias_cache_.data(), pi_b->data, n * sizeof(float));
        }
        if (po_b) {
            size_t n = static_cast<size_t>(po_b->ne[0]);
            proj_out_bias_cache_.resize(n);
            std::memcpy(proj_out_bias_cache_.data(), po_b->data, n * sizeof(float));
        }
        cpu_caches_ready_ = true;
    }

    // Use cached CPU copies (valid even after CUDA migration)
    silence_ptr = silence_latent_cache_.empty() ? nullptr : silence_latent_cache_.data();
    silence_T   = silence_T_cache_;
    proj_in_bias  = proj_in_bias_cache_.empty()  ? nullptr : proj_in_bias_cache_.data();
    proj_out_bias = proj_out_bias_cache_.empty() ? nullptr : proj_out_bias_cache_.data();

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

    // Initialize noise latent [T_latent, out_ch] using Philox4x32-10 to
    // match torch.randn(..., generator=cuda_manual_seed(seed), dtype=bfloat16)
    // exactly. bf16_round=true emulates the bf16 storage round-trip that
    // PyTorch applies on CUDA.
    std::vector<float> noise_latent(static_cast<size_t>(T_latent) * out_ch);
    sampler::philox_randn(noise_seed, noise_latent.data(),
                          static_cast<int>(noise_latent.size()),
                          /*bf16_round=*/true);

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
            // text_to_music: src = detokenized LM codes (PRIMARY musical
            // conditioning). The LM generates music codes conditioned on the
            // caption; the FSQ detokenizer converts them to a 25 Hz latent
            // that serves as the DiT's source context. Matches reference
            // pipeline-synth-ops.cpp:714-718. Frames beyond the detokenized
            // length are padded with silence_latent.
            if (t < lm_src_T_) {
                std::memcpy(row,
                            &lm_src_latents_[static_cast<size_t>(t) * out_ch],
                            static_cast<size_t>(out_ch) * sizeof(float));
            } else {
                int32_t sil_idx = t - lm_src_T_;
                if (silence_ptr && sil_idx < silence_T)
                    std::memcpy(row,
                                silence_ptr + static_cast<size_t>(sil_idx) * out_ch,
                                static_cast<size_t>(out_ch) * sizeof(float));
            }
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

    // Turbo checkpoints have classifier-free guidance distilled into the
    // velocity head (pipeline_ace_step.py:920-925): running CFG on top of
    // that double-counts guidance and shatters the prediction. Mirror the
    // upstream behavior and force gs=1.0 for turbo.
    const bool is_turbo = (cfg_.variant == "turbo" || cfg_.variant.empty());
    const float effective_gs = is_turbo ? 1.0f : req.guidance_scale;
    if (is_turbo && req.guidance_scale > 1.0f) {
        fprintf(stderr,
                "[ace_step] NOTE: requested guidance_scale=%.2f ignored for "
                "turbo checkpoint (guidance is distilled into weights); "
                "using gs=1.0\n",
                req.guidance_scale);
    }

    std::vector<float> hidden(static_cast<size_t>(S_patches) * H);
    std::vector<float> v_hidden(static_cast<size_t>(S_patches) * H);
    std::vector<float> v_latent(static_cast<size_t>(T_latent) * out_ch);

    // DEBUG: env var ACE_STEP_VAE_ONLY=1 skips the DiT loop entirely and
    // decodes silence_latent directly through the VAE. silence_latent is
    // the model's learned "silent source" — VAE decoding it should yield
    // near-silence (a faint hiss at most), NOT noise/whirl/tone. If you
    // hear noise, the VAE itself is broken.
    const bool vae_only_debug = (std::getenv("ACE_STEP_VAE_ONLY") != nullptr);
    if (vae_only_debug && silence_ptr) {
        fprintf(stderr, "[ace_step] DEBUG: ACE_STEP_VAE_ONLY set — bypassing DiT, "
                        "feeding silence_latent directly to VAE\n");
        // Copy silence_latent slice into xt_current
        const int32_t copy_T = std::min(T_latent, silence_T);
        for (int32_t t = 0; t < copy_T; t++) {
            std::memcpy(&xt_current[static_cast<size_t>(t) * out_ch],
                        silence_ptr + static_cast<size_t>(t) * out_ch,
                        static_cast<size_t>(out_ch) * sizeof(float));
        }
        // Pad any remaining frames with silence (zeros)
        for (int32_t t = copy_T; t < T_latent; t++)
            std::memset(&xt_current[static_cast<size_t>(t) * out_ch], 0,
                        static_cast<size_t>(out_ch) * sizeof(float));
        // Dump xt_current for offline comparison with diffusers
        if (const char* p = std::getenv("ACE_STEP_DUMP_INPUT")) {
            FILE* f = std::fopen(p, "wb");
            if (f) {
                uint32_t T32 = (uint32_t)T_latent, D32 = (uint32_t)out_ch;
                std::fwrite(&T32, sizeof(T32), 1, f);
                std::fwrite(&D32, sizeof(D32), 1, f);
                std::fwrite(xt_current.data(), sizeof(float),
                            (size_t)T_latent * out_ch, f);
                std::fclose(f);
                double rms = 0.0; double mn = 1e30, mx = -1e30;
                for (size_t i = 0; i < (size_t)T_latent * out_ch; i++) {
                    double v = xt_current[i];
                    rms += v * v;
                    if (v < mn) mn = v;
                    if (v > mx) mx = v;
                }
                fprintf(stderr, "[ace_step] dumped xt_current [%u, %u] RMS=%.4f range=[%.4f, %.4f] to %s\n",
                        T32, D32, std::sqrt(rms / ((size_t)T_latent * out_ch)),
                        mn, mx, p);
            }
        }
    } else
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

        // Dump context (proj_in input) and hidden_after_proj_in on step 0.
        if (step == 0 && std::getenv("ACE_STEP_DUMP_DIR")) {
            std::string dir = std::getenv("ACE_STEP_DUMP_DIR");
            std::string p = dir + "/proj_in_input.bin";
            if (FILE* f = std::fopen(p.c_str(), "wb")) {
                int32_t ndims = 2, d0 = in_ch, d1 = T_latent;
                std::fwrite(&ndims, sizeof(int32_t), 1, f);
                std::fwrite(&d0, sizeof(int32_t), 1, f);
                std::fwrite(&d1, sizeof(int32_t), 1, f);
                // Upstream ggml layout: ne[0]=in_ch contiguous. Our context
                // buffer IS row-major [T, in_ch] (in_ch contiguous), so write
                // directly without transpose.
                std::fwrite(context.data(), sizeof(float),
                            static_cast<size_t>(in_ch) * T_latent, f);
                std::fclose(f);
            }
        }

        // Dump hidden_after_proj_in on step 0 for upstream comparison.
        if (step == 0 && std::getenv("ACE_STEP_DUMP_DIR")) {
            std::string p = std::getenv("ACE_STEP_DUMP_DIR");
            p += "/hidden_after_proj_in.bin";
            FILE* f = std::fopen(p.c_str(), "wb");
            if (f) {
                int32_t ndims = 2, d0 = H, d1 = S_patches;
                std::fwrite(&ndims, sizeof(int32_t), 1, f);
                std::fwrite(&d0, sizeof(int32_t), 1, f);
                std::fwrite(&d1, sizeof(int32_t), 1, f);
                // Upstream ggml layout: ne[0]=H contiguous, ne[1]=S.
                // Our hidden buffer is row-major [S, H] (H contiguous) — matches.
                std::fwrite(hidden.data(), sizeof(float),
                            static_cast<size_t>(H) * S_patches, f);
                std::fclose(f);
                fprintf(stderr, "[ace_step] dumped hidden_after_proj_in [%d, %d] to %s\n",
                        H, S_patches, p.c_str());
            }
        }

        // DiT forward: hidden [S, H] → v_hidden [S, H]
        //
        // Timbre conditioning reference frame count.
        // Upstream pipeline-synth-ops.cpp:386-390 (ops_encode_timbre):
        //   text2music (no ref_audio): S_ref_timbre = 1, timbre_feats = silence_full[0:64]
        //   ref_audio available:       S_ref_timbre = T_ref (VAE-encoded ref)
        //   ref_latents available:     S_ref_timbre = ref_T_latent
        // For text2music (our only currently supported mode), upstream passes
        // exactly ONE silence frame to the timbre encoder, not 750. With 1
        // frame, self-attention is trivially identity (single-token softmax=1),
        // and the 4-layer transformer degenerates to a fixed nonlinear map of
        // the silence frame. Passing 750 frames instead routes the CLS token
        // through bidirectional attention over all 750 silence frames, which
        // is OOD and produces industrial-machinery DiT output.
        const int32_t T_refer = silence_ptr ? 1 : 0;
        const float* refer_audio = silence_ptr;

        // Lyric conditioning pointers (always populated — even for
        // text2music without lyrics, upstream encodes a default lyric string).
        const float* lyric_ptr = lyric_cond_.empty() ? nullptr : lyric_cond_.data();
        int32_t      T_lyric   = lyric_cond_len_;
        const float* lyric_unc = lyric_uncond_.empty() ? nullptr : lyric_uncond_.data();
        int32_t      T_lyric_unc = static_cast<int32_t>(lyric_uncond_.size() / enc_hs);

        if (!dit_runner_->forward(hidden.data(), t,
                                  cond_ptr, T_cond, enc_hs,
                                  uncond_ptr, T_uncond,
                                  lyric_ptr, T_lyric, enc_hs,
                                  lyric_unc, T_lyric_unc,
                                  refer_audio, T_refer,
                                  effective_gs,
                                  S_patches,
                                  v_hidden.data(), error))
            return false;
        fprintf(stderr, "[ace_step] dit: step %d forward done\n", step);

        // Dump v_hidden (DiT output pre-unpatchify) for upstream comparison.
        // Upstream doesn't dump this directly, but we keep it for diagnostics.
        if (std::getenv("ACE_STEP_DUMP_DIR")) {
            std::string p = std::getenv("ACE_STEP_DUMP_DIR");
            p += "/dit_step" + std::to_string(step) + "_vhid.bin";
            FILE* f = std::fopen(p.c_str(), "wb");
            if (f) {
                int32_t ndims = 2, d0 = H, d1 = S_patches;
                std::fwrite(&ndims, sizeof(int32_t), 1, f);
                std::fwrite(&d0, sizeof(int32_t), 1, f);
                std::fwrite(&d1, sizeof(int32_t), 1, f);
                // ggml layout: ne[0]=H contiguous. v_hidden is [S, H] row-major.
                std::fwrite(v_hidden.data(), sizeof(float),
                            static_cast<size_t>(H) * S_patches, f);
                std::fclose(f);
            }
        }

        // proj_out: v_hidden [S, H] → v_latent [T, 64]  (un-patchify)
        unpatchify_proj_out(v_hidden.data(), S_patches, H, P, out_ch,
                            proj_out_w_f32_.data(), proj_out_bias,
                            v_latent.data());

        // Dump v_latent (post-unpatchify) for comparison with upstream's
        // dit_step{N}_vt.bin (dumped by upstream AFTER compute, BEFORE Euler).
        if (std::getenv("ACE_STEP_DUMP_DIR")) {
            std::string dir = std::getenv("ACE_STEP_DUMP_DIR");
            std::string pv = dir + "/dit_step" + std::to_string(step) + "_vt.bin";
            if (FILE* f = std::fopen(pv.c_str(), "wb")) {
                // Upstream: debug_dump_2d(dbg, name, vt.data(), T, Oc).
                // Shape (T, Oc), data is row-major [T, Oc] — flat C array.
                int32_t ndims = 2, d0 = T_latent, d1 = out_ch;
                std::fwrite(&ndims, sizeof(int32_t), 1, f);
                std::fwrite(&d0, sizeof(int32_t), 1, f);
                std::fwrite(&d1, sizeof(int32_t), 1, f);
                std::fwrite(v_latent.data(), sizeof(float),
                            static_cast<size_t>(T_latent) * out_ch, f);
                std::fclose(f);
            }
        }

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

        // Euler step in LATENT space. Convention matches upstream
        // FlowMatchEulerDiscreteScheduler.step():
        //   sigma_next < sigma, so dt_sched = sigma_next - sigma < 0,
        //   prev = sample + dt_sched * v = sample - |dt| * v.
        // We keep dt positive (dt = t_curr - t_next) and SUBTRACT it.
        // ── Pre-step diagnostic ──
        {
            double sq_v = 0.0, sq_x = 0.0;
            float v_mx = -1e30f, v_mn = 1e30f;
            for (size_t i = 0; i < v_latent.size(); i++) {
                sq_v += (double)v_latent[i] * v_latent[i];
                if (v_latent[i] > v_mx) v_mx = v_latent[i];
                if (v_latent[i] < v_mn) v_mn = v_latent[i];
            }
            for (size_t i = 0; i < xt_current.size(); i++)
                sq_x += (double)xt_current[i] * xt_current[i];
            double rms_v = std::sqrt(sq_v / v_latent.size());
            double rms_x = std::sqrt(sq_x / xt_current.size());
            fprintf(stderr, "[ace_step] step %d: pre_x_rms=%.4f v_rms=%.4f v_range=[%.2f,%.2f] dt=%.4f\n",
                    step, rms_x, rms_v, v_mn, v_mx, dt);
        }
        for (size_t i = 0; i < xt_current.size(); i++) {
            xt_current[i] -= dt * v_latent[i];
        }

        // Dump xt AFTER Euler step to match upstream's dit_step{N}_xt.bin
        // (upstream dumps AFTER the step update). For the final step,
        // upstream writes to a separate `output` buffer as `dit_x0`, but
        // the value is identical (xt - dt*v == xt - t*v when dt==t for
        // last step). We dump both names for the final step.
        if (std::getenv("ACE_STEP_DUMP_DIR")) {
            std::string dir = std::getenv("ACE_STEP_DUMP_DIR");
            std::string px = dir + "/dit_step" + std::to_string(step) + "_xt.bin";
            if (FILE* f = std::fopen(px.c_str(), "wb")) {
                int32_t ndims = 2, d0 = T_latent, d1 = out_ch;
                std::fwrite(&ndims, sizeof(int32_t), 1, f);
                std::fwrite(&d0, sizeof(int32_t), 1, f);
                std::fwrite(&d1, sizeof(int32_t), 1, f);
                std::fwrite(xt_current.data(), sizeof(float),
                            static_cast<size_t>(T_latent) * out_ch, f);
                std::fclose(f);
            }
            if (step == sched.n_steps - 1) {
                std::string p0 = dir + "/dit_x0.bin";
                if (FILE* f = std::fopen(p0.c_str(), "wb")) {
                    int32_t ndims = 2, d0 = T_latent, d1 = out_ch;
                    std::fwrite(&ndims, sizeof(int32_t), 1, f);
                    std::fwrite(&d0, sizeof(int32_t), 1, f);
                    std::fwrite(&d1, sizeof(int32_t), 1, f);
                    std::fwrite(xt_current.data(), sizeof(float),
                                static_cast<size_t>(T_latent) * out_ch, f);
                    std::fclose(f);
                }
            }
        }
        // ── Post-step diagnostic ──
        {
            double sq = 0.0;
            float mx = -1e30f, mn = 1e30f;
            for (size_t i = 0; i < xt_current.size(); i++) {
                sq += (double)xt_current[i] * xt_current[i];
                if (xt_current[i] > mx) mx = xt_current[i];
                if (xt_current[i] < mn) mn = xt_current[i];
            }
            double rms = std::sqrt(sq / xt_current.size());
            fprintf(stderr, "[ace_step] step %d: post_x_rms=%.4f range=[%.2f,%.2f]\n",
                    step, rms, mn, mx);
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
        // TEMP DEBUG: dump final xt_current stats + first frame values.
        // Compare against silence_latent (the "no music" reference) to see
        // whether the DiT produced something silence-like or actually
        // musical. The VAE expects ~[-3, 3] range latents.
        double mn = 1e30, mx = -1e30, sum = 0, abs_sum = 0;
        for (size_t i = 0; i < xt_current.size(); i++) {
            float v = xt_current[i];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += v; abs_sum += std::fabs(v);
        }
        fprintf(stderr,
                "[ace_step] final xt_current: N=%zu range=[%.4f,%.4f] "
                "mean=%.4f mean_abs=%.4f\n",
                xt_current.size(), mn, mx, sum / xt_current.size(),
                abs_sum / xt_current.size());
        // First 8 values of channel 0
        fprintf(stderr, "[ace_step] xt_current[0..7]: ");
        for (int i = 0; i < 8 && i < (int)xt_current.size(); i++)
            fprintf(stderr, "%.4f ", xt_current[i]);
        fprintf(stderr, "\n");
        // Compare with silence_latent first 8 values
        if (silence_ptr) {
            fprintf(stderr, "[ace_step] silence_latent[0..7]: ");
            for (int i = 0; i < 8 && i < silence_T * out_ch; i++)
                fprintf(stderr, "%.4f ", silence_ptr[i]);
            fprintf(stderr, "\n");
        }
        // Dump xt_current to /tmp for offline diffusers comparison
        if (const char* p = std::getenv("ACE_STEP_DUMP_XT")) {
            FILE* f = std::fopen(p, "wb");
            if (f) {
                uint32_t T = (uint32_t)T_latent, D = (uint32_t)out_ch;
                std::fwrite(&T, sizeof(T), 1, f);
                std::fwrite(&D, sizeof(D), 1, f);
                std::fwrite(xt_current.data(), sizeof(float),
                            (size_t)T * D, f);
                std::fclose(f);
                fprintf(stderr, "[ace_step] dumped xt_current [%u, %u] to %s\n",
                        T, D, p);
            }
        }
    }
    {
        auto t_vae0 = std::chrono::steady_clock::now();
        if (!vae_runner_->decode(xt_current.data(), T_latent, pcm_stereo, error))
            return false;
        auto t_vae1 = std::chrono::steady_clock::now();
        double vae_ms = std::chrono::duration<double, std::milli>(t_vae1 - t_vae0).count();
        int nan_pcm = 0; float mx = -1e30f, mn = 1e30f;
        for (size_t i = 0; i < pcm_stereo->size(); i++) {
            if (std::isnan((*pcm_stereo)[i])) nan_pcm++;
            else { if ((*pcm_stereo)[i] > mx) mx = (*pcm_stereo)[i]; if ((*pcm_stereo)[i] < mn) mn = (*pcm_stereo)[i]; }
        }
        fprintf(stderr, "[ace_step] dit: VAE decode: %.0fms (pcm=%zu NaN=%d range=[%.6f,%.6f])\n",
                vae_ms, pcm_stereo->size(), nan_pcm, mn, mx);
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
