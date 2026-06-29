// family.h — ACE-Step music-generation family session.
//
// ACE-Step is LM (Qwen3 5Hz) + text encoder (Qwen3-Embedding) + DiT + VAE.
// The two transformers go through the unified qwen3::Runner (libllama),
// exactly like MOSS's backbone. The DiT and VAE aren't transformers — they
// bind into our own ggml_context, same shape as MOSS's moss.* extensions.
//
// One GGUF directory in → four GgufReader passes:
//   • LM GGUF  → qwen3::Runner (after tensor-name conversion, see below)
//   • TE GGUF  → qwen3::Runner (after conversion)
//   • DiT GGUF → ext_ctx_ (bound by name from acestep.* tensors)
//   • VAE GGUF → ext_ctx_ (bound by name from vae.* tensors)
//
// Why conversion: ACE-Step's LM and TE ship with HF tensor names
// (model.embed_tokens.weight, model.layers.0.self_attn.q_proj.weight, …)
// that libllama refuses. tools/convert_acestep_gguf.py rewrites them once
// to llama.cpp names (token_embd.weight, blk.0.attn_q.weight, …) so the
// SAME qwen3::Runner loads them. There is no second Qwen3 impl in audiocore.
//
// Verified tensor names from ServeurpersoCom/acestep.cpp — see
// docs/GGUF_FORMAT.md → "ACE-Step" section.

#ifndef AUDIOCORE_MODELS_ACE_STEP_FAMILY_H
#define AUDIOCORE_MODELS_ACE_STEP_FAMILY_H

#include <memory>
#include <string>
#include <vector>

#include "audiocore/framework/core/session.h"
#include "audiocore/framework/io/gguf_reader.h"   // complete type — methods take GgufReader&
#include "audiocore/models/qwen3/runner.h"
#include "audiocore/models/ace_step/dit_runner.h"
#include "audiocore/models/ace_step/vae_runner.h"

#include "ggml.h"

struct ggml_context;
struct ggml_tensor;

namespace audiocore::acestep {

// ── ACE-Step config (parsed from DiT GGUF metadata) ────────────────────────
struct AceConfig {
    DitConfig dit;                         // DiT-specific params
    std::string config_json;               // acestep.config_json (raw, upstream fallback)
    std::string variant;                   // "turbo" | "sft" | "xl-turbo" | …
    int32_t     encoder_hidden_size = 1024; // Qwen3-Embedding hidden size (0.6B = 1024)
};

struct MusicRequest {
    std::string caption;             // prompt ("lo-fi ambient piano")
    std::string lyrics;              // optional lyrics text
    float       duration     = 30.0f; // seconds
    int32_t     seed         = 0;
    float       guidance_scale = 7.5f;
    int32_t     n_diffusion_steps = 0;  // 0 → variant default (turbo=8, sft=50)
    float       temperature  = 0.0f;   // LM sampling temp (0=argmax, >0=stochastic)
    float       top_p        = 1.0f;   // LM nucleus sampling threshold
    // ── Repaint / completion conditioning ─────────────────────────────────
    std::vector<float> input_audio;    // stereo PCM interleaved at 48kHz
    float              mask_start = 0.0f;  // normalized position [0,1)
    float              mask_end   = 1.0f;  // normalized position (0,1]
    // ── Mode selection ────────────────────────────────────────────────────
    //   "text_to_music" (default) — full pipeline from text only
    //   "repaint"  — regenerate region [mask_start, mask_end) of input_audio
    //   "completion" — same as repaint but mask covers the end
    //   "cover"   — VAE-encode input_audio, FSQ roundtrip, skip LM, condition
    //               DiT on source latents for style reinterpretation
    //   "stem"    — separate model entirely (BLOCKED)
    //   "lego"    — separate stem-assembler entirely (BLOCKED)
    std::string mode = "text_to_music";
    std::string response_format = "wav"; // "wav" | "mp3"
};

struct MusicResponse {
    std::vector<float> pcm_stereo;   // interleaved L,R,L,R at 48000 Hz
    int32_t            sampling_rate = 48000;
    int32_t            channels      = 2;
    std::string        error;
};

class AceStepSession : public Session {
public:
    ~AceStepSession() override;
    std::string family_name() const override { return "ace_step"; }
    bool load(const std::string& model_path,
              const LoadOptions& opts,
              const BackendConfig& backend_cfg,
              std::string* error = nullptr) override;
    bool run_music(const void* request, void* response,
                   std::string* error = nullptr) override;

    const AceConfig& config() const { return cfg_; }

private:
    // Step 1 of the two-step pipeline. Encodes caption+lyrics via TE, runs
    // the 5Hz LM with classifier-free guidance → music codes. Output:
    // (n_codes,) int32. Cached on the session so step 2 can run separately.
    bool run_lm(const MusicRequest& req, std::vector<int32_t>* music_codes,
                std::string* error);

    // Step 2. DiT diffusion conditioned on music_codes → latents, then VAE
    // decode latents → 48 kHz stereo PCM.
    bool run_dit_and_vae(const MusicRequest& req,
                         const std::vector<int32_t>& music_codes,
                         std::vector<float>* pcm_stereo,
                         std::string* error);

    // Bind every dit.* / vae.* tensor into ext_ctx_, anchoring a few hot
    // tensors on the session for fast access during graph build.
    bool bind_dit_and_vae(const GgufReader& dit,
                          const GgufReader& vae,
                          std::string* error);

    // Refuses if the GGUF at `path` still has HF-style names. libllama only
    // loads llama.cpp-style names; ACE-Step ships the former, so callers must
    // run tools/convert_acestep_gguf.py first.
    bool check_llamacpp_layout(const GgufReader& r,
                               const char* role, std::string* error);

    std::unique_ptr<qwen3::Runner> lm_;   // 5Hz music-code LM
    std::unique_ptr<qwen3::Runner> te_;   // Qwen3-Embedding text encoder
    std::unique_ptr<DiTRunner>     dit_runner_;   // DiT graph builder
    std::unique_ptr<VAERunner>     vae_runner_;   // VAE decoder

    ggml_context*  ext_ctx_   = nullptr;  // DiT (decoder.*) + VAE (vae.decoder.*).
    // Anchored hot tensors. DiT names are unprefixed; VAE names are
    // `vae.`-prefixed at bind time to dodge the `decoder.block.{i}.*`
    // collision with DiT (both upstream files use the same `decoder.*` tree).
    ggml_tensor*   dit_proj_in_   = nullptr;   // decoder.proj_in.1.weight  (bf16 3D)
    ggml_tensor*   dit_proj_out_  = nullptr;   // decoder.proj_out.1.weight (bf16 3D)
    ggml_tensor*   dit_time_embed_= nullptr;   // decoder.time_embed
    ggml_tensor*   vae_conv_in_   = nullptr;   // vae.decoder.conv1 (bound as vae.*)
    ggml_tensor*   vae_conv_out_  = nullptr;   // vae.decoder.conv2
    AceConfig      cfg_;
    bool           owns_ext_ctx_ = false;

    // ── Pre-computed f32 Conv1D weights (bf16 → f32 at load time) ─────────
    std::vector<float> proj_in_w_f32_;   // decoder.proj_in.1.weight  f32 [2, 2048, 192]
    std::vector<float> proj_out_w_f32_;  // decoder.proj_out.1.weight f32 [2, 64, 2048]

    // ── Cached between run_lm() and run_dit_and_vae() ──────────────────────
    std::vector<float> te_cond_;       // TE hidden states: [T_text, encoder_hidden]
    std::vector<float> te_uncond_;     // null-text TE hidden (for CFG)
    int32_t            te_cond_len_ = 0;   // T_text
    int32_t            fsq_code_offset_ = 0;  // base offset for audio code tokens in LM vocab

    // For repaint/completion: VAE-encoded input audio latent [T, 64]
    std::vector<float> repaint_latent_cond_;
    // For cover mode: VAE-encoded + FSQ-roundtripped source audio latent [T, 64]
    std::vector<float> cover_latent_cond_;
};

}  // namespace audiocore::acestep

#endif  // AUDIOCORE_MODELS_ACE_STEP_FAMILY_H
