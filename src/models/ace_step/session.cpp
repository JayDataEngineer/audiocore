// session.cpp — ACE-Step run_music flow.
//
// Two-step pipeline (ServeurpersoCom/acestep.cpp):
//   Step 1 — LM:   caption+lyrics → TE embeddings → 5Hz LM → music codes
//   Step 2 — Synth: DiT (conditioned on music codes) → latents → VAE → PCM
//
// Both transformers go through qwen3::Runner. DiT/VAE are ggml_cgraph builds
// (TODO — see run_dit_and_vae). Each step has a clear seam; the orchestration
// above them is mechanical.

#include "audiocore/models/ace_step/family.h"

#include "ggml.h"

namespace audiocore::acestep {

bool AceStepSession::run_lm(const MusicRequest& req,
                            std::vector<int32_t>* music_codes,
                            std::string* error) {
    // (a) Encode caption + lyrics via TE. forward_tokens on te_ → hidden
    //     states used as encoder outputs for cross-attention in the LM.
    //
    // (b) Prime the LM with TE hidden states + a music-code BOS, then
    //     autoregressively emit n_codes tokens. Per step:
    //       forward_tokens(lm_, prev_codes, n_pos, &logits)
    //       sample next code with classifier-free guidance
    //         (req.guidance_scale combines conditional + unconditional logits)
    //
    // Verified I/O shapes in ServeurpersoCom/acestep.cpp:run_lm().
    //
    // Until the tokenizer (BPE) + the LM forward wrapper land, refuse cleanly.
    (void)req; (void)music_codes;
    if (error) *error = "ACE-Step LM path not yet wired "
                        "(TODO: port acestep.cpp:run_lm — TE encode + "
                        "5Hz LM autoregressive decode with CFG)";
    return false;
}

bool AceStepSession::run_dit_and_vae(const MusicRequest& req,
                                     const std::vector<int32_t>& music_codes,
                                     std::vector<float>* pcm_stereo,
                                     std::string* error) {
    // (a) Build the DiT graph: time_embed → proj_in → N × (attention + MLP)
    //     blocks over cfg_.sliding_window patches → proj_out → noise prediction.
    //     Use classifier-free guidance (req.guidance_scale), step from
    //     n_diffusion_steps (turbo=8, sft=50).
    //
    // (b) VAE decode: decoder.conv1 → decoder.block.{i}.* (Snake activation)
    //     → decoder.conv2. Latents at 75 Hz → PCM at 48 kHz stereo.
    //
    // Porting target: acestep.cpp:run_dit() + run_vae(). Anchor weights are
    // already bound on this session (dit_proj_in_, dit_proj_out_,
    // dit_time_embed_, vae_conv_in_, vae_conv_out_); the per-layer weights
    // (decoder.block.{i}.*, decoder.condition_embedder.*) are reachable by
    // ggml_get_tensor(ext_ctx_, name) at graph-build time.
    (void)req; (void)music_codes; (void)pcm_stereo;
    if (error) *error = "ACE-Step DiT+VAE path not yet wired "
                        "(TODO: port acestep.cpp:run_dit + run_vae — "
                        "DiT diffusion with CFG, VAE decode to 48 kHz)";
    return false;
}

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
    if (owns_ext_ctx_ && ext_ctx_) {
        ggml_free(ext_ctx_);
        ext_ctx_ = nullptr;
    }
}

}  // namespace audiocore::acestep
