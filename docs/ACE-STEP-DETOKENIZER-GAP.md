# ACE-Step Detokenizer Gap

**Status:** Open (text_to_music produces temporally-incoherent noise)
**Found:** 2026-07-05
**Severity:** High — text_to_music mode unusable until resolved

## Summary

The LM (5 Hz FSQ codes) → DiT (25 Hz latents) bridge in our C++ ACE-Step
port is incomplete. The upstream model uses a **2-layer transformer
detokenizer** to project FSQ codes into the 64-D latent space that the DiT
consumes as `src` conditioning; we currently feed raw 6-D FSQ values padded
to 64-D, which the DiT cannot meaningfully refine.

The musical structure (chord progression, rhythm sketch, phrasing) lives in
the LM code sequence — but only after the detokenizer transformer has
*interpreted* those codes. The DiT then refines the detokenized latents;
it cannot infer musical structure from raw FSQ indices on its own.

## Evidence

### Tensor names actually present in our 1.5 Turbo checkpoint

```
detokenizer.embed_tokens.weight      [2048, 2048]   # vocab=2048, dim=2048
detokenizer.special_tokens           [2048, 5]      # per-code special flags
detokenizer.proj_out.weight          [64, 2048]     # 2048 → 64
detokenizer.layers.0.{...}           Qwen3-style block
detokenizer.layers.1.{...}           Qwen3-style block
```

(plus `decoder.fsq_proj.*` keys do **not** exist in this checkpoint)

### Architecture (reconstructed from tensor shapes)

```
Input:    music_codes[N, 5Hz]              # int indices in [0, 2048)
        ↓ embed_tokens (vocab=2048, dim=2048)
        ↓ + special_tokens[code, 0..4] broadcast
Embed:    h[N, 2048]
        ↓ 2× Qwen3-style transformer layers
Hidden:   h[N, 2048]
        ↓ proj_out (2048 → 64)
Latent:   z[N, 64] at 5 Hz
        ↓ repeat each frame 5×
Latent:   z[N*5, 64] at 25 Hz  ← feeds DiT src channels (0..63)
```

### What the codebase does today (`src/models/ace_step/session.cpp`)

The fallback path pads the 6-D FSQ-decoded vector to 64-D and feeds *that*
into the DiT src channels:

```cpp
// No learned weights: simple pad (6 → 64) with the FSQ values
for (int d = 0; d < 6 && d < out_ch; d++)
    projected[d] = f6[d];
```

This is structurally meaningless: the DiT cannot recover chord/rhythm
information from raw FSQ index values, regardless of how many diffusion
steps are run on top.

### Effect

- DiT output per-channel mean ≈ 0, std ≈ random — no temporal structure
- VAE-decoded PCM is white-noise-like
- cloud_vlm reports "loud mechanical or motor-like sound" / "continuous
  unstructured hissing"

Cover mode and repaint mode do **not** hit this gap — they use
`cover_latent_cond_` / `repaint_latent_cond_` which bypass the FSQ path
entirely. Only text_to_music is affected.

## Why the docs said "MLP"

The earlier note (`notes/ace-step-pipeline.md`) describes the FSQ projection
as a "Learned MLP (6→2048→SiLU→LayerNorm→64)". This was likely an inference
from a stale or different checkpoint. The 1.5 Turbo model on disk has the
2-layer transformer described above, not an MLP — verified by inspecting
`detokenizer.*` tensor shapes directly.

## Required work

1. **Bind detokenizer tensors** in `AceStepSession` (or a dedicated
   `detokenizer_runner.cpp` mirroring `vae_runner.cpp`):
   - `detokenizer.embed_tokens.weight`
   - `detokenizer.special_tokens`
   - `detokenizer.proj_out.weight` (+ bias if present)
   - `detokenizer.layers.{0,1}.*` (attention + MLP, Qwen3-style with RoPE)

2. **Build a ggml graph** that:
   - Embeds the FSQ codes via `embed_tokens`
   - Adds the broadcast `special_tokens` row
   - Runs 2 transformer layers (use llama.cpp's existing Qwen3 building
     blocks — same family as our Qwen3 LM)
   - Projects to 64-D via `proj_out`
   - Repeats each 5 Hz frame 5× to reach 25 Hz

3. **Replace the fallback path** in `run_dit_and_vae` so the resulting
   64-D latent at 25 Hz populates `fsq_latent` instead of the padded
   fallback. The downstream src-channel copy (already wired) then does
   the right thing.

4. **Verify** via cloud_vlm on a generated clip: "describe the music"
   should yield musical descriptors (key, tempo, instrumentation) rather
   than noise descriptors.

## Related

- `notes/ace-step-pipeline.md` — older architecture note (now outdated
  on the FSQ path)
- `notes/GAPS.md` — lists text_to_music as ✅; this entry supersedes
  that claim for our 1.5 Turbo port
- `src/models/ace_step/session.cpp` — `run_dit_and_vae` step 1 (FSQ
  decode) is where the fallback path lives
- `src/models/ace_step/vae_runner.cpp` — pattern to mirror for a
  `detokenizer_runner.cpp`
