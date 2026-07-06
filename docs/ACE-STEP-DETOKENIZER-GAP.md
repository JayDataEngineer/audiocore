# ACE-Step Detokenizer Gap

**Status:** Detokenizer transformer implemented (cover-mode ready); text_to_music quality issue is separate (under investigation)
**Found:** 2026-07-05
**Updated:** 2026-07-06

## Summary

The C++ ACE-Step port now has a working `DetokenizerRunner`
(`src/models/ace_step/detokenizer_runner.cpp`) that mirrors the upstream
`AudioTokenDetokenizer` + `ResidualFSQ.project_out` path. It is wired into
`AceStepSession::run_dit_and_vae` but only **invoked for future cover-mode
support** — for text_to_music, upstream uses `silence_latent` as the DiT src,
NOT the detokenized LM codes (verified in
`diffusers/pipelines/ace_step/pipeline_ace_step.py:626` `prepare_src_latents`).

The remaining text_to_music audio-quality issue (weak bass, spectral peaks
at 6/12 kHz) is **NOT** caused by the detokenizer path — output is identical
with seed=42 whether src=silence_latent or src=detokenized codes, suggesting
the DiT is not meaningfully using the src channel for text_to_music.

## Architecture (verified against the 1.5 Turbo checkpoint)

```
detokenizer.embed_tokens.weight      [2048, 2048]   Linear (Q8_0) + bias [2048]
detokenizer.special_tokens           [2048, 5]      P=5 learnable patch offsets (Q8_0)
detokenizer.layers.{0,1}                            2× Qwen3-style encoder layers
  .input_layernorm.weight            [2048]         RMSNorm γ (F32)
  .post_attention_layernorm.weight   [2048]         RMSNorm γ (F32)
  .self_attn.{q,k,v,o}_proj.weight   Q8_0           GQA: 16 Q-heads, 8 KV-heads, hd=128
  .self_attn.{q,k}_norm.weight       [128]          per-head RMSNorm γ (F32)
  .mlp.{gate,up,down}_proj.weight    Q8_0           SwiGLU, intermediate=6144
detokenizer.norm.weight              [2048]         final RMSNorm γ (F32)
detokenizer.proj_out.weight          [2048, 64]     Linear (Q8_0) + bias [64]

tokenizer.quantizer.project_in.weight  [2048, 6]    BF16 — FSQ front projection
tokenizer.quantizer.project_out.weight [6, 2048]    BF16 — FSQ back projection
```

Config constants from `configuration_acestep_v15.py`:
`hidden_size=2048`, `num_attention_heads=16`, `num_key_value_heads=8`,
`head_dim=128`, `intermediate_size=6144`, `pool_window_size=5`,
`audio_acoustic_hidden_dim=64`, `rms_norm_eps=1e-6`, `rope_theta=1e6`,
`fsq_input_levels=[8,8,8,5,5,5]` (64 000 codes).

## Forward pass

```
LM codes [N, 5Hz] (int32 in [0, 64000))
  → fsq_decode_one (mixed-radix → 6-D via FSQ implicit codebook)
  → tokenizer.quantizer.project_out (Linear 6 → 2048, BF16)
  → detokenizer transformer:
      · embed_tokens Linear (2048 → 2048, Q8_0) + bias
      · expand each token to P=5 patches, add learnable special_tokens
      · 2× Qwen3-style encoder layers:
          - pre-norm RMSNorm
          - GQA self-attention (bidirectional within each P=5 group,
            QK-norm, RoPE θ=1e6)
          - residual
          - pre-norm RMSNorm + SwiGLU MLP
          - residual
      · final RMSNorm
      · proj_out Linear (2048 → 64, Q8_0) + bias
  → 25 Hz × 64-D latents
```

The per-group (P=5) bidirectional attention is implemented via
`ggml_flash_attn_ext` with `ne3=N` (batched) so each group of 5 patches
attends only within itself.

## Where the detokenizer IS used (upstream)

`pipeline_ace_step.py:prepare_src_latents`:
1. **audio_codes provided** (cover from codes): parse codes → indices
   `[batch, T, num_quantizers=6]` → `quantizer.get_output_from_indices`
   → `detokenizer(quantized)` → src_latents
2. **src_audio provided + task=cover**: VAE-encode audio → tokenize →
   detokenize → src_latents

## Where the detokenizer is NOT used (upstream)

3. **text_to_music (no source)**: `src_latents = silence_latent` (cropped
   or tiled to target length). The DiT learns "given silence as src +
   text conditioning, generate music."

The LM codes produced in step 1 (run_lm) drive text_to_music generation
only **indirectly** — they were the output of an autoregressive LM that
already happened; the DiT is conditioned on the **text encoder output**,
not on the LM codes or their detokenized reconstruction.

## Why text_to_music output is still poor

Same audio output (seed=42) with src=silence_latent vs src=detokenized
codes → the DiT is not meaningfully using the src channel for
text_to_music. The issue is in the DiT itself or in how we're feeding
the text conditioning. Hypotheses to investigate:

- Text encoder output (TE hidden states) shape/scale mismatch
- proj_in patchify stride correctness
- Time embedding modulation scale
- CFG scale interactions with the turbo variant
- RoPE θ mismatch between our DiT and the checkpoint

These are tracked separately from the detokenizer work.

## Related files

- `src/models/ace_step/detokenizer_runner.cpp` — the implemented detokenizer
- `src/models/ace_step/session.cpp` — `run_dit_and_vae` text_to_music path
- `src/models/ace_step/dit_runner.cpp` — DiT graph builder (under investigation)
- `notes/ace-step-pipeline.md` — older architecture note (FSQ section outdated)
