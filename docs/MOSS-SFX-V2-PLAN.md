# MOSS-SoundEffect-v2.0 — GGML/GGUF Port Plan

Picked up after the active ACE-Step text_to_music investigation ships.
Sibling model: re-uses the DiT / VAE / flow-matching infrastructure built
for `src/models/ace_step/`.

---

## Goal

Port **MOSS-SoundEffect-v2.0** (1.3B DiT + DAC VAE + Qwen3 text encoder +
flow-match scheduler) from PyTorch to GGML/GGUF within the audiocore
framework, using differential testing against dumped intermediate tensors.

## Constraints & Preferences

- Brute-force, layer-by-layer approach — verify each ggml sub-graph against
  PyTorch outputs.
- Model the C++ family after ACE-Step (`src/models/ace_step/`), which already
  has DiT + VAE + flow-matching infrastructure.
- Use existing `qwen3::Runner` for the Qwen3 text encoder (shared with
  MOSS-TTS).
- Tensor format: GGUF with `moss_sfx_v2.*` prefix, BF16/F32 weight storage.
- VAE runs as sequential per-op ggml contexts (not one big graph) to keep
  peak RAM low.

## Progress

### Done

- **Phase 1** — Python tensor-dumping script (`dump_mse2_tensors.py`):
  825 tensors, 1.2 GB at `dump_mse2_tensors/`.
- **Phase 2** — GGUF converter (`tools/convert_mse2.cpp`): produces
  `/tmp/mse2_test.gguf` (5.4 GB, 825 tensors) with correct naming and KV
  metadata.
- **Phase 3** — Headers: `family.h`, `dit_runner.h`, `vae_runner.h`.
- **Phase 4** — Loader (`src/models/moss_sfx_v2/loader.cpp`): GGUF binding,
  KV metadata parsing, `AUDIOCORE_REGISTER_FAMILY`.
- **Phase 5** — DiT graph builder (`src/models/moss_sfx_v2/dit_runner.cpp`):
  30-layer `WanAudioModel` with patch/time/text embedding, AdaLN (6-chunk),
  SA+CA with QK-norm + RoPE, FFN `GELU_tanh`, head modulation, CFG.
- **VAE runner** (`src/models/moss_sfx_v2/vae_runner.cpp`): DAC decoder —
  `WNConv1d`, `Snake`, `WNConvTranspose1d`, 3×`ResUnit` per block, `Tanh`.
  Pre-computes effective weight-norm weights at construction.
- **Session** — `src/models/moss_sfx_v2/session.cpp`: full denoising
  pipeline (FlowMatch Euler schedule, TE context encoding, noise init, CFG
  loop, VAE decode, post-processing with DC offset + gain normalization).
  50 default steps, CFG scale 5.0.
- **family.h** — added `#include "audiocore/models/qwen3/runner.h"` and
  `std::unique_ptr<qwen3::Runner> te_`.
- **loader.cpp** — added TE GGUF loading via `--extras te_path=...`
  (optional), TE cleanup in destructor.
- **CLAUDE.md** — added MSE2 session status table, scheduler/VAE/next-steps
  documentation.
- **Build** — all targets compile cleanly.
- **VAE checkpoint located** — at
  `/mnt/data/models/audio/moss-soundeffect-v2/vae/vae_128d_48k.pth`.
  Metadata: `decoder_dim=2048`, `decoder_rates=[8,5,4,3,2]`, 147 decoder
  keys, F32 weights.

### In Progress

(none — paused while ACE-Step text_to_music is being debugged)

### Blocked

- **VAE runner architecture mismatch** — C++ VAE runner hardcodes
  `decoder_dim=1536` and 4 blocks (strides `8,8,4,2` = 512× upsampling),
  but the actual checkpoint uses `decoder_dim=2048` and 5 blocks (strides
  `8,5,4,3,2` = 960× upsampling). The reference DAC model in
  `dac_vae.py` must be checked to confirm the correct architecture before
  writing the VAE GGUF converter.
- **No VAE GGUF** — cannot test end-to-end until the VAE runner is updated
  and the companion converter script is written.

## Key Decisions

- Use standard `ggml_rope_ext` for RoPE (not the exact 3-band Wan formula).
  Noted as a future parity fix.
- VAE ops use one `ggml_context` per sub-operation (Conv1d, ConvT1d, Snake,
  Tanh) with 64–128 MB scratch buffers.
- VAE weight-norm pre-computed at construction; raw `weight_g`/`weight_v`
  stored in GGUF (same format as PyTorch checkpoint).
- VAE weight layout follows ACE-Step pattern: permuted for `ConvTranspose1d`
  before storage to match `ggml_col2im_1d` expectations.
- Session stores tensors as F32 rather than BF16 for simplicity (VAE ops
  are small — ~172 MB for full decoder at F32).
- Qwen3 TE is loaded from a separate sidecar GGUF via `--extras te_path=...`
  (same pattern as ACE-Step).

## Next Steps

1. **Resolve VAE runner architecture mismatch** — verify the correct DAC
   VAE architecture (`decoder_dim` and `decoder_rates`) from the reference
   `dac_vae.py` model definition, NOT from the `.pth` checkpoint metadata
   alone. Update `vae_runner.cpp` if needed.
2. **Write companion Python script** (`tools/convert_vae.py`) to extract
   DAC VAE weights from `.pth` into GGUF format — once the architecture is
   confirmed.
3. **Write layer-by-layer unit tests**: run each C++ sub-graph against
   dumped `.bin` tensors.
4. **End-to-end integration test**: prompt → Qwen3 → DiT denoise → VAE →
   playable WAV.

## Critical Context

- **Architecture (1.3B):** `dim=1536`, `ffn_dim=8960`, `n_heads=12`,
  `head_dim=128`, `n_layers=30`, `in_dim=128`, `out_dim=128`,
  `text_dim=2048`, `freq_dim=256`, `eps=1e-6`, `patch_size=1`.
- **Blocks** use 2 norms: `norm1` (shared pre-SA and pre-CA), `norm3`
  (pre-FFN, HF `norm2` → renamed). No separate norm for cross-attention.
- **AdaLN modulation**: 6 chunks — `chunks0..2 = shift_msa, scale_msa,
  gate_msa` (self-attn); `chunks3..5 = shift_mlp, scale_mlp, gate_mlp`
  (FFN). Cross-attention shares self-attention's gate (`gate1`).
- **FFN**: `Linear(dim, 2*ffn_dim)` → `GELU_tanh` → `Linear(2*ffn_dim, dim)`.
  Not SwiGLU.
- **DAC VAE**: the checkpoint at
  `/mnt/data/models/audio/moss-soundeffect-v2/vae/vae_128d_48k.pth` has
  `decoder_dim=2048`, `decoder_rates=[8,5,4,3,2]` (960× upsampling, matching
  hop_length). The C++ `vae_runner.cpp` hardcodes `decoder_dim=1536` and
  strides `8,8,4,2` (512× upsampling) — this must be reconciled with the
  reference `dac_vae.py` model definition.
- **FlowMatch scheduler**: `shift=5.0`, `sigma_min=0.0`, `extra_one_step=
  true`, `num_train_timesteps=1000`. Default inference: 50 steps. Euler
  step: `x += dt·v` where `dt = σ_{i+1} − σ_i`.
- **TFloat_match**: `t` (sigma) in `[0,1]`, passes through DiT runner
  which multiplies by 1000 internally (matching PyTorch
  `sinusoidal_embedding_1d` convention).
- `GGML_MAX_NAME=128` set in top-level `CMakeLists.txt` (needed for ~55-char
  tensor names).

## Relevant Files

- `audiocore/src/models/moss_sfx_v2/session.cpp` — Denoising pipeline
  (FlowMatch, CFG, VAE decode, post-processing). ✅ Done.
- `audiocore/include/audiocore/models/moss_sfx_v2/family.h` — `SfxConfig`,
  `SfxSession` with `te_` (`Qwen3::Runner`). ✅ Updated.
- `audiocore/src/models/moss_sfx_v2/loader.cpp` — DiT + VAE + TE weight
  binding. ✅ Updated.
- `audiocore/src/models/moss_sfx_v2/vae_runner.cpp` — DAC decoder (may
  need architecture update to match actual checkpoint).
- `audiocore/src/models/moss_sfx_v2/dit_runner.cpp` — DiT graph builder
  (30 layers, CFG). ✅ Done.
- `audiocore/tools/convert_mse2.cpp` — DiT GGUF converter. ✅ Done.
- `audiocore/tools/convert_vae.py` — Not yet written (next step after
  architecture reconciliation).
- `/mnt/data/models/audio/moss-soundeffect-v2/vae/vae_128d_48k.pth` —
  Actual VAE checkpoint (`decoder_dim=2048`, 5 blocks × 960 upsampling,
  147 tensors).
- `moss-sfx-v2/moss_soundeffect_v2/diffsynth/models/dac_vae.py` —
  Reference DAC VAE model definition (source of truth for architecture).
- `moss-sfx-v2/moss_soundeffect_v2/diffsynth/pipelines/wan_audio.py` —
  Reference denoising pipeline with FlowMatch scheduler.
