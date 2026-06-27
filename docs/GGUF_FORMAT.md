# GGUF tensor naming conventions

Every model family we support needs a documented mapping from GGUF tensor
names → role in the model. Without this, two bad things happen:

1. The loader can't tell whether a tensor is a Qwen3 backbone weight or a
   codec weight when both live in the same GGUF.
2. Different community quantizations of "the same" model end up with
   incompatible tensor names, breaking the loader silently.

This doc is the canonical reference for **our** naming. Community GGUFs
(MOSS-TTS-GGUF, ACE-Step-1.5-GGUF) may use different names; the loader
normalizes them via an alias table.

## MOSS-TTS

Two GGUF files per model:

### `moss-tts-q8_0.gguf` — Qwen3-8B backbone

Tensor name prefix: `backbone.`

| Prefix | Component |
|---|---|
| `backbone.token_embd.weight` | Token embedding |
| `backbone.blk.{i}.attn_q.weight` | Attention Q projection, layer i |
| `backbone.blk.{i}.attn_k.weight` | Attention K projection |
| `backbone.blk.{i}.attn_v.weight` | Attention V projection |
| `backbone.blk.{i}.attn_output.weight` | Attention output projection |
| `backbone.blk.{i}.ffn_gate.weight` | FFN gate |
| `backbone.blk.{i}.ffn_up.weight` | FFN up |
| `backbone.blk.{i}.ffn_down.weight` | FFN down |
| `backbone.blk.{i}.attn_norm.weight` | Pre-attention RMSNorm |
| `backbone.blk.{i}.ffn_norm.weight` | Pre-FFN RMSNorm |
| `backbone.output_norm.weight` | Final RMSNorm |
| `backbone.output.weight` | Output projection (tied to token_embd if tied) |

GGUF general metadata keys we read:

| Key | Type | Example |
|---|---|---|
| `moss.architecture` | string | `"qwen3"` |
| `moss.backbone.parameter_count` | uint64 | `8000000000` |
| `moss.audio_codec.codebook_bits` | array<uint32> | `[5,5,5,5,…]` × 32 |
| `moss.audio_codec.sample_rate_hz` | uint32 | `48000` |

### `moss-tts.extras.gguf` — Codec + auxiliary

| Prefix | Component |
|---|---|
| `codec.encoder.*` | Audio codec encoder (used during training; not needed for inference) |
| `codec.decoder.*` | Audio codec decoder (~1.6B params) |
| `codec.quantizer.{i}.*` | RVQ codebook i, i ∈ [0, 32) |
| `speaker_encoder.*` | ECAPA-TDNN speaker embedding network |

## ACE-Step

Four GGUF files per model configuration; the loader takes a directory and
discovers files by name pattern.

### `acestep-v15-{turbo|sft}-Q8_0.gguf` — DiT (diffusion transformer)

Tensor name prefix: `dit.`

| Prefix | Component |
|---|---|
| `dit.pos_embed.weight` | Positional embedding |
| `dit.t_embedder.mlp.0.*` | Timestep embedding MLP layer 0 |
| `dit.t_embedder.mlp.2.*` | Timestep embedding MLP layer 2 |
| `dit.y_embedder.*` | Conditioning embedding |
| `dit.blocks.{i}.norm1.*` | Pre-attention norm, block i |
| `dit.blocks.{i}.attn.qkv.*` | Fused QKV projection |
| `dit.blocks.{i}.attn.proj.*` | Attention output projection |
| `dit.blocks.{i}.norm2.*` | Pre-FFN norm |
| `dit.blocks.{i}.mlp.fc1.*` | FFN up |
| `dit.blocks.{i}.mlp.fc2.*` | FFN down |
| `dit.final_layer.*` | Final norm + AdaLN |

Metadata:

| Key | Example | Notes |
|---|---|---|
| `acestep.architecture` | `"dit_v1.5"` | |
| `acestep.variant` | `"turbo"` / `"sft"` / `"xl-turbo"` / `"xl-sft"` / `"xl-base"` | Selects inference schedule |
| `acestep.dit.parameter_count` | `1100000000` | ~1.1B for the 1.7B-LM-paired DiT |
| `acestep.dit.inference_steps_default` | `8` | turbo; `50` for sft |

### `acestep-5Hz-lm-{1.7B|4B}-Q8_0.gguf` — music-code LM

Tensor name prefix: `lm.` Same Qwen3-style layout as MOSS-TTS backbone but
with a music-code vocabulary instead of natural-language tokens.

### `Qwen3-Embedding-0.6B-Q8_0.gguf` — text encoder

Tensor name prefix: `te.` Standard Qwen3 embedding layout. Used to encode
the caption + lyrics into conditioning vectors for the LM.

### `vae-BF16.gguf` — audio VAE decoder

Tensor name prefix: `vae.` Converts DiT latents → 48 kHz stereo PCM.

| Prefix | Component |
|---|---|
| `vae.encoder.conv_in.*` | VAE encoder conv (training only; unused at inference) |
| `vae.decoder.conv_in.*` | Decoder conv input |
| `vae.decoder.mid.block_{i}.*` | Mid-block residual layers |
| `vae.decoder.up.{i}.block_{j}.*` | Upsampling stages |
| `vae.decoder.conv_out.*` | Final conv → audio samples |
| `vae.quant_conv.*` | Latent quantization conv (training only) |

## Versioning

Family-specific keys are prefixed with the family name to avoid collisions
when we eventually merge multiple families into a single GGUF bundle (e.g.
`moss_tts.gguf` + `moss_codec.gguf` → `moss_full.gguf`).

A `audiocore.gguf_version` uint32 at the top of every file lets us refuse
incompatible files cleanly:

```cpp
if (meta["audiocore.gguf_version"].u32() != 1) {
    return error("unsupported audiocore GGUF version");
}
```
