# GGUF Weight Layout

## Tensor names

The ECAPA-TDNN weights live under the `speaker.*` namespace. They can
reside in either:
1. A **standalone GGUF** (`qwen3tts-speaker-encoder.gguf`) produced by
   `tools/convert_ecapa.cpp` from
   `marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B`.
2. Bundled inside the talker GGUF alongside `qwen3.*` tensors (legacy
   layout where the official Qwen safetensors were converted including
   the speaker_encoder state_dict).

### Tensor name table

| Layer | Tensor name | Shape (0.6B) | Shape (1.7B) |
|-------|------------|-------------|-------------|
| blk0 weight | `speaker.blocks.0.conv.weight` | [512, 128, 5] | [512, 128, 5] |
| blk0 bias | `speaker.blocks.0.conv.bias` | [512] | [512] |
| blk1 tdnn1 weight | `speaker.blocks.1.tdnn1.conv.weight` | [512, 512, 3] | [512, 512, 3] |
| blk1 tdnn1 bias | `speaker.blocks.1.tdnn1.conv.bias` | [512] | [512] |
| blk1 tdnn2 weight | `speaker.blocks.1.tdnn2.conv.weight` | [512, 512, 3] | [512, 512, 3] |
| blk1 tdnn2 bias | `speaker.blocks.1.tdnn2.conv.bias` | [512] | [512] |
| blk1 se conv1 weight | `speaker.blocks.1.se_block.conv1.weight` | [256, 512, 1] | [256, 512, 1] |
| blk1 se conv1 bias | `speaker.blocks.1.se_block.conv1.bias` | [256] | [256] |
| blk1 se conv2 weight | `speaker.blocks.1.se_block.conv2.weight` | [512, 256, 1] | [512, 256, 1] |
| blk1 se conv2 bias | `speaker.blocks.1.se_block.conv2.bias` | [512] | [512] |
| blk1 res2net[0..6] weight | `speaker.blocks.1.res2net_block.blocks.{0..6}.conv.weight` | [64, 64, 3] × 7 | [64, 64, 3] × 7 |
| blk1 res2net[0..6] bias | `speaker.blocks.1.res2net_block.blocks.{0..6}.conv.bias` | [64] × 7 | [64] × 7 |
| blk2 tdnn1 weight | `speaker.blocks.2.tdnn1.conv.weight` | same as blk1 | same as blk1 |
| blk2 tdnn1 bias | `speaker.blocks.2.tdnn1.conv.bias` | same as blk1 | same as blk1 |
| blk2 tdnn2 weight | `speaker.blocks.2.tdnn2.conv.weight` | same as blk1 | same as blk1 |
| blk2 tdnn2 bias | `speaker.blocks.2.tdnn2.conv.bias` | same as blk1 | same as blk1 |
| blk2 se conv1 weight | `speaker.blocks.2.se_block.conv1.weight` | same as blk1 | same as blk1 |
| blk2 se conv1 bias | `speaker.blocks.2.se_block.conv1.bias` | same as blk1 | same as blk1 |
| blk2 se conv2 weight | `speaker.blocks.2.se_block.conv2.weight` | same as blk1 | same as blk1 |
| blk2 se conv2 bias | `speaker.blocks.2.se_block.conv2.bias` | same as blk1 | same as blk1 |
| blk2 res2net[0..6] weight | `speaker.blocks.2.res2net_block.blocks.{0..6}.conv.weight` | [64, 64, 3] × 7 | [64, 64, 3] × 7 |
| blk2 res2net[0..6] bias | `speaker.blocks.2.res2net_block.blocks.{0..6}.conv.bias` | [64] × 7 | [64] × 7 |
| blk3 tdnn1 weight | `speaker.blocks.3.tdnn1.conv.weight` | same as blk1 | same as blk1 |
| blk3 tdnn1 bias | `speaker.blocks.3.tdnn1.conv.bias` | same as blk1 | same as blk1 |
| blk3 tdnn2 weight | `speaker.blocks.3.tdnn2.conv.weight` | same as blk1 | same as blk1 |
| blk3 tdnn2 bias | `speaker.blocks.3.tdnn2.conv.bias` | same as blk1 | same as blk1 |
| blk3 se conv1 weight | `speaker.blocks.3.se_block.conv1.weight` | same as blk1 | same as blk1 |
| blk3 se conv1 bias | `speaker.blocks.3.se_block.conv1.bias` | same as blk1 | same as blk1 |
| blk3 se conv2 weight | `speaker.blocks.3.se_block.conv2.weight` | same as blk1 | same as blk1 |
| blk3 se conv2 bias | `speaker.blocks.3.se_block.conv2.bias` | same as blk1 | same as blk1 |
| blk3 res2net[0..6] weight | `speaker.blocks.3.res2net_block.blocks.{0..6}.conv.weight` | [64, 64, 3] × 7 | [64, 64, 3] × 7 |
| blk3 res2net[0..6] bias | `speaker.blocks.3.res2net_block.blocks.{0..6}.conv.bias` | [64] × 7 | [64] × 7 |
| MFA weight | `speaker.mfa.conv.weight` | [1536, 1536, 1] | [1536, 1536, 1] |
| MFA bias | `speaker.mfa.conv.bias` | [1536] | [1536] |
| ASP tdnn weight | `speaker.asp.tdnn.conv.weight` | [512, 1536, 3] | [512, 1536, 3] |
| ASP tdnn bias | `speaker.asp.tdnn.conv.bias` | [512] | [512] |
| ASP conv weight | `speaker.asp.conv.weight` | [1, 512, 1] | [1, 512, 1] |
| ASP conv bias | `speaker.asp.conv.bias` | [1] | [1] |
| FC weight | `speaker.fc.weight` | [1024, 3072] | [2048, 3072] |
| FC bias | `speaker.fc.bias` | [1024] | [2048] |

### Total weights

Approximate parameter count:

| Variant | Params | GGUF size (F16) |
|---------|--------|-----------------|
| 0.6B | ~6.3M | ~12 MB |
| 1.7B | ~25M | ~48 MB |

The standalone GGUF saves only `speaker.*` tensors at ~22 MB for the
1.7B variant (some tensors stored as F16 to match ggml conv_1d
requirements, biases as F32).

## KV metadata

The GGUF KV section carries hyperparameters so the loader can configure
the encoder without hard-coded values:

| Key | Type | Value | Description |
|-----|------|-------|-------------|
| `qwen3tts_spk.enc_dim` | u32 | 1024/2048 | Output embedding dimension |
| `qwen3tts_spk.sample_rate` | u32 | 24000 | Expected audio sample rate |
| `qwen3tts.speaker.enc_dim` | u32 | 1024/2048 | Alternative key (CrispASR compat) |
| `qwen3tts.speaker.sample_rate` | u32 | 24000 | Alternative key |

## Weight conversion

### From safetensors (marksverdhei HF repos)

The `tools/convert_ecapa.cpp` converter reads safetensors from
`marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B` and produces a GGUF:

```
safetensors state_dict keys:
  speaker.blocks.0.conv.weight         → speaker.blocks.0.conv.weight
  speaker.blocks.0.conv.bias           → speaker.blocks.0.conv.bias
  speaker.blocks.1.tdnn1.conv.weight   → speaker.blocks.1.tdnn1.conv.weight
  ...                                        ...
  speaker.fc.weight                    → speaker.fc.weight
  speaker.fc.bias                      → speaker.fc.bias
```

No key renaming needed — the safetensors keys match the GGUF tensor
names directly.

### Data type rules

- **rank-1 tensors** (biases): stored as F32 (added to F32 conv outputs).
- **rank-2/3 tensors** (weights): stored as F16 (matches the CPU backend's
  `conv_1d` expected `src0 == F16` requirement).
- `ggml_conv_1d` requires src0 (weights) as F16 or F32; F16 is preferred
  for size reduction.

## Codec encoder tensors (for ICL path)

The codec encoder (SEANet + transformer + RVQ) lives under `codec.enc.*`
namespace in the codec sidecar GGUF. Not part of the speaker encoder
itself, but required for full ICL-based voice cloning.

| Prefix | # tensors | Notes |
|--------|-----------|-------|
| `codec.enc.init_conv.*` | 2 | Causal conv, 1→64, K=7 |
| `codec.enc.res_blocks.*` | ~50 | 4 residual bottleneck blocks |
| `codec.enc.down_samples.*` | 4 | Strided convs, strides 4/5/6/8 |
| `codec.enc.final_conv.*` | 2 | Causal conv, 1024→512, K=3 |
| `codec.enc.transformer.*` | ~40 | 8-layer pre-LN transformer |
| `codec.enc.downsample.*` | 2 | Stride-2 causal conv, K=4 |
| `codec.enc.rvq.semantic.*` | 3 | Codebook 0: project+emb |
| `codec.enc.rvq.acoustic.*` | 45 | Codebooks 1-15: project+emb×15 |

See `codec.cpp` for full tensor name resolution.
