# ECAPA-TDNN Speaker Encoder Architecture

## Overview

The Qwen3-TTS speaker encoder is an ECAPA-TDNN (Emphasized Channel
Attention, Propagation and Aggregation in Time-Delay Neural Network)
x-vector extractor. It converts a short reference audio clip (~3-10 s
of speech) into a fixed-size vector that captures speaker identity.

Two variants exist:

| Model | enc_dim | Params | Source |
|-------|---------|--------|--------|
| 0.6B | 1024 | ~6.3M | Qwen/Qwen3-TTS-12Hz-0.6B-Base |
| 1.7B | 2048 | ~25M | Qwen/Qwen3-TTS-12Hz-1.7B-Base |

Only the **Base** model variant carries the speaker encoder. The
CustomVoice and VoiceDesign variants do NOT have one.

## Full network graph

Shown with exact variable names from `speaker_encoder.cpp:570-598`:

```
  spk_mel:          [128, T]     ← log-mel, channels-first
    │
    ├─ tdnn_block(h, blk0_, d=1)      Conv1d 128→512 K=5 reflect-pad ReLU
    │  spk_blk0_out: [512, T]  ─── NOT in MFA concat
    │
    ├─ se_res2net(h, blk_[0], d=2)    1st SE-Res2Net (512→512)
    │  blk_outs[0]:    [512, T]
    │
    ├─ se_res2net(h, blk_[1], d=3)    2nd SE-Res2Net (512→512)
    │  blk_outs[1]:    [512, T]
    │
    ├─ se_res2net(h, blk_[2], d=4)    3rd SE-Res2Net (512→512)
    │  blk_outs[2]:    [512, T]
    │
    ├─ concat(blk_outs[0], blk_outs[1], blk_outs[2], axis=0)
    │  mfa_in:         [1536, T]     ← 3 × 512
    │
    ├─ tdnn_block(mfa_in, mfa_, d=1)  Conv1d 1536→1536 K=1 ReLU
    │  spk_mfa_out:    [1536, T]
    │
    ├─ asp_block(h, asp_)             Attentive Statistics Pooling
    │                  [3072, 1]      ← 2 × 1536
    │
    └─ fc_w_ @ h + fc_b_              Linear 3072→enc_dim
       spk_emb:        [enc_dim]      (=1024 for 0.6B, 2048 for 1.7B)
```

Key naming note: the code has `blk0_` (the initial conv, NOT in the MFA
concat) and `blk_[0..2]` (three SE-Res2Net blocks whose outputs form
`blk_outs[0..2]`). Only `blk_outs[0..2]` enter the MFA concat — the
`spk_blk0_out` tensor is computed and stored but discarded.

## Building blocks

### same_conv1d

Reflect-padded causal 1D convolution. Mirrors PyTorch `Conv1d` with
`padding_mode='reflect'` and `padding='same'`.

```
input [C_in, T]
  → transpose → [T, C_in]
  → ggml_pad_reflect_1d(pad_left, pad_right) where pad = (K-1)*dilation/2
  → ggml_conv_1d(weights [K, C_out, C_in], stride=1, dilation=d)
  → transpose → [C_out, T]
  → add bias [C_out, 1] (broadcast)
```

### tdnn_block

same_conv1d + ReLU activation.

```
y = ReLU(same_conv1d(x, w, b, dilation))
```

### SE-Res2Net

Squeeze-Excitation + Res2Net residual block. The core building block
of the encoder.

```
residual = x
x = tdnn_block(x, tdnn1, dilation=1)        # 512→512, K=3
x = res2net_block(x, res2net, dilation=d)    # multi-scale split
x = tdnn_block(x, tdnn2, dilation=1)         # 512→512, K=3
x = se_block(x, se)                          # channel attention
return x + residual                           # residual skip connection
```

### Res2Net block

Splits the 512-channel input into 8 chunks of 64 channels each,
then processes with a hierarchical cascade pattern.

```
chunks = split(x, 8, dim=0)    # 8 × [64, T]
out[0] = chunks[0]              # identity passthrough
for i = 1..7:
  in_i = chunks[i] + out[i-1]
  out[i] = tdnn_block(in_i, block[i-1], dilation)
return concat(out[0..7], dim=0)    # [512, T]
```

Each of the 7 inner tdnn blocks has K=3, dilation=d (2/3/4 per layer).

### Squeeze-Excitation block

Channel-wise attention via global statistics.

```
m = global_mean(x, dim=T)       # [C, 1]
h = ReLU(FC1(m))                 # [C→se_channels, 1]
s = Sigmoid(FC2(h))             # [se_channels→C, 1]
return x * s                     # channel-wise scale
```

Where `se_channels = C // 2` (256 for all three SE-Res2Net blocks).

### Multi-layer Feature Aggregation (MFA)

Concatenates the three SE-Res2Net outputs along the channel axis
and applies a pointwise conv to fuse them.

```
mfa_in = concat(blk1_out, blk2_out, blk3_out, dim=0)  # [1536, T]
y = same_conv1d(mfa_in, w_mfa, b_mfa, K=1)
y = ReLU(y)
```

### Attentive Statistics Pooling (ASP)

Collapses the time dimension into a fixed-size vector via
attention-weighted mean and standard deviation.

```
m = mean(x, dim=T)               # [C, 1] broadcast to [C, T]
s = std(x, dim=T)                # [C, 1] broadcast to [C, T]
att_input = concat(x, m, s, dim=0)  # [3C, T] = [1536, T] when C=512
att = tdnn_block(att_input, asp_tdnn, dilation=1)  # [1536→C, T]
att = Tanh(att)
att = Conv1d(att, asp_conv, K=1)  # [C→1, T] → logits per frame
att = Softmax(att, dim=T)        # attention weights over time

weighted_mean = sum(att * x, dim=T)   # [C, 1]
weighted_std  = sqrt(sum(att * (x - weighted_mean)^2, dim=T))  # [C, 1]
return concat(weighted_mean, weighted_std, dim=0)  # [2C, 1] = [1024, 1]
```

The ASP tdnn has K=3 (not K=1), same as the other tdnn_blocks.
The attention conv has K=1.

### Final FC

```
emb = W_fc @ asp_output + b_fc    # [3072→enc_dim, 1]
emb = reshape(enc_dim)            # 1D vector
```

## Mel spectrogram front-end

Computed in host code (not ggml) using standard signal processing before
entering the ECAPA graph.

### Parameters

| Parameter | Value |
|-----------|-------|
| Sample rate | 24000 Hz |
| FFT size (n_fft) | 1024 |
| Hop length | 256 |
| Window length | 1024 |
| Mel bands | 128 |
| Min frequency | 0 Hz |
| Max frequency | 12000 Hz |
| Mel scale | Slaney (librosa default) |
| Window type | Periodic Hann |
| Compression | log(clamp(x, min=1e-5)) |
| Output shape | (T_mel, 128) row-major (time, mel) |

### Algorithm

1. Reflect-pad input: `pad = (n_fft - hop) / 2 = 384` on both sides.
2. Magnitude STFT with `center=False` (the padding is done explicitly
   so the first frame aligns with sample 0).
3. `mel = mel_filterbank @ mag_spec` — 128-band Slaney mel filterbank.
4. `log_mel = log(max(mel, 1e-5))`.

### Input constraints

- Sample rate must be 24 kHz (linear resample from source rate to 24 kHz
  is caller's responsibility).
- Mono audio required.
- Minimum ~0.5 s (192 samples) for a meaningful embedding; ~3-10 s
  recommended for robust speaker identity extraction.

## Weight tensor shapes

See `GGUF_LAYOUT.md` for the complete weight table.
