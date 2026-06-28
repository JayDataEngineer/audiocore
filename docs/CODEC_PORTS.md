# CODEC_PORTS.md — known-good references for the MOSS + Qwen3-TTS codec ports

This document is the canonical reference for **Stage 16 (MOSS codec port)**
and **Stage 17 (Qwen3-TTS codec port)**. Both upstream codecs already have
production-quality GGML implementations. Audiocore's task is to adapt — not
reimplement — them into the `TensorStorage` / `WeightLoader` / `Backend`
abstractions and replace the silence stubs in `MossSession::decode_codec()`
and `Qwen3TtsSession::run_inference()` Phase 4.

The summary tables below are the contract for the port. Every name, shape and
constant comes from a file we have on disk in `<RAY_ROOT>/infra/repos/openmoss/`
or fetched to `/tmp/`.

---

## 1. MOSS-Audio-Tokenizer

### 1.1 Reference implementation

| Field | Value |
|---|---|
| **Repo** | [`pwilkin/openmoss`](https://github.com/pwilkin/openmoss) |
| **License** | Apache-2.0 (compatible with audiocore) |
| **Path on disk** | `<RAY_ROOT>/infra/repos/openmoss/src/codec.cpp` (1087 lines) |
| **Converter** | `convert_hf_to_gguf.py` in same repo (build-time only; not needed at runtime) |
| **Pre-built GGUFs** | [`smcleod/MOSS-TTS-v1.5-GGUF`](https://huggingface.co/smcleod/MOSS-TTS-v1.5-GGUF) (sidecar: `X.gguf` + `X.extras.gguf`); mirror [`ilintar/moss-tts-gguf`](https://huggingface.co/ilintar/moss-tts-gguf) |

### 1.2 Architecture (decoder)

A 4-stage **ProjectedTransformer** with a 32-RVQ LFQ quantizer front-end. All
layers are causal, pre-LN, with per-channel `LayerScale`. From
`openmoss/src/codec.cpp` lines 770–908 (`CodecGraphs::decode`):

```
codes (32, T_audio) int32
   │
   │ per quantizer i ∈ [0,32):
   │   ggml_get_rows(m_codebook[i], codes_in[i])     → (8, T)     f32
   │   mul_mat(m_q_oproj_w[i]) + m_q_oproj_b[i]       → (512, T)   f32
   │   sum across the 32 quantizers
   ▼
sum (512, T)
   │ mul_mat(m_quant_oproj_w) + m_quant_oproj_b        → (768, T)
   ▼
4 × [ stage_s  → patch_upsample(patch_after_s) ]
   │
   ▼
waveform reshape_1d → (T_audio * 1920,)               f32  ≈ 80 ms at T_audio=1
```

**Constants** (top of `codec.cpp`):

```cpp
CODEC_NUM_VQ        = 32
CODEC_CODEBOOK_DIM  = 8        // per-quantizer embedding width
CODEC_QUANT_DIM     = 512      // per-quantizer conv-out channels
CODEC_D_MODEL_ROOT  = 768      // input to stage 0
NORM_EPS            = 1e-5
ROPE_FREQ_BASE      = 10000.0f
ROPE_TYPE           = GGML_ROPE_TYPE_NORMAL  // interleaved-pair
```

**`DECODER_STAGES[4]`** (4 transformer stages; `patch_after` doubles time
resolution at the end of each stage except the last):

| Stage | d_model | ffn_dim | n_heads | n_layers | codebook_dim | quant_dim | patch_after |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 768  | 1280 | 20 | 12  | 8  | 5120 | 2 |
| 1 | 1280 | 1280 | 20 | 12  | 32 | 5120 | 2 |
| 2 | 1280 | 1280 | 20 | 12  | 32 | 5120 | 2 |
| 3 | 1280 | 1280 | 20 | 512 | 32 | 1280 | 0 |

After the four stages, total upsampling = 2 × 2 × 2 = ×8 → final patch is 240,
and the waveform is `T_audio × 1920` samples (1920 = 8 × 240, matching the
MOSS tokenizer's nominal 80 ms / frame at 24 kHz).

### 1.3 Tensor naming (`moss.codec.*` namespace)

Every tensor the codec graph reads lives under `moss.codec.*` in the
sidecar GGUF. From `CodecGraphs::resolve()` (lines 250–400 of codec.cpp):

**Quantizer front-end (per codebook `i ∈ [0,32)`):**

```
moss.codec.quantizer.q.{i}.codebook.weight   (8, 1024) f16   // codebook table
moss.codec.quantizer.q.{i}.oproj.weight      (8, 512)  f16   // conv1d 8→512
moss.codec.quantizer.q.{i}.oproj.bias        (512,)    f32
moss.codec.quantizer.q.{i}.oproj.g           (512,)    f32   // unused on decode
moss.codec.quantizer.q.{i}.oproj.t           (512,)    f32   // unused on decode
```

**RVQ output projection (after summing the 32 quantizers):**

```
moss.codec.quantizer.oproj.weight  (512, 768)  f16
moss.codec.quantizer.oproj.bias    (768,)      f32
```

**Per stage `s ∈ [0,4)`, per layer `l ∈ [0,n_layers)`:**

```
moss.codec.dec.{s}.tr.l.{l}.norm1.weight        (d_model,) f32
moss.codec.dec.{s}.tr.l.{l}.norm1.bias           (d_model,) f32
moss.codec.dec.{s}.tr.l.{l}.attn.inp.0.weight    (d_model, 3*d_model) f16   // fused QKV
moss.codec.dec.{s}.tr.l.{l}.attn.outp.0.weight   (d_model, d_model)   f16
moss.codec.dec.{s}.tr.l.{l}.norm2.weight         (d_model,) f32
moss.codec.dec.{s}.tr.l.{l}.norm2.bias           (d_model,) f32
moss.codec.dec.{s}.tr.l.{l}.linear1.weight       (d_model, ffn_dim)   f16
moss.codec.dec.{s}.tr.l.{l}.linear2.weight       (ffn_dim, d_model)   f16
moss.codec.dec.{s}.tr.l.{l}.layer_scale_1.scale  (d_model,) f32
moss.codec.dec.{s}.tr.l.{l}.layer_scale_2.scale  (d_model,) f32
```

**Stage input/output projections:**

```
moss.codec.dec.{s}.iproj.weight   (d_in, d_out)  f16   // present when d_in ≠ d_out
moss.codec.dec.{s}.oproj.weight   (d_in, d_out)  f16   // present on stage 3 (1280 → 1)
```

**Weight-norm reconstruction.** Several tensors are stored as a direction
tensor `wp1` and a magnitude scalar tensor `wp0` rather than as a fused
weight. The reconstruction formula is:

```
w[o, i] = wp0[o] * wp1[o, i] / sqrt(Σ_i wp1[o, i]^2)
```

This materialises the weight on the host as f16 then uploads it to the
backend. audiocore's port should do the same at tensor-bind time (one-shot,
not per-graph), so the runtime path sees a normal `weight` tensor.

### 1.4 Decoder-side architecture details

Per-layer forward (openmoss codec.cpp lines 660–680):

```
y  = layer_norm(x, norm1) ;
y  = attention(y)          ;   // flash_attn_ext, causal, RoPE interleaved-pair
y *= layer_scale_1         ;
x += y                     ;

y  = layer_norm(x, norm2)  ;
y  = gelu(linear1(y))      ;
y  = linear2(y)            ;
y *= layer_scale_2         ;
x += y                     ;
```

Attention: fused QKV via `attn.inp.0.weight`, head_dim = d_model / n_heads,
RoPE `mode=GGML_ROPE_TYPE_NORMAL` `freq_base=10000`, then
`ggml_flash_attn_ext` with the (T, T) f16 causal mask. K/V are cast to f16
to match the llama.cpp KV path; Q stays f32.

Patch upsample (factor 2): reshapes `(d*h, T_in)` → `(d, h, T_in)` →
permutes to `(d, T_in, h)` → flattens to `(d, T_in * h)`. Triples → no,
doubles the time dimension.

### 1.5 Port plan for audiocore (Stage 16)

**New files:**

- `include/audiocore/models/moss_tts/codec.h` — `class MossCodecGraphs`
  adapted from `CodecGraphs`. Single public method:
  `std::vector<float> decode(const int32_t* codes, int32_t n_vq, int32_t T_audio)`.
- `src/models/moss_tts/codec.cpp` — implementation, adapted verbatim from
  `openmoss/src/codec.cpp` with the substitutions in §1.6 below.

**Modified files:**

- `include/audiocore/models/moss_tts/family.h` — extend the `MossSession`
  private member list with the resolved codec tensors (codebook[32],
  per-stage Layer/Stage structs). The existing `codec_dec_root_` field
  becomes the entry point or is removed in favor of a `MossCodecGraphs
  codec_*` member.
- `src/models/moss_tts/loader.cpp` — call
  `bind_codec_tensors(ext_ctx_, wl)` after the existing
  `bind_extension_tensors()` to populate the codec tensor pointers from
  `moss.codec.*` keys. Weight-norm reconstruction runs here, once per
  load.
- `src/models/moss_tts/session.cpp` — replace the silence stub at line 526
  (`MossSession::decode_codec`) with a forward to
  `codec_graphs_.decode(...)`. The body that emits 1 s silence (lines
  511–523 of `run_tts` step 7) becomes the fallback only when
  `!codec_present()`.
- `CMakeLists.txt` — add `src/models/moss_tts/codec.cpp` to the
  `audiocore_models_moss_tts` source list.

**New tests:**

- `tests/test_moss_codec_graph.cpp` — hermetic. Build a tiny synthetic
  GGUF with the §1.3 tensors (small dims, e.g. `d_model=64,
  ffn_dim=128, n_layers=2`), call `MossCodecGraphs::decode` on a
  known `(32, T)` codes matrix, assert the output has length
  `T * 1920` and is finite. Register via `audiocore_test(...)`.

### 1.6 Substitution table (openmoss → audiocore)

| openmoss | audiocore equivalent |
|---|---|
| `m_owner.aux()->backend` | `backend_->ggml_backend()` (existing `Backend` accessor) |
| `aux->tensors` lookup by name | `ggml_tensor_by_name(ext_ctx_, name)` (already used in `bind_extension_tensors`) |
| `WeightLoader` is N/A (openmoss has none) | `wl->get_tensor(name)` for the host-side copy during weight-norm materialization |
| `m_galloc` per-instance | create one `ggml_gallocr` per `MossCodecGraphs` instance, hold as a member |
| `m_owner.codec_present()` | `codec_present_` member set by `bind_codec_tensors` |
| `convert_hf_to_gguf.py` (build-time) | N/A — audiocore reads the sidecar GGUF produced by openmoss's converter directly. Documented as the canonical path in `models/manifest.json`. |

### 1.7 What the silence stub currently does

`MossSession::decode_codec` (line 526 of `src/models/moss_tts/session.cpp`)
emits 1 s of silence and returns `true`. `run_tts` step 7 (lines 505–523)
warns and continues. Both call sites stay in place as the fallback for
GGUFs without the `moss.codec.*` tensors.

---

## 2. Qwen3-TTS-Tokenizer-12Hz

### 2.1 Reference implementation

| Field | Value |
|---|---|
| **Repo** | [`CrispStrobe/CrispASR`](https://github.com/CrispStrobe/CrispASR) |
| **License** | **MIT** (verified via GitHub API: `spdx_id=MIT`) |
| **Files fetched to** | `/tmp/crispasr_qwen3_tts.cpp` (309 921 bytes), `/tmp/crispasr_qwen3_tts.h` (18 248 bytes) |
| **Pre-built GGUF** | [`cstr/qwen3-tts-tokenizer-12hz-GGUF`](https://huggingface.co/cstr/qwen3-tts-tokenizer-12hz-GGUF) — `tokenizer-f16.gguf` (~342 MB), `tokenizer-q8_0.gguf` (~277 MB) |
| **Avoid** | [`predict-woo/qwen3-tts.cpp`](https://github.com/predict-woo/qwen3-tts.cpp) — **NO LICENSE FILE**, default copyright = all rights reserved. Do not copy from it. (Its architecture description in `/tmp/q3t_audio_tokenizer_decoder.{h,cpp}` is still useful as a reference for shape sanity-checks.) |

### 2.2 Architecture (decoder) — WavTokenizer-class

Per the CrispASR header (`/tmp/crispasr_qwen3_tts.h` lines 1–25) and the
qwen3-tts.cpp reference (architecture is identical; shapes confirmed):

| Field | Value |
|---|---|
| **Codebooks** | **16** (not 32! `0` coarse + `1..15` fine via MTP predictor) |
| `codebook_size` | 2048 |
| `codebook_dim` | 256 |
| `latent_dim` | 1024 |
| `hidden_dim` | 512 |
| **Pre-transformer layers** | 8 |
| `n_heads` | 16 |
| `head_dim` | 128 (= 2048/16) |
| `ffn_dim` | 1024 |
| `decoder_dim` | 1536 |
| `upsample_rates` | `{8, 5, 4, 3}` → **480×** total |
| `hop_length` | 480 (matches product of upsample_rates) |
| Activation | **SnakeBeta** (`α*x + (1-α)*tanh²(β*x)`, learnable α/β) |
| Conv block | **ConvNeXt** (depthwise 5 + pointwise in/out) |
| Final upsample | Transpose-1D convs in `upsample_rates` |

**Frame rate.** 24 kHz / 480 = **50 Hz codec frame rate** = 20 ms/frame.
(Qwen3-TTS-Tokenizer-**12Hz** is the model name; the actual decode rate
matches the 24 kHz WAV output.)

### 2.3 Decoder forward (sketch)

```
codes (16, T_codec) int32
   │ per codebook i ∈ [0,16):
   │   ggml_get_rows(codebook[i], codes_in[i])     → (codebook_dim, T) f32
   │   sum into a single (codebook_dim, T) tensor
   ▼
Σ (256, T_codec)
   │ project → (latent_dim, T_codec) = (1024, T)
   ▼
8 × ConvNeXt + SnakeBeta pre-transformer layers
   │ (depthwise separable convs, GELU FFN, residual)
   ▼
transformer body (16 heads, dim 2048, causal)
   │
   ▼
transpose-conv stack with upsample_rates {8, 5, 4, 3}
   │ → 480 × T_codec samples
   ▼
waveform (T_codec * 480,) f32
```

### 2.4 Metal note (CrispASR perf lesson)

CrispASR patched `kernel_conv_transpose_1d` in `ggml-metal.metal` to avoid
GPU stalls on the transpose-convolution stack. If audiocore targets Metal,
the same shader patch is required for usable decode throughput. The patch
ships in CrispASR's `ggml-metal.metal` fork; the diff is small. Document
in `docs/CODEC_PORTS.md` §2.4 if/when audiocore adds a Metal build.

CUDA / CPU / Vulkan paths need no patch.

### 2.5 Tensor naming (preliminary — confirm against cstr GGUF at port time)

Based on the qwen3-tts.cpp reference and CrispASR's
`qwen3_tts_init_codec_only()` (which loads the GGUF via the standard
gguf API), the tensors live under `qwen3_tts.codec.*` (or `tokenizer.*`
— confirm at port time by reading the actual GGUF KV). Expected groups:

```
qwen3_tts.codec.codebook.{i}.weight       (codebook_size=2048, codebook_dim=256)  f16
qwen3_tts.codec.pre_proj.weight           (codebook_dim*16, latent_dim=1024)      f16
qwen3_tts.codec.pre_proj.bias             (latent_dim,)                           f32
qwen3_tts.codec.pretr.{l}.{conv1,conv2,norm,snake_alpha,snake_beta}.{weight,bias}  for l ∈ [0,8)
qwen3_tts.codec.tr.{l}.{attn_qkv,attn_out,norm1,norm2,ffn1,ffn2}.{weight,bias}     for l ∈ [0,n_layers)
qwen3_tts.codec.up.{i}.{conv_t.weight,conv_t.bias}        for i in {8,5,4,3}
```

The exact prefix is to be confirmed by running `gguf_inspector` against
`cstr/qwen3-tts-tokenizer-12hz-GGUF/tokenizer-f16.gguf` at port time.

### 2.6 Port plan for audiocore (Stage 17)

**New files:**

- `include/audiocore/models/qwen3_tts/codec.h` — `class Qwen3TtsCodec`
  with `std::vector<float> decode(const int32_t* codes, int32_t n_vq,
  int32_t T_codec)`.
- `src/models/qwen3_tts/codec.cpp` — adapted from CrispASR's
  `src/qwen3_tts.cpp` (the codec section only — the talker / MTP /
  speaker-encoder sections already have audiocore equivalents via
  `qwen3::Runner`).

**Modified files:**

- `include/audiocore/models/qwen3_tts/family.h` — extend `Qwen3TtsSession`
  with the codec member and resolved tensor pointers.
- `src/models/qwen3_tts/loader.cpp` — `bind_codec_tensors(ext_ctx_, wl)`.
- `src/models/qwen3_tts/session.cpp` — replace the silence stub in
  Phase 4 of `run_inference` with `codec_.decode(...)`. Silence stays as
  the fallback when `!codec_present_`.
- `CMakeLists.txt` — add `src/models/qwen3_tts/codec.cpp` to
  `audiocore_models_qwen3_tts`.

**Tests:** `tests/test_qwen3_tts_codec_graph.cpp` — same shape as the MOSS
codec test, smaller synthetic dims, assert output length = `T_codec * 480`
and finite values.

### 2.7 Bonus: ECAPA-TDNN speaker encoder

CrispASR's header (`qwen3_tts.h`) reveals it has also ported the
**ECAPA-TDNN speaker encoder** used for Voice Clone mode
(`qwen3_tts_set_voice_prompt*` family of functions). This is the same
model GAPS.md §2.3 item 17 lists as 🚧 blocked. When Stage 17 lands we get
a head start on item 17 from the same upstream — track in GAPS.md §2.3
once the codec port is in.

---

## 3. Status snapshot (post-Stage 16)

| Item | Pre-stage | Post-Stage-15 | Post-Stage-16 |
|---|---|---|---|
| MOSS codec | 🚧 blocked on ground-up port | 📋 reference impl identified (Apache-2.0), pre-built GGUF available, port plan in §1.5 | ✅ **ported** — `src/models/moss_tts/codec.cpp` adapts openmoss; auto-activates when GGUF carries `moss.codec.*`; silence fallback otherwise |
| Qwen3-TTS codec | 🚧 blocked on ground-up port | 📋 reference impl identified (MIT), pre-built GGUF available, port plan in §2.6 | 📋 unchanged — Stage 17 |
| ECAPA-TDNN speaker encoder | 🚧 blocked, new architecture | 📋 same upstream (CrispASR) ports it for free once Stage 17 lands | 📋 unchanged — Stage 17 |
| MOSS pre-built GGUF source | `OpenMOSS-Team/MOSS-TTS-GGUF` only | + `smcleod/MOSS-TTS-v1.5-GGUF`, `ilintar/moss-tts-gguf` | unchanged |
| Qwen3-TTS-Tokenizer GGUF | `QwenLM/Qwen3-TTS-Tokenizer` (safetensors, needs converter) | `cstr/qwen3-tts-tokenizer-12hz-GGUF` (ready-to-load GGUF) | unchanged |

The 🟡 codec stub status for MOSS dialogue / voice_design (codec wired but
mode-specific surface partial) and Qwen3-TTS batch / instruct /
voice_design (still silence-stubbed pending Stage 17) remains. The
**silence fallback stays in place** as a graceful degradation path for
GGUFs without the codec tensors.

### 3.1 Stage 16 verification

- Compile-time: `MossCodecGraphs` builds clean against audiocore's
  `Backend`/`TensorStorage`/`WeightLoader` abstractions and the same
  ggml/libllama stack as the Qwen3 backbone.
- Existing 7 ctest entries still pass (no regressions in the framework).
- Runtime parity: `tests/test_moss_e2e.cpp` exercises the full pipeline
  (load → chat template → AR codec-token generation → delay-pattern →
  codec decode → PCM) when pointed at real codec-bearing weights
  (`smcleod/MOSS-TTS-v1.5-GGUF` or a self-built sidecar). Intentionally
  not in ctest per AGENTS.md — requires ~8 GB VRAM and a real GGUF.
