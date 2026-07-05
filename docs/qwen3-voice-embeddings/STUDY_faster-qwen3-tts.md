# Study: faster-qwen3-tts and its GGML Backend

## Overview

[faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) is a
real-time Qwen3-TTS inference engine. Its primary backend is PyTorch with
CUDA-graph capture (static KV cache, replayed decode graphs). GGML is an
**experimental opt-in** backend that wraps
[qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp) by predict-woo
via the `qwentts-cpp-python` Python package.

## Ecosystem map

```
faster-qwen3-tts (Python CLI + API)
    │
    ├── backend="torch" (default)
    │     └── PyTorch CUDA-graph capture
    │           talker: 28-layer Qwen2 (12ms/step)
    │           predictor: 5-layer (26ms/step for 15 codebooks)
    │           codec: torch chunked_decode
    │
    └── backend="ggml" (experimental, opt-in)
          └── qwentts-cpp-python (Python C extension)
                └── qwen3-tts.cpp (C++17, native ggml)
                      ├── text_tokenizer.{h,cpp} — BPE tokenizer
                      ├── audio_tokenizer_encoder.{h,cpp} — ECAPA-TDNN
                      ├── tts_transformer.{h,cpp} — talker + predictor
                      ├── audio_tokenizer_decoder.{h,cpp} — WavTokenizer
                      └── qwen3_tts.{h,cpp} — pipeline orchestrator
```

## GGUF model format

qwen3-tts.cpp uses a **two-file GGUF layout** (simpler than audiocore's
multi-file approach):

| File | Contents | Size (0.6B) | Size (1.7B) |
|------|----------|-------------|-------------|
| `qwen3-tts-{size}-f16.gguf` | Talker + Code Predictor + Speaker Encoder + BPE tokenizer | ~1.3 GB | ~3.5 GB |
| `qwen3-tts-tokenizer-f16.gguf` | WavTokenizer codec decoder (no encoder) | ~200 MB | ~200 MB |

A single Python converter (`scripts/convert_tts_to_gguf.py`) merges the
HuggingFace safetensors into one GGUF. The tokenizer codec is a separate
converter (`scripts/convert_tokenizer_to_gguf.py`).

Quantization: F16 and Q8_0 are supported.

## Prefill layout (qwen3-tts.cpp)

From the qwen3-tts.cpp README (differs from audiocore's layout):

```
Position  | Content
──────────┼──────────────────────────────────────
0-2       | text_proj(<|im_start|>, assistant, \n)
3-6       | codec embeddings (think tokens, language ID)
7         | TTS pad + speaker embedding      ← injection
8         | TTS BOS + codec pad embedding
9+        | text-projected text tokens + codec BOS/embeddings
```

Compare with audiocore's layout (`session.cpp:331`):

```
Position  | Content
──────────┼──────────────────────────────────────
0-2       | text_proj(bos, assistant, \n)
3         | codec_think/nothink
4         | codec_think_bos
5         | language_id (optional)
6         | codec_think_eos
7         | speaker_embedding               ← injection
8+        | ref_text + codec_pad + ICL codes
```

**Key difference**: audiocore's speaker slot is at position 6 (0-indexed),
qwen3-tts.cpp puts it at position 7. Both use the same slot numbering
scheme after the 3 role tokens + codec bridge — the difference is whether
`codec_think_eos` occupies a separate position.

## Speaker embedding caching (.spk / .rvq)

faster-qwen3-tts caches reference audio processing to avoid recomputing
on every clone request:

| File | Contents | Format | Size |
|------|----------|--------|------|
| `*.spk` | Speaker latent (ECAPA-TDNN x-vector) | raw float32 | 4 KB (1024-dim) / 8 KB (2048-dim) |
| `*.rvq` | Acoustic latent (codec tokens from RVQ encode) | raw int32, [16 × T_ref] | depends on audio length |

After the first `clone` request, both files are written to disk. Subsequent
calls use `--ref-spk` + `--ref-rvq` to bypass WAV loading, ECAPA compute,
and codec encoding.

This is conceptually identical to audiocore's voice-caching pattern
(`speaker_embedding` field + `compute_embedding()` method). The difference
is file format:
- faster-qwen3-tts: two separate binary files (`.spk` + `.rvq`)
- audiocore: single `speaker_embedding` float vector + optional
  `reference_audio` re-encoding for ICL

## Performance characteristics

### qwen3-tts.cpp timing breakdown (92 frames, 7.3s audio, CPU)

From the project's instrumentation (F16, no GPU acceleration):

| Phase | Total | Per frame | % of total |
|-------|-------|-----------|------------|
| Prefill compute | 175.9 ms | — | 0.6% |
| Talker forward_step (graph build) | 21.8 ms | 0.2 ms/frame | 0.08% |
| Talker forward_step (graph alloc) | 34.1 ms | 0.4 ms/frame | 0.12% |
| Talker forward_step (compute) | 7717.4 ms | 83.9 ms/frame | 26.7% |
| Code predictor steps (15 per frame) | 19531.7 ms | 212.3 ms/frame | 67.5% |
| Total generate | 28915.0 ms | 3.2 frames/s | 100% |

**Key insight**: The code predictor accounts for ~71% of total generation
time. Each frame requires 15 sequential forward passes (one per fine
codebook). The talker is ~27%, graph overhead is negligible.

### faster-qwen3-tts CUDA-graph timing (RTX 4090)

| Component | Before | After (CUDA graph) |
|-----------|--------|--------------------|
| Talker (28 layers) | 75ms | 12ms |
| Predictor (15 steps) | 190ms | 26ms |
| Python overhead | 65ms | 16ms |
| Total per step | 330ms | 54ms |

CUDA graphs give a 6.1x speedup by collapsing ~500 kernel launches into
a single replay operation.

### RTF comparison across backends

| GPU | Torch RTF | CUDA-Graph RTF | Speedup |
|-----|-----------|----------------|---------|
| RTX 4090 | 0.82 | 4.78 | 5.8x |
| H100 | 0.44 | 3.88 | 8.9x |
| Jetson AGX Orin | 0.18 | 1.31 | 7.3x |

RTF > 1.0 = faster than real-time. The CUDA-graph backend is 2-9x faster
than baseline PyTorch.

## Streaming approach

faster-qwen3-tts uses **chunked codec decode** with a sliding window:
- Every `chunk_size` AR steps, the accumulated codec codes are decoded
  to PCM using a sliding window with **25-frame left context**.
- This matches the upstream codec's `chunked_decode` pattern to avoid
  boundary artifacts.
- Configurable chunk_size: 1 (83ms audio, highest overhead) to 12 (~1s).
  `chunk_size=2` is the minimum for real-time on Jetson.

Compare with audiocore's streaming (`session.cpp:779-807`): audiocore
also decodes per-frame incrementally during the AR loop, but does NOT
use a sliding window — it decodes the full accumulated code matrix
each time from scratch, which gives correct audio but is less efficient.

## Silence padding fix

faster-qwen3-tts applies a default **0.5s silence padding** to the
reference audio before codec encoding. This fixes the "phoneme bleed"
artifact where the reference ending mid-word bleeds into the generated
speech, since the model literally "continues" the reference audio in
ICL mode.

The reference audio + 0.5s silence is encoded, then the reference
portion is trimmed from the decoded output.

audiocore does NOT currently do this — the ICL path in session.cpp
encodes the raw reference audio as-is.

## GGML adapter architecture (qwentts-cpp-python)

The adapter is a **pure Python ctypes wrapper** over qwentts.cpp's C ABI:

```
faster-qwen3-tts (Python)
  │
  ├── faster_qwen3_tts.ggml_backend (Python adapter)
  │     │
  │     ├── from_pretrained() → downloads GGUF from HF
  │     │     (repo: Serveurperso/Qwen3-TTS-GGUF, BF16)
  │     │
  │     ├── generate_voice_clone() → Python → ctypes → C ABI
  │     ├── generate_custom_voice() → Python → ctypes → C ABI
  │     └── generate_voice_design() → Python → ctypes → C ABI
  │           │
  │           └── qwentts-cpp-python (ctypes wrapper, PyPI wheel)
  │                 │
  │                 ├── libqwen.so (C++ qwen3-tts.cpp, compiled with ggml)
  │                 │     ├── qwen3-tts pipeline (talker + predictor + codec)
  │                 │     ├── ECAPA-TDNN speaker encoder
  │                 │     ├── text tokenizer (BPE from GGUF)
  │                 │     └── WavTokenizer codec decoder
  │                 │
  │                 └── libggml.so (vendored ggml)
  │
  └── Reference cache (~/.cache/faster-qwen3-tts/qwentts_refs/)
        ├── *.spk (speaker embedding, raw float32)
        └── *.rvq (acoustic latents, raw int32 [16 × T_ref])
```

### Wheel distribution

The `qwentts-cpp-python` package is published on PyPI (CUDA 12.8 default)
with additional wheels on HuggingFace for CPU, CUDA 12.4, CUDA 13.0:

| Variant | Platform | Install source |
|---------|----------|---------------|
| `0.3.0` (default) | CUDA 12.8, manylinux | PyPI |
| `0.3.0+cpu` | CPU-only | HF wheelhouse |
| `0.3.0+cu124` | CUDA 12.4 | HF wheelhouse |
| `0.3.0+cu128` | CUDA 12.8, manylinux_2_35 | HF wheelhouse |
| `0.3.0+cu130` | CUDA 13 / DGX Spark | HF wheelhouse |

### ABI gaps (not yet parity with Torch backend)

1. **No `non_streaming_mode` switch**: qwentts.cpp ignores the step-by-step
   text-feed mode and uses its native prompt layout. Requesting it emits
   a warning.
2. **Base-model `instruct` rejected**: qwentts.cpp does not wire the
   instruct parameter for base model voice cloning.
3. **Fixed KV-cache length**: qwentts.cpp uses a fixed buffer rather than
   dynamic allocation.

### Profiling support

The GGML adapter attaches a `ggml_profile` snapshot to the first streamed
chunk's metadata, with markers for:
- Python ctypes parameter packing
- Lock wait time
- Native `qt_synthesize()` entry
- First native audio callback
- Callback copy/queue overhead
- First Python yield

Native phase splits (when libqwen has profiling enabled): prompt build,
first talker prefill, first code predictor step, first emit, first codec decode.

## Speaker/.rvq caching

The GGML adapter caches reference audio automatically:

```
~/.cache/faster-qwen3-tts/qwentts_refs/
  ├── <hash>.spk       # ECAPA-TDNN x-vector (raw float32, 4KB)
  └── <hash>.rvq       # RVQ codec tokens (raw int32, [16 × T_ref])
```

- Created on first clone request for a given reference audio.
- Subsequent requests load the cached files directly — no WAV load,
  no ECAPA compute, no codec encode.
- Hash is computed from the reference audio content.

Manually precomputed references are also supported via `--ref-spk` and
`--ref-rvq` CLI flags / Python kwargs.

## GGML vs audiocore comparison

| Aspect | qwen3-tts.cpp / faster-qwen3-tts (GGML) | audiocore |
|--------|------------------------------------------|-----------|
| **GGUF files** | 2 files (model + codec) | 4 files (talker, predictor, codec, speaker) |
| **Talker backend** | ggml llama.cpp for transformer + manual ggml for conv | `qwen3::Runner` (llama.cpp unified) |
| **Speaker encoder** | `audio_tokenizer_encoder.cpp` (ECAPA, ggml) | `speaker_encoder.cpp` (ECAPA, ggml, CPU) |
| **Codec decoder** | `audio_tokenizer_decoder.cpp` (WavTokenizer, ggml) | `codec.cpp` (WavTokenizer, ggml, GPU) |
| **Codec encoder** | NOT ported (ICL uses .rvq cache) | Ported (SEANet + RVQ for ICL) |
| **Predictor** | manual ggml (5-layer transformer) | `qwen3::Runner` (llama.cpp unified) |
| **Streaming** | chunked decode + sliding window (25-frame ctx) | full decode per chunk (no sliding window) |
| **Embedding cache** | `.spk` + `.rvq` binary files | `speaker_embedding` float vector |
| **Silence padding** | 0.5s appended to ref audio by default | Not implemented |
| **Quantization** | F16, Q8_0 | F16, Q8_0, Q4_0 via llama.cpp |
| **GPU support** | Metal, CUDA (via ggml backend) | CUDA, CPU |
| **Python bindings** | `qwentts-cpp-python` (PyPI wheel) | built-in Python C extension |
| **Prefill speed** | 175.9 ms (CPU, 92 frames) | comparable (GPU backend faster) |
| **AR speed** | 83.9 ms/frame talker + 212.3 ms/frame predictor (CPU) | GPU-accelerated via llama.cpp |

## Key takeaways for audiocore's ggml implementation

### 1. Code predictor is the bottleneck

The 15 sequential forward passes for fine codebooks dominate runtime
(~71% in qwen3-tts.cpp, ~58% in faster-qwen3-tts CUDA graphs).
Any optimization to the predictor yields outsized gains.

### 2. Two-file GGUF is simpler, but four-file gives flexibility

qwen3-tts.cpp bundles everything into one GGUF + one codec GGUF.
audiocore's 4-file approach (talker, predictor, codec, speaker) is
more modular but requires more file discovery logic.

### 3. Chunked codec decode with sliding window

faster-qwen3-tts's sliding-window approach (25-frame left context)
avoids boundary artifacts in streaming. audiocore's full-decode
approach is simpler but wasteful — adopt the sliding window.

### 4. Silence padding prevents ICL bleed

The 0.5s silence prepend is a cheap fix for a noticeable artifact.
Should be added to audiocore's ICL path.

### 5. Speaker latent caching is already handled

audiocore's `speaker_embedding` field + `compute_embedding()` method
cover the same use case as `.spk` files. The `.rvq` acoustic latent
cache is a further optimization — cache the full codec-encoded ref
codes alongside the embedding to skip the codec encoder on repeat calls.

### 6. qwen3-tts.cpp's prefill layout differs slightly

audiocore uses position 6 for the speaker embedding, qwen3-tts.cpp
uses position 7. This is because audiocore has a separate
`codec_think_eos` position. Verify correctness against the upstream
Python reference (the official `qwen-tts` package is the ground truth).
