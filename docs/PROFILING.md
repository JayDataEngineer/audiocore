# Model Profiling & Benchmarks

**Hardware:** NVIDIA RTX 4090 (24 GB VRAM)
**Date:** 2026-07-02
**Engine:** audiocore (ggml/CUDA backend) for GGUF models; standalone PyTorch for MOSS-SoundEffect v2

---

## Summary Table

| Model | Backend | Load | Generate | Audio | RTF | VRAM | Notes |
|-------|---------|------|----------|-------|-----|------|-------|
| Qwen3-TTS 0.6B | ggml/CUDA f16 | 0.3 s | 26.2 s | 327.7 s | 0.08× | ~3 GB | 12.5× faster than realtime |
| MOSS-TTS v1.5 Q8_0 | ggml/CUDA | 8.0 s | 9.7 s | 30.1 s | 0.32× | ~5 GB | 3.1× faster than realtime |
| MOSS-VoiceGen Q8_0 | ggml/CUDA | 1.0 s | 2.5 s | 0.7 s | 3.53× | ~3 GB | Short clips, below realtime |
| MOSS-SoundEffect v2 (3 s) | PyTorch bf16 | 9.9 s | 6.1 s | 3.0 s | 2.04× | 14.9 GB | 50 diffusion steps |
| MOSS-SoundEffect v2 (10 s) | PyTorch bf16 | 8.0 s | 6.1 s | 10.0 s | 0.61× | 14.9 GB | 50 steps, faster than RT |

**RTF** = Real-Time Factor (lower = faster). RTF < 1.0 means faster than realtime.

---

## 1. Qwen3-TTS 0.6B (GGUF, audiocore)

**Path:** C++ ggml engine, f16 weights, CUDA backend.

| Metric | Value |
|--------|-------|
| Load time | 0.29 s |
| Best of 3 runs | 26.16 s |
| Audio duration | 327.68 s (≈5.5 min) |
| Samples | 7,864,320 @ 24 kHz |
| RTF | 0.08 (12.5× faster than realtime) |

### Codec decode — O(T²) → O(T) fix

The codec decoder (WavTokenizer-class) previously used ggml's stock
`conv_transpose_1d` CUDA kernel, which is **O(T²)** in sequence length. At
T=50 codec frames this alone took **29.4 s** — dominating the entire pipeline.

**Fix:** Decomposed transposed conv1d into cuBLAS `mul_mat` + `col2im_1d`
gather, making it **O(T)**.

| Codec frame count | Stock kernel | Fixed (matmul + col2im) | Speedup |
|-------------------|-------------|------------------------|---------|
| T = 50 | 29.4 s | 0.042 s | **700×** |

Correctness verified: 92 dB SNR vs stock kernel in isolated test.

**Root cause of multi-chunk crash:** `ggml_gallocr_is_allocated()` returns
true if `t->data != NULL`. Weight tensors retained dangling device pointers
from the previous decode call's gallocr, causing `cudaMemcpyAsync: invalid
argument` on the second chunk.

**Fix:** `reset_weight_data_()` nulls `tensor->data` and `tensor->buffer` on
all registered weights before each `gallocr_alloc_graph`. The gallocr itself
is now per-call (local variable) rather than a persistent member.

Multi-chunk decode verified: 3 chunks (T=64, T=89, T=47) produce 12 s of
audio with no crashes.

---

## 2. MOSS-TTS v1.5 (GGUF Q8_0, audiocore)

| Metric | Value |
|--------|-------|
| Load time | 8.0 s |
| Best of 3 runs | 9.67 s |
| Audio duration | 30.08 s |
| RTF | 0.32 (3.1× faster than realtime) |

---

## 3. MOSS-VoiceGen (GGUF Q8_0, audiocore)

| Metric | Value |
|--------|-------|
| Load time | 0.96 s |
| Best of 2 runs | 2.54 s |
| Audio duration | 0.72 s |
| RTF | 3.53 (below realtime — short-clip model) |

---

## 4. MOSS-SoundEffect v2.0 (PyTorch, standalone project)

**Project:** `<MOSS_SFX_ROOT>/` (standalone
uv-managed project, not part of audiocore C++ engine)

**Architecture:** Wan-audio DiT (1.3B params) + DAC VAE + Qwen3 text encoder,
flow-matching diffusion, 48 kHz output.

### Benchmark (RTX 4090, bf16, 50 steps)

| Prompt length | Load | Generate | Diffusion rate | Peak VRAM |
|---------------|------|----------|---------------|-----------|
| 3 s audio | 9.9 s | 6.1 s | 10.2 it/s | 14.9 GB |
| 10 s audio | 8.0 s | 6.1 s | 10.2 it/s | 14.9 GB |

Key observations:
- **Diffusion rate is constant** (~10 it/s) regardless of audio length — the
  per-step compute scales with latent frames but overhead dominates at short lengths.
- **VRAM is constant** (14.9 GB) — the DiT parameters (10 GB) dominate; latent
  frames are tiny relative to model size.
- **10 s generation is faster than realtime** (RTF 0.61×). 3 s is not (RTF 2.04×)
  because the 50-step loop overhead is fixed.
- `TORCHDYNAMO_DISABLE=1` is set — torch.compile's first-run compilation cost
  (minutes) isn't worth it for one-shot generation.

### Environment

- Python 3.12, uv-managed (astral), **not conda**
- torch 2.9.0+cu128, diffusers, transformers
- `audiotools` stub replaces descript-audiotools (avoids protobuf≥6 conflict;
  only `AudioSignal` and `ml.BaseModel` referenced, neither called during inference)
- Dockerfile: CUDA 12.8 runtime + uv

---

## Environment Details

### audiocore (C++ GGUF models)

- Build: `cmake --build build --target audiocore_server`
- Backend: ggml CUDA backend, f16 weights (`--quant f16`)
- Weights: GGUF format, mmap'd at runtime

### MOSS-SoundEffect v2 (Python)

```bash
cd <MOSS_SFX_ROOT>
uv sync --extra cu128
MODEL_DIR=/mnt/data/models/audio/moss-soundeffect-v2 \
TORCHDYNAMO_DISABLE=1 \
uv run python benchmark.py --model $MODEL_DIR --seconds 10 --steps 50
```

---

## Reproducing

All benchmark scripts and raw results:

| Artifact | Location |
|----------|----------|
| audiocore benchmark script | `/tmp/bench_audiocore.py` |
| audiocore raw results (JSON) | `/tmp/audiocore_bench_results.json` |
| MOSS-SFX v2 project | `<MOSS_SFX_ROOT>/` |
| MOSS-SFX v2 benchmark | `moss-sfx-v2/benchmark.py` |
| Codec fix source | `src/models/qwen3_tts/codec.cpp` |
