# Model Profiling & Benchmarks

**Hardware:** NVIDIA RTX 4090 (24 GB VRAM)
**Date:** 2026-07-03
**Engine:** audiocore (ggml/CUDA backend) for GGUF models; standalone PyTorch for MOSS-SoundEffect v2

**See also:** [MODELS-SUPPORTED.md](MODELS-SUPPORTED.md) for which models run in
C++ vs Python

---

## Voice cloning modes — Qwen3-TTS (Stage 17b, 2026-07-03)

Two distinct clone paradigms, both wired in C++ via a single ~23 MB
standalone ECAPA-TDNN GGUF
(`qwen3tts-speaker-encoder.gguf`, converted from
`marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B` via `tools/convert_ecapa.cpp`).

| Path | What runs | RTF | Notes |
|------|-----------|-----|-------|
| `tts_batch` with reference_audio | Talker + codec decode | 0.127 (1.7B Base) | Default reference-voice path (no embedding involved) |
| `voice_clone` + reference_audio | ECAPA (~50 ms CPU) + codec encoder (~80 ms GPU) + Talker + codec decode | 0.127 (1.7B Base) | **Full ICL path (Gap K closed 2026-07-03).** ECAPA → 2048-dim speaker vector AND codec encoder → 50 ICL codec tokens, both spliced into the talker prefill. Pass empty `reference_text` (Gap L — text-embedding module not yet converted). |
| `voice_clone` + pre-computed `speaker_embedding` | Talker + codec decode | 0.126 | **Voice-caching pattern.** Compute ECAPA once, reuse the 2048-float vector forever. No per-call WAV I/O, no ECAPA, no codec encoder. RTF identical to Base — embedding injection is free. Exposed as `Session.compute_embedding(wav_path)` in Python bindings (P1, 2026-07-03). |

**ECAPA-TDNN encoder details** — 76 tensors, rank-1 → F32 (biases), rank-2/3
→ F16 (conv_1d weights). Runs on a single CPU backend via direct gallocr
(no scheduler — model is tiny, CPU supports the full conv1d + reflect-pad
op set, and the GPU would just add buffer-copy overhead for a ~46 ms call).
Weight registration mirrors codec.cpp's pattern: loader resolves mmap
pointers via `GgufReader::find()` + `tensor_data_ptr()` and registers them;
`upload_weights_()` copies them into the per-call gallocr buffers after
allocation (without this, all weights are silently zero — latent bug fixed
2026-07-03).

**Emotion conditioning** is text-level via the `instruct` field
(`mode='tts_instruct'`), not a vector — already worked, no porting needed.
The `instruct` field is read mode-agnostically (line 304 of session.cpp)
and prepended to the synthesis text in every non-VoiceDesign mode, so
CustomVoice + emotion, voice_clone + emotion, and tts_batch + emotion all
work. VoiceDesign mode consumes `instruct` for the voice description
(architecturally one slot — same limitation as upstream Qwen3-TTS).

---

## Combo (speaker embedding + emotional instruct + text) — WORKING 2026-07-05

**Status:** the combo (a) pre-computed speaker embedding (.voice file),
(b) separate emotional `instruct` prompt, (c) normal text prompt →
(d) high-quality emotional cloned speech — works end-to-end on **0.6B-Base**
with the freshly rebuilt GGUFs (post `bf16_to_f32` converter fix,
commit 757ce16).

The earlier "Base can't do the combo" claim from the Mastrapasqua blog
("Style control and voice cloning are separate worlds on Base") appears to
have been a description of Base's *training distribution*, not a hard
algorithmic block. Empirically, on the fresh GGUFs:

| Test | Input | Transcript | Tone |
|------|-------|------------|------|
| `09_combo_voice_instruct.wav` | text="I just can't believe you're really gone." + instruct="With deep sadness, whispering, on the verge of tears." + vivian.voice | "just can't believe you're really gone" | "profound grief and disbelief... deep emotional distress" |
| `10_combo_happy.wav` | text="The sun is shining and I feel absolutely wonderful today!" + instruct="Spoken with joyful excitement, bright and cheerful." + vivian.voice | "The sun is shining and I feel absolutely wonderful today!" | "cheerful, enthusiastic, and full of joy" |

### Prerequisites (any one of these was sufficient to break the combo)

1. **GGUF converter fix (commit 757ce16).** The pre-fix converter wrote F32
   norm tensors that pointed at raw BF16 mmap buffers, producing corrupted
   norms (cos 0.55–0.75 vs safetensors). All 0.6B/1.7B talker GGUFs were
   rebuilt; old copies preserved as `*.OLD_buggy.gguf`.
2. **0.6B-native speaker encoder.** The 1.7B ECAPA-TDNN encoder
   (marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B) outputs a 2048-dim vector
   that does NOT match the 0.6B talker's 1024-dim `n_embd`. The previous
   `vc_spk_emb.resize(d, 0.0f)` was a truncation, not a projection. The
   0.6B-Base safetensors bundles its own 1024-dim `speaker_encoder.*`
   tensors; these are extracted via
   `convert_ecapa <0.6B-Base_dir> <out.gguf> --filter-prefix speaker_encoder.`
   into a 16 MB standalone GGUF.
3. **No tts_pad overlay on raw ECAPA (gap A1.b).** Adding the canonical
   `tts_pad_embed` overlay to a raw ECAPA vector at the speaker slot
   collapses output (RMS 0.045 → 0.023, transcription → "Yeah... [beep]").
   Both the canonical HF Qwen3-TTS code and our default omit the overlay
   for raw speaker embeddings. Set `QWEN3TTS_VC_ADD_PAD=1` to A/B test.

### Repro

```bash
# Encode a reference voice once → 1024-dim .voice file (~4 KB)
QWEN3TTS_DIR=/mnt/data/models/audio/qwen3_tts/0.6b-base \
  ./build-debug/qwen_voice encode \
    --model-dir /mnt/data/models/audio/qwen3_tts/0.6b-base \
    --wav weights/reference_output.wav --out vivian.voice

# Apply: voice + emotional instruct + text → high-quality output
QWEN3TTS_DIR=/mnt/data/models/audio/qwen3_tts/0.6b-base \
  ./build-debug/qwen_voice apply \
    --model-dir /mnt/data/models/audio/qwen3_tts/0.6b-base \
    --voice vivian.voice \
    --text "I just can't believe you're really gone." \
    --instruct "With deep sadness, whispering, on the verge of tears." \
    --out webapp/clips/combo.wav \
    --max-new-tokens 200
```

### What still DOESN'T work on Base

- **Voice identity fidelity.** The cloned voice captures prosody and gender
  vaguely but is not a bit-identical timbre match to the reference. The
  Mastrapasqua CV+WDELTA path is required for studio-quality cloning (still
  TODO: TPAD per-frame override + WOVR text_proj/codec_embd override).
- **1.7B-Base.** Requires a rebuild from source safetensors (not present
  locally). The 0.6B-Base results above are with the rebuilt 0.6B GGUFs.

---

## Summary — All Models, All Variants

### Qwen3-TTS Family (audiocore C++/ggml, f16 weights)

| Variant | Params | Load | Generate | Audio out | SR | RTF | Speedup vs RT |
|---------|--------|------|----------|-----------|------|------|---------------|
| 0.6B Base (voice clone) | 0.6B | 0.32 s | 26.5 s | 327.7 s | 24 kHz | 0.081 | **12.4×** |
| 0.6B CustomVoice | 0.6B | 0.28 s | 26.5 s | 327.7 s | 24 kHz | 0.081 | **12.4×** |
| 1.7B Base (voice clone) | 1.7B | 0.62 s | 41.7 s | 327.7 s | 24 kHz | 0.127 | 7.9× |
| 1.7B CustomVoice | 1.7B | 0.51 s | 41.3 s | 327.7 s | 24 kHz | 0.126 | 7.9× |
| 1.7B VoiceDesign | 1.7B | 10.1 s | 41.5 s | 327.7 s | 24 kHz | 0.127 | 7.9× |

Three operating modes across the family:
- **Base** — 3-second rapid voice clone from reference audio.
- **CustomVoice** — style control via instruct text + 9 premium speakers (Vivian, Serena, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee, Uncle_Fu).
- **VoiceDesign** — design voices from natural-language descriptions.

### MOSS Family (audiocore C++/ggml)

| Variant | Quant | Load | Generate | Audio out | SR | RTF | Speedup vs RT |
|---------|-------|------|----------|-----------|------|------|---------------|
| MOSS-TTS-Delay v1.5 | Q8_0 | 8.0 s | 9.7 s | 30.1 s | 24 kHz | 0.322 | **3.1×** |
| MOSS-VoiceGenerator | Q8_0 | 0.96 s | 2.54 s | 0.72 s | 24 kHz | 3.53 | 0.3× |
| MOSS-SoundEffect v1 | f16 | 2.8 s | 9.7 s | 11.7 s | 24 kHz | 0.83 | 1.2× |

- **MOSS-TTS-Delay** — Qwen3-8B backbone + 32 RVQ audio codebooks + 1.6B transformer codec. Delay-pattern autoregressive generation. Supports voice cloning via `--reference`.
- **MOSS-VoiceGenerator** — voice design from text descriptions (n_vq=16, shorter clips).
- **MOSS-SoundEffect v1** — discrete-token AR SFX (n_vq=16). Same delay-pattern architecture as TTS-Delay, trained for sound effects. Same AR code path as `tts` mode — the checkpoint weights determine the output type. Superseded in quality by v2 (below), but runs entirely in audiocore's C++/ggml stack (no Python/PyTorch dependency).

### ACE-Step Family (audiocore C++/ggml, Q8_0)

| Variant | LM | DiT steps | Load | Generate | Audio out | SR | RTF |
|---------|-----|-----------|------|----------|-----------|------|------|
| 1.5 Turbo | 1.7B | 8 | 1.3 s | 39.1 s | 10.0 s stereo | 48 kHz | 3.90 |
| 1.5 SFT | 1.7B | 50 | 2.0 s | 108.1 s | 10.0 s stereo | 48 kHz | 10.81 |
| 1.5 Base (new) | 1.7B | 50 | 1.8 s | 104.9 s | 10.0 s stereo | 48 kHz | 10.49 |
| XL Turbo | 4B | 8 | 4.2 s | 57.2 s | 10.0 s stereo | 48 kHz | 5.72 |
| XL SFT | 4B | 50 | 10.0 s | 230.6 s | 10.0 s stereo | 48 kHz | 23.06 |
| XL Base | 4B | 8 | 1.8 s | 56.8 s | 10.0 s stereo | 48 kHz | 5.68 |

**ScragVAE** (drop-in VAE decoder swap, `scragvae-BF16.gguf` from
`scragnog/Ace-Step-1.5-ScragVAE`): verified end-to-end with turbo DiT —
**RTF 3.70** (vs 3.90 stock, same architecture ⇒ identical decode cost,
difference is run-to-run noise). Output RMS 0.012083 (vs 0.012079 stock —
essentially identical since the turbo DiT produces the same latent; only
the decoder differs). To use: symlink `scragvae-BF16.gguf` → `vae-BF16.gguf`
in the variant's model dir. Verified pure drop-in at the tensor level too:
365 tensors, identical names/shapes, same `acestep-vae` architecture tag.

Text-to-music: Qwen3 text encoder → 5Hz music-code LM → DiT flow-matching diffusion → audio VAE → 48 kHz stereo PCM. Turbo variants use 8 diffusion steps (fast but lower quality); SFT uses 50 steps.

### MOSS-SoundEffect v2 (standalone Python, PyTorch bf16)

| Duration | Load | Generate | RTF | Peak VRAM | Diffusion rate |
|----------|------|----------|-----|-----------|---------------|
| 3 s | 9.9 s | 6.1 s | 2.04× | 14.9 GB | 10.2 it/s |
| 10 s | 8.0 s | 6.1 s | **0.61×** | 14.9 GB | 10.2 it/s |

Wan-audio DiT (1.3B) + DAC VAE + Qwen3 text encoder. Flow-matching diffusion at 50 steps. Runs in standalone project at `../moss-sfx-v2/`.

---

## Codec Decode — O(T²) → O(T) Fix

The Qwen3-TTS codec decoder (WavTokenizer-class) previously used ggml's stock
`conv_transpose_1d` CUDA kernel, which is **O(T²)** in sequence length. At
T=50 codec frames this alone took **29.4 s** — dominating the entire pipeline.

**Fix:** Decomposed transposed conv1d into cuBLAS `mul_mat` + `col2im_1d`
gather, making it **O(T)**.

| Codec frames | Stock kernel | Fixed (matmul + col2im) | Speedup |
|--------------|-------------|------------------------|---------|
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

## ACE-Step GGML_MAX_NAME Fix

ACE-Step DiT GGUFs from Serveurperso/ACE-Step-1.5-GGUF contain tensor names
up to 73 characters (e.g. `double_b.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm1.norm.weight`).
Stock `GGML_MAX_NAME=64` in ggml rejects these, preventing model load.

**Fix:** `add_compile_definitions(GGML_MAX_NAME=128)` in CMakeLists.txt before
`add_subdirectory(third_party/llama.cpp)`. The `#ifndef` guard in ggml.h picks
up the override.

---

## Cross-Model Comparison

### TTS / Speech Generation (RTF — lower is better)

```
Qwen3-TTS 0.6B Base        ████░░░░░░░░░░░░░░░░  0.081  (12.4× RT)
Qwen3-TTS 0.6B CustomVoice ████░░░░░░░░░░░░░░░░  0.081  (12.4× RT)
Qwen3-TTS 1.7B Base        ██████░░░░░░░░░░░░░░  0.127  ( 7.9× RT)
Qwen3-TTS 1.7B CustomVoice ██████░░░░░░░░░░░░░░  0.126  ( 7.9× RT)
Qwen3-TTS 1.7B VoiceDesign ██████░░░░░░░░░░░░░░  0.127  ( 7.9× RT)
MOSS-TTS-Delay v1.5 Q8     ████████████████░░░░  0.322  ( 3.1× RT)
```

### Music Generation (RTF — lower is better)

```
ACE-Step XL Base (4B, 8-step)   ████████████████████  3.90
ACE-Step 1.5 Turbo (8-step)     ████████████████████  3.90
ACE-Step XL Turbo (4B, 8-step)  ████████████████████████░░░░  5.72
ACE-Step 1.5 SFT (50-step)      ████████████████████████████████  10.81
ACE-Step XL SFT (4B, 50-step)   ████████████████████████████████████████████  23.06
```

### Sound Effects (RTF — lower is better)

```
MOSS-SFX v2 (10s)        █████████████░░░░░░░  0.61
MOSS-SFX v1 (f16)        █████████████████░░░  0.83
MOSS-SFX v2 (3s)         ████████████████████  2.04
```

---

## Environment Details

### audiocore (C++ GGUF models)

- Build: `cmake -B build -DENGINE_ENABLE_CUDA=ON && cmake --build build -j`
- Backend: ggml CUDA, f16 weights for Qwen3-TTS, Q8_0 for MOSS/ACE-Step
- Weights: GGUF format, mmap'd at runtime
- Models under `/mnt/data/models/audio/`

### MOSS-SoundEffect v2 (Python)

- Project: `../moss-sfx-v2/` (standalone, uv-managed)
- Python 3.12, torch 2.9.0+cu128, bf16
- `audiotools` stub replaces descript-audiotools (avoids protobuf≥6 conflict)

```bash
cd ../moss-sfx-v2
uv sync --extra cu128
MODEL_DIR=/mnt/data/models/audio/moss-soundeffect-v2 \
TORCHDYNAMO_DISABLE=1 \
uv run python benchmark.py --model $MODEL_DIR --seconds 10 --steps 50
```

---

## Reproducing

| Artifact | Location |
|----------|----------|
| audiocore benchmark script | `/tmp/bench_audiocore.py` |
| audiocore raw results (JSON) | `/tmp/audiocore_bench_results.json` |
| MOSS-SFX v1 benchmark results | `/tmp/moss_sfx_v1_bench.json` |
| ACE-Step benchmark | `tests/test_acestep_e2e.cpp` (env: `ACESTEP_DIR`, `ACESTEP_DIT_VARIANT`, `ACESTEP_LM_VARIANT`) |
| MOSS-SFX v2 project | `../moss-sfx-v2/benchmark.py` |
| Codec fix source | `src/models/qwen3_tts/codec.cpp` |

### Running ACE-Step benchmarks

```bash
export LD_LIBRARY_PATH=build:build/third_party/llama.cpp/build/src
ACESTEP_DIR=/mnt/data/models/audio/acestep-cpp-converted/ \
ACESTEP_DIT_VARIANT=turbo ACESTEP_LM_VARIANT=1.7B \
ACESTEP_DURATION=10 \
./build/tests/test_acestep_e2e
```

### Running Qwen3-TTS / MOSS benchmarks

```bash
# Via Python bindings
PYTHONPATH=build-py/python python3 -c "
import audiocore; audiocore.init()
sess = audiocore.Session.create('qwen3_tts')
sess.load('/mnt/data/models/audio/qwen3_tts/0.6b-base', backend='ggml_cuda', device=0)
pcm, sr = sess.run_tts('Hello world', temperature=0.7, top_p=0.9, seed=42)
print(f'{len(pcm)} samples @ {sr} Hz = {len(pcm)/sr:.1f}s')
"
```

### Running MOSS-SoundEffect v1 benchmark

```bash
export LD_LIBRARY_PATH=build:build/third_party/llama.cpp/build/src
PYTHONPATH=build-py/python python3 -c "
import audiocore; audiocore.init()
sess = audiocore.Session.create('moss_tts')
sess.load('/mnt/data/models/audio/moss-sfx', backend='ggml_cuda', device=0)
pcm, sr = sess.run_tts('heavy rain on tin roof', mode='sfx', temperature=0.8, top_p=0.9, seed=42)
print(f'{len(pcm)} samples @ {sr} Hz = {len(pcm)/sr:.1f}s')
"
```

---

## Codec / Audio Tokenizer Components

These are not standalone generation models — they're codec components used
by the TTS/SFX models above.

| Component | Version | SR | Channels | On Disk | Notes |
|-----------|---------|-----|----------|---------|-------|
| MOSS-Audio-Tokenizer | v1 | 24 kHz | mono | ✅ `moss-audio-tokenizer/` | Used by MOSS-TTS-Delay. Codec encoder/decoder. |
| MOSS-Audio-Tokenizer Nano | v1 | 24 kHz | mono | ✅ `moss-audio-tokenizer-nano/` | Smaller variant. |
| MOSS-Audio-Tokenizer | **v2** | **48 kHz** | **stereo** | ✅ downloaded | Used internally by MOSS-SFX v2 Python project (48 kHz stereo output). Not ported to GGML. |
| Qwen3-TTS-Tokenizer-12Hz | — | 24 kHz | mono | ✅ `qwen3_tts/0.6b-base/tokenizer-f16.gguf` | Used by all Qwen3-TTS variants. GGUF port (codec decode fix above). |

### Release Notes

**MOSS-TTS v1.5** (2026-05-26): Stronger multilingual synthesis with language
tags, more stable voice cloning, better long-reference short-text cloning,
punctuation-following prosody, explicit pause control via `[pause X.Ys]`.

**MOSS-Audio-Tokenizer v2** (2026-06-07): Natively supports 48 kHz stereo
input and output (v1 was 24 kHz mono). Not yet integrated into audiocore.

---

## Models Not Yet Benchmarked

| Model | Status | Blocker |
|-------|--------|---------|
| MOSS-TTSD (dialogue) | Not downloaded | Same Delay architecture — needs HF download + GGUF conversion |
| MOSS-TTS-Local-Transformer-v1.5 | Not downloaded | New MossTTSLocal architecture (4B backbone, 48 kHz stereo) |
| MOSS-TTS-Realtime | Not downloaded | MossTTSRealtime architecture — needs new code path |
| MOSS-TTS-Nano | Not downloaded | MossTTSNano architecture (~100M, CPU-first) |
