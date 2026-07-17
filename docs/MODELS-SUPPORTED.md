# Models We Support — C++ vs Python

Two parallel audio stacks live in this workspace. This file is the canonical
breakdown of which models run where, so you never have to grep to find out.

**C++ stack — `audiocore/`** (this repo, GGUF weights, ggml/CUDA backend)
**Python stack — two projects:**
- `../moss-sfx-v2/` — minimal isolated MOSS-SoundEffect v2 runner
- `../../ray/` — the full ray audio service (`services/audio/` forge adapters
  + `vendor/` PyTorch model implementations). Covers every MOSS architecture,
  ACE-Step, Qwen3-TTS, and ASR.

Last updated: 2026-07-02

---

## At a glance

| Family | Variant | audiocore (C++) | Format | Python (ray) | Format | Python (moss-sfx-v2) | Format |
|--------|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen3-TTS** | 0.6B Base | ✅ RTF 0.081 | GGUF f16 | ✅ | HF safetensors (bf16) | — | — |
| | 0.6B CustomVoice | ✅ RTF 0.081 | GGUF f16 | ✅ | HF safetensors (bf16) | — | — |
| | 1.7B Base | ✅ RTF 0.127 | GGUF f16 | ✅ | HF safetensors (bf16) | — | — |
| | 1.7B CustomVoice | ✅ RTF 0.126 | GGUF f16 | ✅ | HF safetensors (bf16) | — | — |
| | 1.7B VoiceDesign | ✅ RTF 0.127 | GGUF f16 | ✅ | HF safetensors (bf16) | — | — |
| **MOSS-TTS (Delay)** | MOSS-TTS v1.5 | ✅ RTF 0.322 | GGUF Q8_0 | ✅ | GGUF Q8_0 (HTTP→openmoss.cpp) | — | — |
| | MOSS-VoiceGenerator | ✅ RTF 3.53 | GGUF Q8_0 | ✅ | GGUF Q8_0 (HTTP→openmoss.cpp) | — | — |
| | MOSS-SoundEffect v1 | ✅ RTF 0.83 | GGUF f16 | — | — | — | — |
| **MOSS-TTSD (Dialogue)** | v0.7 | ⚠️ GGUF only¹ | GGUF f16 (blocked) | ✅ | HF safetensors (bf16) | — | — |
| **MOSS-TTS-Local** | v1.5 ("new codex") | ❌ | — | ✅ | HF safetensors (bf16) | — | — |
| **MOSS-TTS-Realtime** | streaming agent | ❌ | — | ✅ | HF safetensors (bf16) | — | — |
| **MOSS-SoundEffect v2** | Wan-audio DiT | ❌ | — | ✅ | HF safetensors (bf16) | ✅ RTF 0.61 | HF safetensors (bf16) |
| **ACE-Step** | 1.5 Turbo | ✅ RTF 3.90 | GGUF Q8_0 | ✅ | HF safetensors (bf16) | — | — |
| | 1.5 SFT | ✅ RTF 10.81 | GGUF Q8_0 | ✅ | HF safetensors (bf16) | — | — |
| | 1.5 Base | ✅ RTF 10.49 | GGUF Q8_0 | ✅ | HF safetensors (bf16) | — | — |
| | XL Turbo | ✅ RTF 5.72 | GGUF Q8_0 | ✅ | HF safetensors (bf16) | — | — |
| | XL SFT | ✅ RTF 23.06 | GGUF Q8_0 | ✅ | HF safetensors (bf16) | — | — |
| | XL Base | ✅ RTF 5.68 | GGUF Q8_0 | ✅ | HF safetensors (bf16) | — | — |
| | **ScragVAE** (alt decoder) | ✅ RTF 3.70² | GGUF BF16 (verified drop-in) | ✅ | HF safetensors (bf16) | — | — |
| **ASR (speech-to-text)** | Whisper-class | ❌ | — | ✅ | HF safetensors (bf16) | — | — |
| **Codecs** | MOSS-Audio-Tokenizer v1 | ✅ in C++ | GGUF (codec tensors in sidecar) | ✅ | HF safetensors | — | — |
| | MOSS-Audio-Tokenizer v2 (48 kHz stereo) | ❌ | — | ✅ | HF safetensors | ✅ (internal) | HF safetensors |
| | Qwen3-TTS-Tokenizer-12Hz | ✅ in C++ | GGUF f16 / Q8_0 | — | — | — | — |
| | **Qwen3-TTS-Tokenizer-25Hz** | ❌ Gap F³ | — | ❌ | — | — | — |
| | XY_Tokenizer (TTSD codec) | ❌ | — | ✅ | HF safetensors | — | — |

¹ **Gap G closed (2026-07-02):** ACE-Step 1.5 Base GGUF downloaded (`acestep-v15-base-Q8_0.gguf`, 2.4 GB) and added to manifest. Code patched in `loader.cpp` (`-base` substring detection, excluding `xl-base`) and `session.cpp` (50-step linear schedule for `"base"` variant). Benchmarked: RTF 10.49

² **Gap H closed (2026-07-02):** ScragVAE GGUF downloaded (`scragvae-BF16.gguf`, 322 MB). Verified pure drop-in (365 tensors, identical names/shapes/architecture tag to stock `vae-BF16.gguf`). Added as `alternate_components.scragvae` in manifest. End-to-end run: RTF 3.70 with turbo DiT (same as stock — same architecture, only weights differ). To use: symlink `scragvae-BF16.gguf` → `vae-BF16.gguf` in the variant dir

³ **Gap F (still open):** Qwen3-TTS-Tokenizer-25Hz is described in the Qwen3 technical report but **no public weights exist** on HuggingFace (only 12 Hz variants published). Paper-only release

**Format cheat-sheet:**

| Format | Where it lives | Loaded by |
|---|---|---|
| GGUF (f16 / Q8_0 / Q4_K_M) | audiocore `moss_tts`, `qwen3_tts`, `ace_step` families; also the openmoss HTTP pool that ray's `forge_openmoss.py` talks to | libllama + audiocore's own GGML graphs |
| HF safetensors (bf16 / fp16) | Every Python backend (`ray/vendor/*`, `moss-sfx-v2/`) | `transformers` / `diffusers` `from_pretrained` |
| PyTorch `.bin` / `.pt` / `.ckpt` | Legacy fallback only — acestep's `model_downloader.py` still probes for `pytorch_model.bin` as a secondary | transformers / `torch.load` |

The workspace is **binary-bifurcated on format**: every audiocore model is GGUF, every Python-only model is HuggingFace safetensors. The only place they meet is `forge_openmoss.py`, which is Python-on-the-wire but GGUF-on-the-disk.

---

## audiocore (C++/GGML) — `my-stuff/audiocore/`

Pure C++ engine. Weights are GGUF, the Qwen3 backbones are hosted by
libllama (so you get llama.cpp's quantizations and CUDA/CPU/Vulkan backends
for free), and the codec/embedding/LM-head graphs are audiocore's own GGML.

Three families registered (`include/audiocore/models/`):

### `moss_tts` — MossTTSDelay architecture
Implements the **delay-pattern AR** architecture. Qwen3-8B backbone + 32
RVQ audio codebooks + 1.6B transformer codec. Modes wired in `session.cpp`:
`tts`, `voice_clone`, `voice_design`, `streaming`, `sfx`. (`dialogue` mode
is `not_impl` — TTSD runtime not yet built.)

| Variant | Quant | Status | Benchmark |
|---------|-------|--------|-----------|
| MOSS-TTS v1.5 (flagship) | Q8_0 | ✅ wired | RTF 0.322 (3.1× RT) |
| MOSS-VoiceGenerator | Q8_0 | ✅ wired (voice_design mode) | RTF 3.53 |
| MOSS-SoundEffect v1 | f16 | ✅ wired (sfx mode) | RTF 0.83 |
| MOSS-TTSD v0.7 | f16 | ⚠️ GGUF converts, runtime blocked | — |
| MOSS-TTS-Local-Transformer v1.5 | — | ❌ not ported (MossTTSLocal) | — |
| MOSS-TTS-Realtime | — | ❌ not ported (MossTTSRealtime) | — |
| MOSS-TTS-Nano | — | ❌ not ported (MossTTSNano) | — |

Codec: MOSS-Audio-Tokenizer v1 (24 kHz mono) — integrated in `codec.cpp`.

### `qwen3_tts` — Qwen3-TTS family
Talker (Qwen3 ForCausalLM) + MTP predictor + WavTokenizer-class codec.
Codec decode has the O(T²)→O(T) conv-transpose-1d fix in `codec.cpp`.

| Variant | Status | Benchmark |
|---------|--------|-----------|
| 0.6B Base | ✅ partial (codec wired) | RTF 0.081 (12.4× RT) |
| 0.6B CustomVoice | ✅ partial | RTF 0.081 |
| 1.7B Base | ✅ partial | RTF 0.127 (7.9× RT) |
| 1.7B CustomVoice | ✅ partial | RTF 0.126 |
| 1.7B VoiceDesign | ✅ partial | RTF 0.127 |

Modes: `tts_batch`, `tts_instruct`, `voice_design` partial;
`voice_clone` **wired** (Stage 17b + Gap K closed 2026-07-03 — ECAPA-TDNN
speaker encoder loads from standalone `qwen3tts-speaker-encoder.gguf`
converted from `marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B`. All three
clone paths work end-to-end: cached `speaker_embedding` vector (RTF 0.126),
`reference_audio` → ECAPA + codec encoder ICL (RTF 0.127), and
`tts_batch` + reference_audio. Gap L remains: the text-embedding module
isn't converted, so pass `reference_text=""` for the ICL path.);
`streaming` not_impl.

**Voice-embedding workflow** (2026-07-03): `tools/qwen_voice.cpp` exposes
the marksverdhei-style voice-as-vector flow on disk. Encode a reference
WAV once to a 1024-/2048-d `.voice` file, then reuse it across TTS calls
without re-running ECAPA. Vector-math subcommands (model-free, instant):
`info`, `average` (combine multiple takes of one speaker for a stabler
identity), `mix` (interpolate between two voices — gender/timbre slider),
`add`, `scale`, `direction` (isolate an attribute axis like calm→angry),
`shift` (apply an attribute axis to any voice). Model-backed subcommands:
`encode` (WAV → `.voice`), `apply` (`.voice` + emotional `instruct` → WAV).
Round-trip regression in `tests/test_qwen3_voice_e2e.cpp` verifies the
full flow: encode → save → reload → `synthesize_with_embedding` + instruct
→ audible WAV (4.78s output, RMS=0.040, 1.7B Base, dim=2048, RTF 0.33).
File format: magic `QWEN3VOICE` + u32 version + u32 dim + u32 flags +
u32 reserved + `dim`×float32 values (32-byte header, 8 KB for 2048-d).

Codec: Qwen3-TTS-Tokenizer-12Hz — integrated in C++.

### `ace_step` — ACE-Step text-to-music
Qwen3 text encoder → 5Hz music-code LM → DiT flow-matching → audio VAE →
48 kHz stereo. All 5 DiT variants benchmarked (`tests/test_acestep_e2e.cpp`).

| DiT variant | LM | Steps | Benchmark |
|-------------|-----|-------|-----------|
| 1.5 Turbo | 1.7B | 8 | RTF 3.90 |
| 1.5 SFT | 1.7B | 50 | RTF 10.81 |
| XL Turbo | 4B | 8 | RTF 5.72 |
| XL SFT | 4B | 50 | RTF 23.06 |
| XL Base | 4B | 8 | RTF 5.68 |

Mode: `text_to_music` wired. (`cover`, `repaint`, `stem`, `lego`,
`completion` planned/blocked — see manifest.json.)

Required `add_compile_definitions(GGML_MAX_NAME=128)` in CMakeLists.txt —
ACE-Step GGUF tensor names run up to 73 chars.

### What audiocore does NOT have in C++
- **ASR** — no speech-to-text family
- **MOSS-TTSD dialogue** runtime (GGUF converts, codec/prompt blocker)
- **MOSS-TTS-Local / Realtime / Nano** — not ported
- **MOSS-SoundEffect v2** — Python only (Wan-audio DiT)
- **MOSS-Audio-Tokenizer v2** — Python only (48 kHz stereo)
- **XY_Tokenizer** (TTSD codec) — not ported

Python bindings (`src/python/audiocore_bindings.cpp`, pybind11) wrap the C++
engine — calling Python here does NOT add any new model, it just exposes the
C++ sessions to Python callers.

---

## Python stack — `../moss-sfx-v2/` (minimal, isolated)

Single-purpose: run MOSS-SoundEffect v2.0 end-to-end. No other models.

- **MOSS-SoundEffect v2** — Wan-audio DiT (1.3B) + DAC VAE + Qwen3 text
  encoder, bf16, 50-step flow-matching.
  - 3 s output: RTF 2.04 (peak VRAM 14.9 GB)
  - 10 s output: RTF 0.61 (peak VRAM 14.9 GB)
- Uses MOSS-Audio-Tokenizer v2 internally for 48 kHz stereo output.

Stack: Python 3.12, torch 2.9.0+cu128, diffusers, `audiotools` stub
(replaces descript-audiotools to dodge the protobuf≥6 conflict).

Entry: `benchmark.py --seconds N --steps 50`.

---

## Python stack — `../../ray/` (full audio service)

Three layers:

1. **HTTP forge services** (`ray/services/audio/forge_*.py`) — thin async
   adapters. Each one routes JSON-in/audio-out to a backend (either a
   PyTorch model in `vendor/`, or an external HTTP pool).
2. **Vendored PyTorch models** (`ray/vendor/`) — full upstream Python
   implementations.
3. **External HTTP pools** — e.g. `forge_openmoss.py` actually talks to the
   pwilkin/openmoss C++ binary over HTTP, so that route is C++ under the hood
   even though the forge layer is Python.

### Forge services (`ray/services/audio/`)

| Forge file | Service classes | Backend |
|------------|-----------------|---------|
| `forge_qwentts.py` | `Qwen3TTSBaseForgeService`, `Qwen3TTSBase06BForgeService`, `Qwen3TTSCustomVoiceForgeService`, `Qwen3TTSVoiceDesignForgeService` | HF Transformers (PyTorch) |
| `forge_openmoss.py` | `MossTTSForgeService`, `MossTTSDForgeService`, `MossVoiceGeneratorForgeService` | **HTTP → openmoss.cpp server** (so really C++/GGML) |
| `forge_moss_sfx.py` | `MossSoundEffectForgeService` | `vendor/moss-tts-v2/moss_soundeffect_v2/` (PyTorch) |
| `forge_acestep.py` | `ACEStepForgeService` | `vendor/acestep/` (PyTorch) |
| `forge_asr.py` | `ASRForgeService` | Whisper-class (PyTorch) |

### Vendored PyTorch implementations (`ray/vendor/`)

**`moss-tts-v2/`** — the comprehensive upstream MOSS project. **Covers all
four MOSS architectures** (this is the only place several of them run):

| Subdir | Architecture | Models |
|--------|--------------|--------|
| `moss_tts_delay/` | MossTTSDelay | TTS v1.0/v1.5, VoiceGenerator, SFX v1, **TTSD** |
| `moss_tts_local/` | MossTTSLocal (Global + Local Transformer) | TTS-Local-Transformer v1.0/v1.5 ("new codex") |
| `moss_tts_realtime/` | MossTTSRealtime | Real-time streaming agent (TTFB 180 ms) |
| `moss_soundeffect_v2/` | Wan-audio DiT | SoundEffect v2.0 |
| `moss_audio_tokenizer/` | codec | Audio-Tokenizer v1 + v2 |

**`moss-tts-delay/`** — older standalone vendored copy of just the delay
model (superset by `moss-tts-v2/moss_tts_delay/`).

**`acestep/`** — full PyTorch ACE-Step. Supports all 5 DiT variants via
`models/{base,sft,turbo,xl_base,xl_sft,xl_turbo}/`. Pipeline in
`acestep_v15_pipeline.py`.

### What the Python stack has that C++ doesn't

- **ASR** (speech-to-text) — Whisper-class
- **MOSS-TTSD runtime** — runs in PyTorch via `moss_tts_delay/` modeling
- **MOSS-TTS-Local-Transformer** — runs in PyTorch via `moss_tts_local/`
- **MOSS-TTS-Realtime** — runs in PyTorch via `moss_tts_realtime/`
- **MOSS-SoundEffect v2** — runs in PyTorch (or use the isolated `moss-sfx-v2/`)
- **MOSS-Audio-Tokenizer v2** — 48 kHz stereo codec (PyTorch)
- **XY_Tokenizer** — TTSD codec (PyTorch)

### What C++ has that the Python stack doesn't

- **GGUF quantization** (Q4_K_M, Q8_0) — Python runs bf16/f16 only
- **3.1× real-time MOSS-TTS** — Python is slower (no nanovllm/CUDA-graph
  integration yet — see `ray/spec/moss/OPTIMIZATIONS.md` for the path)
- **CPU/Vulkan backends** — Python needs CUDA
- **12.4× real-time Qwen3-TTS 0.6B** — the O(T²)→O(T) codec fix
- **Single-binary deployment** — no torch/diffusers dependency tree

---

## Why two stacks?

audiocore exists to get C++ inference speed (libllama CUDA graphs, GGUF
quantization, no Python GIL) for the models that matter most — single-speaker
TTS, voice clone, SFX v1, music.

The Python stack exists because (a) it's the upstream reference
implementation for every model, (b) several architectures (Local, Realtime,
SFX v2) aren't ported yet, and (c) ASR isn't in scope for audiocore.

The two stacks converge over HTTP: `forge_openmoss.py` already proves the
pattern (Python adapter → C++ server). As audiocore ports more families,
the matching forge service flips from PyTorch to HTTP-pool without the
client API changing.
