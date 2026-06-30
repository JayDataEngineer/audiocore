# audiocore

**Unified C++ audio inference server for GGUF models** — a single binary that
serves TTS and music generation through an OpenAI-compatible HTTP API, backed by
ggml (CUDA/CPU/Vulkan/Metal) and libllama. GGUF is the only weight format; C++17
is the only implementation language.

| Model | Family | Capabilities | Status |
|---|---|---|---|
| **[MOSS-TTS](https://github.com/pwilkin/openmoss)** (8B) | `moss_tts` | TTS, voice cloning, streaming; (sfx/dialogue/voice_design/realtime fail fast — need dedicated checkpoints we don't ship) | ✅ Flagship backbone fully wired end-to-end with the upstream prompt template and the openmoss codec (`pwilkin/openmoss`, Apache-2.0). `tts` + `voice_clone` + `streaming` verified against real weights. Modes that need separate MOSS checkpoints (TTSD, VoiceGenerator, SoundEffect, Realtime) fail fast with an error naming the missing model. |
| **[Qwen3-TTS](https://huggingface.co/QwenLM/Qwen3-TTS)** (Talker + MTP Predictor) | `qwen3_tts` | Multilingual TTS, style instruct, voice cloning, streaming | ✅ Talker + Code Predictor run through the unified `qwen3::Runner`; 16-codebook codec and ECAPA-TDNN speaker encoder are ggml ports of `CrispStrobe/CrispASR` (MIT). TTS (incl. style-instruct), Voice Clone, and Streaming are wired end-to-end. Voice Design fails fast — it needs the dedicated `Qwen3-TTS-12Hz-1.7B-VoiceDesign` checkpoint, which is not shipped. |
| **[ACE-Step](https://huggingface.co/Serveurperso/ACE-Step-1.5-GGUF)** (DiT + LM) | `ace_step` | Music generation (text-conditional, lyrics), repaint, completion, cover | ✅ `text_to_music` + `repaint` + `completion` + `cover` verified end-to-end across all six DiT checkpoints and both LM sizes (`test_acestep_e2e`, RMS ≈ 0.0229). `stem` and `lego` fail fast — they need a separate Demucs-style stem assembler that is not shipped. |

Run `audiocore_cli --list-supported` for the live mode matrix.

---

## Why audiocore?

Two reference projects solve adjacent halves of the same problem:

| Reference | What it gives us | What it lacks |
|---|---|---|
| [`0xShug0/audio.cpp`](https://github.com/0xShug0/audio.cpp) (Apache-2.0) | Clean framework layout, OpenAI-compatible server, per-family code organization, modular CMake flags | Safetensors only — no GGUF. MOSS/ACE-Step are aspirational, not implemented. |
| [`leejet/stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp) (MIT) | A well-isolated GGUF reader with a `TensorStorage` abstraction that decouples weight formats from model code | Image-only, not audio |

**audiocore** borrows audio.cpp's architecture (clean-room — not a fork) and
vendors sd.cpp's GGUF reader directly (MIT, attributed). The result: a single
C++ binary that serves any collected audio GGUF.

---

## Features

- **OpenAI-compatible HTTP API** — `/v1/audio/speech` (TTS), `/v1/audio/music`, `/v1/models`, `/health`
- **One weight format** — GGUF only. The `WeightLoader` interface is the single seam between file format and model code.
- **One Qwen3 backbone** — MOSS (8B), ACE-Step (1.7B LM + 0.6B TE) and Qwen3-TTS (Talker + Code Predictor) all run through the same `qwen3::Runner` via llama.cpp. No duplicated transformer code.
- **One sampler** — every family (and the qwen3 MTP predictor) samples through `audiocore::sampler::sample_token` in `src/framework/sampling/`.
- **One TtsRequest** — every TTS family speaks the unified `audiocore::TtsRequest` declared in `include/audiocore/framework/runtime/tasks.h`; the server's `/v1/audio/speech` handler has one code path, not one per family.
- **modular backends** — ggml CUDA, CPU, Metal, Vulkan; selectable per model slot
- **multi-model serving** — configure several models in one `server.json`; each gets its own session
- **Delay-pattern sampling** — full port of MOSS-TTS's 32-RVQ delay-pattern state machine with top-k, top-p, repetition penalty
- **Chat template support** — applies model-native chat templates via llama.cpp

---

## Architecture (TL;DR)

```
server.json → main.cpp → FamilyRegistry → Session { WeightLoader + Backend }
                                    │
                                    ▼
                              GGUF reader
                                    │
                                    ▼
                             TensorStorage
                         (format-neutral descriptor)
                                    │
                                    ▼
                           Backend::execute()
                     ggml (CUDA/CPU/Vulkan/Metal)
                     + libllama (Qwen3 inference)
```

Three lean abstractions keep weight formats, transformer inference and
sampling decoupled:

1. **`TensorStorage`** — a format-neutral struct (`{name, type, shape, offset}`).
   Every weight reader produces `vector<TensorStorage>`. Family code never calls
   `gguf_*` directly; the only `gguf_*` calls in the tree live under
   `src/framework/io/`.

2. **`qwen3::Runner`** — the ONE Qwen3 transformer path in audiocore. Loaded by
   every family that needs a Qwen3-style backbone (MOSS, ACE-Step LM/TE) or a
   Qwen3-TTS component (Talker, Code Predictor) via `load_extras(ExtraKind)`.

3. **`audiocore::sampler`** — the ONE token sampler. Top-k / top-p / temperature
   / repetition-penalty / argmax in one place, used by every family and the
   MTP predictor.

Detailed architecture lives in `CLAUDE.md` (in this repo) and the source tree itself.

---

## Quick start

### Prerequisites

- CMake ≥ 3.18
- C++17 compiler (GCC 11+, Clang 14+, MSVC 2022+)
- Git submodules: `git submodule update --init --recursive`
- [Optional] CUDA toolkit ≥ 12 (for `ENGINE_ENABLE_CUDA`)
- [Recommended] [`gitleaks`](https://github.com/gitleaks/gitleaks) ≥ 8.21 — enables the pre-push secret scan (see [Secret handling](#secret-handling))

### Secret handling

Live tokens (HuggingFace, etc.) live in `config/secrets.env`, which is
**gitignored**. A template is shipped at `config/secrets.env.example`:

```bash
cp config/secrets.env.example config/secrets.env
# edit in your real HF_TOKEN
```

Every push is scanned for secrets two ways:

1. **Locally** — install the pre-push hook once after cloning:
   ```bash
   ./scripts/install-git-hooks.sh
   ```
   The hook (`scripts/hooks/pre-push`) runs `gitleaks detect` over the
   commits being pushed and fails closed. Bypass with `git push --no-verify`
   (not recommended). Set `AUDIOCORE_REQUIRE_GITLEAKS=1` to fail-closed
   even when gitleaks isn't installed.

2. **On the remote** — `.github/workflows/gitleaks.yml` runs the same scan
   on every push and pull request, so leaks slip through only if both the
   hook is bypassed *and* the workflow is disabled.

The gitleaks config (`.gitleaks.toml`) extends the default rule set with
explicit HuggingFace rules and allowlists `third_party/`, build artifacts,
and `*.env.example` templates.

### Build

```bash
cmake -S . -B build                 \
  -DENGINE_ENABLE_CUDA=ON           \   # ggml CUDA backend (OFF for CPU-only)
  -DENGINE_BUILD_TESTS=ON           \   # unit + e2e tests
  -DCMAKE_BUILD_TYPE=RelWithDebInfo     # default; Release for production

cmake --build build --parallel

# Or specific targets:
cmake --build build --parallel --target audiocore_server
cmake --build build --parallel --target audiocore_cli
cmake --build build --parallel --target convert_acestep    # ACE-Step tensor renamer
cmake --build build --parallel --target convert_qwen3tts   # Qwen3-TTS safetensors → GGUF
```

#### Build flags

| Flag | Default | Description |
|---|---|---|
| `ENGINE_ENABLE_CUDA` | OFF | ggml CUDA backend |
| `ENGINE_ENABLE_VULKAN` | OFF | ggml Vulkan backend |
| `ENGINE_ENABLE_METAL` | OFF | ggml Metal backend (Apple) |
| `ENGINE_ENABLE_CUDA_GRAPHS` | ON | CUDA graphs (CUDA only) |
| `ENGINE_ENABLE_OPENMP` | ON | host-side parallel work |
| `ENGINE_ENABLE_LLAMAFILE` | ON | llamafile SGEMM (CPU) |
| `ENGINE_BUILD_CLI` | ON | `audiocore_cli` |
| `ENGINE_BUILD_SERVER` | ON | `audiocore_server` |
| `ENGINE_BUILD_TESTS` | OFF | unit + e2e tests |

### Fetch models

audiocore separates **fetch** from **serve** — same pattern as llama.cpp
(`huggingface-cli download` separate from `llama-server`). The bundled
`scripts/fetch_models.sh` is the canonical puller: pure bash + curl, no
Python, no huggingface-cli dependency. It reads `models/manifest.json`
for repo / revision / sha256, downloads via the HF Hub API, optionally
runs `convert_acestep` / `convert_qwen3tts` post-process, and verifies
SHA256 when recorded.

```bash
# Fetch every family + variant the manifest knows about:
./scripts/fetch_models.sh

# Fetch just one family:
./scripts/fetch_models.sh moss_tts

# Fetch one specific variant:
./scripts/fetch_models.sh moss_tts moss-tts-q4-k-m

# Dry run (show what would be fetched):
./scripts/fetch_models.sh --dry-run

# List the mode matrix the manifest describes:
./scripts/fetch_models.sh --list
```

Environment knobs (also honored inside Docker — see "Docker" below):

| Env | Default | Purpose |
|---|---|---|
| `AUDIOCORE_MODELS_DIR` | `./weights/` | **Destination folder for downloaded GGUFs.** Override to target any path. |
| `AUDIOCORE_BUILD_DIR` | `./build/bin` | Where `convert_*` binaries live |
| `HF_TOKEN` | unset | Auth for gated repos (passed as `Authorization: Bearer`) |

After fetching, point the server at the downloaded weights with
`--model`, `--model-dir`, or a `models: [...]` config block.

```bash
./scripts/fetch_models.sh moss_tts
build/bin/audiocore_server --config examples/server.json \
    --model "$AUDIOCORE_MODELS_DIR/moss-tts/MOSS_TTS_Q4_K_M.gguf"
```

### Run the server

**Recommended — `--model` (llama.cpp-style, single model):**

```bash
# Family sniffed from the GGUF (or from the directory name):
build/bin/audiocore_server --config examples/server.json --model /path/to/moss-tts-q8_0.gguf
build/bin/audiocore_server --config examples/server.json --model /path/to/qwen3-tts/

# Override the family if the sniffer can't decide:
build/bin/audiocore_server --config examples/server.json --model /path/to/file.gguf --family moss_tts

# Rename the loaded model for OpenAI client compatibility (clients that
# hardcode "tts-1", "whisper-1", etc.). /v1/models returns the alias,
# /v1/audio/speech accepts it as the model name.
build/bin/audiocore_server --config examples/server.json \
    --model /path/to/moss-tts-q8_0.gguf --alias tts-1
```

One model per process. To swap, stop the server and restart with a different
`--model`. `--alias` only works in single-model mode (errors out if the
config has 0 or >1 models). The `examples/server.json` config supplies
`host` / `port` / `device` / `threads` — only one entry is needed even
though `models` is an array (for backwards compat). When `--model` is set,
it overrides any `models: [...]` block in the JSON.

**Backwards-compatible — `models: [...]` or `--model-dir`:**

```bash
build/bin/audiocore_server --config examples/server.json
build/bin/audiocore_server --config examples/server.json --model-dir /models
```

`examples/server.json` (`models` array form, still supported):

```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "device": 0,
  "threads": 4,
  "models": [
    {
      "id": "moss-tts",
      "family": "moss_tts",
      "path": "/models/moss-tts/moss-tts-q8_0.gguf",
      "backend": "ggml_cuda"
    },
    {
      "id": "qwen3-tts",
      "family": "qwen3_tts",
      "path": "/models/qwen3-tts/",
      "backend": "ggml_cuda",
      "extras": {
        "talker_path":    "qwen3_tts_talker.q5_k.gguf",
        "predictor_path": "qwen3_tts_predictor.q8_0.gguf",
        "n_gpu_layers":   "99"
      }
    },
    {
      "id": "ace-step",
      "family": "ace_step",
      "path": "/models/ace-step/",
      "backend": "ggml_cuda"
    }
  ]
}
```

### Docker

The repo ships two multi-stage Dockerfiles:

| File | Base | Size | Use |
|---|---|---|---|
| `Dockerfile` | `nvidia/cuda:13.1-devel-ubuntu24.04` (build) → `…-runtime-…` | ~1.1 GB | Production. GPU inference via ggml CUDA backend. |
| `Dockerfile.cpu` | `ubuntu:24.04` | ~350 MB | Dev, CI, or CPU-only hosts. |

Both expect the git submodules to be populated at clone time:

```bash
git clone https://github.com/JayDataEngineer/audiocore.git
cd audiocore
git submodule update --init --recursive   # llama.cpp + ggml
```

Then build and run:

```bash
# GPU image (needs Docker with --gpus all + a matching NVIDIA driver)
docker build -t audiocore .

# CPU image
docker build -f Dockerfile.cpu -t audiocore:cpu .

# Recommended — llama.cpp-style single-model. Mount one model and point
# --model at it. To swap models, stop the container and restart with a
# different mount + --model.
docker run --gpus all -p 8080:8080 \
    -v "$PWD/weights/moss-tts-q8_0.gguf:/model.gguf:ro" \
    audiocore --model /model.gguf

# Multi-file family (e.g. qwen3-tts needs talker + predictor + codec) —
# mount the whole directory and pass it to --model:
docker run --gpus all -p 8080:8080 \
    -v "$PWD/weights/qwen3-tts:/model:ro" \
    audiocore --model /model

# OpenAI client compat — rename the loaded model so a client that hardcodes
# "tts-1" (or any other name) hits it without code changes. Pairs with --model.
docker run --gpus all -p 8080:8080 \
    -v "$PWD/weights/moss-tts-q8_0.gguf:/model.gguf:ro" \
    audiocore --model /model.gguf --alias tts-1

# Auto-discover everything under /models (one subdir per family):
docker run --gpus all -p 8080:8080 \
    -v "$PWD/weights:/models:ro" \
    audiocore

# No models mounted — server boots idle (/health → 200, /v1/models → []).
# Useful for image smoke-tests; /v1/audio/* returns 404 until a model loads.
docker run --rm -p 8080:8080 audiocore:cpu
```

Once `curl http://localhost:8080/health` returns `{"status":"ok"}`, the API is
live — see [API](#api) below. The image exposes these mount points and env
vars:

| Mount | Purpose |
|---|---|
| `/models` | GGUF weights (auto-discovered at boot; mount **without** `:ro` if you use `AUDIOCORE_PREPULL=1`) |
| `/etc/audiocore/server.json` | Override the default config (optional) |

| Env | Default | Purpose |
|---|---|---|
| `AUDIOCORE_CONFIG` | `/etc/audiocore/server.json` | Path to the config file |
| `AUDIOCORE_PORT` | `8080` | Used by the in-image `HEALTHCHECK` |
| `LD_LIBRARY_PATH` | `/opt/audiocore/lib` | Where the ggml/llama `.so`s live |
| `AUDIOCORE_PREPULL` | `0` | Set `1` to run `fetch_models.sh` before the server boots (see below) |
| `AUDIOCORE_MODELS_DIR` | `/models` | Destination folder for the prepull — **target any path** |
| `AUDIOCORE_PREPULL_FAMILY` | unset | Restrict prepull to one family (e.g. `moss_tts`) |
| `AUDIOCORE_PREPULL_VARIANT` | unset | Restrict prepull to one variant (e.g. `moss-tts-q4-k-m`) |
| `HF_TOKEN` | unset | Auth for gated repos; passed through to `fetch_models.sh` |

`AUDIOCORE_ALLOW_EMPTY` from older revisions is tolerated but no longer
required — the shipped default config now auto-discovers `/models` and boots
idle if nothing is mounted, so it is always safe to seed.

#### Boot-time model pull (`AUDIOCORE_PREPULL=1`)

The image ships `scripts/fetch_models.sh` and `models/manifest.json`. Set
`AUDIOCORE_PREPULL=1` and the entrypoint runs the fetcher before starting
the server, populating `AUDIOCORE_MODELS_DIR` (default `/models`) and then
letting auto-discovery pick up the results.

```bash
# Pull every family the manifest knows about, then serve:
docker run --gpus all -p 8080:8080 \
    -e AUDIOCORE_PREPULL=1 \
    -e HF_TOKEN=hf_xxx \
    -v audiocore-weights:/models \
    audiocore

# Pull just one family into a specific host folder:
docker run --gpus all -p 8080:8080 \
    -e AUDIOCORE_PREPULL=1 \
    -e AUDIOCORE_PREPULL_FAMILY=moss_tts \
    -e AUDIOCORE_MODELS_DIR=/models \
    -e HF_TOKEN=hf_xxx \
    -v /data/audiocore/weights:/models \
    audiocore

# Bypass the prepull and serve whatever is already on disk (the primary
# workflow once weights are present):
docker run --gpus all -p 8080:8080 \
    -v /data/audiocore/weights:/models:ro \
    audiocore
```

The prepull runs as the `audiocore` user (UID 10000), so `/models` must be
writable by that UID — mount without `:ro`. If `fetch_models.sh` exits
non-zero, the entrypoint refuses to start the server.

#### Model directory layout

The default config sets `"model_dir": "/models"`. At boot the server walks
that directory one level deep and registers one model per subdirectory whose
name matches a registered family (after `kebab-case` → `snake_case`
normalization). Each family's loader picks its own files within the
subdirectory — you don't have to enumerate them.

```
/models/
  moss-tts/                     → id="moss-tts",        family="moss_tts"
    moss-tts-q8_0.gguf            (moss_tts loader picks its .gguf itself)
  qwen3-tts/                    → id="qwen3-tts",       family="qwen3_tts"
    talker.q5_k.gguf              (loader resolves siblings by name)
    predictor.q8_0.gguf
    tokenizer-f16.gguf
  ace-step/                     → id="ace-step",        family="ace_step"
    acestep-v15-turbo-*.gguf      (loader finds its 4 GGUFs by pattern)
    5Hz-lm-1.7B-*.gguf
    Qwen3-Embedding-*.gguf
    vae-*.gguf
```

Subdirectories whose names don't match a registered family are skipped with
a warning, so a stray `README.md` or unrelated folder won't break boot.
Discovery only fires when the config's `models` array is empty — an explicit
`models: [...]` block always takes precedence.

Outside Docker, pass `--model-dir /path` on the server CLI (or set
`"model_dir": "/path"` in the JSON) to point at a different root. The flag
overrides the JSON value when both are set.

#### Build knobs

```bash
docker build \
  --build-arg CUDA_VERSION=12.4.1 \
  --build-arg UBUNTU=22.04 \
  --build-arg BUILD_TYPE=RelWithDebInfo \
  --build-arg BUILD_JOBS=8 \
  -t audiocore:cuda-12.4 .
```

`CUDA_VERSION` must match a tag on
[`nvidia/cuda`](https://hub.docker.com/r/nvidia/cuda/tags) and the runtime
must be backed by a driver that supports it (CUDA 13.x → driver ≥ 580, CUDA
12.x → driver ≥ 545). The container's major CUDA version must match the host
driver — the ggml CUDA backend links against `libcudart.so.<major>`.

### Test

```bash
ctest --test-dir build
```

Runs: GGUF reader round-trips, family registry, audio-head projection parity,
unified sampler behavior, server HTTP e2e, ACE-Step converter e2e. The MOSS
and Qwen3-TTS full load+run tests (`test_moss_e2e`, `test_qwen3tts_e2e`)
require weights mounted at their configured paths and are not registered as
ctest entries.

---

## Models manifest & download

`models/manifest.json` is the canonical record of every family/variant with
HuggingFace repo, revision, file list, license, supported modes, and
`min_vram_gb`. Two consumers:

```bash
# Print the mode matrix from the manifest
scripts/fetch_models.sh --list

# Download one variant + run any required converters
scripts/fetch_models.sh ace_step ace-step-1.5-turbo

# Self-describe from the binary (same matrix)
audiocore_cli --list-supported
```

Environment overrides: `AUDIOCORE_MODELS_DIR` (default `./weights/`),
`AUDIOCORE_BUILD_DIR` (default `./build/bin`), `HF_TOKEN` (for gated repos).

---

## API

All endpoints serve from the same binary.

### `POST /v1/audio/speech` (TTS)

**Models**: `moss_tts`, `qwen3_tts`. The server parses the JSON body into the
unified `audiocore::TtsRequest` once and hands it to whichever family the
`model` id resolves to. Fields the family doesn't use are ignored.

```json
{
  "model": "moss-tts",
  "input": "Hello, this is a test of the MOSS-TTS voice generation system.",
  "voice": "default",
  "language": "zh",
  "seed": 42,
  "temperature": 0.7,
  "top_p": 0.9
}
```

Qwen3-TTS-specific fields (`instruct`, `speaker`, `speed`, `reference_audio`)
are accepted on the same endpoint and read by `qwen3_tts::Qwen3TtsSession`.
Both `max_tokens` and `max_new_tokens` are accepted as aliases.

Returns `audio/wav` (24 kHz mono, 16-bit PCM).

### `POST /v1/audio/speech/stream` (TTS, chunked)

Same JSON body and same dispatch as `/v1/audio/speech`, but the WAV is
emitted via chunked transfer encoding in ~64 KiB chunks. Transport-level
scaffold only — the family still renders the full PCM before the first
byte ships. True incremental streaming (frames as the autoregressive
loop produces them) is open.

### `POST /v1/audio/music` (Music generation)

**Model**: `ace_step`

```json
{
  "model": "ace-step",
  "caption": "A gentle piano melody with ambient pads",
  "lyrics": "",
  "duration": 10.0,
  "seed": 42,
  "guidance_scale": 7.5,
  "steps": 50,
  "mode": "text_to_music"
}
```

`mode` defaults to `text_to_music`. The five other advertised modes
(`cover`, `repaint`, `stem`, `lego`, `completion`) fail fast with a
pointer at the error string instead of silently running text-to-music.

Returns `audio/wav` (48 kHz stereo, 16-bit PCM).

### `GET /v1/models`

Lists configured model slots with family and load status.

### `GET /health`

Returns `{"status": "ok"}`.

---

## Supported models

### MOSS-TTS (`moss_tts`)

| Property | Detail |
|---|---|
| **Source** | [OpenMOSS-Team/MOSS-TTS](https://huggingface.co/OpenMOSS-Team/MOSS-TTS), [pwilkin/openmoss](https://github.com/pwilkin/openmoss) |
| **Backbone** | Qwen3-8B (GGUF, via the unified `qwen3::Runner` over llama.cpp) |
| **Audio codec** | 32-RVQ delay-pattern sampling. Codec-token → PCM decoder is a ggml port of `openmoss/src/codec.cpp` (Apache-2.0) at `src/models/moss_tts/codec.cpp`. Binds automatically when the GGUF carries `moss.codec.*` tensors (e.g. `smcleod/MOSS-TTS-v1.5-GGUF` sidecar). The codec encoder is also ported and drives `voice_clone` (real WAV → codes → splice). Missing codec tensors is a hard error, not a silence fallback. |
| **Sampling** | Delay-pattern autoregressive: top-k, top-p, temperature, repetition penalty (via `audiocore::sampler`) |
| **Output** | 24 kHz mono PCM |
| **Weight formats** | Single GGUF (community), sidecar pair (`X.gguf` + `X.extras.gguf`, smcleod/MOSS-TTS-v1.5-GGUF), or backbone GGUF + `.npy` embedding/lm_head dirs |
| **Status** | ✅ End-to-end verified: `test_moss_e2e` (TTS, RMS 0.160), `test_moss_voice_clone_e2e` (38.8 s output, RMS 0.185), streaming (~1.3 s cold-start then 80 ms/frame). |
| **Reference** | `pwilkin/openmoss` (C++, Apache-2.0) — parity target for byte-identical audio; pre-built sidecar GGUFs at `smcleod/MOSS-TTS-v1.5-GGUF` |

GGUF tensor map: see the family `loader.cpp` and `tools/inspect_gguf`.

### Qwen3-TTS (`qwen3_tts`)

| Property | Detail |
|---|---|
| **Source** | [QwenLM/Qwen3-TTS](https://huggingface.co/QwenLM/Qwen3-TTS) |
| **Backbone** | Talker (qwen3tts arch) + Code Predictor (qwen3tts_cp), both via the unified `qwen3::Runner` with `load_extras(ExtraKind::Talker/Predictor)` |
| **Audio codec** | 16-codebook matrix (1 coarse + 15 MTP fine). Codec-token → PCM decoder is `Qwen3TtsCodecGraphs` in `src/models/qwen3_tts/codec.cpp` (adapted from `CrispStrobe/CrispASR`, MIT). Auto-binds when the codec sidecar GGUF is discovered (`extras["codec_path"]` or `tokenizer-{f16,q8_0}.gguf` next to the talker); pre-built GGUFs at `cstr/qwen3-tts-tokenizer-12hz-GGUF`. Missing codec is a hard error. |
| **Speaker encoder** | ECAPA-TDNN at `src/models/qwen3_tts/speaker_encoder.cpp` (adapted from CrispASR, MIT). Loads from `speaker.*` tensors in the talker GGUF; drives Voice Clone by extracting a 192-d embedding from a reference WAV. Missing encoder tensors is a hard error for Voice Clone. |
| **Sampling** | Top-p / temperature via `audiocore::sampler`; MTP predictor uses the same sampler for fine-codebook draws |
| **Output** | 24 kHz mono PCM |
| **Weight formats** | Two GGUFs (talker + predictor) produced by `tools/convert_qwen3tts` from the official safetensors; pre-built codec GGUF from `cstr/qwen3-tts-tokenizer-12hz-GGUF` (no conversion) |
| **Status** | ✅ End-to-end verified: TTS (incl. style-instruct), Voice Clone, and Streaming (`test_qwen3tts_e2e` against Lunavox/CrispASR-derived talker + 12 Hz codec sidecar). Voice Design fails fast — dedicated checkpoint not shipped. |
| **Reference** | `QwenLM/Qwen3-TTS` (Python) — parity target. Codec + ECAPA parity: `CrispStrobe/CrispASR` (C++, MIT) |

### ACE-Step (`ace_step`)

| Property | Detail |
|---|---|
| **Source** | [Serveurperso/ACE-Step-1.5-GGUF](https://huggingface.co/Serveurperso/ACE-Step-1.5-GGUF), [ServeurpersoCom/acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp) |
| **Pipeline** | Text encoder (Qwen3-Embedding) → 5Hz LM (Qwen3-1.7B/4B) → DiT → VAE |
| **Components** | 4 separate GGUFs (DiT, LM, TE, VAE) |
| **Output** | 48 kHz stereo PCM |
| **Status** | ✅ End-to-end verified: Text-to-Music (`test_acestep_e2e`, RMS ≈ 0.0229) across all six DiT checkpoints (v1.5 turbo/sft, XL base/sft/turbo) and both LM sizes (1.7B, 4B). `repaint`, `completion`, and `cover` modes ported from HOT-Step (correct `[src|mask|xt]` channel layout, stride=P patchify, latent-space Euler, per-step repaint injection + boundary blend). `stem` and `lego` modes fail fast — they need a separate Demucs-style stem assembler that is not shipped. |
| **Reference** | `ServeurpersoCom/acestep.cpp` (single-purpose C++ server) |

> **Note**: ACE-Step ships with HuggingFace-style tensor names. The
> [`convert_acestep`](tools/convert_acestep.cpp) tool rewrites them to
> llama.cpp naming so the unified runner can load them.

---

## Project layout

```
├── CMakeLists.txt                    # Build flags + targets
├── include/audiocore/
│   ├── framework/                    # Public API (headers only)
│   │   ├── core/                     #   Backend, Session
│   │   ├── io/                       #   GGUF reader, TensorStorage, WeightLoader
│   │   ├── sampling/                 #   Unified sampler (Params + sample_token)
│   │   └── runtime/                  #   FamilyRegistry + unified tasks.h
│   ├── models/                       # Per-family public types
│   │   ├── moss_tts/                 #   MossConfig + aliases to unified TtsRequest
│   │   ├── qwen3_tts/                #   Qwen3TtsConfig + aliases to unified TtsRequest
│   │   ├── qwen3/                    #   qwen3::Runner (the ONE Qwen3 path)
│   │   └── ace_step/                 #   MusicRequest, MusicResponse, AceStepConfig
│   └── server/                       #   HTTP server factory
├── src/
│   ├── cli/main.cpp                  # CLI entry point
│   ├── framework/                    # Framework implementation
│   │   ├── io/                       #   gguf_reader, weight_loader (only gguf_* calls)
│   │   ├── sampling/                 #   unified sampler.cpp
│   │   ├── core/                     #   backend, session
│   │   ├── models/qwen3/             #   Qwen3 runner (libllama wrapper + extras)
│   │   └── runtime/                  #   registry
│   ├── models/                       # Per-family code
│   │   ├── moss_tts/                 #   loader, session, projection, delay_state, sampler (shim)
│   │   ├── qwen3_tts/                #   loader, session, codec (Stage 17: Tokenizer-12Hz), speaker_encoder (Stage 17b: ECAPA-TDNN)
│   │   └── ace_step/                 #   loader, session, dit_runner, vae_runner
│   └── server/                       # main.cpp, server.cpp (routes + WAV encoding)
├── tests/                            # Unit + e2e tests (one binary per file)
│   ├── test_framework.h              #   Header-only test macros
│   ├── synthetic_gguf.h/.cpp         #   Hermetic GGUF fixture builder
│   ├── test_gguf_reader.cpp
│   ├── test_registry.cpp
│   ├── test_projection.cpp
│   ├── test_sampler.cpp              #   Unified sampler: argmax/top-k/top-p/temp/rep penalty
│   ├── test_server_e2e.cpp
│   ├── test_convert_acestep.cpp
│   ├── test_moss_e2e.cpp             #   Full load + run (requires weights)
│   └── test_qwen3tts_e2e.cpp         #   Full load + run (requires weights)
├── tools/
│   ├── convert_acestep.cpp           # HF → llama.cpp tensor rename
│   └── convert_qwen3tts.cpp          # Qwen3-TTS safetensors → GGUF
├── examples/
│   └── server.json                   # Reference server config
├── scripts/
│   └── reference_config.yaml         # Upstream Python pipeline config
└── third_party/                      # Vendored + submoduled deps
    ├── llama.cpp/                    #   llama.cpp (MIT, submodule) — vendors ggml/
    ├── httplib/                      #   cpp-httplib (MIT, vendored)
    └── nlohmann/                     #   nlohmann/json (MIT, vendored)
```

---

## Adding a new model family

1. Create `src/models/<family>/`.
2. Write `loader.cpp` — reads weights via `WeightLoader`, builds tensor bindings.
3. Write `session.cpp` — subclass `Session`, override `run_*` methods.
4. Add `AUDIOCORE_REGISTER_FAMILY(<name>, factory)` to `loader.cpp`.
5. Add an entry to `examples/server.json` for testing.
6. (Optional) Add a parity test against a reference C++ implementation.

See [`CLAUDE.md`](CLAUDE.md) for full conventions and design rationale.

---

## Dependencies

| Component | Source | License |
|---|---|---|
| llama.cpp (Qwen3 inference) | `third_party/llama.cpp` (submodule) | MIT |
| ggml (tensor library) | via llama.cpp submodule | MIT |
| GGUF reader | `leejet/stable-diffusion.cpp` (vendored) | MIT |
| cpp-httplib (HTTP) | `third_party/httplib` (vendored) | MIT |
| nlohmann/json | `third_party/nlohmann` (vendored) | MIT |

---

## License

Apache-2.0. Vendored code in `third_party/` retains its original license headers:

- GGUF reader (vendored from `leejet/stable-diffusion.cpp`) — MIT, Copyright (c) 2023 leejet.
- `ggml` — MIT, Copyright (c) 2022–2024 llama.cpp contributors.
- `llama.cpp` — MIT.
- cpp-httplib — MIT.
- nlohmann/json — MIT.

---

## References

- [`0xShug0/audio.cpp`](https://github.com/0xShug0/audio.cpp) — architectural inspiration (clean-room).
- [`leejet/stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp) — GGUF reader source.
- [`ServeurpersoCom/acestep.cpp`](https://github.com/ServeurpersoCom/acestep.cpp) — ACE-Step C++ reference.
- [`pwilkin/openmoss`](https://github.com/pwilkin/openmoss) — MOSS-TTS C++ reference.
- [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) — LLM backbone runtime.
