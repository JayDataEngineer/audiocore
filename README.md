# audiocore

**Unified C++ audio inference server for GGUF models** — a single binary that
serves TTS and music generation through an OpenAI-compatible HTTP API, backed by
ggml (CUDA/CPU/Vulkan/Metal) and libllama. GGUF is the only weight format; C++17
is the only implementation language.

| Model | Family | Capabilities | Status |
|---|---|---|---|
| **[MOSS-TTS](https://github.com/pwilkin/openmoss)** (8B) | `moss_tts` | TTS, dialogue (TTSD-style), voice design (VoiceGenerator-equivalent), sound effects, voice cloning | 🟡 5 of 6 modes parse + run end-to-end; codec → PCM is a silence stub pending a ggml port |
| **[Qwen3-TTS](https://huggingface.co/QwenLM/Qwen3-TTS)** (Talker + MTP Predictor) | `qwen3_tts` | Multilingual TTS, instructable voice design, 9 default speakers (CustomVoice) | 🟡 Talker + predictor wired; variant detection + speaker routing wired; codec → PCM is a silence stub pending a ggml port |
| **[ACE-Step](https://huggingface.co/Serveurperso/ACE-Step-1.5-GGUF)** (DiT + LM) | `ace_step` | Music generation (text-conditional, lyrics) | ✅ Text-to-Music fully wired end-to-end; 5 other modes fail fast with a pointer at GAPS.md |

Run `audiocore_cli --list-supported` for the live mode matrix, or see
[`GAPS.md`](GAPS.md) for the full per-family audit.

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

Detailed architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Quick start

### Prerequisites

- CMake ≥ 3.18
- C++17 compiler (GCC 11+, Clang 14+, MSVC 2022+)
- Git submodules: `git submodule update --init --recursive`
- [Optional] CUDA toolkit ≥ 12 (for `ENGINE_ENABLE_CUDA`)

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

### Run the server

```bash
build/bin/audiocore_server --config examples/server.json
```

Example `server.json` (`examples/server.json`):

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
loop produces them) is open; see `GAPS.md` §1.2 / §2.2.

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
pointer at `GAPS.md` §3.2 instead of silently running text-to-music.

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
| **Audio codec** | 32-RVQ delay-pattern sampling. Codec-token → PCM decoder is a silence stub pending a ggml port of the speech-tokenizer graph; weights will live in the same GGUF as the backbone. |
| **Sampling** | Delay-pattern autoregressive: top-k, top-p, temperature, repetition penalty (via `audiocore::sampler`) |
| **Output** | 24 kHz mono PCM |
| **Weight formats** | Single GGUF (community), or backbone GGUF + `.npy` embedding/lm_head dirs |
| **Status** | 🚧 Generation works; codec → PCM is a silence stub pending a ggml port |
| **Reference** | `pwilkin/openmoss` (C++) — parity target for byte-identical audio |

GGUF tensor map: [`docs/GGUF_FORMAT.md`](docs/GGUF_FORMAT.md).

### Qwen3-TTS (`qwen3_tts`)

| Property | Detail |
|---|---|
| **Source** | [QwenLM/Qwen3-TTS](https://huggingface.co/QwenLM/Qwen3-TTS) |
| **Backbone** | Talker (qwen3tts arch) + Code Predictor (qwen3tts_cp), both via the unified `qwen3::Runner` with `load_extras(ExtraKind::Talker/Predictor)` |
| **Audio codec** | 32-codebook matrix (1 coarse + 31 MTP fine). Codec-token → PCM decoder is a silence stub pending a ggml port. |
| **Sampling** | Top-p / temperature via `audiocore::sampler`; MTP predictor uses the same sampler for fine-codebook draws |
| **Output** | 24 kHz mono PCM |
| **Weight formats** | Two GGUFs (talker + predictor) produced by `tools/convert_qwen3tts` from the official safetensors |
| **Status** | 🚧 Talker + predictor load and run; codec → PCM is a silence stub pending a ggml port |
| **Reference** | `QwenLM/Qwen3-TTS` (Python) — parity target |

### ACE-Step (`ace_step`)

| Property | Detail |
|---|---|
| **Source** | [Serveurperso/ACE-Step-1.5-GGUF](https://huggingface.co/Serveurperso/ACE-Step-1.5-GGUF), [ServeurpersoCom/acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp) |
| **Pipeline** | Text encoder (Qwen3-Embedding) → 5Hz LM (Qwen3-1.7B/4B) → DiT → VAE |
| **Components** | 4 separate GGUFs (DiT, LM, TE, VAE) |
| **Output** | 48 kHz stereo PCM |
| **Status** | 🚧 Weight loading + family registration done. `run_lm()` and `run_dit_and_vae()` are scaffolds returning "not yet wired". |
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
│   │   ├── qwen3_tts/                #   loader, session
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
├── docs/
│   ├── ARCHITECTURE.md               # Two-seam deep-dive
│   └── GGUF_FORMAT.md                # Tensor maps for each family
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
   Document the family in a `README.md` in the same directory.
4. Add `AUDIOCORE_REGISTER_FAMILY(<name>, factory)` to `loader.cpp`.
5. Document the GGUF tensor map in `docs/GGUF_FORMAT.md`.
6. Add an entry to `examples/server.json` for testing.
7. (Optional) Add a parity test against a reference C++ implementation.

See [`AGENTS.md`](AGENTS.md) for full conventions and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
for the design rationale.

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
