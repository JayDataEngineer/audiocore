# audiocore

**Unified C++ audio inference server for GGUF models** — a single binary that
serves TTS and music generation through an OpenAI-compatible HTTP API, backed by
ggml (CUDA/CPU/Vulkan/Metal) with an architecture designed to accept ONNX Runtime
as a peer backend.

| Model | Family | Capabilities | Status |
|---|---|---|---|
| **[MOSS-TTS](https://github.com/pwilkin/openmoss)** (8B) | `moss_tts` | TTS, dialogue-TTS, voice generation, sound effects | ✅ Full pipeline |
| **[ACE-Step](https://huggingface.co/Serveurperso/ACE-Step-1.5-GGUF)** (DiT + LM) | `ace_step` | Music generation (text-conditional, lyrics) | 🚧 Scaffolded (inference paths in progress) |

---

## Why audiocore?

Two reference projects solve adjacent halves of the same problem:

| Reference | What it gives us | What it lacks |
|---|---|---|
| [`0xShug0/audio.cpp`](https://github.com/0xShug0/audio.cpp) (Apache-2.0) | Clean framework layout, OpenAI-compatible server, per-family code organization, modular CMake flags | Safetensors only — no GGUF. MOSS/ACE-Step are aspirational, not implemented. |
| [`leejet/stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp) (MIT) | A well-isolated GGUF reader with a `TensorStorage` abstraction that decouples weight formats from model code | Image-only, not audio |

**audiocore** borrows audio.cpp's architecture (clean-room — not a fork) and
vendors sd.cpp's GGUF reader directly (MIT, attributed). The result: a single
C++ binary that serves any collected audio GGUF, with the seams in place to
swap ggml for ONNX Runtime later.

---

## Features

- **OpenAI-compatible HTTP API** — `/v1/audio/speech` (TTS), `/v1/audio/music`, `/v1/models`, `/health`
- **Two weight-format tiers** — GGUF (primary) with a path to ONNX Runtime (Phase 2)
- **One Qwen3 backbone** — both MOSS (8B) and ACE-Step (1.7B LM + 0.6B TE) use the same `qwen3::Runner` via llama.cpp. No duplicated transformer code.
- **modular backends** — ggml CUDA, CPU, Metal, Vulkan; selectable per model slot
- **multi-model serving** — configure several models in one `server.json`; each gets its own session
- **ONNX Runtime codec decode** — MOSS audio codec decoder runs via ONNX Runtime (GPU-capable; stub builds gracefully without the SDK)
- **Delay-pattern sampling** — full port of MOSS-TTS's 32-RVQ delay-pattern state machine with top-k, top-p, repetition penalty
- **Chat template support** — applies model-native chat templates via llama.cpp

---

## Architecture (TL;DR)

```
server.json → main.cpp → FamilyRegistry → Session { WeightLoader + Backend }
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
             GGUF reader     safetensors (P2)   ONNX (P2)
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
                             TensorStorage
                         (format-neutral descriptor)
                                    │
                                    ▼
                           Backend::execute()
                     ggml (CUDA/CPU/Vulkan/Metal)
                     ONNX Runtime           (Phase 2)
```

Two lean abstractions keep weight formats and execution engines decoupled:

1. **`TensorStorage`** — a format-neutral struct (`{name, type, shape, offset}`).
   Every weight reader produces `vector<TensorStorage>`. Model code never calls
   `gguf_*` directly.

2. **`Backend`** — an execution runtime interface. `Session` owns one and
   submits inference graphs to it. ggml today; ONNX Runtime as a peer later.

Detailed architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Quick start

### Prerequisites

- CMake ≥ 3.18
- C++17 compiler (GCC 11+, Clang 14+, MSVC 2022+)
- Git submodules: `git submodule update --init --recursive`
- [Optional] CUDA toolkit ≥ 12 (for `ENGINE_ENABLE_CUDA`)
- [Optional] ONNX Runtime system install (for MOSS codec GPU decode)

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
| `ENGINE_ENABLE_ONNXRUNTIME` | OFF | ONNX Runtime backend (Phase 2) |
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
server HTTP e2e, ACE-Step converter e2e. The MOSS full load+run test
(`test_moss_e2e`) requires weights mounted at their configured path.

---

## API

All endpoints serve from the same binary.

### `POST /v1/audio/speech` (TTS)

**Model**: `moss_tts`

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

Returns `audio/wav` (24 kHz mono, 16-bit PCM).

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
  "steps": 50
}
```

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
| **Backbone** | Qwen3-8B (GGUF, via llama.cpp) |
| **Audio codec** | 32-RVQ with 1.6B decoder (ONNX Runtime) |
| **Sampling** | Delay-pattern autoregressive: top-k, top-p, temperature, repetition penalty |
| **Output** | 24 kHz mono PCM |
| **Weight formats** | Single GGUF (community), or backbone GGUF + `.npy` embedding/lm_head dirs |
| **Status** | ✅ Full pipeline — TTS, TTSD, SoundEffect. Tested end-to-end. |
| **Reference** | `pwilkin/openmoss` (C++) — parity target for byte-identical audio |

GGUF tensor map: [`docs/GGUF_FORMAT.md`](docs/GGUF_FORMAT.md).

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
│   │   ├── runtime/                  #   FamilyRegistry
│   │   ├── audio/                    #   (future)
│   │   └── text/                     #   (future)
│   ├── models/                       # Per-family public types
│   │   ├── moss_tts/                 #   TtsRequest, TtsResponse, CodecConfig
│   │   └── ace_step/                 #   MusicRequest, MusicResponse, AceStepConfig
│   └── server/                       #   HTTP server factory
├── src/
│   ├── cli/main.cpp                  # CLI entry point
│   ├── framework/                    # Framework implementation
│   │   ├── io/                       #   gguf_reader, weight_loader
│   │   ├── core/                     #   backend, session
│   │   ├── models/qwen3/             #   Qwen3 runner (llama.cpp wrapper)
│   │   └── runtime/                  #   registry
│   ├── models/                       # Per-family code
│   │   ├── moss_tts/                 #   loader, session, projection, delay_state, sampler, codec
│   │   └── ace_step/                 #   loader, session
│   └── server/                       # main.cpp, server.cpp (routes + WAV encoding)
├── tests/                            # Unit + e2e tests (one binary per file)
│   ├── test_framework.h              #   Header-only test macros
│   ├── synthetic_gguf.h/.cpp         #   Hermetic GGUF fixture builder
│   ├── test_gguf_reader.cpp
│   ├── test_registry.cpp
│   ├── test_projection.cpp
│   ├── test_server_e2e.cpp
│   ├── test_convert_acestep.cpp
│   └── test_moss_e2e.cpp            #   Full load + run (requires weights)
├── tools/
│   ├── convert_acestep.cpp           # HF → llama.cpp tensor rename
│   └── convert_qwen3tts.cpp          # Qwen3-TTS safetensors → GGUF
├── docs/
│   ├── ARCHITECTURE.md               # Two-seam deep-dive
│   ├── GGUF_FORMAT.md                # Tensor maps for each family
│   └── ONNX_ROADMAP.md               # Phase 2 plan
├── examples/
│   └── server.json                   # Reference server config
├── scripts/
│   └── reference_config.yaml         # Upstream Python pipeline config
└── third_party/                      # Vendored + submoduled deps
    ├── ggml/                         #   ggml (MIT, submodule)
    ├── llama.cpp/                    #   llama.cpp (MIT, submodule)
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
| ONNX Runtime (codec decode) | system install (optional) | MIT |

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
