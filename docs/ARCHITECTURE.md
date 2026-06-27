# audiocore architecture

## The two seams

audiocore is shaped around two abstractions, both borrowed from
`leejet/stable-diffusion.cpp`:

### Seam 1 — `TensorStorage` (format-neutral weight descriptor)

```cpp
struct TensorStorage {
    std::string name;
    ggml_type type;
    int64_t ne[5];
    size_t file_index;   // multi-file model dirs
    uint64_t offset;     // lazy-load on demand
};
```

Every weight reader produces `vector<TensorStorage>`. Model code requests
tensors by name; the framework materializes them lazily into the active
backend. Adding a new weight format (ONNX, NPZ, …) means writing one new
reader — model code is untouched.

### Seam 2 — `Backend` (execution runtime)

```cpp
class Backend {
    virtual bool execute(void* graph_handle, void* io_bindings, …) = 0;
};
```

Today only ggml CUDA/CPU/Vulkan/Metal backends exist. Phase 2 adds ONNX
Runtime as a sibling backend. The `Session` owns one `Backend` and exposes
task methods (`run_tts`, `run_music`, …) that build graphs in the backend's
native shape and submit them.

## How the pieces fit

```
              ┌──────────────────────────────────────────────────────────┐
              │  src/server/main.cpp                                    │
              │  reads server.json → spawns one Session per model id    │
              └────────────────────────────┬─────────────────────────────┘
                                           │ FamilyRegistry::create(family)
                                           ▼
              ┌──────────────────────────────────────────────────────────┐
              │  Session (one per configured model id)                  │
              │  • owns WeightLoader (reads weights on demand)          │
              │  • owns Backend (submits graphs for execution)          │
              │  • exposes run_tts / run_music / run_asr                │
              └────────────────────────────┬─────────────────────────────┘
                                           │
              ┌────────────────────────────┴─────────────────────────────┐
              │                                                          │
              ▼                                                          ▼
   ┌─────────────────────────┐                          ┌──────────────────────────┐
   │ src/models/<family>/    │                          │ src/framework/io/        │
   │   loader.cpp            │ ←── asks tensors by name │   weight_loader.cpp      │
   │   session.cpp           │                          │   gguf_reader.cpp        │
   │   (model-specific code) │                          │   (safetensors, onnx: P2)│
   └─────────────────────────┘                          └──────────────────────────┘
                                                                       │
                                                                       ▼
                                                          ┌──────────────────────────┐
                                                          │ src/framework/core/      │
                                                          │   backend.cpp            │
                                                          │   (ggml now; ORT: P2)    │
                                                          └──────────────────────────┘
```

## What's borrowed, what's fresh

| Component | Origin | Reason |
|---|---|---|
| Framework layout (engine + framework + models + app) | `0xShug0/audio.cpp` | Their organization is correct; reinventing it adds nothing. Clean-room — no code copied. |
| Server API (`/v1/audio/speech`, `server.json` shape) | `0xShug0/audio.cpp` | OpenAI-compatible is what consumers expect. |
| CLI shape (`--task --family --model --backend`) | `0xShug0/audio.cpp` | Familiar surface; matches what `audiocpp_cli` already does. |
| `TensorStorage` struct | `leejet/stable-diffusion.cpp` (MIT) | The seam that lets us add ONNX without touching model code. Vendored with attribution. |
| `gguf_reader.cpp` + `gguf_reader_ext.h` | `leejet/stable-diffusion.cpp` (MIT) | Proven, small, MIT. ~360 lines total. |
| Per-family `loader.cpp` + `session.cpp` pattern | `0xShug0/audio.cpp` | Clear separation between "what tensors this model needs" (loader) and "how to run a task" (session). |
| MOSS-TTS family code | fresh | audio.cpp only has Nano-100M aspirationally; we need 8B Qwen3 backbone. |
| ACE-Step family code | fresh | audio.cpp has none; only `ServeurpersoCom/acestep.cpp` exists as a single-purpose server. |
| Backend abstraction | fresh (mirrors audio.cpp's `ENGINE_ENABLE_*` CMake flags) | Needs to support ONNX Runtime later, so we can't lock onto ggml. |

## What changes when Phase 2 (ONNX) lands

Three additions, no changes to existing code:

1. `src/framework/io/onnx_reader.cpp` — produces `vector<TensorStorage>` from
   an `.onnx` file (the ONNX Runtime API exposes weight tensors as initializers;
   we just enumerate them).
2. `src/framework/core/backend_onnx.cpp` — implements `Backend::execute()` by
   building an `Ort::Session` from the graph handle and running it.
3. `ENGINE_ENABLE_ONNXRUNTIME` CMake flag.

Model code is unchanged because it already speaks in terms of `TensorStorage`
and `Backend::execute()`. A MOSS-TTS family that today runs on ggml CUDA can
run on ONNX Runtime tomorrow by changing one line in `server.json`:

```json
{ "backend": "onnxruntime" }   // was "ggml_cuda"
```

## What changes for each new model family

Adding a new family (e.g. `kokoro_tts`, `spark_tts`, `whisper`):

1. Create `src/models/<family>/{loader.cpp,session.cpp,…}`.
2. Document the tensor map in `docs/GGUF_FORMAT.md`.
3. Add one `AUDIOCORE_REGISTER_FAMILY(<name>, factory)` call in `loader.cpp`.
4. Add an entry to `examples/server.json` for parity testing.

No framework changes. No server changes. No CMake changes (the family's
sources get added to `audiocore_framework`'s source list once at the project
level, or moved into a CMake `glob` if we want zero-friction additions).

## Test strategy

Two-tier, mirroring audio.cpp:

1. **Unit tests** (`tests/`) — `TensorStorage`, GGUF reader, tokenizer, codec
   I/O. Pure framework layer, no model weights required.
2. **Parity tests** (`tests/parity/`) — generate audio for fixed prompts +
   seeds and compare against the reference C++ implementations:
   - `pwilkin/openmoss/build/moss-tts-cli` for MOSS
   - `ServeurpersoCom/acestep.cpp/build/ace-server` for ACE-Step

   These need the actual weights mounted and run in CI nightly, not on every
   commit.
