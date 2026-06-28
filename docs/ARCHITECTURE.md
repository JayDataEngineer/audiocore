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
tensors by name via `WeightLoader::find()` + `materialize()`; the framework
handles format specifics. GGUF is the only weight format audiocore ships
today, and the only one it will ever ship — `TensorStorage` exists so model
code never calls `gguf_*` directly, not to enable foreign formats.

### Seam 2 — `Backend` (execution runtime)

```cpp
class Backend {
    virtual bool execute(void* graph_handle, void* io_bindings, …) = 0;
};
```

ggml CUDA/CPU/Vulkan/Metal backends. The `Session` owns one `Backend` and
exposes task methods (`run_tts`, `run_music`, …) that build graphs in the
backend's native shape and submit them. The Qwen3 transformer path
(`qwen3::Runner`) talks to libllama directly for KV-cache + sampling, with
graph execution still going through ggml.

### Seam 3 — `audiocore::sampler` (unified token sampling)

```cpp
struct Params {
    float temperature        = 1.0f;
    float top_p              = 1.0f;
    int   top_k              = 0;
    float repetition_penalty = 1.0f;
    bool  do_sample          = true;
};
int32_t sample_token(const float* logits, int32_t vocab_size,
                     const Params& p, …);
```

Top-k / top-p / temperature / repetition-penalty / argmax in one place, in
`src/framework/sampling/`. Every family and the `qwen3::Runner` MTP
predictor sample through it. Before Stage 5 four near-duplicate samplers
lived in `moss_tts/sampler.cpp`, `qwen3/runner.cpp`, `qwen3_tts/session.cpp`,
and `ace_step/session.cpp`; all four now delegate here.

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
              │  • exposes run_tts / run_music                          │
              └────────────────────────────┬─────────────────────────────┘
                                           │
              ┌────────────────────────────┴─────────────────────────────┐
              │                                                          │
              ▼                                                          ▼
   ┌─────────────────────────┐                          ┌──────────────────────────┐
   │ src/models/<family>/    │                          │ src/framework/io/        │
   │   loader.cpp            │ ←── asks tensors by name │   weight_loader.cpp      │
   │   session.cpp           │                          │   gguf_reader.cpp        │
   │   (model-specific code) │                          │   (only gguf_* calls in  │
   └─────────────────────────┘                          │    the tree live here)   │
                                                          └──────────────────────────┘
                                                                       │
                                                                       ▼
                                                          ┌──────────────────────────┐
                                                          │ src/framework/core/      │
                                                          │   backend.cpp            │
                                                          │   (ggml CUDA/CPU/Vulkan/ │
                                                          │    Metal)                │
                                                          └──────────────────────────┘
```

## What's borrowed, what's fresh

| Component | Origin | Reason |
|---|---|---|
| Framework layout (engine + framework + models + app) | `0xShug0/audio.cpp` | Their organization is correct; reinventing it adds nothing. Clean-room — no code copied. |
| Server API (`/v1/audio/speech`, `server.json` shape) | `0xShug0/audio.cpp` | OpenAI-compatible is what consumers expect. |
| CLI shape (`--task --family --model --backend`) | `0xShug0/audio.cpp` | Familiar surface; matches what `audiocpp_cli` already does. |
| `TensorStorage` struct | `leejet/stable-diffusion.cpp` (MIT) | The seam that keeps format specifics out of model code. Vendored with attribution. |
| `gguf_reader.cpp` + `gguf_reader_ext.h` | `leejet/stable-diffusion.cpp` (MIT) | Proven, small, MIT. ~360 lines total. |
| Per-family `loader.cpp` + `session.cpp` pattern | `0xShug0/audio.cpp` | Clear separation between "what tensors this model needs" (loader) and "how to run a task" (session). |
| MOSS-TTS family code | fresh | audio.cpp only has Nano-100M aspirationally; we need 8B Qwen3 backbone. |
| Qwen3-TTS family code | fresh | Talker + Code Predictor both run through the unified `qwen3::Runner`. |
| ACE-Step family code | fresh | audio.cpp has none; only `ServeurpersoCom/acestep.cpp` exists as a single-purpose server. |
| Backend abstraction | fresh (mirrors audio.cpp's `ENGINE_ENABLE_*` CMake flags) | ggml is the only execution engine; the abstraction exists so family code doesn't hardcode `ggml_*` calls. |

## What changes for each new model family

Adding a new family (e.g. `spark_tts`, `chatterbox`):

1. Create `src/models/<family>/{loader.cpp,session.cpp,…}`.
2. Document the tensor map in `docs/GGUF_FORMAT.md`.
3. Add one `AUDIOCORE_REGISTER_FAMILY(<name>, factory)` call in `loader.cpp`.
4. Add an entry to `examples/server.json` for parity testing.

No framework changes. No server changes. No CMake changes (the family's
sources get added to `audiocore_framework`'s source list once at the project
level, or moved into a CMake `glob` if we want zero-friction additions).

## Test strategy

Two-tier, mirroring audio.cpp:

1. **Unit tests** (`tests/`, registered via ctest) — `TensorStorage`, GGUF
   reader round-trip, family registry, audio-head projection parity,
   unified sampler (argmax / top-k / top-p / temperature / rep penalty /
   determinism), server HTTP e2e, ACE-Step converter e2e. Pure framework
   layer, no model weights required.
2. **Parity tests** (run manually; not in ctest) — `test_moss_e2e` and
   `test_qwen3tts_e2e` load the real weights and run the full pipeline.
   These need the actual weights mounted at their configured paths.
