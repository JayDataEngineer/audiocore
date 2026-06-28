# AGENTS.md

## Repository layout

- `include/audiocore/framework/` — public API. Headers only.
- `include/audiocore/server/` — HTTP server factory API (shared between
  the binary and the e2e test).
- `src/framework/` — framework implementation (io, core, runtime, …).
- `src/models/<family>/` — per-family code. One subdirectory per model
  family with a consistent `loader.cpp` + `session.cpp` layout.
- `src/server/` — HTTP server factory + the binary entry point.
- `src/cli/` — CLI entry point.
- `third_party/` — vendored deps. Each subdir keeps its own LICENSE.
- `tools/` — weight converters, quantizers, utility scripts.
- `docs/` — architecture, GGUF format spec.
- `examples/` — `server.json` example configs.
- `tests/` — unit + parity + e2e tests (see "Build / test" below).

## Adding a new model family

1. Create `src/models/<family>/`.
2. Implement `loader.cpp` (reads weights via `WeightLoader`, builds
   `ggml_context` for the active backend).
3. Implement `session.cpp` (subclass of `Session`, overrides the relevant
   `run_*` methods). Document the family in a `README.md` in the same dir.
4. Add `AUDIOCORE_REGISTER_FAMILY(<name>, factory)` to `loader.cpp`.
5. Document the GGUF tensor map in `docs/GGUF_FORMAT.md`.
6. Add a parity test against a reference implementation if one exists.

## Conventions

- C++17, no compiler extensions.
- `camelCase` for variables/functions, `PascalCase` for classes/structs,
  `snake_case` for files.
- All vendored third-party code keeps its original license header.
  Never strip attribution.
- Weight-loading code only ever speaks to `TensorStorage`. Do not call
  `gguf_*` APIs from family code — go through the `WeightLoader` interface
  (the only `gguf_*` calls left in the tree live under `src/framework/io/`).
- TTS request/response types are unified in
  `include/audiocore/framework/runtime/tasks.h`. Do not add family-specific
  TtsRequest/TtsResponse structs — extend the unified one if you need a new
  field. Music keeps its own type since only ACE-Step serves it.
- Token sampling goes through `audiocore::sampler::sample_token`
  (`include/audiocore/framework/sampling/sampler.h`). Do not reimplement
  softmax/top-p/temperature locally.

## Build / test

```bash
cmake -S . -B build -DENGINE_ENABLE_CUDA=ON -DENGINE_BUILD_TESTS=ON
cmake --build build --parallel --target audiocore_server audiocore_cli
ctest --test-dir build
```

### Writing tests

Tests live in `tests/`, one executable per file (so a crash in one test
doesn't hide the rest). Each links against `audiocore_test_helpers`, which
provides:

- `tests/test_framework.h` — tiny header-only framework (TEST, CHECK,
  CHECK_EQ, FAIL). No GoogleTest dep; the framework prints `[PASS]`/`[FAIL]`
  per case and exits non-zero on any failure.
- `tests/synthetic_gguf.h/.cpp` — writes a tiny in-memory GGUF with
  caller-specified tensors + KV metadata, so reader tests don't need
  fixture files. Builders: `tspec(name, ne, start, step)` for tensors,
  `kv_i32` / `kv_str` / `kv_bool` for KV entries.

Declare new tests via the helper functions in `tests/CMakeLists.txt`:

```cmake
audiocore_test(my_unit)                       # plain test binary
audiocore_test_with_bin(my_e2e, convert_acestep)  # if it shells out to a binary
```

The `convert_acestep` tool is the C++ rewrite of the audit-only Python
script — it rewrites ACE-Step HF tensor names to llama.cpp names in place,
preserving KV. `convert_qwen3tts` is a C++ safetensors→GGUF converter for
Qwen3-TTS (replaces the Python original). All Python tooling has been
removed from the project. The HTTP server is also testable in-process via
`audiocore::build_server(slots)`; see `test_server_e2e.cpp`.

## Family feature matrix

### MOSS-TTS (`moss_tts`)

| Mode | Status | HTTP endpoint | Notes |
|------|--------|---------------|-------|
| TTS (zero-shot) | Done | `POST /v1/audio/speech` | Text → speech, default handler |
| Sound effects | Done | `POST /v1/audio/speech {"mode":"sfx"}` | Different system prompt + lower temps |
| Voice cloning | Done | `POST /v1/audio/speech {"mode":"voice_clone","voice":"path.codes"}` | Requires pre-encoded `.codes` file (int32le: n_frames + n_frames×32 codes) |
| Streaming | Not yet | — | Requires chunked HTTP response |
| ggml codec graph | Not yet | — | Codec decoder is a silence stub pending a ggml port of the speech-tokenizer graph; weights will live in the same GGUF as the backbone |

**Codec token format** (`.codes` binary): `[n_frames: i32le] [codes: n_frames × 32 × i32le]`.

### Qwen3-TTS (`qwen3_tts`)

| Mode | Status | HTTP endpoint | Notes |
|------|--------|---------------|-------|
| TTS (talker + MTP predictor) | Scaffolded | `POST /v1/audio/speech` | Talker + Code Predictor load and run; codec → PCM decoder is a silence stub pending a ggml port |
| Voice cloning | Not yet | — | Speaker encoder (ECAPA-TDNN) needs a GGUF port |
| Streaming | Not yet | — | Requires chunked HTTP response |

Both transformers (talker + predictor) run through the unified `qwen3::Runner`
— the same class MOSS and ACE-Step use. Weights: official
`QwenLM/Qwen3-TTS` safetensors converted via `tools/convert_qwen3tts`.

### ACE-Step (`ace_step`)

| Model variant | Status | n_steps default | Notes |
|---------------|--------|----------------|-------|
| turbo (v15) | Done | 8 | Shifted-cosine schedule |
| sft | Done | 50 | Linear schedule |
| xl-turbo | Auto‑detected | 8 (override with `steps`) | Config read from GGUF KV metadata |

| Parameter | Status | Field |
|-----------|--------|-------|
| caption | Done | `caption` |
| lyrics | Done | `lyrics` |
| duration | Done | `duration` |
| seed | Done | `seed` |
| guidance_scale | Done | `guidance_scale` (DiT-side CFG) |
| n_diffusion_steps | Done | `steps` |
| temperature | Done | `temperature` (LM, 0=argmax) |
| top_p | Done | `top_p` (LM nucleus, 1.0=off) |

Endpoint: `POST /v1/audio/music`

## Reference implementations

When adding a family, cross-check against the existing reference C++ for
that model. The reference is the parity target — byte-identical audio
output (modulo quantization noise).

- MOSS-TTS: `pwilkin/openmoss`
- Qwen3-TTS: `QwenLM/Qwen3-TTS` (official Python reference)
- ACE-Step: `ServeurpersoCom/acestep.cpp`
