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
| TTS (zero-shot) | ✅ Stage 16 | `POST /v1/audio/speech` | Text → speech, default handler. Codec-token → PCM via `MossCodecGraphs` when GGUF carries `moss.codec.*`; silence fallback otherwise. |
| Sound effects | ✅ Stage 16 | `POST /v1/audio/speech {"mode":"sfx"}` | Different system prompt + lower temps. Same codec path. |
| Voice cloning | ✅ Stage 16 | `POST /v1/audio/speech {"mode":"voice_clone","voice":"path.codes"}` | Requires pre-encoded `.codes` file (int32le: n_frames + n_frames×32 codes). Same codec path. |
| Dialogue (TTSD) | 🟡 Stage 11 | `POST /v1/audio/speech {"mode":"dialogue"}` | TTSD-style system prompt + dialogue sampling defaults. Single `text` becomes opening turn; true multi-turn pending. Codec wired (Stage 16). |
| Voice design | 🟡 Stage 11 | `POST /v1/audio/speech {"mode":"voice_design","instruct":"a calm deep female voice"}` | Voice description in `instruct` routes through flagship backbone with voice-design system prompt. Best-effort fallback for the dedicated VoiceGenerator model. Codec wired (Stage 16). |
| Streaming | ✅ Stage 18 | `POST /v1/audio/speech {"mode":"realtime"}` | Per-frame streaming: incremental codec decode during AR loop, PCM emitted via stream callback. Requires non-null `stream.on_audio`. First frame after ~1.3s delay (N_VQ × 80ms), then 80ms/frame. Response PCM empty in streaming mode. |
| ggml codec graph | ✅ Stage 16 | — | `src/models/moss_tts/codec.cpp` adapts `pwilkin/openmoss/src/codec.cpp` (Apache-2.0). Auto-binds when GGUF carries `moss.codec.*`; silence fallback otherwise. Pre-built sidecar GGUFs at `smcleod/MOSS-TTS-v1.5-GGUF`. Architecture in `docs/CODEC_PORTS.md` §1. |

**Codec token format** (`.codes` binary): `[n_frames: i32le] [codes: n_frames × 32 × i32le]`.

### Qwen3-TTS (`qwen3_tts`)

| Mode | Status | HTTP endpoint | Notes |
|------|--------|---------------|-------|
| TTS (talker + MTP predictor) | ✅ Stage 17 | `POST /v1/audio/speech` | Talker + Code Predictor load and run. Codec-token → PCM via `Qwen3TtsCodecGraphs` when codec sidecar GGUF is discovered (`extras["codec_path"]` or `tokenizer-{f16,q8_0}.gguf` next to talker); silence fallback otherwise. |
| TTS with style instructions | ✅ Stage 17 | `POST /v1/audio/speech {"speaker":"Vivian","instruct":"whispered"}` | Speaker routing injects `<|spk_NAME|>` codec token; `instruct` summed into the text embedding. Same codec path. |
| Voice design | ✅ Stage 17 | `POST /v1/audio/speech {"mode":"voice_design","instruct":"young female, energetic"}` | Instruct prefixed with the official VoiceDesign template. Best-effort on Base backbone. Same codec path. |
| Voice clone | ✅ Stage 18 | `POST /v1/audio/speech {"mode":"voice_clone","reference_audio":"path.wav","reference_text":"the reference text"}` | ECAPA-TDNN speaker encoder + ICL prefill (`reference_text` extends prefill with phonetic context). Requires speaker encoder tensors in talker GGUF; fail-fast with GAPS.md pointer otherwise. |
| Streaming | ✅ Stage 19 | `POST /v1/audio/speech/stream {"mode":"streaming"}` | Per-frame streaming: codec frames decoded incrementally during AR loop and emitted via callback. Requires non-null `stream.on_audio`. First frame after ~80ms, then 80ms/frame. Response PCM empty in streaming mode. |
| Variant detection | ✅ Stage 10 | — | Set `extras["variant"]` = `Base` / `CustomVoice` / `VoiceDesign`, or rely on directory-name substring match. |
| ggml codec graph | ✅ Stage 17 | — | `src/models/qwen3_tts/codec.cpp` adapts `CrispStrobe/CrispASR`'s Qwen3-TTS codec section (MIT). Auto-binds when codec sidecar GGUF is discovered; silence fallback otherwise. Pre-built sidecar GGUFs at `cstr/qwen3-tts-tokenizer-12hz-GGUF`. Architecture in `docs/CODEC_PORTS.md` §2. |

Both transformers (talker + predictor) run through the unified `qwen3::Runner`
— the same class MOSS and ACE-Step use. Weights: official
`QwenLM/Qwen3-TTS` safetensors converted via `tools/convert_qwen3tts`.

### ACE-Step (`ace_step`)

| Model variant | Status | n_steps default | Notes |
|---------------|--------|----------------|-------|
| turbo (v15) | ✅ Done | 8 | Shifted-cosine schedule |
| sft | ✅ Done | 50 | Linear schedule |
| xl-turbo | ✅ Auto‑detected | 8 (override with `steps`) | Config read from GGUF KV metadata |

| Parameter | Status | Field |
|-----------|--------|-------|
| caption | ✅ Done | `caption` |
| lyrics | ✅ Done | `lyrics` |
| duration | ✅ Done | `duration` |
| seed | ✅ Done | `seed` |
| guidance_scale | ✅ Done | `guidance_scale` (DiT-side CFG) |
| n_diffusion_steps | ✅ Done | `steps` |
| temperature | ✅ Done | `temperature` (LM, 0=argmax) |
| top_p | ✅ Done | `top_p` (LM nucleus, 1.0=off) |
| mode | ✅ Stage 13 | `mode` (`text_to_music` default; `cover`/`repaint`/`stem`/`lego`/`completion` fail fast with a pointer at GAPS.md §3.2) |

Endpoint: `POST /v1/audio/music`

## Models manifest & download

`models/manifest.json` is the canonical record of every family/variant
audiocore supports, with HuggingFace repo, revision, file list, supported
modes, and per-variant `min_vram_gb`. Consumers:

- `scripts/fetch_models.sh` — pure bash + curl downloader. Reads the
  manifest, downloads per-variant, optionally verifies SHA256, invokes
  `convert_acestep` / `convert_qwen3tts` as requested. Try
  `scripts/fetch_models.sh --list` for the mode matrix,
  `scripts/fetch_models.sh --dry-run ace_step` to preview a fetch.
- `audiocore_cli --list-supported` — same matrix from the binary.
  Searches for `models/manifest.json` relative to the executable and
  the CWD; falls back to `FamilyRegistry::list()` if absent.

Adding a new family or variant = add a new entry to the manifest, in
parallel with the loader.cpp registration. The CLI's `--list-supported`
picks it up with no extra wiring.

## Reference implementations

When adding a family, cross-check against the existing reference C++ for
that model. The reference is the parity target — byte-identical audio
output (modulo quantization noise).

- MOSS-TTS: `pwilkin/openmoss` (Apache-2.0)
- MOSS-Audio-Tokenizer codec: `pwilkin/openmoss/src/codec.cpp` — full
  encoder/decoder/RVQ graphs in pure ggml. Pre-built sidecar GGUFs at
  `smcleod/MOSS-TTS-v1.5-GGUF` and `ilintar/moss-tts-gguf`. Port plan
  in `docs/CODEC_PORTS.md` §1.
- Qwen3-TTS: `QwenLM/Qwen3-TTS` (official Python reference)
- Qwen3-TTS-Tokenizer-12Hz codec: `CrispStrobe/CrispASR` (MIT) — full
  codec + ECAPA-TDNN speaker encoder in pure ggml. Pre-built GGUFs at
  `cstr/qwen3-tts-tokenizer-12hz-GGUF`. **Stage 17 port wired**
  (`src/models/qwen3_tts/codec.cpp`, codec section only). **Stage 17b**
  **port wired** (`src/models/qwen3_tts/speaker_encoder.cpp`, ECAPA-TDNN
  section only). Architecture and substitution tables in
  `docs/CODEC_PORTS.md` §2 and §4. Do **not** use
  `predict-woo/qwen3-tts.cpp` as a reference — it has no license file.
- ACE-Step: `ServeurpersoCom/acestep.cpp`
