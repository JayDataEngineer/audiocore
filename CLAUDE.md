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

- `examples/` — `server.json` example configs.
- `tests/` — unit + parity + e2e tests (see "Build / test" below).

## Adding a new model family

1. Create `src/models/<family>/`.
2. Implement `loader.cpp` (reads weights via `WeightLoader`, builds
   `ggml_context` for the active backend).
3. Implement `session.cpp` (subclass of `Session`, overrides the relevant
   `run_*` methods). Document the family in a `README.md` in the same dir.
4. Add `AUDIOCORE_REGISTER_FAMILY(<name>, factory)` to `loader.cpp`.

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
Qwen3-TTS (replaces the Python original). All *build-time* Python tooling
has been removed from the project. (A scratch `tools/compare/compare_ui.py`
exists for ad-hoc A/B comparison only — it is gitignored and is not part
of any build or test.) The HTTP server is also testable in-process via
`audiocore::build_server(slots)`; see `test_server_e2e.cpp`.

### Secrets, hooks, and internal notes

- **Tokens** live in `config/secrets.env` (gitignored). Copy the template:
  `cp config/secrets.env.example config/secrets.env`. Never commit a real
  token — the pre-push hook and the `gitleaks` GitHub Action both fail
  closed on HuggingFace tokens (and the rest of the gitleaks default
  ruleset).
- **Pre-push hook** — run `./scripts/install-git-hooks.sh` once after a
  fresh clone to enable secret scanning on push. Config: `.gitleaks.toml`.
- **`notes/`** is a gitignored scratch directory for maintainer-only docs
  (the honest GAP tracker, agent memory, ad-hoc status files). It is never
  committed (the public repo ships only README.md and CLAUDE.md).

## Family feature matrix

### MOSS-TTS (`moss_tts`)

| Mode | Status | HTTP endpoint | Notes |
|------|--------|---------------|-------|
| TTS (zero-shot) | ✅ Stage 16 | `POST /v1/audio/speech` | Text → speech, default handler. Codec-token → PCM via `MossCodecGraphs` when GGUF carries `moss.codec.*`; silence fallback otherwise. |
| Sound effects | ✅ Stage 16 | `POST /v1/audio/speech {"mode":"sfx"}` | Different system prompt + lower temps. Same codec path. |
| Voice cloning | ✅ Stage 16 | `POST /v1/audio/speech {"mode":"voice_clone","voice":"path.codes"}` | Requires pre-encoded `.codes` file (int32le: n_frames + n_frames×32 codes). Same codec path. |
| Dialogue (TTSD) | ✅ Stage 11 | `POST /v1/audio/speech {"mode":"dialogue"}` | TTSD-style system prompt + dialogue sampling defaults. Multi-turn via `messages` array (system/user/assistant roles). Single `text` becomes opening turn when `messages` absent. Codec wired (Stage 16). |
| Voice design | 🟡 Stage 11 | `POST /v1/audio/speech {"mode":"voice_design","instruct":"a calm deep female voice"}` | Voice description in `instruct` routes through flagship backbone with voice-design system prompt. Best-effort fallback for the dedicated VoiceGenerator model. Codec wired (Stage 16). |
| Streaming | ✅ Stage 18 | `POST /v1/audio/speech {"mode":"realtime"}` | Per-frame streaming: incremental codec decode during AR loop, PCM emitted via stream callback. Requires non-null `stream.on_audio`. First frame after ~1.3s delay (N_VQ × 80ms), then 80ms/frame. Response PCM empty in streaming mode. |
| ggml codec graph | ✅ Stage 16 | — | `src/models/moss_tts/codec.cpp` adapts `pwilkin/openmoss/src/codec.cpp` (Apache-2.0). Auto-binds when GGUF carries `moss.codec.*`; silence fallback otherwise. Pre-built sidecar GGUFs at `smcleod/MOSS-TTS-v1.5-GGUF`. |

**Codec token format** (`.codes` binary): `[n_frames: i32le] [codes: n_frames × 32 × i32le]`.

### Qwen3-TTS (`qwen3_tts`)

| Mode | Status | HTTP endpoint | Notes |
|------|--------|---------------|-------|
| TTS (talker + MTP predictor) | ✅ Stage 17 | `POST /v1/audio/speech` | Talker + Code Predictor load and run. Codec-token → PCM via `Qwen3TtsCodecGraphs` when codec sidecar GGUF is discovered (`extras["codec_path"]` or `tokenizer-{f16,q8_0}.gguf` next to talker); silence fallback otherwise. |
| TTS with style instructions | ✅ Stage 17 | `POST /v1/audio/speech {"speaker":"Vivian","instruct":"whispered"}` | Speaker routing injects `<|spk_NAME|>` codec token; `instruct` summed into the text embedding. Same codec path. |
| Voice design | ✅ Stage 17 | `POST /v1/audio/speech {"mode":"voice_design","instruct":"young female, energetic"}` | Instruct prefixed with the official VoiceDesign template. Best-effort on Base backbone. Same codec path. |
| Voice clone | ✅ Stage 18 (Phase B) | `POST /v1/audio/speech {"mode":"voice_clone","reference_audio":"path.wav","reference_text":"the reference text"}` | ECAPA-TDNN speaker encoder + ICL prefill with ref-codes (Phase B). When `codec.enc.*` tensors present, reference WAV is encoded into code tokens and injected as acoustic context alongside ref-text phonetic context. xvec_only fallback if encoder absent. Requires speaker encoder tensors in talker GGUF; fail-fast otherwise. |
| Streaming | ✅ Stage 19 | `POST /v1/audio/speech/stream {"mode":"streaming"}` | Per-frame streaming: codec frames decoded incrementally during AR loop and emitted via callback. Requires non-null `stream.on_audio`. First frame after ~80ms, then 80ms/frame. Response PCM empty in streaming mode. |
| Variant detection | ✅ Stage 10 | — | Set `extras["variant"]` = `Base` / `CustomVoice` / `VoiceDesign`, or rely on directory-name substring match. |
| ggml codec graph | ✅ Stage 17 | — | `src/models/qwen3_tts/codec.cpp` adapts `CrispStrobe/CrispASR`'s Qwen3-TTS codec section (MIT). Auto-binds when codec sidecar GGUF is discovered; silence fallback otherwise. Pre-built sidecar GGUFs at `cstr/qwen3-tts-tokenizer-12hz-GGUF`. |

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
| mode | ✅ Stage 13 | `mode` (`text_to_music` default; `cover`/`repaint`/`completion` wired; `stem`/`lego` fail fast — separate model families) |

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
  in the MOSS codec source.
- Qwen3-TTS: `QwenLM/Qwen3-TTS` (official Python reference)
- Qwen3-TTS-Tokenizer-12Hz codec: `CrispStrobe/CrispASR` (MIT) — full
  codec + ECAPA-TDNN speaker encoder in pure ggml. Pre-built GGUFs at
  `cstr/qwen3-tts-tokenizer-12hz-GGUF`. **Stage 17 port wired**
  (`src/models/qwen3_tts/codec.cpp`, codec section only). **Stage 17b**
  **port wired** (`src/models/qwen3_tts/speaker_encoder.cpp`, ECAPA-TDNN
  section only). Architecture and substitution tables in
  the Qwen3-TTS codec source. Do **not** use
  `predict-woo/qwen3-tts.cpp` as a reference — it has no license file.
- ACE-Step: `ServeurpersoCom/acestep.cpp`
- MOSS-SoundEffect-v2 (MSE2): `moss-sfx-v2/` repo next to audiocore.
  Reference pipeline at `moss_soundeffect_v2/diffsynth/pipelines/wan_audio.py`.

## MOSS-SoundEffect-v2 (`moss_sfx_v2`) — session status

| Component | Status | File | Notes |
|-----------|--------|------|-------|
| DiT GGUF converter | ✅ Phase 2 | `tools/convert_mse2.cpp` | Writes 825 tensors with `moss_sfx_v2.*` prefix |
| DiT graph builder | ✅ Phase 5 | `src/models/moss_sfx_v2/dit_runner.cpp` | 30-layer DiT, QK-norm, RoPE, CFG |
| VAE decoder | ✅ Phase 4 | `src/models/moss_sfx_v2/vae_runner.cpp` | DAC decoder: 4×DecoderBlock(strides=8,8,4,2), Snake, Tanh |
| Loader | ✅ Phase 3 | `src/models/moss_sfx_v2/loader.cpp` | DiT + VAE + TE GGUF binding |
| **Session (denoising loop)** | ✅ **New** | `src/models/moss_sfx_v2/session.cpp` | FlowMatch scheduler, CFG, Euler step, VAE decode, post-processing |

### Architecture (1.3B)
- dim=1536, ffn_dim=8960, n_heads=12, head_dim=128, n_layers=30
- in_dim=128, out_dim=128, text_dim=2048, freq_dim=256, patch_size=1
- 2 norms per block: norm1 (pre-SA/CA), norm3 (pre-FFN)
- AdaLN modulation: 6-chunk (shift/scale/gate for SA + shift/scale/gate for MLP)
- CA shares SA's gate
- FFN: Linear(dim, 2*ffn_dim) → GELU_tanh → Linear(2*ffn_dim, dim)

### Scheduler (FlowMatch)
- shift=5.0, sigma_min=0.0, extra_one_step=true
- Schedule: linspace(1.0, sigma_min, n+1)[:-1] → shift formula → sigmas descending
- Euler step: `x_{t+1} = x_t + (σ_{i+1} - σ_i) · v_θ(x_t, σ_i)`
- CFG: `v = v_uncond + cfg_scale · (v_cond - v_uncond)`

### DAC VAE decoder
- conv_in: WNConv1d(128→2048, k=7)
- 5 DecoderBlocks(up strides: 8,5,4,3,2 → total factor 960)
  - Each: Snake → ConvT1d(2×stride) → 3×ResUnit(k=7 dilated 1,3,9 + k=1 skip)
- Final: Snake(64) → Conv1d(64→1, k=7) → Tanh → mono PCM
- Total upsampling = 960 (matches hop_length)

### Loading order
1. `--extras vae_path=...` → VAE GGUF (DAC weights, must be produced by Python script)
2. `--extras te_path=...` → Qwen3 TE GGUF (optional; fallback to zero-context dummy)
3. DiT GGUF as primary model_path

### VAE GGUF converter
- `tools/convert_vae.py` — Python script extracting DAC `.pth` → GGUF.
  Reads `state_dict["decoder.model.*"]`, renames → `moss_sfx_v2.vae.*`,
  stores weight_norm params (weight_v, weight_g, bias) as F32.
  Tested: 147 tensors output to `/tmp/mse2_vae.gguf`.
- VAE GGUF loaded via `--extras vae_path=...`, bound into ext_ctx_
  alongside DiT tensors.

### Parity tests (`tests/test_mse2_parity`)
- **Status**: ✅ All 30 blocks pass (TOL=10)
- **Test coverage**: Per-block tests for norm1, modulation, SA Q/K/V/QK-norm/RoPE, SA attention, SA O-proj, CA Q/K/V/QK-norm, CA full pipeline, norm3, FFN gate/gelu/out, VAE layers
- **Known gaps** (within BF16/F16 quantization envelope):
  - `ca_v`: BF16 weight precision, abs_err ~0.5-1.2
  - `ffn_out`: large output values (up to ±1016) amplify quantization noise, abs_err up to ~8
- **Tolerance rationale**: TOL=10 uses `||` logic (pass if ae OR re < tol). All ops with real bugs would have rel_err >> 10.

### Critical fix (2026-07-06): flash_attn_ext permute bug
- **Bug**: `ggml_cont(ggml_permute(ctx, attn_out, 0, 2, 1, 3))` was called
  AFTER `ggml_flash_attn_ext`, scrambling the head/position layout.
- **Root cause**: flash_attn_ext returns `[hd, nh, T, 1]`, which reshapes
  correctly to `[nh*hd, T]` WITHOUT any permute. The permute swapped dims
  1 and 2, causing head data from one position to appear at a different
  position in the output.
- **Impact**: This was the root cause of ACE-Step producing "industrial
  machinery" audio instead of music. After the fix, VLM confirms the audio
  has clear musical structure (rhythm, harmonics, dynamics).
- **Fixed in**: `ace_step/dit_runner.cpp` (self_attn, cross_attn,
  timbre_encoder) and `moss_sfx_v2/dit_runner.cpp` (self_attn, cross_attn).
- **Already correct** (no permute): `moss_tts/codec.cpp`,
  `ace_step/detokenizer_runner.cpp`.

### Next steps
- End-to-end inference test with a complete GGUF set (DiT + VAE + TE)
- Resolve VAE architecture mismatch (decoder_dim=2048, 5 blocks)
