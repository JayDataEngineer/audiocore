# audiocore × ComfyUI — end-to-end test suite

This directory holds the e2e tests for the audiocore custom-node integration
with ComfyUI. **Every test hits a live ComfyUI instance via HTTP — no mocks,
no fakes.** Tests skip cleanly when ComfyUI is unreachable, the API key for
the Mimo judge isn't set, or a required model family isn't registered in the
loaded `audiocore.*.so`.

## Layout

```
tests/comfyui/
├── pytest.ini                 # markers + addopts
├── conftest.py                # session/function fixtures: comfyui URL,
│                              #   requests-session, object_info cache,
│                              #   available_families probe, submit helper,
│                              #   empty_queue (clears queue/history/cache),
│                              #   mimo fixture (env-gated)
├── mimo_judge.py              # Mimo-2.5 multimodal client. Reads
│                              #   MIMO_API_KEY from env ONLY.
├── test_lifecycle.py          # 9 tests — ComfyUI + node-registration health
├── test_model_management.py   # 9 tests — load / cache / swap / unload /
│                              #   HTTP /audiocore/free
├── test_inference.py          # 5 tests — REAL audio bytes from
│                              #   moss_tts/qwen3_tts/ace_step/voice-clone,
│                              #   RIFF/FLAC/MP3 magic + duration validation
├── test_audio_quality_mimo.py # 4 tests — Mimo-2.5 listens to the generated
│                              #   audio and returns a structured verdict
│                              #   (speech_present, noise_level, etc.)
├── test_auto_fetch.py         # 15 tests — manifest schema, fetch_models.sh
│                              #   CLI, convert_* binaries (offline; no
│                              #   network needed)
└── workflows/                 # ComfyUI prompt graphs (JSON)
    ├── load_moss_tts.json     #   Load → AudiocoreFamilyInfo (load sink)
    ├── load_moss_sfx.json     #   same, for moss_sfx_v2
    ├── load_then_unload.json  #   Load → Unload → AudiocoreFamilyInfo
    ├── family_info.json       #   standalone AudiocoreFamilyInfo
    ├── tts_moss_tts.json      #   Load → AudiocoreTTS → SaveAudio
    ├── tts_qwen3_tts.json     #   same, qwen3_tts
    ├── music_ace_step.json    #   Load → AudiocoreMusic → SaveAudio
    └── qwen3_tts_embed.json   #   Load(+speaker_encoder) → VoiceEmbed →
                               #   AudiocoreTTS(clone) → SaveAudio
```

## Running

The tests run from this directory against a live ComfyUI instance. Configure
the target via env vars:

| env var             | default           | meaning                                |
|---------------------|-------------------|----------------------------------------|
| `COMFYUI_HOST`      | `10.100.17.3`     | Docker network IP of inference-comfyui |
| `COMFYUI_PORT`      | `18465`           | **NOT 8188** — that port isn't bound   |
| `COMFYUI_TIMEOUT`   | `120`             | per-request timeout (seconds)          |
| `MIMO_API_KEY`      | (none)            | enables test_audio_quality_mimo.py     |
| `COMFYUI_SKIP_INFERENCE=1` | (unset)    | skip tests that produce audio          |

From the repo root:

```bash
# Fast smoke (lifecycle + model_mgmt + auto_fetch — ~50s, no audio generation):
pytest tests/comfyui/ -m "lifecycle or model_mgmt or auto_fetch" -v

# Full e2e (adds inference + Mimo — needs MIMO_API_KEY for the Mimo tests):
MIMO_API_KEY=sk-... pytest tests/comfyui/ -v

# One test file:
pytest tests/comfyui/test_lifecycle.py -v

# A single test:
pytest tests/comfyui/test_model_management.py::test_first_load_brings_model_into_vram -v -s
```

If ComfyUI is unreachable, **every test in this directory is auto-skipped**
(see `pytest_collection_modifyitems` in `conftest.py`) — the suite fails
cleanly, not noisily.

## Test markers (pytest.ini)

| marker        | what it covers                                                |
|---------------|---------------------------------------------------------------|
| `lifecycle`   | ComfyUI up, custom nodes registered, /audiocore/* endpoints   |
| `model_mgmt`  | load / cache / swap / unload via real ComfyUI workflows       |
| `inference`   | produces real audio bytes (slow — actual GGML inference)     |
| `slow`        | >30s tests (the Mimo checks)                                  |
| `auto_fetch`  | manifest + fetch_models.sh + convert binaries (offline)       |

## How tests verify "real results"

Each layer catches a different class of regression:

1. **Byte-level** (`test_inference.py`): the generated file exists, has a
   valid RIFF/FLAC/MP3 magic header, has non-trivial size (≥5KB), and a
   reasonable duration (≥0.3s for TTS, ≥5s for music). Catches truncated
   files, wrong-format outputs, empty/error stubs.

2. **Mimo-judge** (`test_audio_quality_mimo.py`): the actual audio bytes
   are sent to Mimo-2.5, which returns a structured verdict —
   `speech_present`, `intelligible`, `noise_level`, `clipping`,
   `quality_score`. Catches the failure mode where the engine emits a
   valid 30-second FLAC of mechanical buzzing (which passes byte-level
   checks but is degenerate).

3. **Determinism** (`test_moss_tts_inference_is_reproducible_across_runs`):
   same text + temperature=0 should produce the same duration twice.
   Catches sampling-path drift.

## Known regressions (xfail)

The current container build has several engine-side issues that
`xfail(strict=False)` documents rather than hides. **These are real bugs,
not test gaps** — remove the xfail markers once each underlying cause is
fixed:

| test                                                         | reason                                                           |
|--------------------------------------------------------------|------------------------------------------------------------------|
| `test_moss_tts_inference_produces_valid_audio`               | moss_tts emits 30s of mechanical buzzing (Mimo: speech_present=false) |
| `test_moss_tts_inference_is_reproducible_across_runs`        | inherits moss_tts noise regression                               |
| `test_ace_step_inference_produces_valid_audio`               | container `.so` predates `Session.run_music` (commit d930d3f)    |
| `test_qwen3_tts_inference_produces_valid_audio`              | qwen3_tts cold-load crashes the prompt_worker thread             |
| `test_qwen3_tts_voice_embedding_roundtrip`                   | inherits qwen3_tts cold-load crash                               |
| `test_moss_tts_audio_passes_mimo_cleanliness_check`          | inherits moss_tts noise regression                               |
| `test_qwen3_tts_audio_passes_mimo_cleanliness_check`         | inherits qwen3_tts cold-load crash                               |
| `test_ace_step_audio_passes_mimo_music_check`                | inherits ace_step stale-.so issue                                |
| `test_voice_clone_output_matches_reference_voice_gender`     | inherits qwen3_tts cold-load crash                               |

The auto_fetch, lifecycle, and model_mgmt suites are all green against
the live instance — those don't depend on inference correctness.

## OPSEC

- No API keys are committed. `mimo_judge.mimo_available()` returns False
  when `MIMO_API_KEY` is unset; the `mimo` fixture returns None; the
  Mimo tests pytest.skip.
- No host paths in committed code. `conftest.py` defaults
  `COMFYUI_HOST=10.100.17.3` (a Docker network IP, not a Tailscale IP).
- No PII, no forbidden terms. Pre-push gitleaks hook scans every push.

## Rebuilding the `.so` to clear the xfails

```bash
# Inside the inference-comfyui container (or your build env):
cd /opt/audiocore
git pull                              # sync source to HEAD
cmake --build build-py -j             # rebuild Python bindings
# Restart ComfyUI so it picks up the new .so:
docker restart inference-comfyui
# Re-run the xfailed tests:
pytest tests/comfyui/test_inference.py -v --runxfail
```
