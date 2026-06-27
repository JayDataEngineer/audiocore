# audiocore

Unified C++ audio inference server for **GGUF** models, with a path to **ONNX Runtime** as an alternative backend.

Focused initially on two model families that have no clean GGUF home today:

- **MOSS-TTS** (OpenMOSS) — TTS / TTSD / VoiceGenerator / SoundEffect
- **ACE-Step** — music generation (LM → DiT → VAE two-step pipeline)

## Why this exists

Two reference projects, each solving half the problem:

| Reference | What it gives us | What it lacks |
|---|---|---|
| [`0xShug0/audio.cpp`](https://github.com/0xShug0/audio.cpp) (Apache-2.0) | Framework layout, OpenAI-compatible server API, CLI shape, per-model code organization, CMake modular flags | Safetensors only — no GGUF support. MOSS/ACE-Step are README-aspirational, not implemented. |
| [`leejet/stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp) (MIT) | A clean GGUF reader that layers cleanly on top of a framework — the `TensorStorage` abstraction is the seam that lets format readers stay isolated from model code | Image-only, not audio |

**audiocore** borrows audio.cpp's architecture (clean room — not a fork) and vendors sd.cpp's GGUF reader directly (with attribution, per MIT). The result: a single C++ binary that serves any audio GGUF we collect, with the framework seams in place to swap ggml for ONNX Runtime later.

## Status

Pre-alpha. Currently scaffolding:

- [x] Project tree, license, build system skeleton
- [x] `TensorStorage` + abstract `WeightLoader` + GGUF reader (ported from sd.cpp)
- [x] Backend abstraction (ggml CUDA/CPU now, ONNX Runtime later)
- [ ] MOSS-TTS family: loader, session, codec wiring
- [ ] ACE-Step family: loader, two-step pipeline (LM / DiT / VAE)
- [ ] Server: `/v1/audio/speech`, `/v1/audio/music`, `/health`, `/v1/models`
- [ ] CLI: `--task tts --family moss_tts --model /path …`
- [ ] Weight converter: `safetensors → gguf` for MOSS + ACE-Step
- [ ] ONNX Runtime backend (Phase 2)

## Architecture (short version)

```
                    ┌─────────────────────────────────────────┐
                    │  Per-family model code                  │
                    │  src/models/<moss_tts|ace_step|…>/      │
                    │  (loader.cpp, session.cpp, …)           │
                    └──────────────────┬──────────────────────┘
                                       │ asks for tensors by name
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │  Weight loader (format-agnostic)        │
                    │  TensorStorage = {name, type, ne[5],    │
                    │                   file_index, offset}  │
                    └──────────────────┬──────────────────────┘
                                       │ dispatches by file magic
                          ┌────────────┼────────────┐
                          ▼            ▼            ▼
                     ┌────────┐   ┌──────────┐  ┌──────────┐
                     │  GGUF  │   │ safeten- │  │  ONNX    │
                     │ reader │   │ sors     │  │  reader  │
                     │(sd.cpp)│   │ (later)  │  │ (later)  │
                     └────────┘   └──────────┘  └──────────┘
                                       │ materializes into
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │  Backend (execution)                    │
                    │  • ggml (CUDA / CPU / Vulkan / Metal)   │
                    │  • ONNX Runtime            (Phase 2)    │
                    └─────────────────────────────────────────┘
```

The seam between "weights" and "execution" is the same as sd.cpp's: format readers produce `TensorStorage` metadata, model code lazy-loads tensors into whichever backend is active. Swapping ggml for ONNX Runtime touches the backend layer only.

## Build (planned)

```bash
cmake -S . -B build -DENGINE_ENABLE_CUDA=ON
cmake --build build --parallel --target audiocore_server audiocore_cli
```

## Run (planned)

```bash
build/bin/audiocore_server --config examples/server.json
```

See `examples/server.json` for the model configuration format.

## License

Apache-2.0. Vendored code in `third_party/` retains its original license headers:
- `stable-diffusion.cpp` GGUF reader — MIT, Copyright (c) 2023 leejet.
- `ggml` — MIT, Copyright (c) 2022-2024 llama.cpp contributors.

## References

- [0xShug0/audio.cpp](https://github.com/0xShug0/audio.cpp) — architectural inspiration.
- [leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) — GGUF reader source.
- [ServeurpersoCom/acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp) — existing ACE-Step C++ (single-purpose, no framework).
- [pwilkin/openmoss](https://github.com/pwilkin/openmoss) — existing MOSS-TTS C++ on llama.cpp.
