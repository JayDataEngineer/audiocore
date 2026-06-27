# ACE-Step family

Serves: music generation. Two-step pipeline (LM → DiT → VAE), same as
`ServeurpersoCom/acestep.cpp`.

## Source weights

- **GGUF (community)**: `hf://Serveurperso/ACE-Step-1.5-GGUF/`
  - `acestep-v15-turbo-Q8_0.gguf` (~2.5 GB) — DiT, 8-step distilled
  - `acestep-v15-sft-Q8_0.gguf` (~2.5 GB) — DiT, 50-step high-quality
  - `acestep-v15-xl-{turbo,sft,base}-Q8_0.gguf` (~5 GB) — XL 4B variants
  - `acestep-5Hz-lm-1.7B-Q8_0.gguf` (~1.9 GB) — music-code LM
  - `acestep-5Hz-lm-4B-Q8_0.gguf` (~4.2 GB) — larger LM
  - `Qwen3-Embedding-0.6B-Q8_0.gguf` (~750 MB) — text encoder
  - `vae-BF16.gguf` (~320 MB) — audio VAE decoder
- **Reference C++**: `ServeurpersoCom/acestep.cpp`

## What we need to build

### 1. Tensor map

Five separate GGUF files make up one model variant. Document every tensor
name and which file it lives in → `docs/GGUF_FORMAT.md` → ACE-Step section.

Files involved for the turbo + 1.7B LM configuration:

| File | Role | Tensors |
|---|---|---|
| `acestep-v15-turbo-Q8_0.gguf` | DiT (8-step) | dit.* |
| `acestep-5Hz-lm-1.7B-Q8_0.gguf` | Music-code LM (5 Hz) | lm.* |
| `Qwen3-Embedding-0.6B-Q8_0.gguf` | Text encoder | te.* |
| `vae-BF16.gguf` | Audio VAE decoder | vae.* |

The XL variants swap in a 4B DiT (~5 GB file) and optionally a 4B LM.

### 2. Loader (`loader.cpp`)

- Take a directory path; discover all GGUF files by name pattern.
- Open each via `GgufReader`, merge into a unified TensorStorage view
  (the `file_index` field on TensorStorage disambiguates).
- Parse `acestep.variant` from DiT file metadata (`turbo` / `sft` / `xl-turbo`).
- Materialize each component into its own ggml_context (DiT and VAE share
  a CUDA context; LM and text encoder get their own because they alternate
  with DiT execution and we want explicit VRAM accounting).
- Register with `AUDIOCORE_REGISTER_FAMILY(ace_step, …)`.

### 3. Session (`session.cpp`)

Implements `run_music(request, response)`. Two-step pipeline, async:

**Step 1 — LM** (`/lm` in upstream):
1. Encode caption + lyrics via Qwen3-Embedding.
2. Run the 5Hz LM with classifier-free guidance → music codes (token ids).
3. Codes cached on the session keyed by request id.

**Step 2 — Synth** (`/synth` in upstream):
1. Load DiT (turbo = 8 steps, sft = 50 steps).
2. Diffuse from noise, conditioned on the LM codes.
3. Decode latents → 48 kHz stereo WAV via the VAE.
4. Encode WAV (or MP3) into the response buffer.

The server exposes both as one `POST /v1/audio/music` (blocking) and as
`POST /v1/audio/music/submit` + `GET /v1/audio/music/job?id=…` (async, same
shape as acestep.cpp's current API).

### 4. Tokenizers

ACE-Step uses two: Qwen3-Embedding's BPE for captions/lyrics, and a
music-code vocabulary for the LM. Both are SentencePiece-shaped; vendor
once, configure per-tokenizer.

## Validation

Parity target: bit-equivalent LM output codes for the same caption + lyrics
+ seed. Audio output may differ very slightly due to DiT scheduling but
should be perceptually identical.

```bash
# Reference (ServeurpersoCom/acestep.cpp two-step)
curl -X POST http://localhost:8080/synth -d '{
  "caption": "ambient lo-fi", "lyrics": "", "duration": 30,
  "dit_model": "acestep-v15-turbo-Q8_0.gguf"
}' | jq .id
# … poll /job?id=… → ref.mp3

# Ours
./build/bin/audiocore_cli --task music --family ace_step \
  --model /mnt/data/models/audio/acestep-cpp \
  --dit acestep-v15-turbo-Q8_0.gguf \
  --lm  acestep-5Hz-lm-1.7B-Q8_0.gguf \
  --caption "ambient lo-fi" --duration 30 --out ours.mp3
```

## License note

ACE-Step model weights: Apache-2.0 (Serveurperso GGUF community release).
Our implementation: Apache-2.0.
