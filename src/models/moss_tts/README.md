# MOSS-TTS family

Serves: TTS, TTSD (dialogue), VoiceGenerator (voice design), SoundEffect.

## Source weights

- **Original**: `hf://OpenMOSS-Team/MOSS-TTS` (safetensors, ~16 GB)
- **GGUF (community)**: `hf://OpenMOSS-Team/MOSS-TTS-GGUF`
  - `moss-tts-q8_0.gguf` (~8.7 GB) — Qwen3-8B backbone quantized
  - `moss-tts.extras.gguf` (~3.9 GB) — 32-RVQ audio codec + 1.6B codec aux
- **Reference C++**: `pwilkin/openmoss` (llama.cpp-based, single server)

## What we need to build

### 1. Tensor map

Document every tensor name we read from the GGUF and what role it plays in
the model. This becomes `docs/GGUF_FORMAT.md` → MOSS-TTS section.

Initial inventory (cross-reference `pwilkin/openmoss/moss-tts-info`):

- Qwen3-8B backbone (`backbone.*`, `token_embd.*`, `output.*`, `blk.*.attn_*`,
  `blk.*.ffn_*`, `blk.*.attn_norm.*`, …)
- Audio codec codebooks (32 RVQ layers, stored in `moss-tts.extras.gguf`)
- Codec decoder (~1.6B params)

### 2. Loader (`loader.cpp`)

- Open both GGUF files (`moss-tts-q8_0.gguf` + `moss-tts.extras.gguf`)
  via two `GgufReader` instances; merge into one TensorStorage view.
- Materialize Qwen3-8B backbone into ggml_context on the active backend.
- Materialize codec weights into a separate context (CPU-side today; codec
  runs on CPU per OpenMOSS-Team/MOSS-Audio-Tokenizer convention).
- Parse general metadata (`moss.audio_codec.*`, `moss.sample_rate`,
  `moss.codebook_bits`) and store on the Session.
- Register with `AUDIOCORE_REGISTER_FAMILY(moss_tts, …)`.

### 3. Session (`session.cpp`)

Implements `run_tts(request, response)`:

1. Tokenize input text via Qwen3 tokenizer (SentencePiece + chat template).
2. Generate token stream through Qwen3-8B backbone (ggml_cgraph per step).
3. Map generated tokens → codec codebook indices.
4. Decode codebook indices → 48 kHz stereo PCM via the audio codec.
5. Encode PCM as WAV (or MP3 if requested) into the response buffer.

Voice cloning variants add a reference-audio path: extract speaker embedding
from `voice_path`, condition generation. VoiceGenerator + SoundEffect have
their own request shapes but share the same backbone.

### 4. Tokenizer

MOSS uses a Qwen3 tokenizer with a chat template. SentencePiece is the
natural fit — vendor `google/sentencepiece` (audio.cpp does this in
`external/sentencepiece/`) and add a thin wrapper.

## Validation

Parity target: byte-identical audio output (modulo quantization) to
`pwilkin/openmoss/build/moss-tts-cli` for the same prompt + seed.

```bash
# Reference
docker run --rm -v /mnt/data/models/audio/moss-tts:/models \
  pwilkin/openmoss moss-tts-cli -m /models/moss-tts-q8_0.gguf \
    --text "Hello, world." -o ref.wav

# Ours
./build/bin/audiocore_cli --task tts --family moss_tts \
  --model /mnt/data/models/audio/moss-tts/moss-tts-q8_0.gguf \
  --text "Hello, world." --out ours.wav

# Compare
diff <(sha256sum ref.wav) <(sha256sum ours.wav)
```

## License note

MOSS-TTS model weights: Apache-2.0 (OpenMOSS-Team).
Our implementation: Apache-2.0.
