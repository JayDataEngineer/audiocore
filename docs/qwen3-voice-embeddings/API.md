# API / Serving Integration

## HTTP API — speaker_embedding parameter

The `/v1/audio/speech` endpoint accepts a `speaker_embedding` field
that bypasses reference audio loading entirely. The embedding is a
base64-encoded float32 vector.

### Request format (JSON)

```json
{
  "model": "qwen3-tts-0.6b-base",
  "input": "Hello, this is a voice cloned message.",
  "mode": "voice_clone",
  "speaker_embedding": "<base64-encoded float32[1024]>",
  "response_format": "wav",
  "language": "en"
}
```

### Base64 encoding

The embedding is serialized as raw float32 little-endian bytes, then
base64-encoded. No JSON array of floats — that would be ~8 KB for a
1024-dim vector; base64 is ~1.4 KB and avoids JSON parsing overhead.

Python reference:
```python
import base64
import struct
import numpy as np

def encode_embedding(embedding: np.ndarray) -> str:
    """Encode float32 vector as base64 string for HTTP API."""
    assert embedding.dtype == np.float32
    raw = embedding.tobytes()  # little-endian float32
    return base64.b64encode(raw).decode('ascii')

def decode_embedding(b64: str) -> np.ndarray:
    """Decode base64 string back to float32 vector."""
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype=np.float32)
```

### Server-side decoding (audiocore)

From `src/server/server.cpp`:

```
if (body.contains("speaker_embedding") && body["speaker_embedding"].is_string()) {
    string b64 = body["speaker_embedding"];
    raw = base64_decode(b64);                    // → bytes
    n_floats = raw.size() / sizeof(float);
    tr.speaker_embedding.resize(n_floats);
    memcpy(tr.speaker_embedding.data(), raw.data(), n_floats * sizeof(float));
}
```

### Validation

- `speaker_embedding` is mutually exclusive with `reference_audio`.
  If both are provided, return 400.
- `speaker_embedding` implies `x_vector_only_mode = True` — no
  `reference_text` required.
- Empty embedding is rejected (400).
- NaN or Inf values in the float vector are rejected (400).
- Non-Base task types (CustomVoice, VoiceDesign) reject
  `speaker_embedding` (400) — only Base model supports voice cloning.

## Named voice registration (vllm-omni /v1/audio/voices)

A `speaker_embedding` can be registered as a named voice:

```
POST /v1/audio/voices
{
  "voice_id": "my_custom_voice",
  "speaker_embedding": [1024 float array or base64...],
  "embedding_dim": 1024
}
```

GET response includes:
```
{
  "voice_id": "my_custom_voice",
  "embedding_source": "direct",  // or "audio" if from sample
  "embedding_dim": 1024,
  "cache_status": "ready"        // immediate for direct embeddings
}
```

## Streaming endpoint

The `/v1/audio/speech/stream` endpoint also accepts `speaker_embedding`
with the same base64-encoded format. When combined with `mode: "streaming"`,
the embedding is injected into the prefill normally, and the per-frame
codec decode emits incremental PCM chunks via the stream callback. The
first audio frame arrives after ~80 ms (vs ~1.3 s for MOSS streaming).

## vllm-omni compatibility

The `speaker_embedding` parameter was added to vllm-omni via PR #1227
(merged March 2026). The protocol is:

- Field: `speaker_embedding` (list of floats in JSON, OR base64 string).
- JSON array: `"speaker_embedding": [0.123, -0.456, ...]` — 1024 or
  2048 floats. Simpler for scripting but ~8 KB in the request body.
- Base64 string: `"speaker_embedding": "AAAAAABBgMCA..."` — preferred
  for production (~1.4 KB for 1024-dim).
- The embedding dimension must match the model's `n_embd`:
  - 0.6B: 1024
  - 1.7B: 2048
- Mismatched dimensions produce a warning and the embedding is
  zero-padded or truncated to match.

## Python API (audiocore bindings)

```python
# Extract embedding from WAV → reuse in TTS calls
session = audiocore.Session("qwen3_tts", model_path="/path/to/model")

# Method 1: standalone extract
embedding = session.compute_embedding("reference.wav")
# embedding is list[float] of length 1024

# Method 2: extract + synthesize in one call
result = session.run_tts({
    "text": "Hello world",
    "mode": "voice_clone",
    "reference_audio": "reference.wav",
    "reference_text": "the reference text",
})

# Method 3: pre-computed embedding (no WAV)
embedding = session.compute_embedding("reference.wav")
result = session.run_tts({
    "text": "Hello world",
    "mode": "voice_clone",
    "speaker_embedding": embedding,  # list[float]
})
```

## CLI tool (qwen_voice.cpp)

The `tools/qwen_voice.cpp` CLI provides:

```
# Extract embedding
qwen_voice --model /path/to/model extract --wav ref.wav -o embedding.raw

# Synthesize with pre-computed embedding
qwen_voice --model /path/to/model tts \
  --text "Hello world" \
  --embedding embedding.raw \
  --output out.wav

# Interpolate two embeddings
qwen_voice --model /path/to/model interpolate \
  --emb-a male.raw --emb-b female.raw --t 0.5 --output blended.raw
```

## Notes for ggml implementation

- **Graph per call**: build a fresh ggml graph for each `run_on_mel()`
  call. The model is tiny (~12 MB for 0.6B) and graph construction
  takes <1 ms. No persistent graph caching needed.

- **Weight handling**: weights come from a GGUF mmap with `no_alloc`.
  The per-call `gallocr` allocates fresh zero-initialized CPU buffers.
  Copy the mmap'd weight data into these buffers after allocation
  via `ggml_backend_tensor_set()`. Without this step, weights are
  silently zero and the output is an all-zero embedding.

- **CPU backend only**: use `ggml_backend_cpu_init()`. The ECAPA model
  is too small to benefit from GPU. CPU supports the full op set
  (conv_1d with F16 weights, pad_reflect_1d, etc.).

- **Mel in host code**: compute the 128-bin log-mel spectrogram in
  ordinary C++ (librosa-style Slaney filterbank). Don't put the STFT
  in the ggml graph — it's cheaper to compute on the host and feed
  the mel tensor as input.

- **Embedding injection**: after extracting the embedding, it's a
  simple `memcpy` into the talker's codec_embedding slot at position 6.
  No projection or transformation needed (the ECAPA output dimension
  matches the talker's `n_embd` by design).
