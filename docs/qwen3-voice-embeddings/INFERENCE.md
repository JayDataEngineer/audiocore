# Inference Guide

## Standalone embedding extraction

Extract a speaker embedding from a reference WAV file, without running
full TTS. The embedding can be saved and reused later.

### ggml flow

```
Input: 24 kHz mono 16-bit PCM WAV (or 32-bit float WAV)
  │
  ├─ load_wav(path)
  │    Parses RIFF header, supports 16-bit int and 32-bit float.
  │    Multi-channel → average to mono.
  │    Resampling to 24 kHz is NOT done — caller must provide
  │    audio at the correct sample rate (or resample beforehand).
  │
  ├─ compute_mel(samples, n_samples)
  │    Reflect-pad (n_fft - hop) / 2 = 384 each side
  │    Magnitude STFT: n_fft=1024, hop=256, periodic Hann window
  │    Slaney mel filterbank: 128 bands, fmin=0, fmax=12000
  │    Log: clamp(mel, 1e-5) → natural log
  │    Output: float[T_mel * 128] row-major (T_mel, 128)
  │
  └─ run_on_mel(mel_TC, T_mel)
       Transpose mel to [128, T_mel] (channels-first)
       Build ECAPA ggml graph (fresh per call)
       Allocate via gallocr (single CPU backend)
       Upload weight data from mmap sources
       Set mel tensor as input → compute
       Extract output tensor → float[enc_dim]
```

### Expected performance

| Audio length | ECAPA compute time (CPU, ~6.3M params) |
|-------------|---------------------------------------|
| 3 s (T_mel ≈ 281) | ~15-30 ms |
| 5 s (T_mel ≈ 469) | ~25-50 ms |
| 10 s (T_mel ≈ 938) | ~50-100 ms |

The encoder runs on CPU. GPU transfer overhead (~1 ms for a 6.3M model)
is not worth it for a model this small. The CPU CPU backend supports
full conv1d + reflect-pad ops natively.

### Embedding caching

Save embeddings as raw float32 binary for later reuse:

```
fwrite(embedding.data(), sizeof(float), enc_dim, file);
// Or: write as 1024/2048 base64-encoded floats for API use
```

For the HTTP API, the embedding is base64-encoded float32 bytes (see API.md).

## Full TTS with voice clone

### x-vector-only mode (no reference audio needed after embedding)

```
1. Compute embedding (once):
     emb = ecapa.compute_embedding("reference.wav")
     save "my_voice.emb" → float[1024]

2. Synthesize (many times, no WAV needed):
     req = {
       text: "Hello world",
       mode: "voice_clone",
       speaker_embedding: emb,     // ← pre-computed float[1024]
     }
     resp = session.run_tts(req)
```

The embedding is injected at codec slot 6 in the prefill. No reference
audio is loaded or encoded. This is the "voice caching" pattern.

### ICL mode (best quality)

```
1. Compute embedding:
     emb = ecapa.compute_embedding("reference.wav")

2. Encode reference audio to code tokens:
     ref_codes = codec.encode(ref_pcm, pcm_len)  // [16, T_ref]
     cb0_codes = ref_codes[0, :]                  // codebook 0 only

3. Build prefill with both:
     prefill = [
       ...role_tokens, codec_bridge_tokens,
       speaker_embedding,                        // slot 6
       ...ref_text_tokens,
       ...codec_emb(cb0_codes),
       ...syn_text_tokens + codec_bos,
     ]

4. Run talker prefill + AR loop as usual.
```

ICL mode gives better voice fidelity because the talker sees both the
voice identity (embedding) and an acoustic example (ref codes + text).
But it requires:
- The codec encoder (SEANet + transformer + RVQ) to be available.
- `reference_text` for the reference audio transcript.

If the codec encoder is absent, the path falls back to x-vector-only
with a warning.

### Full pipeline pseudocode

```
function run_tts(req):
  // Phase 0: Embedding
  if req.mode == "voice_clone":
    if req.speaker_embedding not empty:
      vc_spk_emb = req.speaker_embedding           // pre-computed
    else if req.reference_audio not empty:
      vc_spk_emb = speaker_encoder.compute_embedding(req.reference_audio)
    else: fail

    // Phase 0b: ICL ref codes (optional)
    if codec_encoder present and req.reference_audio not empty:
      ref_pcm = load_wav(req.reference_audio)
      ref_codes = codec_encoder.encode(ref_pcm)    // [16, T_ref]
      cb0 = ref_codes[0, :]

  // Phase 1: Build prefill embeddings
  emb_table = stack of text_emb(tok) + codec_emb(codec_tok)
  emb_table[6] = vc_spk_emb                        // speaker slot

  // Phase 2: Talker forward (prefill)
  hidden = talker.forward_embeddings(emb_table)

  // Phase 3: AR loop
  for step in 1..max_steps:
    logits = codec_head(hidden[-1])                 // codebook 0
    code_0 = sample(logits, temperature, top_p)
    fine_codes = predictor.predict_one_step(hidden[-1], code_0)  // 15 more
    if streaming: emit decoded audio frame
    hidden = talker.forward_embeddings(next_emb)

  // Phase 4: Codec decode
  audio = codec_decoder.decode(code_matrix)         // [16, T] → PCM
  return audio
```

## Embedding arithmetic

The embedding vector lives in a semantically meaningful latent space.
You can perform linear operations:

### SLERP interpolation

```python
def slerp(v0, v1, t):
    """Spherical linear interpolation between two embeddings."""
    v0_n = v0 / (norm(v0) + 1e-8)
    v1_n = v1 / (norm(v1) + 1e-8)
    omega = arccos(clip(dot(v0_n, v1_n), -1, 1))
    if omega < 1e-6:
        return (1 - t) * v0 + t * v1
    s0 = sin((1 - t) * omega) / sin(omega)
    s1 = sin(t * omega) / sin(omega)
    return s0 * v0 + s1 * v1

blended = slerp(male_voice, female_voice, t=0.5)  # androgynous mix
```

### Averaging

```python
avg_emb = mean([emb1, emb2, emb3], axis=0)  # consensus voice
```

### Dimensional editing

Voice characteristics (gender, pitch, emotion) are encoded in subspaces
of the embedding vector. You can identify correlated dimensions via
sparse autoencoders or simple PCA on an annotated dataset, then modify
those dimensions to steer voice attributes.

Note: The embedding space is not fully disentangled. Editing one
attribute (e.g., pitch) may bleed into others (e.g., timbre).
