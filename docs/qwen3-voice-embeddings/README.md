# Qwen3 Voice Embeddings

Documentation for the Qwen3-TTS voice embedding system — the ECAPA-TDNN
speaker encoder, embedding injection into the talker, inference modes, and
API integration. Written to support a pure ggml implementation.

## Files

| File | What |
|------|------|
| `ARCHITECTURE.md` | ECAPA-TDNN network architecture, mel front-end, weight layout |
| `EMBEDDING_FLOW.md` | How embeddings flow through the inference pipeline (injection into talker, x-vector-only vs ICL) |
| `INFERENCE.md` | Standalone embedding extraction, full TTS synthesis, embedding arithmetic |
| `GGUF_LAYOUT.md` | Tensor names, KV metadata, dimension tables, weight conversion |
| `API.md` | HTTP API (`speaker_embedding` field), base64 encoding, vllm-omni compatibility |

## Quick overview

```
Reference Audio (24 kHz mono)
  │
  ├─→ load_wav() → float samples
  │
  └─→ compute_mel() → 128-bin log-mel spectrogram
       │  [n_fft=1024, hop=256, fmin=0, fmax=12000, Slaney scale]
       │
       └─→ ECAPA-TDNN (ggml graph, CPU backend)
            │
            ├─ blk0: Conv1d 128→512 k=5 d=1 reflect-pad ReLU
            ├─ 3× SE-Res2Net (d=2/3/4): tdnn1→res2net→tdnn2→SE→+residual
            ├─ MFA: concat[blk0, blk1, blk2] → Conv1d 1536→1536 k=1 ReLU
            ├─ ASP: attentive statistics pooling → [3072, 1]
            └─ FC: 3072→enc_dim (1024 for 0.6B, 2048 for 1.7B)
            │
            └─→ speaker embedding (float[enc_dim])
                 │
                 └─→ injected into talker prefill at codec slot 6
                      (summed with text embedding at that position)
```

## Key design points

- **~6.3M params** (0.6B) / ~25M params (1.7B) — tiny enough to run on CPU
  even for real-time use.
- **Mel computation in host code** (not ggml) — STFT + mel filterbank using
  standard signal-processing primitives. The graph only starts after the
  mel spectrogram is computed.
- **Fresh ggml graph per call** — no persistent graph state. Allocate,
  build, compute, extract, free. The model is small enough that graph
  overhead is negligible (~1 ms).
- **Two operating modes**: x-vector-only (embedding alone, no ref audio
  codes) and ICL (embedding + ref audio code tokens + ref text for
  acoustic context).
- **Pre-computed embedding pass-through**: caller supplies the float vector
  directly via `speaker_embedding` — bypasses WAV load and ECAPA compute
  entirely. Enables voice caching, interpolation, and API passthrough.
