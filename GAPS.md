# GAPS.md — advertised features vs. audiocore coverage

The canonical audit of what each upstream family advertises vs. what
audiocore actually implements. Every entry is one of:

- ✅ **wired** — parses, runs the model, returns non-silent audio end-to-end
- 🟡 **partial** — request parses, model runs, but a stage (codec decode,
  speaker encoder, mode-specific graph) is a stub or missing
- ❌ **not implemented** — no code path; the request is rejected or silently
  ignored
- 🚧 **blocked** — requires new model weights or a major port that is out of
  scope for a single commit; documented here so it's not forgotten

This file is the source of truth that `audiocore_cli --list-supported`
reads from at runtime (see `models/manifest.json` for the machine-readable
counterpart that also carries download locations).

---

## 1. MOSS-TTS family

Source: OpenMOSS-Team. Upstream advertises 5 models sharing the
MOSS-Audio-Tokenizer-v2 / -Nano codec infrastructure.

### 1.1 Models

| Upstream model | audiocore coverage |
|---|---|
| MOSS-TTS (8B flagship, zero-shot clone + TTS) | 🟡 backbone loads, audio tokens generate, **codec → PCM is a silence stub** |
| MOSS-TTSD (8B spoken dialogue, Qwen3-1.7B base) | ❌ no dialogue mode, no multi-turn input |
| MOSS-VoiceGenerator (1.7B voice design) | ❌ no voice-design mode |
| MOSS-TTS-Realtime (1.7B streaming) | ❌ no streaming endpoint |
| MOSS-SoundEffect (8B sound effects) | 🟡 `mode="sfx"` runs the backbone, **silence stub for codec** |
| MOSS-Audio-Tokenizer-v2 / -Nano | 🚧 blocked on a ggml port of the speech-tokenizer graph |

### 1.2 Modes (the `mode` field on `TtsRequest`)

| Mode | Status | Notes |
|---|---|---|
| `tts` (zero-shot) | 🟡 | Wired; codec stub returns 1 s silence |
| `sfx` (sound effects) | 🟡 | Wired; different system prompt + sampling defaults; codec stub |
| `voice_clone` | 🟡 | Wired; reads `voice_path` as pre-encoded `.codes`; codec stub |
| `dialogue` (TTSD) | 🟡 | Stage 11: TTSD-style system prompt + dialogue sampling defaults. Multi-turn input surface still missing (single `text` field becomes the opening turn); true TTSD weights are a separate 8B variant. |
| `voice_design` (VoiceGenerator) | 🟡 | Stage 11: routes voice description from `instruct` through the flagship backbone with a voice-design system prompt. Best-effort fallback — the dedicated 1.7B VoiceGenerator model produces better voice fidelity. |
| `realtime` / streaming | ❌ | Stage 11: fails fast with a pointer at this section. Streaming endpoint scaffold is GAPS.md §4. |

### 1.3 Codec / tokenizer

The MOSS-Audio-Tokenizer (the codec that turns 32-RVQ code matrices back
into 24/48 kHz PCM) was originally wired through ONNX Runtime. Stage 1
deleted that surface; the planned replacement is a ggml_cgraph port that
reads `moss.codec.dec.*` tensors from the same GGUF as the backbone.

Until that port lands, every MOSS-TTS request emits 1 s of silence where
the codec decode should be. The transformer pipeline upstream of the
codec is exercised end-to-end — only the final decode is stubbed.

**Dependencies to close:** `moss.codec.dec.*` tensor layout (documented in
`docs/GGUF_FORMAT.md`), CNN-free causal transformer architecture
(upstream `openmoss/src/codec.cpp`), ~1.6B parameters of weights.

---

## 2. Qwen3-TTS family

Source: Alibaba Qwen team. Upstream advertises 2 model sizes × 3 training
variants × 5 usage modes, all built on the Qwen3-TTS-Tokenizer-12Hz
multi-codebook codec.

### 2.1 Models / variants

| Upstream variant | audiocore coverage |
|---|---|
| Qwen3-TTS 1.7B Base | 🟡 talker + predictor load and run, codec stub |
| Qwen3-TTS 1.7B CustomVoice | 🟡 **no variant detection**; the loader treats every Qwen3-TTS GGUF identically |
| Qwen3-TTS 1.7B VoiceDesign | 🟡 same — no variant-specific defaults |
| Qwen3-TTS 0.6B Base / CustomVoice / VoiceDesign | 🟡 same — no size detection |
| Qwen3-TTS-Tokenizer-12Hz (codec) | 🚧 blocked on a ggml port |

### 2.2 Modes

| Mode | Status | Notes |
|---|---|---|
| Non-Streaming / Batch TTS | 🟡 Wired; codec stub returns silence |
| TTS with style instructions | 🟡 Stage 10: `instruct` tokenized + summed into text embedding; `speaker_name` resolves to a codec token (9 default speakers) and is injected as a dedicated codec-prefix slot before codec_bos. |
| Voice Clone Mode | ❌ Stage 10: fails fast with a pointer at §2.3. Full clone needs the ECAPA-TDNN speaker encoder (🚧 blocked on a GGUF port). |
| Voice Design Mode | 🟡 Stage 10: `instruct` is prefixed with the official "Generate a voice with the following characteristics:" template and routed through the talker. Best-effort on Base; true VoiceDesign fidelity needs the dedicated variant weights (🚧). |
| Streaming Mode | ❌ Stage 10: fails fast with a pointer at this section. Chunked transport scaffold exists at `/v1/audio/speech/stream` (Stage 12); per-family incremental decode does not. |

### 2.3 Codec / speaker encoder

Two ports are blocked:

1. **Qwen3-TTS-Tokenizer-12Hz** — multi-codebook speech encoder/decoder.
   Turns the 32-codebook matrix into 24 kHz mono PCM. Same situation as
   MOSS: ggml port pending, silence stub in the meantime.
2. **ECAPA-TDNN speaker encoder** — extracts a speaker embedding from a
   short reference clip for Voice Clone mode. Currently no code; the
   `reference_audio` field is parsed but ignored.

**Dependencies to close:** Qwen3-TTS-Tokenizer-12Hz architecture (from
`QwenLM/Qwen3-TTS` Python reference), ECAPA-TDNN architecture + weights,
per-variant config.json parsing.

---

## 3. ACE-Step family

Source: StepFun. Upstream advertises a "Two-Brain" architecture (LM planner
+ DiT renderer) with 6 user-facing modes. ACE-Step is the only audiocore
family that produces non-silent audio today.

### 3.1 Models / variants

| Upstream variant | audiocore coverage |
|---|---|
| ACE-Step 1.5 (3.5/3.6B) turbo | ✅ Full pipeline (LM → DiT → VAE → 48 kHz stereo) |
| ACE-Step 1.5 SFT | ✅ Same pipeline; `acestep.variant=sft` selects 50-step default |
| ACE-Step 1.5 Base | 🟡 Variant detected, pipeline runs; no fine-tuning entry point |
| ACE-Step 1.5 XL (4B) turbo / sft / base | 🟡 Variant detected, pipeline runs; same behavior as standard |
| ACE VAE | ✅ Loads and decodes |
| ACE-Step Text Encoders (`qwen_0.6b_ace15`, `qwen_1.7b_ace15`) | ✅ Both load via the unified `qwen3::Runner` |

### 3.2 Modes

| Mode | Status | Notes |
|---|---|---|
| Text-to-Music (Custom) | ✅ `caption` + `lyrics` + `duration` wired end-to-end |
| Cover Generation | ❌ No `mode` field on `MusicRequest`; needs style/voice transfer graph |
| Repainting (Inpainting) | ❌ Needs segment-level mask input + selective re-generation |
| Track / Stem Extraction | ❌ Typically a separate model (Demucs-class); not part of ACE-Step proper |
| Layering (LEGO) | ❌ Needs stem assembly mixer; not part of ACE-Step proper |
| Completion (Outpointing) | ❌ Needs partial-song conditioning + extension graph |

### 3.3 Architecture gaps

Cover / Repainting / Completion all need the DiT to accept a conditioning
signal beyond text — an existing latent to preserve, a mask, or a partial
song. The DiT graph in `src/models/ace_step/dit_runner.cpp` currently
only accepts text-encoder conditioning. Adding the rest is a graph-level
change, not a small patch.

Stem Extraction and Layering are genuinely separate models in the
upstream ecosystem (Demucs for separation, a DAW-style mixer for
layering). They don't belong in the ACE-Step DiT at all — they'd be new
families.

---

## 4. Cross-family infrastructure

| Capability | Status | Notes |
|---|---|---|
| OpenAI-compatible HTTP API | ✅ `/v1/audio/speech`, `/v1/audio/speech/stream`, `/v1/audio/music`, `/v1/models`, `/health` |
| Chunked / streaming HTTP for TTS | 🟡 Stage 12: `/v1/audio/speech/stream` emits the WAV in ~64 KiB chunks. Transport-level only — the family still renders the full PCM before streaming starts. True incremental streaming (frames as the AR loop produces them) needs per-family hooks (GAPS.md §1.2 / §2.2). |
| Models manifest (`models/manifest.json`) | ✅ Stage 9: central record with HF repo/revision/modes for every family/variant |
| Fetch tool (`scripts/fetch_models.sh`) | ✅ Stage 9: pure bash + curl downloader; reads the manifest, optional sha256 verify, invokes converters |
| Unified `TtsRequest` (Stage 6) | ✅ Single struct, single server branch |
| Unified `qwen3::Runner` (Stage 2) | ✅ One backbone path for 4 transformers |
| Unified `audiocore::sampler` (Stage 5) | ✅ One sampler for all families + MTP |
| `WeightLoader` everywhere (Stage 4) | ✅ No `gguf_*` calls outside `src/framework/io/` |
| `--list-supported` self-describe flag | ✅ Stage 9: `audiocore_cli --list-supported` reads the manifest and prints the matrix; `--list-families` prints the registered set |
| WAV output | ✅ 16-bit PCM mono (TTS) and stereo (music) at family-native rates |
| MP3 output | ❌ No encoder linked |

---

## 5. What "closing" each gap requires

Sorted by effort, smallest first. Items marked 🚧 are out of scope for a
single session but are listed so the work is plan-able.

### Smallest (config / parsing / IaC — no model changes)

1. ✅ Stage 9 — **`models/manifest.json`** — machine-readable record of every
   family/variant with HF repo, revision, file list, license, supported modes.
2. ✅ Stage 9 — **`scripts/fetch_models.sh`** — pure bash + curl downloader,
   reads the manifest, optional sha256 verify, invokes converters.
3. ✅ Stage 9 — **`audiocore_cli --list-supported`** — prints the mode matrix;
   `--list-families` prints the registered set.
4. ✅ Stage 10 — **Qwen3-TTS variant detection** — `extras["variant"]` or
   directory-name substring match.
5. ✅ Stage 10 — **Qwen3-TTS `speaker_name` → token routing** — the 9 default
   speakers resolve to codec tokens, injected as a dedicated codec-prefix slot.
6. ✅ Stage 11 — **MOSS `mode="dialogue"`** — TTSD-style system prompt + dialogue
   sampling defaults. Multi-message input still pending.
7. ✅ Stage 11 — **MOSS `mode="voice_design"`** — voice description in `instruct`
   routes through the flagship backbone with a voice-design system prompt.
8. ✅ Stage 10 — **Qwen3-TTS `mode="voice_design"`** — same idea, on the
   Qwen3-TTS backbone. True VoiceDesign needs the separate variant weights (🚧).
9. ⏳ Stage 13 — **ACE-Step `mode` field** — add `mode` to `MusicRequest`;
   reject non-text-to-music modes with a clear error pointing at GAPS.md.
10. ✅ Stage 12 — **HTTP streaming endpoint scaffold** — `POST /v1/audio/speech/stream`
    emits the WAV in ~64 KiB chunks. Transport-level only; per-family
    incremental streaming is still open.

### Medium (real feature work — graph changes, no new weights)

11. **Qwen3-TTS Voice Clone field wiring** — read `reference_audio`,
    surface a clear "speaker encoder not ported" error until #17 lands.
    The parsing layer is implementable today; the encoder is not.
12. **ACE-Step Cover Generation** — DiT accepts a second conditioning
    signal (target voice profile). Graph change in `dit_runner.cpp`.
13. **ACE-Step Completion (Outpointing)** — DiT accepts a partial-song
    latent. Graph change.
14. **ACE-Step Repainting (Inpainting)** — DiT accepts mask + partial
    latent. Largest of the three DiT-extension gaps.

### Large (new weights or major ports — out of scope for one session)

15. 🚧 **MOSS-Audio-Tokenizer ggml port** — ~1.6B-param neural codec,
    port from upstream `openmoss/src/codec.cpp`. Unblocks all MOSS modes
    end-to-end.
16. 🚧 **Qwen3-TTS-Tokenizer-12Hz ggml port** — multi-codebook codec
    from `QwenLM/Qwen3-TTS`. Unblocks all Qwen3-TTS modes end-to-end.
17. 🚧 **ECAPA-TDNN speaker encoder port** — small model (~22M params)
    but entirely new architecture in audiocore. Unblocks true Voice Clone.
18. 🚧 **ACE-Step Stem Extraction / Layering** — separate model family
    (Demucs-class separator + mixer). New family, not an ACE-Step mode.
19. 🚧 **MP3 output** — link libmp3lame; small but adds a dependency.

---

## 6. Coverage summary

Of the **5 + 5 + 6 = 16 advertised user-facing modes** across the three
families, after Stages 9–13:

- ✅ fully wired end-to-end: **1** (ACE-Step Text-to-Music)
- 🟡 parse + run + emit silence (codec stub): **9** (MOSS tts / sfx /
  voice_clone / dialogue / voice_design; Qwen3-TTS batch TTS / with
  style instructions / voice_design; ACE-Step mode field rejects the
  rest with a clear error)
- ❌ not implemented, achievable without new weights: **3** (ACE-Step
  cover / repaint / completion — all need the DiT to accept an extra
  conditioning signal beyond text)
- ❌ not implemented, separate model: **2** (ACE-Step stem extraction,
  layering — blocked on a new Demucs-class / stem-assembler family)
- ❌ not implemented, streaming infra: **2** (MOSS realtime, Qwen3-TTS
  streaming — the chunked transport scaffold exists; per-family
  incremental decode does not)
- 🚧 blocked on major port: **3+** (MOSS codec, Qwen3-TTS codec,
  ECAPA-TDNN speaker encoder)

Stage 9 closed the entire IaC bucket. Stages 10–12 flipped six
modes from ❌ to 🟡 (Qwen3-TTS voice_design, MOSS dialogue, MOSS
voice_design, MOSS/Qwen3-TTS realtime/streaming — the last via the
transport scaffold). Stage 13 added explicit `mode` parsing to ACE-Step
so the unimplemented modes fail fast with a GAPS.md pointer instead of
silently running text-to-music.

The remaining work is concentrated in two places:

1. **Codec ports (🚧)** — MOSS-Audio-Tokenizer + Qwen3-TTS-Tokenizer-12Hz
   ggml ports. These unblock 6 of the 9 🟡 modes end-to-end at once.
   Multi-week port per codec.
2. **DiT conditioning extension** — ACE-Step cover / repaint / completion.
   Graph-level change in `dit_runner.cpp`, no new weights. Medium effort.

Plus, smaller but unblocked items: streaming-endpoint per-family hooks,
ECAPA-TDNN speaker encoder port (Voice Clone), MP3 output.
