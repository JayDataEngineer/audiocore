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
| MOSS-Audio-Tokenizer-v2 / -Nano | ✅ Stage 16: ggml port wired (`src/models/moss_tts/codec.cpp`, adapted from `pwilkin/openmoss` Apache-2.0). Activates automatically when the GGUF carries `moss.codec.*` tensors (e.g. the `smcleod/MOSS-TTS-v1.5-GGUF` sidecar); community backbone-only GGUFs still get the silence fallback. |

### 1.2 Modes (the `mode` field on `TtsRequest`)

| Mode | Status | Notes |
|---|---|---|
| `tts` (zero-shot) | ✅ Stage 16 | Wired; codec decode via `MossCodecGraphs` when GGUF carries `moss.codec.*`, silence fallback otherwise |
| `sfx` (sound effects) | ✅ Stage 16 | Wired; different system prompt + sampling defaults; same codec path |
| `voice_clone` | ✅ Stage 16 | Wired; reads `voice_path` as pre-encoded `.codes`; same codec path |
| `dialogue` (TTSD) | 🟡 | Stage 11: TTSD-style system prompt + dialogue sampling defaults. Multi-turn input surface still missing (single `text` field becomes the opening turn); true TTSD weights are a separate 8B variant. Codec wired (Stage 16). |
| `voice_design` (VoiceGenerator) | 🟡 | Stage 11: routes voice description from `instruct` through the flagship backbone with a voice-design system prompt. Best-effort fallback — the dedicated 1.7B VoiceGenerator model produces better voice fidelity. Codec wired (Stage 16). |
| `realtime` / streaming | ✅ Stage 18 | Per-frame incremental codec decode during AR loop. First frame after ~1.3s (N_VQ delay steps), then 80ms/frame. Response PCM empty in streaming mode — all audio through callback. |

### 1.3 Codec / tokenizer

The MOSS-Audio-Tokenizer (the codec that turns 32-RVQ code matrices back
into 24/48 kHz PCM) was originally wired through ONNX Runtime. Stage 1
deleted that surface; Stage 16 landed the ggml_cgraph port that reads
`moss.codec.dec.*` tensors from the same GGUF as the backbone.

**Stage 16 (complete):** `src/models/moss_tts/codec.cpp` is an
adaptation of `pwilkin/openmoss/src/codec.cpp` (Apache-2.0, 1087 lines
upstream). Audiocore's `MossCodecGraphs` reproduces the openmoss
architecture verbatim — 4-stage ProjectedTransformer, 32-RVQ LFQ
quantizer, weight-norm reconstruction, fused QKV, RoPE interleaved-pair,
flash attention via `ggml_flash_attn_ext`, per-channel LayerScale —
swapped onto audiocore's `Backend` / `TensorStorage` /
`WeightLoader` abstractions. The codec binds automatically during
`MossSession::load()` when the GGUF carries `moss.codec.*` tensors
(e.g. the `smcleod/MOSS-TTS-v1.5-GGUF` sidecar); community
backbone-only GGUFs continue to get the 1 s silence fallback
(documented behavior, not an error). See `docs/CODEC_PORTS.md` §1 for
the architecture and substitution table.

The transformer pipeline upstream of the codec was already exercised
end-to-end; with Stage 16, MOSS-TTS requests now produce real audio
when codec-bearing weights are loaded. Runtime parity is verified by
`tests/test_moss_e2e.cpp` (requires real weights; intentionally not in
ctest per AGENTS.md).

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
| Qwen3-TTS-Tokenizer-12Hz (codec) | ✅ Stage 17: ggml port wired (`src/models/qwen3_tts/codec.cpp`, adapted from `CrispStrobe/CrispASR` MIT). Activates automatically when the codec sidecar GGUF is discovered (`extras["codec_path"]` or `tokenizer-{f16,q8_0}.gguf` next to the talker); setups without the sidecar still get the silence fallback. |

### 2.2 Modes

| Mode | Status | Notes |
|---|---|---|
| Non-Streaming / Batch TTS | ✅ Stage 17: codec decode via `Qwen3TtsCodecGraphs` when codec GGUF discovered, silence fallback otherwise |
| TTS with style instructions | ✅ Stage 17: `instruct` tokenized + summed into text embedding; `speaker_name` resolves to a codec token (9 default speakers) and is injected as a dedicated codec-prefix slot before codec_bos. Codec decode wired (Stage 17). |
| Voice Clone Mode | ✅ Stage 17b: ECAPA-TDNN speaker encoder ported (`Qwen3TtsSpeakerEncoder` in `src/models/qwen3_tts/speaker_encoder.cpp`, adapted from CrispASR MIT). Requires reference WAV at 24 kHz mono; speaker embedding is injected into the codec bridge. Silently skips if no `speaker.*` tensors in the talker GGUF (Base variant — Voice Clone still fails fast with a clear error). |
| Voice Design Mode | 🟡 Stage 10: `instruct` is prefixed with the official "Generate a voice with the following characteristics:" template and routed through the talker. Best-effort on Base; true VoiceDesign fidelity needs the dedicated variant weights (🚧). Codec decode wired (Stage 17). Speaker encoder does NOT apply to VoiceDesign — that mode's voice description is a text-only instruct, not a reference-audio embedding. |
| Streaming Mode | ✅ Stage 19 | Per-frame streaming: codec frames decoded incrementally during AR loop and emitted via callback. First frame after ~80ms, then 80ms/frame. Response PCM empty in streaming mode. |

### 2.3 Codec / speaker encoder

Codec port complete (Stage 17); speaker encoder remains:

1. **Qwen3-TTS-Tokenizer-12Hz** — multi-codebook speech encoder/decoder.
   Turns the 16-codebook matrix into 24 kHz mono PCM.
   **Stage 17 (complete):** `src/models/qwen3_tts/codec.cpp` is an
   adaptation of `CrispStrobe/CrispASR`'s `src/qwen3_tts.cpp` codec
   section (MIT). Audiocore's `Qwen3TtsCodecGraphs` reproduces the
   CrispASR architecture verbatim — 16-codebook RVQ front-end (1 first +
   15 rest, each with a Conv1d k=1 out_proj), 8-layer pre-LN RMSNorm
   transformer (non-fused QKV, NEOX RoPE, SwiGLU, LayerScale,
   `ggml_flash_attn_ext`), 2 ConvNeXt upsample stages (dw causal conv +
   LayerNorm + GELU + LayerScale + residual), `in_conv` → 4 decoder
   blocks (SnakeBeta → transposed-conv stride 8/5/4/3 → 3× residual
   unit, dilations 1/3/9), final SnakeBeta → out_conv → clamp →
   `T_codec × 1920` samples @ 24 kHz. Single-pass decode; the chunked /
   sliding-window optimization from CrispASR (kept VRAM-constant for
   long sequences) is intentionally omitted for the initial port. The
   codec lives in its own sidecar GGUF (`cstr/qwen3-tts-tokenizer-12hz-GGUF`,
   discovered via `extras["codec_path"]` or `tokenizer-{f16,q8_0}.gguf`
   next to the talker); talker+predictor emit 32 codebooks and the codec
   consumes the first 16. Silence fallback retained for codec-less
   setups. See `docs/CODEC_PORTS.md` §2 for the architecture and
   substitution table.
2. **ECAPA-TDNN speaker encoder** — extracts a speaker embedding from a
   short reference clip for Voice Clone mode.
   **Stage 17b (complete):** `src/models/qwen3_tts/speaker_encoder.cpp` is
   an adaptation of CrispASR's ECAPA-TDNN section (MIT). Audiocore's
   `Qwen3TtsSpeakerEncoder` reproduces the CrispASR architecture verbatim
   — 128-mel→Conv1d→3×SE-Res2Net(d=2/3/4)→MFA→ASP→FC→speaker embedding.
   Mel spectrogram (n_fft=1024, hop=256, Hann, magnitude STFT, Slaney
   mel filterbank, ln) is computed in host code. The speaker encoder
   weights live in the talker GGUF under the `speaker.*` namespace; the
   loader opens the talker file with a separate GgufReader to resolve them
   (llama.cpp skips unrecognized tensor names). Voice Clone injects the
   embedding into the codec bridge at the speaker-position slot (same
   mechanism as CustomVoice's speaker_name lookup, but with an on-the-fly
   ECAPA computation from the reference WAV). WAV input must be 24 kHz
   mono 16-bit PCM. Silence fallback retained for setups without the
   `speaker.*` tensors (Voice Clone fails fast with a clear error).

**Dependencies to close:** Full ICL (in-context learning) prefill builder
that pairs the speaker embedding with ref-text AND ref-codes for improved
clone fidelity — Stage 18 added the xvec_only ICL path (ref-text via text_proj,
injected between spk and codec_bos), which gives the talker phonetic context.
The full ICL path with ref-codes (requiring the codec encoder to produce them
locally from the reference WAV) remains unported.

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
| Chunked / streaming HTTP for TTS | ✅ | `/v1/audio/speech/stream` with per-family incremental decode. MOSS (Stage 18) and Qwen3-TTS (Stage 19) both emit frames during the AR loop via stream callback. WAV/MP3 chunked response. |
| Models manifest (`models/manifest.json`) | ✅ Stage 9: central record with HF repo/revision/modes for every family/variant |
| Fetch tool (`scripts/fetch_models.sh`) | ✅ Stage 9: pure bash + curl downloader; reads the manifest, optional sha256 verify, invokes converters |
| Unified `TtsRequest` (Stage 6) | ✅ Single struct, single server branch |
| Unified `qwen3::Runner` (Stage 2) | ✅ One backbone path for 4 transformers |
| Unified `audiocore::sampler` (Stage 5) | ✅ One sampler for all families + MTP |
| `WeightLoader` everywhere (Stage 4) | ✅ No `gguf_*` calls outside `src/framework/io/` |
| `--list-supported` self-describe flag | ✅ Stage 9: `audiocore_cli --list-supported` reads the manifest and prints the matrix; `--list-families` prints the registered set |
| WAV output | ✅ 16-bit PCM mono (TTS) and stereo (music) at family-native rates |
| MP3 output | ✅ libmp3lame, `ENGINE_ENABLE_MP3` (default ON) |

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

### Large (major ports — now scoped, references identified)

15. ✅ **Stage 16 — MOSS-Audio-Tokenizer ggml port** — `src/models/moss_tts/codec.cpp`
    adapts `pwilkin/openmoss/src/codec.cpp` (Apache-2.0) onto audiocore's
    `Backend`/`TensorStorage`/`WeightLoader`. Auto-activates when the GGUF
    carries `moss.codec.*` tensors; silence fallback otherwise. Unblocks
    MOSS tts / sfx / voice_clone / dialogue / voice_design end-to-end
    (when fed codec-bearing weights like `smcleod/MOSS-TTS-v1.5-GGUF`).
16. ✅ **Stage 17 — Qwen3-TTS-Tokenizer-12Hz ggml port** — `src/models/qwen3_tts/codec.cpp`
    adapts `CrispStrobe/CrispASR`'s `qwen3_tts.cpp` codec section (MIT) onto
    audiocore's `Backend`/`TensorStorage`/`WeightLoader`. Auto-activates
    when the codec sidecar GGUF is discovered (`extras["codec_path"]` or
    `tokenizer-{f16,q8_0}.gguf` next to the talker); silence fallback
    otherwise. Unblocks Qwen3-TTS batch TTS / style instructions /
    voice_design end-to-end.
17. ✅ **Stage 17b — ECAPA-TDNN speaker encoder port** — `src/models/qwen3_tts/speaker_encoder.cpp`
    adapts CrispASR's ECAPA-TDNN section (MIT) onto audiocore's Backend /
    TensorStorage abstractions. Weights live in the talker GGUF (`speaker.*`
    tensors). Auto-activates when the talker GGUF carries `speaker.*` tensors;
    Voice Clone mode fails fast with a clear error otherwise. Unblocks
    Qwen3-TTS Voice Clone end-to-end.
18. 🚧 **ACE-Step Stem Extraction / Layering** — separate model family
    (Demucs-class separator + mixer). New family, not an ACE-Step mode.
19. ✅ **MP3 output** — `ENGINE_ENABLE_MP3` (default ON) links libmp3lame.
    New `response_format` field on TTS/Music request ("wav" | "mp3").

---

## 6. Coverage summary

Of the **5 + 5 + 6 = 16 advertised user-facing modes** across the three
families, after Stages 9–19:

- ✅ fully wired end-to-end: **10** (ACE-Step Text-to-Music; MOSS tts /
  sfx / voice_clone / realtime — when fed codec-bearing weights like
  `smcleod/MOSS-TTS-v1.5-GGUF`; Qwen3-TTS batch TTS / TTS with style
  instructions / voice_design / voice_clone / streaming — when fed a codec
  sidecar like `cstr/qwen3-tts-tokenizer-12hz-GGUF` AND the talker GGUF
  carries speaker encoder tensors; Voice Clone falls back to a clear
  fail-fast error with a GAPS.md pointer when the speech encoder is absent)
- 🟡 parse + run, partial elsewhere: **2** (MOSS dialogue / voice_design
  — codec wired but mode-specific surface partial)
- ❌ not implemented, achievable without new weights: **3** (ACE-Step
  cover / repaint / completion — all need the DiT to accept an extra
  conditioning signal beyond text)
- ❌ not implemented, separate model: **2** (ACE-Step stem extraction,
  layering — blocked on a new Demucs-class / stem-assembler family)
- ✅ per-frame streaming: **2** (MOSS realtime + Qwen3-TTS streaming —
  both emit frames during AR loop via callback; response PCM empty)
- 🚧 blocked on major port: **1** (ACE-Step stem/lego — separate
  Demucs-class model)
- 📋 reference impl identified, port scoped: **0** (all ported) — minor
  refinements remain: full ICL prefill builder (ref-text + ref-codes) for
  improved Voice Clone fidelity; MOSS dialogue multi-turn input surface

Stage 9 closed the entire IaC bucket. Stages 10–12 flipped six
modes from ❌ to 🟡 (Qwen3-TTS voice_design, MOSS dialogue, MOSS
voice_design, MOSS/Qwen3-TTS realtime/streaming — the last via the
transport scaffold). Stage 13 added explicit `mode` parsing to ACE-Step
so the unimplemented modes fail fast with a GAPS.md pointer instead of
silently running text-to-music. **Stage 15 identified production-quality
GGML reference implementations (Apache-2.0 + MIT) for both blocked
codecs, plus pre-built GGUFs that skip the converter step entirely.
Stage 16 ported the MOSS-Audio-Tokenizer; Stage 17 ports the
Qwen3-TTS-Tokenizer-12Hz. Stages 18–19 added per-frame streaming for both
MOSS and Qwen3-TTS.**

The remaining work is concentrated in the following places:

1. **DiT conditioning extension** — ACE-Step cover / repaint / completion.
   Graph-level change in `dit_runner.cpp`, no new weights. Medium effort.
2. **Full ICL prefill builder** — Voice Clone now supports the xvec_only
   ICL path (Stage 18: ref-text phonetic context via text_proj). The full
   ICL (in-context learning) path with ref-codes pairs the embedding with
   both ref-text AND ref-codes for improved clone fidelity, as CrispASR's
   `build_icl_prefill_embeds` implements. Requires the codec encoder to
   produce ref-codes locally from the reference WAV.
