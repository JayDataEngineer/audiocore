# audiocore — Honest Gap Tracker

> **Purpose.** This file tracks what is *actually* working, what is *partially*
> working, what is a documented fallback, and what is an **outright fraud**
> (code that pretends to support a feature it does not, by silently routing
> the request through an unrelated model with a system prompt).
>
> The rule: if a feature path emits audible output but uses the wrong model
> or a system-prompt disguise, it is a **fraud** and must be marked as such
> — not as a "fallback." Frauds fail-fast in any quality-sensitive use case.
>
> Last updated: 2026-06-30.

---

## 0. Legend

| Tag | Meaning |
|---|---|
| ✅ WORKS | Verified end-to-end against real weights; output quality matches upstream. |
| 🟡 PARTIAL | Real implementation but incomplete (missing modes, untested configs, etc.). |
| 🔇 SILENCE | Returns silent PCM by design when a tensor file is missing. Documented. |
| ⚠️ FRAUD | Routes the request through a *different model* with a system prompt disguise. Output is intelligible but is NOT what the request asked for. |
| ❌ MISSING | No code path at all; the request fails fast or silently no-ops. |
| 💤 DORMANT | Weights are on disk but no test ever loads them. Code may or may not exist. |

---

## 1. MOSS-TTS family (`src/models/moss_tts/`)

### 1.0 Upstream model matrix — what the family promises vs what we ship

Upstream MOSS is an **8-model pipeline** (per OpenMOSS / MOSI.AI). We ship 1.

| Upstream model | Purpose | On disk? | Code path? | Test? | Status |
|---|---|---|---|---|---|
| **MOSS-TTS** 8B (v1, v1.5) | Flagship TTS + zero-shot clone, 30+ languages | ✅ v1 + v1.5 GGUF | ✅ Delay architecture | 🟡 1 happy-path test | 🟡 PARTIAL |
| **MOSS-TTSD** 8B | Multi-speaker spoken dialogue | ❌ Never downloaded | ❌ No code path | ❌ None | ⚠️ FRAUD #4 (`mode="dialogue"` runs flagship TTS instead) |
| **MOSS-VoiceGenerator** 1.7B | Voice design from text description | ❌ Never downloaded | ❌ No code path | ❌ None | ⚠️ FRAUD #3 (`mode="voice_design"` runs flagship TTS instead) |
| **MOSS-SoundEffect-v1** 8B | Text-to-SFX (Delay backbone) | ❌ Never downloaded | ❌ No code path | ❌ None | ⚠️ FRAUD #2 path A |
| **MOSS-SoundEffect-v2.0** 1.3B | Text-to-SFX (DiT + Flow Matching) | ✅ Raw safetensors at `moss-soundeffect-v2/` | ❌ No converter, no DiT loader | ❌ None | ⚠️ FRAUD #2 path B (`mode="sfx"` runs flagship TTS instead of either SFX model) |
| **MOSS-TTS-Realtime** 1.7B | Incremental-synth voice agent backend | ❌ Never downloaded | ❌ No code path | ❌ None | ⚠️ FRAUD #5 (`mode="realtime"` streams the Delay flagship) |
| **MOSS-TTS-Nano-100M** | CPU/edge TTS | ❌ Never downloaded | ❌ No pure-AR nano code | ❌ None | ❌ MISSING |
| **MOSS-Audio-Tokenizer** v1 + v2 | Codec (CNN-free causal Transformer) | ✅ v1 safetensors, ✅ v2 safetensors | ✅ Decoder in `codec.cpp`; 🟡 Encoder ported 2026-06-29 but **not wired** | 🟡 Decoder exercised by flagship TTS test | 🟡 PARTIAL |

**Result: 1 of 8 upstream models actually runs.** Six of the remaining seven
have no code path at all and three are actively impersonated by the flagship
backbone via system-prompt hacks (§1.2).

### 1.1 Architecture coverage — **1 of 4 upstream architectures implemented**

| Upstream architecture | Released models | Our coverage |
|---|---|---|
| **MossTTSDelay** (multi-head parallel RVQ + delay pattern) | MOSS-TTS-v1, MOSS-TTS-v1.5, MOSS-TTSD, MOSS-VoiceGenerator, MOSS-SoundEffect-v1 | 🟡 We load **only the flagship MOSS-TTS-v1.5 backbone**. Other Delay checkpoints (TTSD, VoiceGenerator, SFX-v1) are NOT separately loaded. |
| **MossTTSLocal** (time-synchronous RVQ + depth transformer) | MOSS-TTS-Local-Transformer-v1 (1.7B), v1.5 (4B, 48 kHz stereo) | ❌ No code. |
| **MossTTSRealtime** (hierarchical text+audio for incremental synth) | MOSS-TTS-Realtime-1.7B | ❌ No code. `mode="realtime"` runs the Delay flagship with a streaming callback — NOT the Realtime architecture. |
| **MossSoundEffectPipeline** (DiT + Flow Matching) | MOSS-SoundEffect-v2-1.3B | ❌ No code. Weights are on disk at `/mnt/data/models/audio/moss-soundeffect-v2/` and **untouched** (safetensors, no converter written). |
| (pure-AR nano) | MOSS-TTS-Nano-100M | ❌ No code. |

### 1.2 Mode-by-mode on the **flagship** backbone

Confirmed by reading `src/models/moss_tts/session.cpp` lines 209, 221–239:

```cpp
case Mode::Sfx:         system_prompt = "You are a sound effects generator.";      break;
case Mode::Dialogue:    system_prompt = "You are a spoken-dialogue assistant…";    break;
case Mode::VoiceDesign: system_prompt = "You are a voice cloning assistant. Speak…"; break;
case Mode::Realtime:
case Mode::Tts:
default:                system_prompt = "You are a helpful voice assistant.";     break;
```

These strings are passed to `backbone_->apply_chat_template(msgs, …)` and used
as the system message. The model is **not** the one the user requested.

| `mode=` | Tag | What happens | Honest status |
|---|---|---|---|
| `tts` | plain TTS | Real flagship backbone + real codec decoder. Single happy-path e2e test passes. | 🟡 PARTIAL — prompt builder uses the **chat template + system prompt** hack (see §1.3), not the upstream `<user_inst>` template. Output is intelligible but does not match upstream prompting → wrong pause/style behaviour, wrong instruction slotting. |
| `voice_clone` | voice cloning | Loads a pre-encoded **`.codes` binary file** from `voice_path`. | ⚠️ FRAUD #1. Upstream cloning takes a **WAV**, runs it through the codec ENCODER, and delay-pattern-splices the codes into the user-side audio block of the chat prompt. We skip the encoder entirely and load raw int32 codes from disk — which the user cannot produce without a separate Python tool. The clone quality is therefore untested against upstream. **Codec encoder port is in progress** (see `codec.cpp` `encode()`); once it lands this mode still needs the upstream prompt builder to be honest. |
| `sfx` (sound effects) | sound effects | Runs the **flagship TTS backbone** with system prompt `"You are a sound effects generator."` | ⚠️ FRAUD #2. There are **two** real upstream SFX models (v1 Delay 8B, v2 DiT 1.3B). We load neither. We ask the speech model to "be a sound model." Output is muffled speech that describes the sound, not the sound. Sampling defaults (`audio_temperature=1.2`, `top_k=20`) were fabricated with no upstream basis. |
| `voice_design` | voice design | Runs the **flagship TTS backbone** with system prompt `"You are a voice cloning assistant. Speak the user's text in a voice matching this description: <instruct>."` | ⚠️ FRAUD #3. There is a real `MOSS-VoiceGenerator` 1.7B Delay checkpoint. We do not load it. We pass the voice description as a system prompt to the TTS model and hope it changes its voice. It does not, in any controlled way. Code comment at `session.cpp:207` admits this. |
| `dialogue` | spoken dialogue (MOSS-TTSD) | Runs the **flagship TTS backbone** with system prompt `"You are a spoken-dialogue assistant. Continue the conversation..."`. Optional `messages` array is fed through the chat template. | ⚠️ FRAUD #4. There is a real `MOSS-TTSD-v1.0` 8B model (separate checkpoint, same Delay architecture). We do not load it. Sampling defaults (`text_temperature=1.6`, `audio_top_p=0.85`, `repetition_penalty=1.05`) were fabricated with no upstream basis. |
| `realtime` / `streaming` | streaming TTS | Runs the **flagship Delay backbone** with an incremental codec-decode callback that fires after the n_vq-step delay window. The AR loop emits frames at ~80 ms each. | ⚠️ FRAUD #5. There is a real `MossTTSRealtime` 1.7B architecture with hierarchical text-audio inputs designed for incremental synth (TTFB ~180 ms). We do not load it. We are streaming the Delay backbone, which has ~1.3 s cold-start (n_vq × 80 ms) — acceptable for batch, wrong for voice agents. |

### 1.3 The prompt-builder hack (affects every flagship mode)

The upstream `processing_moss_tts.py` builds the user turn as:

```
<|im_start|>user
<user_inst>
- Reference(s):
[S1]:
<audio_start><|audio_user_slot|>...<|audio_delay_slot|>...<audio_end>
- Instruction:
- Tokens:
- Quality:
- Sound Event: None
- Ambient Sound: None
- Language: en
- Text: <the actual text>
</user_inst><|im_end|>
<|im_start|>assistant
<audio_start>
```

Our `session.cpp` instead calls `backbone_->apply_chat_template()` with a
generic `messages` list (system + user), prepends one of the mode-specific
system prompts above, then appends `<audio_start>`. This means:

- **No `<user_inst>` block** → the model never sees the Reference/Instruction/
  Tokens/Quality/Sound Event/Ambient Sound/Language/Text slots upstream trained
  on. Style and language hints are ignored at the architectural level.
- **No reference-audio splice** → even if voice cloning loaded codes, they would
  not be inserted at the `[S1]:` position with delay-pattern shift; they go into
  ad-hoc "GEN_SLOT + codec tokens" rows appended after the text tokens, which
  does not match upstream positional encodings.
- **`instruction` and `language` request fields are dropped on the floor** for
  the prompt-template path; only the fraud `voice_design` mode uses `instruct`,
  and only as a system-prompt string.

The fix is to port `openmoss/src/pipeline.cpp` (`build_user_inst`,
`build_prompt_text`, `build_prompt_grid`, `apply_delay_pattern`) verbatim.
~150 lines. **Status: NOT DONE.**

### 1.4 Feature gaps vs upstream

| Upstream feature | Our status |
|---|---|
| **Token-level duration control** (`- Tokens:` slot carries per-token duration hints) | ❌ MISSING. The request struct has no `tokens` field; the `- Tokens:` slot is never written. |
| **Pinyin control** (phoneme-level override) | ❌ MISSING. No `pinyin` field in `MossTtsRequest`. |
| **Instruction-slot control** (style/emotion via `- Instruction:`) | ❌ MISSING. `req.instruct` exists but is only routed to the fraud `voice_design` system prompt; never placed in the upstream `<user_inst>` slot. |
| **Quality slot** (`- Quality:`) | ❌ MISSING. No field, no slot write. |
| **Sound Event / Ambient Sound slots** (non-speech audio embedding within TTS) | ❌ MISSING. Both slots hard-coded to `"None"`. |
| **Language slot** | ❌ MISSING — `req.language` is never written into the prompt; the chat template does not know about it. |
| **Multilingual (30+ languages in v1.5)** | ❌ UNTESTED — the only e2e test is English. Even if the backbone supports it, the prompt-builder hack (no `- Language:` slot) likely breaks code-switching. |
| **Modular pipeline** ("design voice with VG, speak with TTS, layer SFX") | ❌ MISSING — there is no API surface to chain models, and only one model is loadable anyway. |
| **Reference-audio splice** (the `[S1]:` block in `<user_inst>`) | ❌ MISSING until prompt-builder port + codec-encoder wiring land. |
| **Ultra-long stable speech** (MOSS-TTS strength) | ❌ UNTESTED — test caps `max_new_tokens`. No long-form test. |

### 1.5 Other MOSS gaps

- 🔊 **Codec encoder** (needed for real voice cloning): ported in
  `codec.cpp::encode()` on 2026-06-29 but **not yet wired** into
  `session.cpp` — `voice_clone` still loads `.codes` files.
- 🔊 **Codec decoder silence fallback**: documented — if the GGUF has no
  `moss.codec.*` tensors, `decode_codec()` returns 1 s of silence rather
  than failing. Test `test_moss_e2e.cpp` accepts `rms > 1e-6` as pass and
  prints a `[WARN] silent output` otherwise, which historically masked
  broken codec binds.
- 🔇 **Standalone Audio-Tokenizer tool**: upstream ships the tokenizer as a
  reusable component (`MOSS-Audio-Tokenizer` v1, v2). We expose it only as
  an internal detail of `moss_tts::session`; there is no `encode_wav →
  codes` CLI, no `decode_codes → wav` CLI. Blocks the community GGUF/MLX
  porting workflow that upstream enables.
- 📉 **Delay state machine** uses `INT_MAX` sentinel with a subtly different
  off-by-one from `openmoss/src/delay.cpp` (we use `T = total_len - N_VQ + 1`,
  openmoss uses `T_audio = T - n_vq`). Outputs happen to match because both
  conventions drop the same flush frame, but the divergence is unflagged.
- 🚫 **Multi-GPU** unsupported: we rely on libllama defaults;
  `openmoss/src/model.cpp::pick_gpu_device` does an explicit "most-free-VRAM"
  pick and pins via `LLAMA_SPLIT_MODE_NONE`.
- 🚫 **CPU/edge coverage** (MOSS-TTS-Nano): zero. We have no Nano code path,
  no Nano GGUF, and no plan to support pure-AR-100M inference. The family
  entirely excludes the on-device use case upstream advertises.

---

## 2. ACE-Step family (`src/models/ace_step/`)

### 2.0 Upstream model matrix — 7 checkpoints on disk, 1 ever exercised

Upstream ACE-Step (ACE Studio × StepFun) ships the checkpoints below. All
seven GGUFs are present on disk at `/mnt/data/models/audio/acestep-cpp/`,
plus the 0.6B text encoder and the VAE.

| Checkpoint on disk | Upstream role | Loaded by `AceStepRunner`? | Test coverage |
|---|---|---|---|
| `acestep-v15-turbo-Q8_0.gguf` | v1.5 3.5B turbo (8-step) | ✅ Yes | ✅ 4 configs (the only one tested) |
| `acestep-v15-sft-Q8_0.gguf` | v1.5 SFT (close style/lyric alignment) | ✅ Yes | ✅ Exercised 2026-06-30 (RMS=0.022915) |
| `acestep-v15-xl-base-Q8_0.gguf` | XL 4B base | ✅ Yes | ✅ Exercised 2026-06-30 (RMS=0.022984) — required condition_embedder bias-repeat fix |
| `acestep-v15-xl-sft-Q8_0.gguf` | XL 4B SFT | ✅ Yes | ✅ Exercised 2026-06-30 (RMS=0.022984) |
| `acestep-v15-xl-turbo-Q8_0.gguf` | XL 4B turbo | ✅ Yes | ✅ Exercised 2026-06-30 (RMS=0.022987) |
| `acestep-5Hz-lm-1.7B-Q8_0.gguf` | v1 1.7B LM | ✅ Yes | ✅ Exercised via all above tests |
| `acestep-5Hz-lm-4B-Q8_0.gguf` | v1 4B LM | ✅ Yes (after `convert_acestep`) | ✅ Exercised 2026-06-30 with turbo DiT (RMS=0.022914) |
| `Qwen3-Embedding-0.6B-Q8_0.gguf` | Text encoder (TE) | ✅ Yes | ✅ Exercised via turbo test |
| `vae-BF16.gguf` | Music VAE | ✅ Yes | ✅ Exercised via turbo test |

**Result: all 7 music checkpoints verified.** The `find_gguf` substring-match
bug (turbo also matched xl-turbo) was fixed via exact variant-tag extraction.
The XL DiTs required a condition_embedder bias-repeat fix (bias tensor is
`H`-dim not `encoder_hidden`-dim; was being repeated against `proj` output
instead of `ce_out` output — these coincide only when H==encoder_hidden=2048).
The 4B LM required conversion with `convert_acestep` to llama.cpp tensor names.

### 2.1 Upstream feature matrix

| Upstream feature | Our status |
|---|---|
| **Text-to-music** (tags + lyrics) | 🟡 PARTIAL — runs, but lyrics path only smoke-tested with one short prompt; tag/lyric adherence never compared to upstream output. |
| **Repainting** (select start/stop, regenerate that section only) | ✅ FIXED (rev 5) — ported HOT-Step repaint injection + boundary blend; latent-space Euler loop with per-step proj_in/DiT/proj_out. |
| **Style remixing** (upload song, change genre, keep melody) | ❌ MISSING. There is no `remix_strength` or `melody_retention` parameter; no path that re-encodes user audio, strips style, and re-decodes with new tags. |
| **Vocals-to-BGM** | ❌ MISSING. No vocal-removal path. |
| **Multilingual lyric alignment (50+ languages, stochastic Romanization)** | ❌ UNTESTED. Only English lyrics in the one test prompt. The TE may or may not handle Romanization; we have no test. |
| **Structured lyric tags** (`[verse]`, `[chorus]`, `[bridge]`) | ❌ UNTESTED — no test parses structural markers; tag handling in `family.h:53` is just a string field, never validated against the TE's expected markup. |
| **LoRA adapter loading** | ❌ MISSING — no LoRA path in `AceStepRunner`; upstream's `ACEStep-XL-Regrind-V1` (a three-file resonance-suppression package: baked base + VAE decoder regrind + LoRA) cannot be applied. The regrind VAE is on the roadmap in §5 but not shipped. |
| **BPM / key-scale control** | ❌ UNTESTED — request fields exist but no test verifies that `bpm=120 key=C minor` actually changes the output. |
| **Long-track resonance suppression** (the Regrind package's whole point) | ❌ MISSING — without LoRA support, long tracks accumulate the harmonic hum Regrind was published to fix. |

### 2.2 Modes

| Mode | Status |
|---|---|
| `text_to_music` | ✅ WORKS — RMS ≈ 0.0229, correct patchify stride=P=2, channel layout `[src(64)|mask(64)|xt(64)]`, per-step proj_in+DiT+proj_out with latent-space Euler. |
| `repaint` | ✅ FIXED (rev 5) — ported HOT-Step `sampler_repaint_inject` + `sampler_repaint_blend`. Context building matches `ops_build_context`: silence src + mask=1 inside zone, actual latent + mask=0 outside. Per-step injection on xt for first 50% of steps. |
| `completion` | ✅ FIXED (rev 5) — same code path as repaint (mask covers end of track). |
| `cover` | ✅ FIXED (rev 5) — correct channel layout with cover_latent as src, mask=1.0 everywhere. |
| `stem` | ❌ MISSING — separate model family (Demucs). |
| `lego` | ❌ MISSING — separate stem assembler. |

### 2.3 Variants / backends

| Variant | Status |
|---|---|
| v1.5 turbo (3.5B, 8 steps) | ✅ WORKS — all 4 test configs use this. |
| v1.5 SFT (3.5B, 50 steps) | ✅ WORKS — exercised 2026-06-30, RMS=0.022915. |
| XL base (4B) | ✅ WORKS — exercised 2026-06-30, RMS=0.022984. Condition_embedder bias-repeat bug fixed. |
| XL SFT (4B) | ✅ WORKS — exercised 2026-06-30, RMS=0.022984. |
| XL turbo (4B) | ✅ WORKS — exercised 2026-06-30, RMS=0.022987. find_gguf variant-tag bug fixed. |
| v1 5Hz-LM 1.7B | ✅ WORKS — exercised as default LM in all above tests. |
| v1 5Hz-LM 4B | ✅ WORKS — exercised 2026-06-30 after convert_acestep renaming. |
| CUDA backend | ✅ WORKS. |
| CPU backend (`ACESTEP_DEVICE=ggml_cpu`) | ❌ Untested. |
| Duration > 20 s | ❌ Untested — 4 GB VAE scratch buffer would OOM; **tiled VAE decode NOT implemented**. |
| Sampling temperature > 0 | ❌ Untested — all e2e tests use `temperature=0` (argmax). |

### 2.4 DiT alignment with reference (`ServeurpersoCom/acestep.cpp`)

The DiT graph works (produces non-silent audio) but diverges from the
reference implementation in three places that affect quality at the
margin:

1. **Missing SiLU before `time_proj`** — reference applies `silu(time_embedder(t))` before the linear time projection; we apply the linear directly.
2. **Missing timestep scaling by 1000** — reference scales the input timestep `t → t * 1000` before passing to the embedding; we pass the raw [0,1] value.
3. **Missing F16 K/V cast for `flash_attn_ext`** — reference casts K/V to F16 inside attention; we leave them at F32. This works in CUDA but diverges from the validated kernel path.

These do not break the pipeline (output RMS is healthy) but they shift the
noise schedule and attention numerics away from upstream. **Status: known
divergence, not yet fixed.**

---

## 3. Qwen3-TTS family (`src/models/qwen3_tts/`)

### 3.0 Upstream model matrix — 5 variants upstream, 1 on disk, 1 exercised

Upstream Qwen3-TTS (Alibaba) ships the variants below.

| Upstream variant | Purpose | On disk? | Loaded? | Test? |
|---|---|---|---|---|
| `Qwen3-TTS-12Hz-1.7B-Base` | Flagship zero-shot clone | ✅ (`talker.q5_k.gguf` + `predictor.q8_0.gguf`) | ✅ | 🟡 1 happy-path test |
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | Novel voice from text description | ❌ Never downloaded | ❌ | ⚠️ FRAUD #7 (see §3.1) |
| `Qwen3-TTS-12Hz-1.7B-CustomVoice` | Preset timbres, deep style control | ❌ Never downloaded | ❌ | ❌ MISSING |
| `Qwen3-TTS-12Hz-0.6B-Base` | Lightweight zero-shot clone | ❌ Never downloaded | ❌ | ❌ MISSING |
| `Qwen3-TTS-12Hz-0.6B-CustomVoice` | Lightweight preset voice | ❌ Never downloaded | ❌ | ❌ MISSING |
| `Qwen-TTS-Tokenizer-12Hz` | 16-codebook, 12.5 Hz streaming codec | ✅ Both f16 + q8_0 | ✅ q8_0 in test | 🟡 Exercised via Base test |
| `Qwen-TTS-Tokenizer-25Hz` | Single-codebook, Qwen-Audio integration | ❌ Never downloaded | ❌ | ❌ MISSING |

**Result: 1 of 5 talker variants runs, with 1 codec (the 12 Hz) and 1 happy-path
test.** CustomVoice, VoiceDesign, both 0.6B variants, and the 25 Hz codec are
entirely absent.

### 3.1 Modes

| Mode | Status |
|---|---|
| `TtsBatch` (plain batch TTS) | 🟡 PARTIAL — real codec path works; **single happy-path test** in `test_qwen3tts_e2e.cpp` (text="Hello, this is a test…", max_new_tokens=50, speed=1.0, language="en"). No variation across language, speed, temperature, or voice. |
| `VoiceDesign` | ⚠️ FRAUD #7 — confirmed at `src/models/qwen3_tts/session.cpp:120–124` and `:244–252`. The code literally prepends a fixed prefix to `req.instruct` and runs it through the **Base talker** as an ordinary instruct string. The comment admits: *"a best-effort fallback — output will be intelligible but less polished"*. Real VoiceDesign needs the `Qwen3-TTS-12Hz-1.7B-VoiceDesign` checkpoint, which is not on disk and not downloadable by the runner. |
| `VoiceClone` (ECAPA-TDNN speaker encoder, Stage 17b) | 🟡 PARTIAL — the speaker encoder exists (`speaker_encoder.cpp`, 569 lines) and unit tests pass, but the e2e voice-clone path is **not exercised by any test**. Reference-audio → embedding → conditioning is wired in `session.cpp` but not validated end-to-end. Also: upstream advertises **3-second minimum reference audio** for cloning; we have no test that verifies the minimum length is enforced or that shorter clips fail gracefully. |
| `Streaming` (Stage 19) | 🟡 PARTIAL — implemented, not tested. The e2e test does not set `req.stream`. **Critical:** upstream advertises a **Dual-Track hybrid streaming architecture** with end-to-end latency as low as **97 ms** (audio output starts after receiving a single character). Our path is a single-track incremental codec-decode callback — there is no dual-track (semantic-track + acoustic-track) split, no character-level text ingestion, no 97 ms TTFB measurement, no test. |

### 3.2 Variants

| Variant | Status |
|---|---|
| 1.7B Base | 🟡 PARTIAL — single config verified (see above). |
| 1.7B VoiceDesign | ❌ Weights not on disk. Fraud-system-prompt path runs instead. |
| 1.7B CustomVoice | ❌ Weights not on disk. No code path. |
| 0.6B Base | ❌ Weights not on disk. No code path. |
| 0.6B CustomVoice | ❌ Weights not on disk. No code path. |

### 3.3 Feature gaps vs upstream

| Upstream feature | Our status |
|---|---|
| **10-language support** (zh, en, ja, ko, de, fr, ru, pt, es, it) | ❌ UNTESTED — test uses `language="en"` only. No multilingual smoke test, no language-tag routing. |
| **Regional dialects** (Cantonese, Sichuanese, Wu, Hokkien) | ❌ MISSING — no dialect field, no test, no dialect-tagged prompting. |
| **97 ms Dual-Track streaming** | ❌ MISSING — see §3.1. Single-track only. No latency benchmark. |
| **Adaptive Contextual Expressiveness** (rate/emotion/prosody from semantic cues or NL prompts like "speak happily", "whisper softly") | ❌ MISSING — `req.instruct` exists but is not validated for emotion control; no test asserts that "speak happily" changes f0/energy. |
| **3-second minimum reference for zero-shot clone** | ❌ UNTESTED — no length check, no test with short references. |
| **VoiceDesign from natural-language description** | ⚠️ FRAUD #7 — see §3.1. |
| **CustomVoice preset timbres** | ❌ MISSING — no preset voice registry, no preset-id routing. |
| **25 Hz codec integration** (with Qwen-Audio) | ❌ MISSING — only 12 Hz codec loadable. |

### 3.4 Other Qwen3 gaps

- 🔇 **Silence fallbacks everywhere** — `session.cpp` falls back to 1 s of
  silence in **five** places:
  - `n_frames <= 0` after AR loop
  - codec not bound
  - codec expects more codebooks than the talker produces
  - codec decode exception
  - (older) Lunavox no-text-embedding branch
- ⚠️ **Lunavox fallback** — when the talker has no text embedding table,
  we fall back to token-mode with `codec_bos`. This is a community-model
  workaround that has nothing to do with upstream Qwen3-TTS; it is kept
  alive for `wkwong/Lunavox-Qwen3-TTS-GGUF` (the only publicly available
  talker GGUF). The official Qwen3-TTS talker GGUF is **not tested**.
- 📉 **Codec/talker codebook mismatch** — talker+predictor produce 32 codebooks
  (QwenLM original architecture); the 12Hz codec consumes only 16. We slice
  the first 16 and discard the rest. Unverified against upstream truncation
  behavior.

---

## 4. Cross-family systemic issues

1. **No upstream prompt-template port for MOSS.** This is the largest single
   fraud. The flagship backbone is loaded correctly but is being prompted in a
   way the model was not trained on. Until `openmoss/src/pipeline.cpp` is
   ported, **every MOSS mode produces off-distribution output**, including the
   one mode (`tts`) that we call WORKS.

2. **Silence is treated as success.** Multiple test harnesses (`test_moss_e2e`,
   `test_qwen3tts_e2e` historically) accept `rms > 1e-6` or print `[WARN]` and
   return 0. This hides broken codec binds and broken AR loops. A real test
   must compare against a baseline RMS or mel-similarity, not just "non-zero."

3. **Mode strings without model loading.** `TtsRequest::mode` is a string that
   the family interprets however it likes. There is no contract that
   `mode="voice_design"` actually loads the VoiceDesign checkpoint. The
   unified API surface advertises capabilities the families do not have.

4. **Codec encoder is not a first-class citizen.** Until 2026-06-29 the MOSS
   encoder was "intentionally NOT ported" (codec.h comment). Voice cloning
   across families is therefore broken: MOSS does it via `.codes` files,
   Qwen3-TTS does it via speaker embeddings, neither does it the way upstream
   does (WAV → codec encoder → splice).

5. **No regression baseline.** Tests check "non-silent," never "matches
   upstream output within X dB." Output quality drift cannot be detected.

6. **Dead-weight checkpoints.** 6 of 7 ACE-Step music checkpoints, 1 of 2
   ACE-Step VAEs (the Regrind package is not loadable), 4 of 5 Qwen3-TTS
   talker variants, 1 of 2 Qwen3-TTS codecs, 7 of 8 MOSS models — all on
   disk (or upstream-available) and never loaded by any test. The "test
   everything" directive is unmet by a factor of ~10x.

7. **No long-form test in any family.** Upstream MOSS sells "ultra-long stable
   speech"; upstream Qwen3 sells 97 ms streaming; upstream ACE-Step sells full
   songs. We test 5–20 s clips, max_new_tokens=50, and a single non-streaming
   batch. The hard cases (where these models are supposed to shine) are
   exactly the cases we do not touch.

8. **No multilingual validation in any family.** MOSS advertises 30+
   languages, Qwen3 advertises 10 + dialects, ACE-Step advertises 50+. We test
   English only across the board.

9. **No upstream-feature exposure for control surfaces.** MOSS's
   Tokens/Pinyin/Quality/Sound-Event/Ambient-Sound slots; Qwen3's dialect tags,
   emotion instruct presets, CustomVoice registry, dual-track streaming;
   ACE-Step's BPM/key adherence, lyric structural tags, remix_strength,
   melody_retention, LoRA adapters — **none of these are reachable through
   our request structs**, even though they are the headline features upstream
   advertises.

---

## 5. Priorities (what to do first)

Ranked by "how much of the fraud surface area" the fix removes:

1. **Port `openmoss/src/pipeline.cpp` into MOSS `session.cpp`.** Fixes
   FRAUDs #1 (partially — still need encoder wiring), the prompt-builder
   hack, and makes `mode="tts"` actually honest. Also unblocks Tokens /
   Pinyin / Quality / Sound Event / Ambient Sound / Language slots.
2. **Wire the MOSS codec encoder into `voice_clone`** (read WAV via
   `reference_audio`, encode, delay-pattern splice). Completes the FRAUD #1
   fix.
3. **Delete `sfx`, `voice_design`, `dialogue`, `realtime` modes from MOSS**
   until we actually ship those checkpoints. Replace with fail-fast errors
   that name the missing model. Fixes FRAUDs #2, #3, #4, #5.
4. **Delete `voice_design` mode from Qwen3-TTS** until we ship a
   VoiceDesign checkpoint. Fixes FRAUD #7.
5. **Fix the ACE-Step repaint latent projection** (FRAUD #6) — ✅ DONE (rev 5).
   Ported HOT-Step's `sampler_repaint_inject`, `sampler_repaint_blend`, and
   the correct `[src(64)|mask(64)|xt(64)]` channel layout. Also fixed two
   deeper bugs: stride=1→stride=P=2 patchify, and hidden-space→latent-space
   Euler (proj_in+proj_out now run every step).
6. **Exercise dormant checkpoints.** ✅ DONE (2026-06-30) — all 6 dormant
   ACE-Step checkpoints exercised and verified: sft (RMS=0.022915), xl-base
   (RMS=0.022984), xl-sft (RMS=0.022984), xl-turbo (RMS=0.022987), lm-1.7B
   (default in all tests), lm-4B (RMS=0.022914 after convert_acestep).
   Two bugs fixed en route: `find_gguf` variant-tag substring collision
   (turbo matched xl-turbo), and condition_embedder bias-repeat target for
   XL variants (H=2560 ≠ encoder_hidden=2048). `ACESTEP_DIT_VARIANT` /
   `ACESTEP_LM_VARIANT` env vars added to `test_acestep_e2e`.
7. **Convert and ship MOSS-SoundEffect-v2.** Weights are on disk in
   safetensors form at `/mnt/data/models/audio/moss-soundeffect-v2/`.
   Reuses ACE-Step DiT infrastructure. Fixes FRAUD #2 for real.
8. **Replace silence fallbacks with fail-fast errors** in test harnesses
   and in MOSS/Qwen3 family code. A missing codec tensor is a configuration
   error, not a successful silence.
9. **Add a regression baseline** (RMS or mel-similarity window) so quality
   drift is detectable.
10. **DiT alignment with HOT-Step** — ✅ DONE (rev 5): SiLU before time_proj,
    timestep × 1000, fixed sin/cos embedding order + frequency formula.
    F16 K/V cast is not used by HOT-Step (only F16 masks); not applicable.
11. **Implement tiled VAE decode** for ACE-Step durations > 20 s.
12. **Multilingual + long-form test sweep.** At minimum: MOSS with zh/ja/de,
    Qwen3 with zh/ko/ja + a Cantonese dialect tag, ACE-Step with non-English
    lyrics + structural tags + ≥60 s generation. No family ships as
    "multilingual" without this.
13. **Download missing talker variants.** `Qwen3-TTS-12Hz-1.7B-VoiceDesign`
    and `-CustomVoice`, plus the 0.6B pair. Without weights the fraud modes
    cannot be made honest.
14. **Dual-Track streaming for Qwen3-TTS** (semantic + acoustic split) to
    approach the 97 ms TTFB upstream advertises. Current single-track path
    is not competitive for voice-agent use.

---

## 6. Changelog

- **2026-06-30 (rev 6) — ACE-Step dormant checkpoints exercised**:
  - **All 6 dormant ACE-Step checkpoints verified**: sft, xl-base, xl-sft,
    xl-turbo DiTs and lm-4B LM all produce audible non-silent output through
    the existing runner. No new architecture code required.
  - **`find_gguf` variant-tag bug fixed** (`loader.cpp`): naive substring
    matching would pick `acestep-v15-xl-turbo` when caller requested `turbo`
    (both contain the substring). Replaced with exact variant-tag extraction
    (strip pattern prefix + quant suffix) so each prefer= string matches only
    its intended file.
  - **Condition_embedder bias-repeat bug fixed** (`dit_runner.cpp`): the
    `ceb` tensor is `H`-dimensional (2048 for standard, 2560 for XL variants).
    It was being `ggml_repeat`'d against `proj` (the text_projector output,
    always 2048-dim), which asserts when H≠2048. Fixed by repeating against
    `ce_out` (the mul_mat output after the condition_embedder weight). Both
    the conditioned and null-condition branches were affected.
  - **LM 4B converted**: `convert_acestep` used to rename HF-style tensor
    names to llama.cpp convention. Converted file stored at
    `weights/ace_step/acestep-5Hz-lm-4B-Q8_0.gguf`.
  - **`ACESTEP_DIT_VARIANT` / `ACESTEP_LM_VARIANT` env vars** added to
    `test_acestep_e2e` for exercising specific variant combinations without
    rebuilding.

- **2026-06-29 (rev 5) — ACE-Step FRAUD #6 fixed + deep architectural bugs**:
  - **FRAUD #6 fixed** (repaint broken projection): the pad-and-conv1d hack
    is replaced with a full port of HOT-Step's repaint algorithm.
  - **Channel layout fixed** (affected ALL modes): changed from the wrong
    `[noise(64)|cond(64)|mask(64)]` to the correct `[src(64)|mask(64)|xt(64)]`
    per HOT-Step `dit-graph.h:462` and `pipeline-synth-ops.cpp:939–1064`.
  - **Patchify stride fixed**: `conv1d_k2_s1_p0` (stride=1, T→T-1 patches)
    replaced with `patchify_proj_in` (stride=P=2, T→T/P patches) and
    `unpatchify_proj_out` (ConvTranspose1d equivalent, T/P→T frames).
  - **Euler step moved to latent space**: proj_in→DiT→proj_out now runs every
    step (velocity prediction in latent space), matching HOT-Step's loop.
  - **Repaint injection + boundary blend ported verbatim** from
    `sampler-repaint.h`.
  - **Timestep embedding fixed** (3 bugs):
    - Sin/cos swapped: our code had sin first, cos second; ggml (and
      HOT-Step) has cos first, sin second.
    - Frequency denominator: was `/(half-1)`, should be `/half` per
      `ggml_compute_forward_timestep_embedding_f32`.
    - Missing `× 1000` timestep scaling (diffusion convention).
  - **SiLU before time_proj**: was missing; now matches
    HOT-Step `dit-graph.h:144`.
  - **E2E verified**: `test_acestep_e2e` produces 10.0 s @ RMS 0.0229.
    MOSS and Qwen3 tests unchanged.
- **2026-06-29 (rev 4) — Qwen3 frauds fixed**:
  - **FRAUD #7 fixed** (`voice_design` instruct-prefix fraud): the
    `kVoiceDesignInstructPrefix` template string and the silent-acceptance
    block at `qwen3_tts/session.cpp:120–127, 244–254` are deleted.
    `voice_design` mode now fails fast unless the loaded talker variant is
    `Qwen3TtsVariant::VoiceDesign`, with an error naming the missing
    `Qwen3-TTS-12Hz-1.7B-VoiceDesign` checkpoint.
  - **Silence fallbacks eliminated** at four sites in `run_inference`:
    - `n_frames <= 0` after AR loop → now returns `false` with a clear error.
    - Codec not bound → now returns `false` (was 1 s silence + `return true`).
    - Codebook-count mismatch → now returns `false` with diagnostic.
    - Codec decode exception → now returns `false` with exception text.
  - **E2E verified**: `test_qwen3tts_e2e` still produces 4.00 s @ RMS 0.0895
    through the now-honest path; nothing was broken.
- **2026-06-29 (rev 3) — MOSS frauds fixed**:
  - **FRAUD #1 fixed** (`voice_clone` .codes hack): rewritten to read a WAV
    via the new `audiocore::io::read_wav_mono` (ported verbatim from
    `openmoss/src/wav.cpp`) and run the real codec encoder. Verified end-to-end
    (`test_moss_voice_clone_e2e` produces 38.8 s @ RMS 0.185).
  - **FRAUD #2/#3/#4/#5 fixed** (sfx/voice_design/dialogue/realtime
    impersonation): the system-prompt injection code at `session.cpp:209–239`
    is deleted. Unsupported modes now fail fast with an error that names the
    missing dedicated checkpoint (`test_moss_modes` verifies all four).
  - **Prompt-builder hack fixed**: `session.cpp` now uses a verbatim port of
    `openmoss/src/pipeline.cpp` (`build_user_inst`, `build_prompt_text`,
    `build_reference_audio_block`, `apply_delay_pattern`, `build_prompt_grid`).
    The model now sees the upstream `<user_inst>` template it was trained on.
    Plain `mode="tts"` verified (`test_moss_e2e` produces 0.32 s @ RMS 0.160).
  - **Silence fallback removed**: missing `moss.codec.*` tensors is now a
    hard error in both `run_tts` and `decode_codec`. `test_moss_e2e`
    converted from `[WARN] silent output` to `CHECK(rms > 1e-6)`.
  - **Codec encoder runtime-verified**: `codec.cpp::encode()` (ported
    2026-06-29 rev 1) is now wired into `voice_clone` and produces real
    codes that drive the AR loop through the spliced prompt grid.
  - **Fabricated per-mode sampling configs deleted**: only the TTS defaults
    remain; the fabricated sfx/dialogue/voice_design tables at the old
    `session.cpp:399–446` are gone.
  - **Shared WAV utility added**: `src/framework/io/wav.cpp` +
    `include/audiocore/framework/io/wav.h` (Apache-2.0 port of openmoss).
    Available for the Qwen3 and ACE-Step families to use in subsequent work.
- **2026-06-29 (rev 2)**: Expanded all three family sections with
  upstream-vs-shipped model matrices and feature matrices per the user's
  upstream-feature inventory. Added 💤 DORMANT tag for the 6 unexercised
  ACE-Step checkpoints, the 4 missing Qwen3 talker variants, and the 25 Hz
  codec. Added systemic issues #6–#9 (dead-weight checkpoints, no long-form
  test, no multilingual validation, no control-surface exposure). Expanded
  priorities from 10 to 14 items. Cross-referenced `session.cpp` line
  numbers for every fraud claim with direct grep evidence.
- **2026-06-29 (rev 1)**: Initial honest tracker. Listed 7 frauds across 3
  families. Codec encoder source ported to `codec.cpp::encode()` but not
  yet wired into `voice_clone`. Prompt-builder port not started.

---

## Appendix A: Fraud summary table

| # | Family | Requested | Actually runs | Code location |
|---|---|---|---|---|
| 1 | MOSS | `mode=voice_clone` (WAV reference) | `.codes` binary loader on flagship TTS | `session.cpp:294–307` |
| 2 | MOSS | `mode=sfx` (SFX model) | Flagship TTS + "You are a sound effects generator." | `session.cpp:224` |
| 3 | MOSS | `mode=voice_design` (VoiceGenerator 1.7B) | Flagship TTS + "You are a voice cloning assistant…" | `session.cpp:235–238` |
| 4 | MOSS | `mode=dialogue` (TTSD 8B) | Flagship TTS + "You are a spoken-dialogue assistant…" | `session.cpp:230–233` |
| 5 | MOSS | `mode=realtime` (Realtime 1.7B, ~180 ms TTFB) | Flagship Delay streamed (~1.3 s cold-start) | `session.cpp:realtime branch` |
| 6 | ACE-Step | `mode=repaint` (in-paint a region) | ✅ FIXED (rev 5) — full HOT-Step port: correct channel layout, stride=P patchify, latent-space Euler, per-step repaint injection | `session.cpp` (rev 5 rewrite) |
| 7 | Qwen3 | `mode=voice_design` (VoiceDesign 1.7B) | Base talker with instruct-prefix | `session.cpp:120–124, 244–252` |

## Appendix B: Dormant-weight inventory

Weights that exist on local disk but are loaded by **no** test:

| Path | Family |
|---|---|
| ~~`/mnt/data/models/audio/acestep-cpp/acestep-v15-sft-Q8_0.gguf`~~ | ACE-Step — exercised 2026-06-30 |
| ~~`/mnt/data/models/audio/acestep-cpp/acestep-v15-xl-base-Q8_0.gguf`~~ | ACE-Step — exercised 2026-06-30 |
| ~~`/mnt/data/models/audio/acestep-cpp/acestep-v15-xl-sft-Q8_0.gguf`~~ | ACE-Step — exercised 2026-06-30 |
| ~~`/mnt/data/models/audio/acestep-cpp/acestep-v15-xl-turbo-Q8_0.gguf`~~ | ACE-Step — exercised 2026-06-30 |
| ~~`/mnt/data/models/audio/acestep-cpp/acestep-5Hz-lm-1.7B-Q8_0.gguf`~~ | ACE-Step — exercised 2026-06-30 |
| ~~`/mnt/data/models/audio/acestep-cpp/acestep-5Hz-lm-4B-Q8_0.gguf`~~ | ACE-Step — exercised 2026-06-30 (after convert_acestep) |
| `/mnt/data/models/audio/qwen3-tts/tokenizer-f16.gguf` | Qwen3-TTS (only q8_0 is tested) |
| `/mnt/data/models/audio/moss-tts/moss-tts-f16.gguf` | MOSS-TTS (only q8_0 is tested) |
| `/mnt/data/models/audio/moss-tts/moss-tts-v1.5-q8_0.extras.gguf` | MOSS-TTS v1.5 extras (test uses old extras) |
| `/mnt/data/models/audio/moss-soundeffect-v2/` (raw safetensors) | MOSS-SFX v2 — never converted to GGUF |
| `/mnt/data/models/audio/moss-audio-tokenizer/` (v1 safetensors) | MOSS codec v1 — never converted as standalone |
| `/mnt/data/models/audio/moss-audio-tokenizer-nano/` (safetensors) | MOSS codec nano — never converted |
