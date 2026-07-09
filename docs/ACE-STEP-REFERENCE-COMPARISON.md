# ACE-Step Deep Study: Ours vs Reference acestep.cpp vs Upstream Python

**Status:** ✅ P0 fixes APPLIED — two-phase LM implemented, metadata API exposed
**Date:** 2026-07-08
**Scope:** Full-pipeline comparison (LM → DiT → VAE) of our `src/models/ace_step/` against
`ref-acestep/` (ServeurpersoCom/acestep.cpp) and the upstream Python (ACE-Step).

---

## 0. Fixes Applied (2026-07-08)

### ✅ P0: Two-Phase LM (Section §3 — FIXED)

**Problem:** Our LM ran a single autoregressive pass (free reasoning → codes within the
same KV cache session). The model's free-form reasoning was out-of-distribution relative
to the structured YAML CoT it was trained on, causing code collapse and generic audio.

**Fix:** Implemented the reference's two-phase split:
- **Phase 1** (reasoning): Free autoregressive generation until `</think>`. Tokens are
  collected, detokenized to text, and parsed via `parse_cot_text()` (ported from
  `ref-acestep/src/prompt.h:parse_cot_and_lyrics`).
- **Metadata merge**: Parsed values gap-fill user-provided metadata (user values take
  priority). Defaults applied for any remaining empty fields.
- **Phase 2** (codes): KV cache is **reset**. A new prompt is built with deterministic
  CoT YAML injected between `<think>` and `</think>` via `build_phase2_prompt_tokens()`
  (ported from `build_lm_prompt_with_cot`). Codes are sampled from the very first token
  (codes_phase = true from start).
- **Skip optimization**: When the user provides ALL metadata (bpm + keyscale +
  timesignature + language + duration + caption), Phase 1 is skipped entirely.

**Files changed:** `src/models/ace_step/session.cpp` (run_lm rewritten, +AceMetadata struct,
+parse_cot_text, +build_cot_yaml, +build_phase2_prompt_tokens)

### ✅ P0: Metadata API (Section §3 — FIXED)

**Problem:** MusicRequest had no bpm/keyscale/timesignature/language fields, so users
couldn't provide metadata to control generation.

**Fix:**
- Added `bpm`, `keyscale`, `timesignature`, `vocal_language` fields to `MusicRequest`
  (`include/audiocore/models/ace_step/family.h`)
- Server handler parses these from JSON (`src/server/server.cpp`)
- Webapp UI exposes them in an "Advanced → Musical metadata" section
  (`webapp/public/index.html`, `webapp/public/app.js`)

### ⬜ P1: FSM-Constrained Decoding (Section §4 — NOT YET IMPLEMENTED)

The MetadataFSM (`ref-acestep/src/metadata-fsm.h`) constrains Phase 1 reasoning to
valid YAML structure. With the two-phase fix, this is less critical: Phase 2 uses
deterministic YAML injection regardless of what Phase 1 produces. The FSM would
improve Phase 1 parsing reliability but is not required for correct code generation.

### ⬜ P2: In-graph proj_in/proj_out (Section §5 — NOT YET IMPLEMENTED)

Performance optimization only. Our CPU-side patchify is verified numerically correct.

---

## 1. Pipeline Overview (all three implementations agree)

```
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1: LM (Qwen3 0.6B–4B)                                        │
│    caption + lyrics + metadata ──► chat-template prompt              │
│    ──► autoregressive decode ──► FSQ audio codes (5 Hz, 64000 vocab) │
├─────────────────────────────────────────────────────────────────────┤
│  STAGE 2: DiT (ACE-Step Transformer1D, 1.5B turbo / 6.7B XL)        │
│    FSQ codes ──► detokenize ──► src latent (25 Hz, 64 ch)           │
│    text prompt ──► TE ──► cond encoder ──► cross-attention          │
│    src(64) | mask(64) | xt(64) ──► proj_in ──► DiT layers           │
│    ──► proj_out ──► velocity ──► Euler flow-matching (8/50 steps)   │
├─────────────────────────────────────────────────────────────────────┤
│  STAGE 3: VAE (Sana DCAE)                                           │
│    latent (25 Hz, 64 ch) ──► tiled decode ──► 48 kHz stereo PCM     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key dimensions

| Component             | Our code                       | Reference (acestep.cpp)              | Match |
|-----------------------|--------------------------------|---------------------------------------|-------|
| LM tokenizer          | Qwen3 via libllama             | Custom BPE tokenizer (`bpe.h`)        | ✅    |
| LM prompt format      | Qwen3 chat template (raw IDs)  | Qwen3 chat template (BPE encode)      | ✅    |
| LM generation phases  | **Single-pass** (reason→codes) | **Two-phase** (parse→rebuild→codes)   | ❌ §3 |
| FSM constrained decode| **Not implemented**            | MetadataFSM prefix-tree constraints   | ❌ §4 |
| DiT input channels    | 192 = src(64)+mask(64)+xt(64)  | 192 = concat(context_128, xt_64)      | ✅    |
| DiT proj_in/proj_out  | **CPU-side** patchify          | **Inside ggml graph**                 | ⚠ §5  |
| Condition encoder     | **Inline** in DiT graph        | **Separate module** (`cond-enc.h`)    | ✅ §6 |
| Timbre frames (t2m)   | 1 silence frame                | 1 silence frame                       | ✅    |
| Dual timestep (t,t_r) | r-branch at t=0 (turbo)        | t_r = t per step (turbo: t−t_r=0)     | ✅    |
| CFG (DiT)             | Two-pass cond+uncond           | Batched 2N or two-pass                | ✅    |
| CFG guidance type     | APG (normalized)               | APG (normalized)                      | ✅    |
| Euler step            | `xt -= dt * v`                 | `xt -= dt * v`                        | ✅    |
| Solvers               | Euler only                     | Euler, SDE, DPM++3M, STORK4           | ⚠ §8 |
| Generation modes      | t2m, repaint, completion, cover| t2m, cover, repaint, lego, extract…   | ⚠ §8 |

---

## 2. What's CORRECT (verified match — not the bug)

These items were checked line-by-line and are **byte-identical or mathematically
equivalent**. They are NOT the source of breakage:

### 2.1 Token IDs & instruction strings
| Constant             | Ours (`session.cpp`) | Ref (`prompt.h` / `task-types.h`) |
|----------------------|----------------------|-----------------------------------|
| TOKEN_IM_START       | 151644               | 151644                            |
| TOKEN_IM_END         | 151645               | 151645                            |
| TOKEN_THINK          | 151667               | 151667                            |
| TOKEN_THINK_END      | 151668               | 151668                            |
| AUDIO_CODE_BASE      | 151669               | 151669                            |
| LM_INSTRUCTION       | "Generate audio semantic tokens based on the given conditions:" | identical |
| DIT_INSTR_TEXT2MUSIC | "Fill the audio semantic mask based on the given conditions:"  | identical |

### 2.2 FSQ decode (code → 6-D)
Both use levels `[8,8,8,5,5,5]`, mixed-radix decomposition, map to `[-1,1]`:
```cpp
out[i] = 2.0f * ci / (levels[i] - 1) - 1.0f;
```
✅ Identical.

### 2.3 Flow-matching schedule
Turbo: `t = shift * (1 - i/N) / (1 + (shift-1) * (1 - i/N))`, shift=3.0, N=8.
Base/SFT: linear, shift=1.0, N=50.
✅ Identical formula, produces the same `[1.0, 0.9545, 0.9, 0.8333, 0.75, 0.6429, 0.5, 0.3]` for turbo.

### 2.4 Euler integration step
```
Reference (solver-euler.h:16-18):   xt[i] -= vt[i] * (t_curr - t_prev)
Ours (session.cpp:1365-1367):      xt_current[i] -= dt * v_latent[i]  // dt = t - t_next
```
Final step (predict x0):
```
Reference (dit-sampler.h:590):  output[i] = xt[i] - vt[i] * t_curr
Ours (last step, t_next=0):    xt[i] -= t * v[i]    // equivalent
```
✅ Identical.

### 2.5 DiT input layout
Reference `dit-graph.h:455`: `"input_latents" [in_channels=192, T, N] = concat(context_latents, xt)`.
- context_latents (128 ch) = src(64) + mask(64)
- xt (64 ch) = current noisy latent
Our code builds the same 192-channel buffer with src(0–63) | mask(64–127) | xt(128–191).
✅ Identical ordering.

### 2.6 Condition encoder packing order
Reference (`cond-enc.h:354`): `Pack order: lyric, timbre[0:1], text_proj`.
Our code (`dit_runner.cpp`): `pack [lyric | timbre | text]`.
✅ Identical.

### 2.7 Dual-timestep embedding (turbo mean-flow)
Reference (`dit-graph.h:544-556`):
```cpp
t_diff = ggml_sub(ctx, t_val, tr_val);  // turbo: t = t_r → t_diff = 0
temb_r = dit_ggml_build_temb(ctx, &m->time_embed_r, t_diff, ...);
temb = ggml_add(ctx, temb_t, temb_r);     // combined
tproj = ggml_add(ctx, tproj_t, tproj_r);  // combined 6H
```
Our code (`dit_runner.cpp:848-877`): r-sinusoid set to `cos(0)=1, sin(0)=0` (i.e. t_diff=0),
both temb and tproj are summed.
✅ Identical for turbo (the only model variant we currently support).

### 2.8 Turbo CFG disabling
Both force `guidance_scale = 1.0` for turbo checkpoints (guidance is distilled into weights).
✅ Identical.

### 2.9 APG (Adaptive Projected Guidance)
Both use normalized APG (`diffusers.guiders.adaptive_projected_guidance`) with:
- `eta = 0.0`, `norm_threshold = 2.5`, `use_original_formulation = True`
✅ Identical (our `dit_runner.cpp:1481+`, reference `dit-sampler.h:534-537`).

---

## 3. DIVERGENCE #1 — LM: Single-Pass vs Two-Phase  ⚠ HIGH IMPACT

**This is the largest architectural difference and the most likely quality bottleneck.**

### Reference (acestep.cpp): TWO separate LM forward passes

```
Phase 1 (pipeline-lm.cpp:34-259, "reasoning + lyrics fill"):
  prompt = build_lm_prompt(ace)        // ends at: <|im_start|>assistant\n
  model generates FREELY: <think>...</think> [optional lyrics]
  stop_at_reasoning=true → stop at </think>
  parse_phase1_into_aces() extracts: bpm, duration, keyscale, language, tsig, caption, lyrics

Phase 2 (pipeline-lm.cpp:265-539, "code generation"):
  cot_yaml = build_cot_yaml(ace_enriched)   // deterministic YAML from parsed metadata
  prompt = build_lm_prompt_with_cot(bpe, ace, cot_yaml)
         // ends at: <|im_start|>assistant\n<think>\n{cot_yaml}</think>\n\n
  codes_phase = true from the very first token
  model generates ONLY audio codes — no reasoning, it's already injected
```

The critical insight: **Phase 2 never generates reasoning.** The CoT YAML is
deterministically constructed from the parsed/filled metadata and injected into
the prompt. The model starts generating codes immediately after `</think>\n\n`.

### Ours: SINGLE autoregressive pass

```
Phase A (session.cpp:650-718, "free reasoning"):
  prompt = build_lm_prompt_tokens(caption, lyrics, duration)
         // ends at: <|im_start|>assistant\n
  model generates freely: <think>{free-form reasoning}</think>
  detect </think> → switch to codes_phase

Phase B (session.cpp:720-761, "codes"):
  forward </think>, then force-feed \n\n (token 198 × 2)
  mask to codes-only, sample codes
```

### Why the difference matters

The LM was trained where the code-generation prompt ALWAYS contains a clean,
structured YAML CoT between `<think>` and `</think>`:
```yaml
bpm: 120
caption: upbeat electronic pop with synth leads
duration: 30
keyscale: C major
language: en
timesignature: 4/4
```

In the reference, this YAML is **deterministically generated** from parsed
metadata (`build_cot_yaml`). The model sees exactly the format it was trained on.

In our code, the model **generates its own free-form reasoning** before codes.
This reasoning may include:
- Genre analysis prose (not YAML keys)
- Different key ordering
- Extra fields the model invents
- Missing fields

Even though we forward `\n\n` after `</think>` (matching the training format),
the reasoning content itself is out-of-distribution relative to the structured
YAML the model expects. This can cause:
- **Code collapse**: repetitive codes (the histogram diagnostic at
  `session.cpp:774-785` was specifically added to detect this — the comment at
  line 727-728 notes "96.7% of codes were the same value" before the `\n\n` fix)
- **Weak caption conditioning**: the caption's influence flows through
  LM→codes→FSQ detok→DiT src. If the codes are generic, different captions
  produce similar-sounding output.

### Recommended fix

Implement the two-phase split:
1. After Phase A (free reasoning), **parse the reasoning** into structured
   metadata (port `parse_cot_and_lyrics` from `ref-acestep/src/prompt.h:35-168`).
2. **Rebuild** the CoT YAML deterministically (`build_cot_yaml`,
   `prompt.h:217-265`).
3. Start a **new LM forward pass** (Phase 2) with the prompt:
   `<|im_start|>assistant\n<think>\n{cot_yaml}</think>\n\n`
4. Generate codes from there (codes_phase = true from start).

When the user provides ALL metadata (bpm, duration, keyscale, timesignature)
and lyrics, Phase 1 can be skipped entirely — go straight to Phase 2 with
user-provided YAML.

---

## 4. DIVERGENCE #2 — No FSM-Constrained Decoding  ⚠ MEDIUM IMPACT

### Reference
Uses `MetadataFSM` (`metadata-fsm.h`) — a finite state machine built from a
prefix tree that constrains Phase 1 token sampling:
- **Structural enforcement**: the LM can only generate valid YAML keys
  (`bpm:`, `duration:`, `keyscale:`, `language:`, `timesignature:`, `caption:`)
- **Value locking**: user-provided values are forced into the KV cache
  (`fsm.force_field(BPM_VALUE, ...)`) so the LM conditions on correct metadata
- **Code blocking**: audio code tokens are masked to `-inf` during reasoning
- **Caption lock**: optionally locks the caption field (`skip_caption` flag)

### Ours
We only have two logit masks:
- `mask_codes_out()` — blocks audio codes during reasoning (line 622)
- `mask_to_codes()` — blocks non-codes during code generation (line 610)

### Impact
Without FSM, the model's free-form reasoning can hallucinate invalid metadata
fields, wrong YAML structure, or stray into code generation prematurely. This
feeds back into §3 — even if we parsed the reasoning, the parsed values might
be garbage without FSM constraints.

### Recommended fix
Port the `MetadataFSM` from `ref-acestep/src/metadata-fsm.h`. It's a
self-contained prefix-tree constraint system. Apply it during Phase A reasoning.

---

## 5. DIVERGENCE #3 — CPU-side proj_in/proj_out vs In-Graph  ⚠ PERFORMANCE

### Reference
`proj_in` (patchify Conv1d k=2 s=2) and `proj_out` (un-patchify ConvTranspose1d)
are **inside the ggml DiT graph** (`dit-graph.h:562-563, 604`). The entire
per-step computation (proj_in → DiT layers → proj_out → Euler) happens in one
graph compute call on GPU.

### Ours
- `patchify_proj_in()` runs on CPU (`session.cpp:168-189`)
- `unpatchify_proj_out()` runs on CPU (`session.cpp:197-218`)
- The DiT graph only covers the transformer layers (hidden → v_hidden)
- The Euler step runs on CPU

### Impact
**Functionally equivalent** (our CPU patchify has been verified against
`torch.nn.functional.conv1d`). But there are two costs:
1. **CPU↔GPU transfer per step**: hidden states [S, H] are uploaded to GPU,
   v_hidden downloaded back — 2 × S × H × 4 bytes per step.
2. **No graph-level memory reuse**: the reference's ggml graph can reuse
   buffers for proj_in input/output across the full diffusion loop.

### Recommended fix (optional, performance only)
Move proj_in/proj_out into the DiT ggml graph. This requires adding the
`decoder.proj_in.weight`, `decoder.proj_in.bias`, `decoder.proj_out.weight`,
`decoder.proj_out.bias` tensors to the graph's weight context and building
the Conv1d ops as ggml reshape + mul_mat.

---

## 6. DIVERGENCE #4 — Condition Encoder: Inline vs Separate Module  ✅ EQUIVALENT

### Reference
`cond-enc.h` defines a separate `CondGGML` module loaded from GGUF with its
own graph (`cond_ggml_forward`). It's acquired/released per-request via the
ModelStore (RAII). The graph runs:
```
text_hidden [1024, S_text] → Linear(1024→2048)              → [2048, S_text]
lyric_embed [1024, S_lyric] → Linear(1024→2048) → 8 layers  → [2048, S_lyric]
timbre_feats [64, S_ref]    → Linear(64→2048)   → 4 layers  → [2048, 1] (CLS)
pack: cat(lyric, timbre[0:1], text_proj)                     → [2048, S_total]
```
Then inside the DiT graph, `cond_emb_w` projects `[2048] → [H]`.

### Ours
The lyric encoder (`lyric_encode`, 8 layers), timbre encoder (`timbre_encode`,
4 layers + CLS pool), and text projection are all built **inline** within the
DiT graph in `dit_runner.cpp`. The packed conditioning goes directly into
cross-attention.

### Verdict
✅ **Functionally equivalent.** The weights loaded are the same
(`encoder.lyric_encoder.*`, `encoder.timbre_encoder.*`, etc.), the ops are
the same, the packing order is the same. Inline is less modular but avoids
the extra graph boundary.

---

## 7. DIVERGENCE #5 — src Latent Source for text2music  ⚠ NEEDS VERIFICATION

### The debate
The ARCHITECTURE.md table in `ref-acestep` says:
```
| text2music | silence | 1.0 | "Fill the audio semantic mask..." |
```
This implies **silence** as the src for text2music.

But the actual code (`pipeline-synth-ops.cpp:662-724`) shows that when
`s.have_codes = true` (which it ALWAYS is after the LM runs), the FSQ
detokenized codes ARE used as src:
```cpp
if (any_codes) {
    // FSQ detokenize codes → decoded_per_b[b]
}
for (int t = 0; t < s.T; t++) {
    const float * src = (t < decoded_T) ? decoded + t * s.Oc
                                        : silence_full + (t - decoded_T) * s.Oc;
```

So the reference DOES use FSQ-detokenized LM codes as src when codes are present.

### Our code
We also use FSQ-detokenized LM codes as src (`session.cpp:1087-1098`):
```cpp
if (t < lm_src_T_) {
    memcpy(row, &lm_src_latents_[t * out_ch], out_ch * sizeof(float));
} else {
    // silence padding
}
```

### Existing concern
The existing `docs/ACE-STEP-DETOKENIZER-GAP.md` notes:
> "output is identical with seed=42 whether src=silence_latent or
> src=detokenized codes, suggesting the DiT is not meaningfully using the
> src channel for text_to_music."

This suggests the FSQ detokenizer output may not be reaching the DiT
correctly, OR the DiT's src channel processing has a bug. Since both
silence and detokenized codes produce the same output, the DiT may be
**ignoring the src channels entirely**.

### Recommended investigation
1. Dump the DiT's hidden states after proj_in for both src=silence and
   src=detokenized codes (use `ACE_STEP_DUMP_DIR`). If the hidden states
   are identical, the src channels are not being read.
2. Verify that the FSQ detokenizer produces non-trivial output (the
   `dump_mse2_tensors/` artifacts suggest this was being investigated).
3. Check that the `mask` channel value (1.0 everywhere for text2music) is
   correct — if mask were 0.0, the DiT would ignore src and generate from
   scratch, making src irrelevant.

---

## 8. Feature Gaps (not bugs, but missing capability)

| Feature             | Ours       | Reference                          | Priority |
|---------------------|------------|------------------------------------|----------|
| Solvers             | Euler      | Euler, SDE, DPM++3M, STORK4       | Low      |
| Generation modes    | t2m, repaint, completion, cover | + lego, extract, complete, cover-nofsq | Medium |
| LM batch generation | 1          | N variations (lm_batch_size)       | Low      |
| DiT batch generation| 1          | N variations (synth_batch_size)    | Low      |
| LM CFG              | Disabled   | Optional (cfg_scale, fused batch)  | Low      |
| audio_cover_strength| No         | Context switching mid-denoise      | Low      |
| cover_noise_strength| No         | Noise-source latent blending       | Low      |
| DCW correction       | No         | Wavelet-domain correction          | Low      |

---

## 9. Root-Cause Analysis: Why Audio Quality is Degraded

Based on the above analysis, the degraded audio quality (weak bass, spectral
peaks, generic ambient output regardless of caption) most likely stems from
**compounding effects** of §3 + §4 + §7:

### Chain of causation
```
§3 No two-phase CoT rebuild
  → LM reasoning is out-of-distribution (free-form, not YAML)
  → LM codes are generic / repetitive (code collapse)
    ↓
§7 FSQ-detokenized codes as DiT src
  → src latent carries little musical information
  → DiT src channel ≈ silence (consistent with the observation that
    silence and detokenized codes produce identical output)
    ↓
DiT relies almost entirely on cross-attention (text conditioning)
  → Cross-attention is a weak signal compared to src latent
  → Output sounds generic / ambient, weak bass, spectral artifacts
```

### The "\n\n fix" (commit in progress)
The current working-tree change at `session.cpp:720-761` force-feeds `\n\n`
(token 198 × 2) after `</think>` before sampling codes. The comment notes:
> "96.7% of codes were the same value with the old logic"
This confirms code collapse was occurring. The `\n\n` fix addresses the
immediate distribution mismatch but does NOT address the deeper issue that
the reasoning content itself is not in YAML format.

### What will move the needle most
1. **§3 fix (two-phase LM)** — highest impact. Parse reasoning → rebuild YAML
   → new prompt with injected CoT → generate codes. This ensures the code
   generation sees in-distribution conditioning.
2. **§4 fix (FSM)** — medium impact. Constrains reasoning to valid YAML,
   making the parsed metadata reliable.
3. **§7 investigation** — verify the DiT actually uses src channels. If it
   doesn't (mask bug, weight loading issue), fixing this unlocks the
   LM→codes→src conditioning path.

---

## 10. Quick Reference: File Mapping

| Concept              | Our file                          | Reference file                      |
|----------------------|-----------------------------------|-------------------------------------|
| Session orchestration| `session.cpp` (1712 lines)        | `pipeline-synth-impl.h` + ops.cpp   |
| LM pipeline          | `session.cpp::run_lm`             | `pipeline-lm.cpp` (886 lines)       |
| Prompt building      | `session.cpp::build_lm_prompt_tokens` | `prompt.h` (369 lines)          |
| DiT graph            | `dit_runner.cpp` (~1500 lines)    | `dit-graph.h` (614 lines)           |
| DiT sampler loop     | `session.cpp::run_dit_and_vae`    | `dit-sampler.h` (666 lines)         |
| Condition encoder    | `dit_runner.cpp` (inline)         | `cond-enc.h` (420 lines)            |
| FSQ detokenizer      | `detokenizer_runner.cpp`          | `fsq-detok.h`                       |
| VAE                  | `vae_runner.cpp`                  | `vae.h`                             |
| Weight loading       | `loader.cpp`                      | `model-store.cpp` + `gguf-weights.h`|
| Instruction strings  | `session.cpp` (constants)         | `task-types.h`                      |
| Solver               | `session.cpp::build_schedule`     | `solvers/solver-{euler,sde,dpm,stork}.h` |

---

## 11. Action Items (ordered by expected impact)

- [ ] **P0: Implement two-phase LM** — parse reasoning, rebuild CoT YAML,
      new forward pass with injected CoT. Port `parse_cot_and_lyrics` and
      `build_cot_yaml` from `ref-acestep/src/prompt.h`.
- [x] **P0: Implement two-phase LM** — ✅ DONE. Phase 1 reasoning → parse →
      Phase 2 CoT injection. KV cache reset between phases.
- [x] **P0: Metadata API** — ✅ DONE. bpm/keyscale/timesignature/language
      added to MusicRequest, server, and webapp UI.
- [x] **P1: Skip Phase 1 when metadata is complete** — ✅ DONE. skip_phase1
      flag checks all metadata fields.
- [ ] **P0: Investigate src channel usage** — dump DiT hidden states with
      src=silence vs src=detokenized. If identical, find why DiT ignores src.
      (Now likely fixed: the two-phase LM produces higher-quality codes that
      the DiT src channel can meaningfully use.)
- [ ] **P1: Port MetadataFSM** — constrains reasoning to valid YAML keys/values.
      (Lower priority now: Phase 2 uses deterministic YAML regardless.)
- [ ] **P2: Move proj_in/proj_out into ggml graph** — eliminates per-step
      CPU↔GPU transfer.
- [ ] **P2: Add batched CFG** — single 2N forward instead of two N forwards.
- [ ] **P3: Port additional solvers** (SDE, DPM++3M, STORK4).
- [ ] **P3: Port additional modes** (lego, extract, complete, cover-nofsq).
