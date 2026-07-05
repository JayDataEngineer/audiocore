# Embedding Flow Through the Inference Pipeline

This document describes how the speaker embedding enters the talker's input
sequence and how it conditions the autoregressive generation.

## Pipeline status (2026-07-04)

**0.6B-Base: WORKING** — verified end-to-end via cloud_vlm (MiMo-V2.5):

| Demo | Input | cloud_vlm verdict |
|------|-------|-------------------|
| Speaker embedding + text | `vivian.voice` + "Hello world, this is a voice cloning demonstration." | "Hello world, this is a voice cloning demonstration." · **Excellent** · 100% |
| Speaker embedding + HAPPY emotion + text | + `--instruct "Speak with a happy, cheerful, warm emotion."` | "...demonstration. **[laughs]**" · Emotion: "**Cheerful and playful with a theatrical laugh**" · **Excellent** · 100% |
| Longer text | "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs." | Verbatim match · **Excellent** · 100% |

The user's four-part requirement is met for 0.6B-Base:
- (a) Speaker embeddings (raw 1024-dim ECAPA vector from `*.voice`) ✓
- (b) Separate emotional prompt (`--instruct`, injected as system-role overlay) ✓
- (c) Normal text prompt ✓
- (d) High-quality output (Excellent quality, 100% intelligibility) ✓

**1.7B-Base: STILL BROKEN** — talker produces gibberish despite identical
prefill assembly. Diagnostics (QWEN3TTS_NO_MTP=1) confirm the bug is in
the talker forward pass, not the predictor. See "1.7B-specific notes"
below.

## How `--instruct` (emotion) is injected

The instruct string is the EMOTION/STYLE cue. It must NEVER be concatenated
to the synthesis text (a prior bug caused the model to literally read the
instruct aloud, e.g., cloud_vlm transcribed *"Vit speak with a happy
cheerful emotion. Hello"*).

Instead, the instruct is wrapped in a system-role turn and projected
through `text_proj` into a separate prefill segment that is prepended
**before** the `<|im_start|>assistant\n` role tokens:

```
[instruct_overlay(n_instruct)]      ← ONLY if --instruct is non-empty
  Tokens: <|im_start|> system \n {instruct} <|im_end|> \n
  Each token projected via text_proj → n_instruct × n_embd floats

[role_embed(3)]                     ← always present
  Tokens: <|im_start|> assistant \n

[codec_overlay(overlay_len)]        ← bridge + speaker + (optional ICL)
  Tokens: codec_nothink, codec_think_bos, codec_think_eos,
          [speaker_embedding] (raw ECAPA vector at position 6),
          [ref_text + codec_pad × L_ref]  (ICL only),
          [ref_codec × T_ref]              (ICL only),
          codec_pad (transition)

[text_region(text_region_len)]      ← synthesis text
  Streaming mode: 1 frame (first_text + codec_bos)
  Compact mode  : N text frames + tts_eos + transition
```

Why the **system** role (not user)? The Qwen3-TTS Base variant was trained
on the assistant-turn format. Adding a `user` turn puts the input
out-of-distribution and the talker either collapses to silence (RMS≈0.003)
or hallucinates content. The `system` turn is treated as a soft style
cue — cloud_vlm confirms the emotion comes through (e.g., a "happy"
cue produced `[laughs]` and was rated "Cheerful and playful"). The
conditioning is SOFT on a Base model (it nudges; it does not deterministically
control), so for reproducible strong emotion use the Instruct / VoiceDesign
variant.

## Talker input layout (prefill assembly)

The Qwen3-TTS talker (a `qwen3::Runner` with dual embeddings — text and
codec) builds a prefill sequence where every position has two overlays:

```
input[pos] = text_embedding[tok[pos]] + codec_embedding[codec_tok[pos]]
```

The prefill layout is:

```
Pos  Content                              Role
───  ───────                              ────
0    text_proj(<|im_start|>)              Role: system start
1    text_proj(assistant)                 Role: tag
2    text_proj(\n)                        Role: newline

3    codec_think/nothink                  Codec bridge: thinking token
4    codec_think_bos                      Codec bridge: begin special
5    codec_think_eos                      Codec bridge: end special
6    ~~~ SPEAKER EMBEDDING ~~~            <- injection point
7..  ref_text tokens                      ICL: reference transcript
     + codec_pad × L_ref
     + codec_emb(ref_code) × T_ref        ICL: reference codec frames
...  codec_pad + tts_bos                  Transition: text region start
...  text_tok_0 + codec_bos              First synthesis text token
...  text_tok_1..N + codec_pad           Synthesis text continues
...  tts_eos                              End of synthesis
```

## Embedding injection at position 6

The speaker embedding (float vector of length `enc_dim`) is written
directly into the codec_embedding slot at position 6. It is NOT looked
up from an embedding table — it is a raw vector assignment.

In code (`session.cpp` line ~485-510):

```
// vc_spk_emb = speaker_embedding (from ECAPA or pre-computed)
// Position 6 is the speaker slot in the prefill layout
codec_prompt_embeddings[6 * n_embd..(6+1) * n_embd] = vc_spk_emb
```

The embedding dimension (`enc_dim`) must match `talker->n_embd()`:
- 0.6B talker: `n_embd = 1024`
- 1.7B talker: `n_embd = 2048`

If `enc_dim != n_embd`, the vector is zero-padded or truncated (with
a warning). This should not happen in practice — the encoder is always
paired with its matching talker.

## Two voice-clone modes

### x-vector-only mode (speaker_embedding alone)

```
Positions in prefill:
  [0..5]  Role + codec bridge tokens
  [6]     Speaker embedding
  [N..]   Synthesis text + codec_bos/codec_pad

No reference text prepended.
No reference codec tokens prepended.
ref_text NOT required. x_vector_only_mode = true.
```

This mode is simpler and faster (shorter prefill). The talker conditions
on the speaker embedding alone to reproduce the voice. Quality is slightly
reduced compared to ICL mode — the talker has less acoustic context.

When to use:
- Pre-computed embeddings (voice caching)
- Ephemeral/synthetic embeddings (SLERP blends)
- When you don't have a reference transcript

Implied automatically when `speaker_embedding` is provided without
`reference_audio` or `reference_text`.

### ICL mode (embedding + reference codes)

```
Positions in prefill:
  [0..5]  Role + codec bridge tokens
  [6]     Speaker embedding
  [7..]   ref_text token embeddings
  [+ L]   codec_emb(ref_code_0) .. codec_emb(ref_code_T-1)
  [+ T]   codec_pad + tts_bos (transition)
  [+..]   Synthesis text + codec_bos/codec_pad
```

The reference audio is encoded into code tokens via the codec encoder
(`codec_graphs_.encode()`) producing a `[16, T_ref]` code matrix. Only
codebook 0 codes are injected as ICL context (codes from the first 16
codebooks are the full code matrix, but the ICL path only extracts
codebook 0 tokens for the prefill). The full `[16, T_ref]` matrix is
used later if the talker generates fine codes in-context.

When to use:
- Full voice clone with `reference_audio` + `reference_text`
- Best quality — provides both voice identity (embedding) and
  acoustic/prosodic context (ref codes + ref text)

## How the embedding conditions generation

1. **Prefill pass**: The full prefill (including the speaker embedding at
   position 6) is fed through the talker transformer in one forward pass.
   The attention mechanism propagates the speaker identity from position 6
   to all subsequent positions.

2. **AR decoding**: For each autoregressive step, the talker generates the
   next hidden state conditioned on all prior positions (including the
   speaker embedding via KV cache). The codec head projects the hidden
   state to codebook-0 logits, which are sampled to produce the first
   code token.

3. **MTP (Multi-Token Prediction)**: For each AR step, the code predictor
   generates fine codebooks 1-15 (15 fine, plus the talker-sampled cb0 =
   16 total). The full code vector `[code_0, code_1, ..., code_15]` at
   step t defines one codec frame, matching the codec GGUF `rvq=16`
   metadata and the predictor's `codec_embd.{0..14}.weight` tensor count.

4. **Codec decode**: The accumulated `[16, T_final]` code matrix is decoded
   to 24 kHz PCM via the WavTokenizer codec decoder (8 transformer layers
   + ConvNeXt upsample + snake-beta decoder blocks).

## Named speaker injection (CustomVoice variant)

For the CustomVoice variant, 9 default speakers are available as
pre-trained codec embedding slots:

| Name | Slot position |
|------|--------------|
| Vivian | slot 3000 |
| Ryan | slot 3001 |
| ... | slots 3002-3008 |

These are baked into the `codec_embedding` weight matrix at specific
rows. At inference time, `speaker_name → speaker_token` resolves the
name to a slot index, and the embedding is looked up from
`codec_embedding[speaker_token]` instead of being injected from the
ECAPA encoder.

This is separate from the ECAPA embedding path. CustomVoice models do
NOT have a speaker encoder — they use fixed speaker tokens only.

## Dimension compatibility

| Component | 0.6B Base | 1.7B Base | 1.7B VoiceDesign |
|-----------|-----------|-----------|------------------|
| Talker n_embd | 1024 | 2048 | 2048 |
| ECAPA enc_dim | 1024 | 2048 | N/A (no encoder) |
| Codec vocab | 3072 | 3072 | 3072 |
| Codec embd dim | 1024 | 1536 | 1536 |
| Text vocab | 151936 | 151936 | 151936 |
| Text proj dim | 2048→1024 | 2048→2048 | 2048→2048 |
