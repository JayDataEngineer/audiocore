# GGUF tensor naming conventions

Canonical reference for the tensor names each family reads, and the GGUF KV
metadata keys it expects. Names here are **verified** against:

- **MOSS**: [`pwilkin/openmoss`](https://github.com/pwilkin/openmoss) → `src/model.cpp`, `src/codec.cpp`
- **ACE-Step**: [`ServeurpersoCom/acestep.cpp`](https://github.com/ServeurpersoCom/acestep.cpp) → `qwen3-lm.h`, `vae.h`, `dit.h`

The family code in `src/models/<family>/loader.cpp` is the source of truth —
if this doc and the loader disagree, the loader wins. Please update both
together.

## Why a single naming doc

Two failure modes it prevents:

1. The loader can't tell a Qwen3 backbone tensor from a family-specific
   extension tensor when both live in the same GGUF. The naming convention
   makes the role unambiguous from the prefix.
2. Community quantizations of "the same" model end up with different tensor
   names, breaking the loader silently. Naming this doc as the reference gives
   us one place to push back from.

---

## MOSS-TTS

**One file.** The community MOSS GGUF (`moss-tts-q8_0.gguf`) ships the Qwen3-8B
backbone AND the audio extensions in a single file. We never split it.

### Backbone — standard llama.cpp Qwen3 layout

libllama loads the backbone directly. Tensor names are the standard llama.cpp
Qwen3 names — NO `moss.` prefix on these:

| Tensor | Role |
|---|---|
| `token_embd.weight` | Token embedding |
| `blk.{i}.attn_norm.weight` | Pre-attention RMSNorm |
| `blk.{i}.attn_q.weight` | Attention Q |
| `blk.{i}.attn_k.weight` | Attention K |
| `blk.{i}.attn_v.weight` | Attention V |
| `blk.{i}.attn_output.weight` | Attention output |
| `blk.{i}.ffn_norm.weight` | Pre-FFN RMSNorm |
| `blk.{i}.ffn_gate.weight` | FFN gate (Qwen3 SwiGLU) |
| `blk.{i}.ffn_up.weight` | FFN up |
| `blk.{i}.ffn_down.weight` | FFN down |
| `output_norm.weight` | Final RMSNorm |
| `output.weight` | Output projection (tied to `token_embd` if tied) |

**We do not bind these ourselves** — `qwen3::Runner::load()` hands the file
to `llama_model_load_from_file()` and libllama owns them.

### Audio extensions — `moss.*` prefix

These are the tensors libllama doesn't know about. `loader.cpp` binds them
into a separate `ggml_context` via the framework's `GgufReader`.

| Tensor | Shape | Role |
|---|---|---|
| `moss.audio_embed.{i}.weight` | `(audio_vocab_size+1, hidden_size)` | Embeds codec token i into the Qwen3 hidden space; i ∈ [0, `moss.n_vq`) |
| `moss.audio_head.{i}.weight`  | `(audio_vocab_size+1, hidden_size)` | Projects a hidden state → codec-token logits for stream i |
| `moss.codec.enc.<spec>.*` | (varies) | Codec encoder (training only, unused at inference) |
| `moss.codec.dec.<spec>.*` | (varies) | Codec decoder (~1.6B params) — PCM ← codec tokens |
| `moss.codec.quantizer.q.{i}.codebook.weight` | `(codebook_size, hidden)` | RVQ codebook i |
| `moss.codec.quantizer.q.{i}.{iproj,oproj}.{wp0,wp1,bias}` | (varies) | Per-stream projections |
| `moss.codec.quantizer.{iproj,oproj}.{wp0,wp1,bias}` | (varies) | Global quantizer projections |

### KV metadata (read by `loader.cpp`)

Required — missing any of these is a load error:

| Key | Type | Example |
|---|---|---|
| `moss.n_vq` | i32 | `32` |
| `moss.audio_vocab_size` | i32 | `65536` |
| `moss.sampling_rate` | i32 | `24000` |

Optional — defaults live in `MossConfig`:

| Key | Type | Default |
|---|---|---|
| `moss.audio_pad_code` | i32 | `0` |
| `moss.downsample_rate` | i32 | `0` |
| `moss.token.audio_start` | i32 | `0` |
| `moss.token.audio_end` | i32 | `0` |
| `moss.token.user_slot` | i32 | `0` |
| `moss.token.audio_gen_slot` | i32 | `0` |
| `moss.token.audio_delay_slot` | i32 | `0` |
| `moss.token.im_start` | i32 | `0` |
| `moss.token.im_end` | i32 | `0` |
| `moss.token.pad` | i32 | `0` |
| `moss.codec.present` | i32 | `0` |

---

## ACE-Step

**Four files per model directory.** Each file is independent; the loader
locates them by filename pattern.

### `acestep-5Hz-lm-{1.7B|4B}-Q8_0.gguf` — music-code LM

Qwen3 transformer. ACE-Step **ships these with HF PyTorch names** that
libllama refuses:

| HF name (shipped) | llama.cpp name (required) |
|---|---|
| `model.embed_tokens.weight` | `token_embd.weight` |
| `model.norm.weight` | `output_norm.weight` |
| `model.layers.{i}.input_layernorm.weight` | `blk.{i}.attn_norm.weight` |
| `model.layers.{i}.self_attn.q_proj.weight` | `blk.{i}.attn_q.weight` |
| `model.layers.{i}.self_attn.k_proj.weight` | `blk.{i}.attn_k.weight` |
| `model.layers.{i}.self_attn.v_proj.weight` | `blk.{i}.attn_v.weight` |
| `model.layers.{i}.self_attn.o_proj.weight` | `blk.{i}.attn_output.weight` |
| `model.layers.{i}.post_attention_layernorm.weight` | `blk.{i}.ffn_norm.weight` |
| `model.layers.{i}.mlp.gate_proj.weight` | `blk.{i}.ffn_gate.weight` |
| `model.layers.{i}.mlp.up_proj.weight` | `blk.{i}.ffn_up.weight` |
| `model.layers.{i}.mlp.down_proj.weight` | `blk.{i}.ffn_down.weight` |

`tools/convert_acestep_gguf.py` rewrites them **once** at download time.
After conversion the file loads via `qwen3::Runner::load()` exactly like a
vanilla Qwen3 GGUF. There is no second Qwen3 implementation in audiocore.

### `Qwen3-Embedding-0.6B-Q8_0.gguf` — text encoder

Same HF→llama.cpp rename table as the LM above. The text encoder is a
Qwen3 transformer too — same runner, same libllama, same path.

### `acestep-v15-{turbo|sft|xl-*}-Q8_0.gguf` — DiT (diffusion transformer)

NOT a Qwen3 transformer — we bind these into `ext_ctx_` ourselves. Native
PyTorch names (no prefix), verified from `acestep.cpp`:

| Tensor | Role |
|---|---|
| `decoder.time_embed` | Timestep embedding |
| `decoder.time_embed_r` | Timestep embedding (second branch) |
| `decoder.proj_in.1.weight` / `.bias` | Patch projection in |
| `decoder.condition_embedder.weight` / `.bias` | Conditioning embedder |
| `decoder.norm_out.weight` | Final norm |
| `decoder.scale_shift_table` | AdaLN scale/shift table |
| `decoder.proj_out.1.weight` / `.bias` | Patch projection out |
| `null_condition_emb` | CFG null conditioning embedding |
| `decoder.block.{i}.*` | Per-block attention + MLP weights |

### `vae-BF16.gguf` — audio VAE decoder

Also bound into `ext_ctx_`. Verified names from `acestep.cpp:vae.h`. The
loader prefixes these with `vae.` at bind time to avoid collision with DiT's
`decoder.*` namespace (see naming-collision note below). VAE graph code looks
up `vae.decoder.*`; the table shows names as shipped in the GGUF:

| Tensor (shipped) | Bound as | Role |
|---|---|---|
| `decoder.conv1` / `decoder.conv1.bias` | `vae.decoder.conv1` | Initial conv |
| `decoder.block.{i}.*` | `vae.decoder.block.{i}.*` | Residual blocks (Snake activation) |
| `decoder.snake1.alpha` / `decoder.snake1.beta` | `vae.decoder.snake1.{alpha,beta}` | Snake activation params |
| `decoder.conv2` | `vae.decoder.conv2` | Final conv → PCM |

**Naming collision note:** DiT and VAE both ship with `decoder.*` trees.
Within any one GGUF file there's no collision, but `loader.cpp` binds both
into a single `ext_ctx_` so codec graph code can reach both. We resolve the
collision by prefixing VAE tensor names with `vae.` at bind time — no
converter step required, no rename at the file level. DiT tensors stay
unprefixed because the DiT block path is the larger graph and we'd rather
not retag every `ggml_get_tensor` call inside it.

### KV metadata (DiT file, read by `loader.cpp`)

| Key | Type | Notes |
|---|---|---|
| `acestep.in_channels` | i32 | |
| `acestep.audio_acoustic_hidden_dim` | i32 | |
| `acestep.patch_size` | i32 | |
| `acestep.sliding_window` | i32 | |
| `acestep.config_json` | string | Full upstream config (fallback) |
| `acestep.variant` | string | `turbo` / `sft` / `xl-turbo` / `xl-sft` |

---

## Qwen3-TTS

Qwen3-TTS ships two GGUFs produced by `tools/convert_qwen3tts` from the
official safetensors at `QwenLM/Qwen3-TTS`:

### `qwen3_tts_talker.gguf` — qwen3tts architecture

The backbone loads natively via libllama. The extra tensors the talker needs
to do dual-embedding (text + codec) and codec-head projection are pulled out
of the same GGUF by `qwen3::Runner::load_extras(ExtraKind::Talker)`, going
through `WeightLoader` like every other family:

| Tensor | Shape | Used by |
|---|---|---|
| `text_embd.weight` | `[text_vocab, 2048]` | Text-side input embedding |
| `text_proj.0.weight` / `text_proj.0.bias` | `[2048, 2048]` / `[2048]` | Text projection MLP layer 0 |
| `text_proj.1.weight` / `text_proj.1.bias` | `[1024, 2048]` / `[1024]` | Text projection MLP layer 1 (→ hidden_size) |
| `token_embd.weight` | `[codec_vocab, 1024]` | Codec-side input embedding (aliased as `codec_embedding` by the runner) |
| `output.weight` | `[codec_vocab, 1024]` | Codec head (coarse codebook 0 logits) |

If none of these are present (Lunavox-style GGUFs that only ship the
backbone), the talker falls back to plain token-mode forward.

### `qwen3_tts_predictor.gguf` — qwen3tts_cp architecture

The predictor backbone also loads via libllama. The MTP extras — 31 fine
codebook tables + 31 lm heads + a small→MTP projection — are pulled by
`qwen3::Runner::load_extras(ExtraKind::Predictor, n_fine_books=31)`:

| Tensor | Shape | Used by |
|---|---|---|
| `codec_embd.{i}.weight` (i ∈ [0, 31)) | `[fine_vocab, 1024]` | Per-codebook fine input embedding |
| `lm_head.{i}.weight` (i ∈ [0, 31)) | `[1024, fine_vocab]` | Per-codebook fine head |
| `small_to_mtp.weight` / `small_to_mtp.bias` | `[hidden, hidden]` / `[hidden]` | Projects talker hidden state into the MTP conditioner |

`fine_vocab` is inferred from `codec_embd.0.weight`'s element count at load
time. If none of the MTP tensors are present the predictor falls back to
stock `forward_tokens` (Lunavox-style predictor without MTP).

---

## Why no `audiocore.gguf_version`

We don't write our own GGUFs (yet). When we do — quantizer tool, merged
multi-family bundles — a top-level `audiocore.gguf_version` uint32 will be
added at that point and validated at load. Today every file we read was
written by upstream tools (llama.cpp's quantizer, the community ACE-Step
release, our own `convert_qwen3tts`), so a version header we never wrote
would just refuse valid input.
