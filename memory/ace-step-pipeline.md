---
name: ace-step-pipeline
description: ACE-Step music generation pipeline architecture
metadata:
  type: reference
---

ACE-Step is a 4-component music generation pipeline: LM (Qwen3 5Hz) + TE (Qwen3-Embedding) + DiT + VAE.

**Architecture**: Text → TE encodes prompt → 5Hz LM autoregressively generates music codes → FSQ detokenizer decodes codes to 6D → upsample 5→25Hz → proj_in → DiT flow-matching denoises → proj_out → VAE decodes to 48kHz stereo PCM.

**LM**: 5Hz music-code LM (1.7B, Qwen3-based). Uses [[qwen3-runner]] for `forward_tokens()`. Argmax over 64K audio codes appended to Qwen3 vocab.

**TE**: Qwen3-Embedding (0.6B, Qwen3-based). Uses [[qwen3-runner]] for `forward_get_embeddings()`. Cached as `te_cond_`/`te_uncond_` vectors on the session.

**DiT**: 24-layer Diffusion Transformer with AdaLN modulation, GQA self-attention + RoPE, cross-attention to TE condition, SwiGLU MLP. Uses `ggml_graph_compute_with_ctx` per forward (512MB temp context). CFG by running two graphs (cond/uncond) and blending. Flow matching with Euler integration: `x_{t-dt} = x_t + dt · v(x_t, t, cond)`. Supported schedules: turbo (8 steps, shifted), sft (50 steps, linear), custom.

**FSQ**: 6D codebook [8,8,8,5,5,5] = 64K codes. Mixed-radix: cumulative products 1, 8, 64, 512, 2560, 12800. Learned MLP (6→2048→SiLU→LayerNorm→64) on the FSQ vectors if weights present. Fallback: identity pad/truncate.

**VAE** (implemented): Full ConvTranspose1d + Snake + ResUnit architecture matching acestep.cpp. 5 decoder blocks with strides [10,6,4,4,2] = 1920× total upsampling from 25Hz to 48kHz. Channels: 64→2048→1024→512→256→128→2(stereo). Each block: Snake(x + sin²(e^α·x)/e^β) → conv_t1d (mul_mat + col2im_1d with pre-permuted 2D weights) → 3× ResUnit (Snake → Conv1d(k=7,dilated) → Snake → Conv1d(k=1) + skip). Final Snake + Conv1d(k=7) → stereo PCM. Memory: per-block sub-graphs with 256MB shared buffer, CPU handoff between blocks.

**Key details**: Conv_t1d weights are 2D [IC, K·OC] (pre-permuted by acestep.cpp converter for mul_mat). Conv1d weights are 3D [kH, IC, OC] (native ggml_conv_1d format). Snake params pre-computed as exp(alpha) and 1/exp(beta) in the GGUF. All VAE tensors prefixed with `vae.` in ext_ctx_ to avoid `decoder.*` collision with DiT. No tiling yet — single-pass decode works for moderate durations (≤~10s at 25Hz).

**Key contributor**: The two Qwen3 transformers go through the unified [[qwen3-runner]] (libllama) — neither has a separate Qwen3 implementation. GGUFs must be in llama.cpp tensor-name layout (converted via `tools/convert_acestep_gguf.py`).

**Why**: The session layer does proj_in/proj_out via CPU matmul (`manual_linear()`) rather than in the ggml graph — simpler, avoids graph complexity for the MVP.
