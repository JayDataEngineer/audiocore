# ACE-Step E2E Test Status Tracker

## Pipeline Stages

| Stage | Status | Notes |
|-------|--------|-------|
| Model loading (4 GGUFs) | ✅ PASS | DiT+LM+TE+VAE all load |
| TE forward (text encoder) | ✅ PASS | 1024-dim embeddings, 8 tokens |
| TE null-text (for CFG) | ✅ PASS | Unconditional embedding |
| LM prefill | ✅ PASS | 8 prompt tokens |
| LM autoregressive decode | ✅ PASS | 50 codes at 5Hz (10s) |
| proj_in Conv1D (192→2048) | ✅ PASS | bf16→f32 weight conversion |
| DiT graph build | ✅ PASS | 24 layers, 4096-node graph, 3GB ctx |
| DiT flow matching (8 turbo steps) | ✅ PASS | CFG guidance=7.5, RoPE+QK-norm |
| proj_out Conv1D (2048→64) | ✅ PASS | bf16→f32 weight conversion |
| VAE decode (5 blocks) | ✅ PASS | 248 latents → 952320 samples |
| **Output quality** | ✅ **PASS** | **RMS=0.0117, non-silent stereo PCM, WAV written** |

## Bug Fixes Applied

| Fix | File | Description |
|-----|------|-------------|
| UINT32 KV parsing | gguf_reader.cpp | get_kv_i32 accepts UINT32 (ACE-Step stores as UINT32) |
| ne[2] fallback | loader.cpp | Post-bind hidden_size from ne[2] not ne[0] |
| Text projector | dit_runner.cpp | Added encoder.text_projector (1024→2048) before condition_embedder |
| RoPE position IDs | dit_runner.cpp | ggml_rope_ext requires int32 pos tensor (API change) |
| RoPE 3D reshape | dit_runner.cpp | Reshape Q/K/V to [hd,nh,T] before RoPE |
| Graph capacity | dit_runner.cpp | ggml_new_graph_custom(4096) — default 2048 too small |
| Conv1D input layout | vae_runner.cpp | Transpose input to [T,IC] for ggml_conv_1d |
| Conv1D weight F16 | vae_runner.cpp | im2col CPU requires F16 kernel type |
| Conv1D bias broadcast | vae_runner.cpp | Reshape bias to [1,OC] for ggml_add |
| VAE snake tensor names | vae_runner.cpp | Block snake = "snake1" not empty (macros had wrong path) |
| VAE buffer size | vae_runner.cpp | 256MB → 2GB scratch for 10s audio |
| ResUnit use-after-free | vae_runner.cpp | h_ru destroyed each iter; added h_prev to keep alive |
| **BF16→F32 conversion** | loader/vae/dit/session | **CRITICAL**: BF16 is F32 truncated to top 16 bits — correct conv is `uint32_t(bits) << 16` (was double-biasing with +127 exponent, producing ~2^127 overflow → inf/NaN) |
| **Weight normalization formula** | vae_runner.cpp | **CRITICAL**: Reference uses `w = g*v/||v||` (weight norm, L2) NOT `w = g*(v-μ)/σ` (weight standardization). Wrong formula produced huge values (1e+33) → F16 im2col overflow → NaN in ru_conv2 |
| **Conv1d weight layout** | vae_runner.cpp | GGUF stores Conv1d weight_v as ne=[K,IC,OC] (PyTorch [OC,IC,K]→ggml), ConvTranspose1d as ne=[K,OC,IC]. Added `input_is_K_IC_OC` flag to compute_wsconv to handle both |
| **conv_t1d permute layout** | vae_runner.cpp | Output must be ne=[IC,K*OC] ggml row-major (data[ic+k_oc*IC]); was writing transposed [K*OC,IC] |
| Snake params use exp() | vae_runner.cpp | Snake alpha/beta go through exp() / 1/exp() (reference vae_load_snake/_inv). Direct use of raw alpha/beta is wrong |
| DiT context 3GB→6GB | dit_runner.cpp | 15s (n_frames=375) needs more than 3GB for bias broadcast tensors |
| VAE scratch 2GB→4GB | vae_runner.cpp | 15s block 4 ResUnit im2col needs > 2GB |
| buf_size ULL suffix | vae_runner.cpp | `4096u * 1024u * 1024u` overflows unsigned (32-bit); must use `4096ULL * 1024 * 1024` |

## Test Configurations Verified

All on CUDA backend with turbo variant (8 flow-matching steps):

| Seed | Duration | Guidance | Caption | RMS | Result |
|------|----------|----------|---------|-----|--------|
| 42   | 10s      | 7.5      | lo-fi ambient piano with soft rain | 0.0117 | ✅ PASS |
| 123  | 5s       | 3.5      | rock music | 0.0119 | ✅ PASS |
| 999  | 20s      | 1.0      | ambient music | 0.0117 | ✅ PASS |
| 7    | 15s      | 15.0     | drums music | 0.0118 | ✅ PASS |

Output WAVs are 16-bit stereo 48kHz, ~1.9MB/10s.

## Modes to Test

| Mode | Status | Notes |
|------|--------|-------|
| text_to_music | ✅ PASS | RMS=0.0117, 9.92s stereo WAV, multiple configs verified |
| repaint | ❌ Untested | Requires input_audio + VAE encode |
| completion | ❌ Untested | Same as repaint, different mask |
| cover | ❌ Untested | Requires style encoder (may fail-fast) |
| stem | ❌ N/A | Separate model family (Demucs) |
| lego | ❌ N/A | Separate stem-assembler |

## Configurations to Test

| Config | Status | Notes |
|--------|--------|-------|
| Turbo variant (8 steps) | ✅ PASS | Default path, all tests use turbo |
| SFT variant (50 steps) | ❌ Untested | Different schedule, need SFT weights |
| XL-turbo variant | ❌ Untested | Larger model, need XL weights |
| CPU backend | ❌ Untested | ACESTEP_DEVICE=ggml_cpu (will be slow) |
| CUDA backend | ✅ PASS | All tests run on CUDA |
| Duration 5s | ✅ PASS | seed=123, guid=3.5, "rock music" |
| Duration 10s | ✅ PASS | seed=42, guid=7.5, "lo-fi" (default) |
| Duration 15s | ✅ PASS | seed=7, guid=15.0, "drums music" |
| Duration 20s | ✅ PASS | seed=999, guid=1.0, "ambient music" |
| Different seeds | ✅ PASS | 42, 123, 999, 7 all verified |
| CFG guidance = 1.0 | ✅ PASS | No-CFG path works (20s ambient) |
| CFG guidance = 15.0 | ✅ PASS | High guidance works (15s drums) |
| Duration > 20s | ❌ Untested | May need larger DiT ctx or tiled decode |
| Temperature > 0 | ❌ Untested | Sampling vs argmax |
