# Patches against vendored submodules

Each subdirectory holds the patches `scripts/apply-submodule-patches.sh`
re-applies after `git submodule update --init --recursive` so that a
fresh clone reproduces the exact tree the maintainer builds with.

Patches are named `<NN>-<slug>.patch` and applied in lexical order.

## `llama.cpp/`

- **0001-ggml-max-name-128.patch** — bump `GGML_MAX_NAME` 64 → 128 so
  long Qwen3-TTS / MOSS tensor names are not truncated.
- **0002-qwen3-k-gqa-tensor-dim.patch** — pass `n_embd_k_gqa` (not
  `n_embd_gqa`) to `create_tensor_qkv` so Qwen3-TTS Talker weights load
  with the correct K/V dimension.
