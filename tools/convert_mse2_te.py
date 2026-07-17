#!/usr/bin/env python3
"""Convert MSE2 Qwen3 text encoder (HF safetensors) → GGUF.

Usage:
    python3 tools/convert_mse2_te.py \
        /mnt/data/models/audio/moss-soundeffect-v2/text_encoder/ \
        /tmp/mse2_te.gguf
"""

import json
import os
import sys
import numpy as np

from gguf import GGUFWriter, GGMLQuantizationType


def load_safetensors_dir(model_dir: str):
    """Load all .safetensors files in a directory, return {tensor_name: np.array}."""
    state = {}
    from safetensors.torch import load_file as st_load
    import torch
    for fname in sorted(os.listdir(model_dir)):
        if not fname.endswith(".safetensors"):
            continue
        path = os.path.join(model_dir, fname)
        print(f"  Loading {fname} ...")
        sd = st_load(path)
        for key, t in sd.items():
            state[key] = t.to(dtype=torch.float32).numpy()
    return state


HF_TO_GGUF = {
    "model.embed_tokens.weight":         "token_embd.weight",
    "model.norm.weight":                 "output_norm.weight",
    "lm_head.weight":                    "output.weight",
}

HF_PER_LAYER = {
    "input_layernorm.weight":             ("attn_norm", "weight"),
    "post_attention_layernorm.weight":    ("ffn_norm", "weight"),
    "self_attn.q_proj.weight":            ("attn_q", "weight"),
    "self_attn.k_proj.weight":            ("attn_k", "weight"),
    "self_attn.v_proj.weight":            ("attn_v", "weight"),
    "self_attn.o_proj.weight":            ("attn_output", "weight"),
    "self_attn.q_norm.weight":            ("attn_q_norm", "weight"),
    "self_attn.k_norm.weight":            ("attn_k_norm", "weight"),
    "mlp.gate_proj.weight":               ("ffn_gate", "weight"),
    "mlp.up_proj.weight":                 ("ffn_up", "weight"),
    "mlp.down_proj.weight":               ("ffn_down", "weight"),
}


def _is_norm_tensor(gguf_name: str) -> bool:
    """Norm weights must be F32 (llama.cpp convention; matches moss_tts).

    The ggml CUDA binbcast dispatcher for `ggml_mul` (used by build_norm to
    apply the per-channel scale after ggml_rms_norm) has an implicit
    assumption that when src0 and dst are both F32, src1 is also F32. If src1
    is F16 the kernel asserts with `nb10 % sizeof(src1_t) == 0` because the
    F16 stride (2 bytes) is not a multiple of sizeof(float) (4 bytes). The
    TE forward hits this on every layer's `attn_norm` mul.

    Norm weights are small (hidden_size or head_dim floats per layer, ~1 MB
    total for a 28-layer Qwen3-TE) so storing them as F32 has no meaningful
    size cost and matches what llama.cpp's own convert scripts emit.
    """
    # Match all RMS/LayerNorm weights, including the q/k per-head norms.
    return gguf_name.endswith("_norm.weight") or "_norm." in gguf_name


def convert(args):
    model_dir = args.model_dir
    out_path = args.output

    print(f"Loading safetensors from {model_dir} ...")
    state = load_safetensors_dir(model_dir)

    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)

    hidden_size = config["hidden_size"]
    n_layers = config["num_hidden_layers"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config.get("num_key_value_heads", n_heads)
    head_dim = config.get("head_dim", hidden_size // n_heads)
    ffn_size = config["intermediate_size"]
    vocab_size = config.get("vocab_size", len(state.get("model.embed_tokens.weight", [])))
    max_seq_len = config.get("max_position_embeddings", 4096)
    rope_theta = config.get("rope_theta", 1000000.0)
    eps = config.get("rms_norm_eps", 1e-6)

    print(f"  hidden_size={hidden_size}, layers={n_layers}, heads={n_heads}")
    print(f"  kv_heads={n_kv_heads}, head_dim={head_dim}, ffn={ffn_size}")
    print(f"  vocab={vocab_size}, max_seq_len={max_seq_len}")

    # ── Create GGUF writer ────────────────────────────────────────────
    arch_str = "qwen3"
    writer = GGUFWriter(out_path, arch_str)

    writer.add_context_length(max_seq_len)
    writer.add_embedding_length(hidden_size)
    writer.add_block_count(n_layers)
    writer.add_feed_forward_length(ffn_size)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv_heads)
    writer.add_rope_dimension_count(head_dim)
    writer.add_rope_freq_base(rope_theta)
    writer.add_layer_norm_rms_eps(eps)
    writer.add_file_type(1)  # F16

    # Tokenizer — extract from tokenizer.json
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen2")
    bos_id = config.get("bos_token_id", 151643)
    eos_id = config.get("eos_token_id", 151645)
    tok_path = os.path.join(os.path.dirname(model_dir.rstrip("/")), "tokenizer")
    tok_json_path = os.path.join(tok_path, "tokenizer.json")
    if os.path.exists(tok_json_path):
        with open(tok_json_path) as f:
            tok_data = json.load(f)
        vocab = tok_data.get("model", {}).get("vocab", {})
        merges_list = tok_data.get("model", {}).get("merges", [])
        # Sort vocab by id
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        tokens = [item[0] for item in sorted_vocab]
        # Pad tokens to match model's vocab_size (embedding layer expects 151936)
        embedding_vocab = config.get("vocab_size", len(tokens))
        if len(tokens) < embedding_vocab:
            # Add the added_tokens from tokenizer_config.json if available, else pad with <unk>
            tok_cfg_path = os.path.join(tok_path, "tokenizer_config.json")
            pad_tokens = {}
            if os.path.exists(tok_cfg_path):
                with open(tok_cfg_path) as f:
                    tcfg = json.load(f)
                at = tcfg.get("added_tokens", [])
                pad_tokens = {a.get("id"): a.get("content") for a in at if a.get("id") is not None and a.get("id") >= len(tokens)}
            for i in range(len(tokens), embedding_vocab):
                if i in pad_tokens:
                    tokens.append(pad_tokens[i])
                else:
                    tokens.append(f"<|reserved_{i}|>")
        scores = [0.0] * len(tokens)   # Qwen3 uses 0.0 for all token scores
        # Token types: 1=normal, 3=control, 2=unknown, etc.
        token_types = [1] * len(tokens)
        # Mark BOS/EOS as control tokens
        if bos_id < len(token_types):
            token_types[bos_id] = 3
        if eos_id < len(token_types):
            token_types[eos_id] = 3

        # Write tokenizer data
        writer.add_token_list(tokens)
        writer.add_token_scores(scores)
        writer.add_token_types(token_types)

        # Add merges as string array (tokenizer.json stores them as [pair, pair] lists)
        if merges_list:
            string_merges = [m if isinstance(m, str) else ' '.join(m) for m in merges_list]
            writer.add_token_merges(string_merges)

    writer.add_bos_token_id(bos_id)
    writer.add_eos_token_id(eos_id)

    # ── Write tensors ──────────────────────────────────────────────────
    written = 0

    def to_f16(arr: np.ndarray) -> np.ndarray:
        return arr.astype(np.float16) if arr.dtype != np.float16 else arr

    def to_f32(arr: np.ndarray) -> np.ndarray:
        return arr.astype(np.float32) if arr.dtype != np.float32 else arr

    def add_weight(writer, gguf_name, arr):
        """Write a model weight with the llama.cpp-convention dtype.

        Norm weights → F32 (see _is_norm_tensor for rationale).
        Everything else → F16 (matches existing F16 layout).
        """
        if _is_norm_tensor(gguf_name):
            t = to_f32(arr)
            writer.add_tensor(gguf_name, t, raw_dtype=GGMLQuantizationType.F32)
        else:
            t = to_f16(arr)
            writer.add_tensor(gguf_name, t, raw_dtype=GGMLQuantizationType.F16)
        return t

    # Global tensors
    for hf_name, gguf_name in HF_TO_GGUF.items():
        if hf_name in state:
            t = add_weight(writer, gguf_name, state[hf_name])
            written += 1
            print(f"  [{written}] {gguf_name} ({t.shape}, {t.dtype})")
        else:
            print(f"  SKIP (not found): {hf_name}")

    # Per-layer tensors
    for layer_idx in range(n_layers):
        for hf_suffix, (gguf_base, gguf_suffix) in HF_PER_LAYER.items():
            hf_name = f"model.layers.{layer_idx}.{hf_suffix}"
            if hf_name not in state:
                continue
            gguf_name = f"blk.{layer_idx}.{gguf_base}.{gguf_suffix}"
            t = add_weight(writer, gguf_name, state[hf_name])
            written += 1
            if written <= 20 or written % 50 == 0:
                print(f"  [{written}] {gguf_name} ({t.shape}, {t.dtype})")

    print(f"\n  Total tensors written: {written}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nDone: {out_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="HF model directory (containing config.json + .safetensors)")
    ap.add_argument("output", help="Output GGUF path")
    convert(ap.parse_args())
