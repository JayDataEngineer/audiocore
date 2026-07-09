#!/usr/bin/env python3
"""MSE2 tensor dump — layer-by-layer intermediates for differential testing.

Saves every sub-layer's input + output as raw f32 binary blobs so the
C++ port can verify each building block in isolation."""

import argparse, os, math, json, struct
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from moss_soundeffect_v2 import MossSoundEffectPipeline

DTYPE = torch.bfloat16
DEVICE = "cuda"

def save_f32(name, t, out_dir):
    """Save a torch tensor as raw f32 binary + shape metadata."""
    t = t.detach().float().cpu().contiguous()
    # Save raw binary
    t.numpy().tofile(out_dir / f"{name}.bin")
    # Save shape
    (out_dir / f"{name}.shape").write_text(" ".join(str(s) for s in t.shape))


def hook_block(block_idx, out_dir, blocks_data):
    """Return (pre_hook, post_hook) for a DiTBlock that captures
    the 5 key intermediate states."""
    pre_dir = out_dir / f"block_{block_idx}_pre"
    post_dir = out_dir / f"block_{block_idx}_post"
    pre_dir.mkdir(parents=True, exist_ok=True)
    post_dir.mkdir(parents=True, exist_ok=True)

    data = blocks_data[block_idx]

    def pre_hook(m, args):
        x, context, t_mod, freqs = args
        data["in_x"] = x.clone()
        data["in_context"] = context.clone()

        # Capture modulation split BEFORE the block uses it
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        mod_plus_t = m.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        chunks = mod_plus_t.chunk(6, dim=chunk_dim)
        if has_seq:
            data["shift_msa"], data["scale_msa"], data["gate_msa"] = [c.squeeze(2) for c in chunks[:3]]
            data["shift_mlp"], data["scale_mlp"], data["gate_mlp"] = [c.squeeze(2) for c in chunks[3:]]
        else:
            data["shift_msa"], data["scale_msa"], data["gate_msa"] = chunks[:3]
            data["shift_mlp"], data["scale_mlp"], data["gate_mlp"] = chunks[3:]

    def post_hook(m, args, output):
        nonlocal data
        save_f32(f"in_x", data["in_x"], pre_dir)
        save_f32(f"in_context", data["in_context"], pre_dir)

        # Modulated inputs
        mod_x = (m.norm1(data["in_x"]) * (1 + data["scale_msa"]) + data["shift_msa"]).to(torch.float32)
        save_f32(f"modulated_norm1", mod_x, pre_dir)

        # self-attn sub-layers
        x_sa = m.self_attn
        q = x_sa.norm_q(x_sa.q(mod_x))
        k = x_sa.norm_k(x_sa.k(mod_x))
        v = x_sa.v(mod_x)
        save_f32(f"self_attn_q", q, pre_dir)
        save_f32(f"self_attn_k", k, pre_dir)
        save_f32(f"self_attn_v", v, pre_dir)

        save_f32(f"output", output, post_dir)

    return pre_hook, post_hook


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/data/models/audio/moss-soundeffect-v2")
    ap.add_argument("--out", default="dump_mse2_tensors")
    ap.add_argument("--prompt", default="heavy rain on a tin roof")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pipeline...")
    pipe = MossSoundEffectPipeline.from_pretrained(
        args.model,
        torch_dtype=DTYPE,
        device=DEVICE,
    )
    dit = pipe.transformer
    vae = pipe.vae
    print(f"DiT: {dit.__class__.__name__}, layers={len(dit.blocks)}")
    print(f"VAE: {vae.__class__.__name__}, latent_dim={vae.latent_dim}, hop={vae.hop_length}")
    print(f"text_encoder dim: {pipe.text_encoder.dim}")

    # ── Build dummy input ────────────────────────────────────────────────
    n_seconds = args.seconds
    sample_rate = pipe.sample_rate
    n_samples = int(sample_rate * n_seconds)
    # Round to hop_length boundary
    hop = vae.hop_length
    n_samples = (n_samples // hop) * hop
    n_latent_frames = n_samples // hop

    B = 1
    in_dim = dit.config.get("in_dim", vae.latent_dim)
    latent_shape = (B, in_dim, n_latent_frames)

    # Fixed seed for reproducibility
    torch.manual_seed(args.seed)
    dummy_latents = torch.randn(latent_shape, dtype=DTYPE, device=DEVICE)
    dummy_timestep = torch.tensor([500], dtype=DTYPE, device=DEVICE)
    text_dim = pipe.text_encoder.dim
    dummy_context = torch.randn(B, 512, text_dim, dtype=DTYPE, device=DEVICE)

    # Save dummy inputs
    save_f32("dummy_latents", dummy_latents, out_dir)
    save_f32("dummy_timestep", dummy_timestep, out_dir)
    save_f32("dummy_context", dummy_context, out_dir)

    # ── 1. Time embedding ────────────────────────────────────────────────
    from ..diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
    freq_dim = dit.config.get("freq_dim", 256)
    t_sin = sinusoidal_embedding_1d(freq_dim, dummy_timestep)
    save_f32("time_sinusoidal", t_sin, out_dir)

    t = dit.time_embedding(t_sin)
    save_f32("time_embedding_out", t, out_dir)

    t_mod = dit.time_projection(t).unflatten(1, (6, dit.config["dim"]))
    save_f32("time_projection_out", t_mod, out_dir)

    # ── 2. Text embedding ────────────────────────────────────────────────
    context = dit.text_embedding(dummy_context)
    save_f32("text_embedding_out", context, out_dir)

    # ── 3. Patchify ──────────────────────────────────────────────────────
    x = dummy_latents.clone()
    save_f32("pre_patchify", x, out_dir)
    x, (f,) = dit.patchify(x)
    save_f32("post_patchify", x, out_dir)
    print(f"  patchify: {dummy_latents.shape} -> {x.shape}")

    # ── 4. Build RoPE freqs ──────────────────────────────────────────────
    freqs = torch.cat([
        dit.freqs_cis_0[:f].view(f, -1).expand(f, -1),
        dit.freqs_cis_1[:f].view(f, -1).expand(f, -1),
        dit.freqs_cis_2[:f].view(f, -1).expand(f, -1),
    ], dim=-1).reshape(f, 1, -1)
    save_f32("freqs", freqs, out_dir)
    print(f"  freqs: {freqs.shape}")

    # ── 5. Register hooks and run blocks ─────────────────────────────────
    blocks_data = [{} for _ in range(len(dit.blocks))]
    hooks = []
    for i, block in enumerate(dit.blocks):
        pre_hook, post_hook = hook_block(i, out_dir, blocks_data)
        hooks.append(block.register_forward_pre_hook(pre_hook))
        hooks.append(block.register_forward_hook(post_hook))

    print(f"  Running {len(dit.blocks)} blocks with hooks...")
    for block in tqdm(dit.blocks):
        x = block(x, context, t_mod, freqs)

    # Remove hooks
    for h in hooks:
        h.remove()

    # ── 6. Head ──────────────────────────────────────────────────────────
    save_f32("pre_head", x, out_dir)
    x = dit.head(x, t)
    save_f32("post_head", x, out_dir)

    # ── 7. Unpatchify ────────────────────────────────────────────────────
    save_f32("pre_unpatchify", x, out_dir)
    x = dit.unpatchify(x, (f,))
    save_f32("post_unpatchify", x, out_dir)

    # ── 8. VAE Decoder ───────────────────────────────────────────────────
    print(f"\n  Running VAE decoder on {x.shape}...")
    # Register hooks on VAE decoder submodules
    decoder = vae.decoder.model
    for i, layer in enumerate(decoder):
        name = f"vae_layer_{i}"
        def make_vae_hook(name):
            def hook(m, args, output):
                save_f32(name, output, out_dir)
            return hook
        if isinstance(layer, torch.nn.Sequential):
            for j, sub in enumerate(layer):
                sub.register_forward_hook(make_vae_hook(f"{name}_{j}"))
        else:
            layer.register_forward_hook(make_vae_hook(name))

    audio = vae.decode(x)
    save_f32("vae_output", audio, out_dir)
    print(f"  VAE output: {audio.shape}")

    # ── 9. Save config for the C++ port ─────────────────────────────────
    config = {
        "dim": dit.config["dim"],
        "in_dim": dit.config["in_dim"],
        "out_dim": dit.config["out_dim"],
        "ffn_dim": dit.config["ffn_dim"],
        "text_dim": dit.config["text_dim"],
        "freq_dim": dit.config["freq_dim"],
        "eps": dit.config.get("eps", 1e-6),
        "patch_size": list(dit.config["patch_size"]),
        "num_heads": dit.config["num_heads"],
        "num_layers": dit.config["num_layers"],
        "has_image_input": dit.config.get("has_image_input", False),
        "vae_type": dit.config.get("vae_type", "dac"),
        "head_dim": dit.config["dim"] // dit.config["num_heads"],
        "n_latent_frames": n_latent_frames,
        "n_seconds": n_seconds,
        "n_samples": n_samples,
        "sample_rate": sample_rate,
        "vae_latent_dim": vae.latent_dim,
        "vae_hop_length": vae.hop_length,
        "vae_sample_rate": vae.sample_rate,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"\n{'='*50}")
    print(f"All tensors dumped to {out_dir}/")
    print(f"  Block inputs:  block_{i}_pre/  for i=0..{len(dit.blocks)-1}")
    print(f"  Block outputs: block_{i}_post/  for i=0..{len(dit.blocks)-1}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
