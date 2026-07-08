#!/usr/bin/env python3
"""Convert DAC VAE checkpoint from .pth to GGUF format.

Usage:
    python tools/convert_vae.py /path/to/vae_128d_48k.pth [/path/to/output.gguf]

The output GGUF uses prefix "moss_sfx_v2.vae." and stores raw weight_norm
parameters (weight_v, weight_g, bias) as F32 — the C++ VAE runner computes
the effective weight at bind time.

For continuous-mode VAEs (continuous=True), the post_quant_conv weights
(plain Conv1d, no weight-norm) are also included under
    moss_sfx_v2.vae.post_quant_conv.weight
    moss_sfx_v2.vae.post_quant_conv.bias

The input .pth must be a DAC VAE decoder checkpoint with keys like:
    decoder.model.0.weight_v, decoder.model.0.weight_g, decoder.model.0.bias
    decoder.model.1.block.0.alpha
    decoder.model.1.block.1.weight_v, ...
    ...
    decoder.model.7.bias
    post_quant_conv.weight, post_quant_conv.bias   (continuous=True only)
"""
import argparse
import os
import sys

import gguf
import numpy as np
import torch


def convert_vae(input_path: str, output_path: str) -> None:
    # weights_only=False: the checkpoint has a nested metadata dict with
    # lists (decoder_rates, etc.) that weights_only=True rejects.
    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)

    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        meta = ckpt.get("metadata", {})
        kwargs = meta.get("kwargs", {})
    else:
        sd = ckpt
        kwargs = {}

    decoder_sd = {k: v for k, v in sd.items() if k.startswith("decoder.model.")}
    if not decoder_sd:
        print("Error: no decoder.model.* keys found in checkpoint", file=sys.stderr)
        sys.exit(1)

    # Metadata
    latent_dim = kwargs.get("latent_dim", 128)
    decoder_dim = kwargs.get("decoder_dim", 2048)
    decoder_rates = kwargs.get("decoder_rates", [8, 5, 4, 3, 2])
    sample_rate = kwargs.get("sample_rate", 48000)
    continuous = kwargs.get("continuous", False)
    hop_length = int(np.prod(decoder_rates))

    # post_quant_conv (continuous VAE only): plain Conv1d, no weight-norm.
    post_quant_sd = {}
    if continuous:
        post_quant_sd = {
            k: v for k, v in sd.items() if k.startswith("post_quant_conv.")
        }
        if not post_quant_sd:
            print("Error: continuous=True but no post_quant_conv.* keys found",
                  file=sys.stderr)
            sys.exit(1)

    # Write GGUF
    writer = gguf.GGUFWriter(output_path, "mse2-vae")

    # KV metadata — the C++ loader reads these to set VAEConfig
    writer.add_uint32("moss_sfx_v2.vae.latent_dim", latent_dim)
    writer.add_uint32("moss_sfx_v2.vae.decoder_dim", decoder_dim)
    writer.add_uint32("moss_sfx_v2.vae.hop_length", hop_length)
    writer.add_uint32("moss_sfx_v2.vae.sample_rate", sample_rate)
    writer.add_bool("moss_sfx_v2.vae.continuous", continuous)

    # Convert decoder tensors: rename decoder.model. → moss_sfx_v2.vae.
    n_tensors = 0
    for k, v in decoder_sd.items():
        new_key = k.replace("decoder.model.", "moss_sfx_v2.vae.")
        arr = v.detach().numpy().astype(np.float32)
        writer.add_tensor(new_key, arr)
        n_tensors += 1

    # Convert post_quant_conv tensors (plain Conv1d weight/bias).
    for k, v in post_quant_sd.items():
        new_key = "moss_sfx_v2.vae." + k  # post_quant_conv.weight → moss_sfx_v2.vae.post_quant_conv.weight
        arr = v.detach().numpy().astype(np.float32)
        writer.add_tensor(new_key, arr)
        n_tensors += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"[vae] wrote {n_tensors} tensors to {output_path}")
    print(f"[vae]   latent_dim={latent_dim}, decoder_dim={decoder_dim}, "
          f"hop_length={hop_length}, rates={decoder_rates}, "
          f"continuous={continuous}")
    if post_quant_sd:
        print(f"[vae]   post_quant_conv: "
              f"{post_quant_sd['post_quant_conv.weight'].shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DAC VAE .pth → GGUF")
    parser.add_argument("input", help="Path to .pth checkpoint")
    parser.add_argument("output", nargs="?", default=None,
                        help="Output .gguf path (default: input path with .gguf extension)")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.splitext(args.input)[0] + ".gguf"

    convert_vae(args.input, args.output)


if __name__ == "__main__":
    main()
