#!/usr/bin/env python3
"""Convert MOSS-TTS / ACE-Step safetensors weights to the GGUF format
audiocore consumes.

Stage 1 of the project requires existing GGUF releases that the community
already produced (OpenMOSS-Team/MOSS-TTS-GGUF, Serveurperso/ACE-Step-1.5-GGUF).
This converter exists for two reasons:

  1. Regenerating GGUFs from a fresh safetensors release the community
     hasn't quantized yet.
  2. Producing our own quantization variants (Q5_K, Q4_K_M, …) when we
     need to fit a tighter VRAM budget than Q8_0.

Implementation strategy: thin Python wrapper around llama.cpp's
`gguf.py` + `convert.py` machinery, with MOSS/ACE-Step-specific tensor
naming + metadata writing. Mirrors how `pwilkin/openmoss/scripts/convert_hf_to_gguf.py`
works today.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path


def convert_moss_tts(safetensors_dir: Path, extras_dir: Path | None,
                     output: Path, quant: str = "q8_0") -> None:
    # TODO(Phase 2): port pwilkin/openmoss's conversion logic.
    raise NotImplementedError("MOSS-TTS conversion not yet implemented")


def convert_ace_step(safetensors_dir: Path, output_dir: Path,
                     quant: str = "q8_0") -> None:
    # TODO(Phase 2): port ServeurpersoCom/acestep.cpp's conversion logic.
    raise NotImplementedError("ACE-Step conversion not yet implemented")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", required=True, choices=["moss_tts", "ace_step"])
    p.add_argument("--input",  required=True, type=Path,
                   help="safetensors directory")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--extras", type=Path,
                   help="(MOSS only) codec auxiliaries directory")
    p.add_argument("--quant",  default="q8_0",
                   choices=["f16", "q8_0", "q5_k_m", "q4_k_m", "q4_0"])
    args = p.parse_args()

    if args.family == "moss_tts":
        convert_moss_tts(args.input, args.extras, args.output, args.quant)
    else:
        convert_ace_step(args.input, args.output, args.quant)
    return 0


if __name__ == "__main__":
    sys.exit(main())
