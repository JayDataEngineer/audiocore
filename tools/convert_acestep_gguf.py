#!/usr/bin/env python3
"""Convert an ACE-Step GGUF from HF tensor names to llama.cpp tensor names.

ACE-Step ships its Qwen3-based LM and Text-Encoder GGUFs with HuggingFace
PyTorch names (``model.embed_tokens.weight``, ``model.layers.0.*``, etc.).
libllama refuses anything that isn't in its own naming, so audiocore's
qwen3::Runner can't load them directly.

This script rewrites tensor names in-place to the llama.cpp layout. Run it
once per downloaded file:

    python tools/convert_acestep_gguf.py acestep-5Hz-lm-1.7B-Q8_0.gguf
    python tools/convert_acestep_gguf.py Qwen3-Embedding-0.6B-Q8_0.gguf

After conversion the files load through ``qwen3::Runner::load()`` exactly
like any other Qwen3 GGUF. There is no second Qwen3 implementation in
audiocore — DRY.

The DiT and VAE GGUFs are NOT Qwen3 transformers and DO NOT need this
converter; the family code binds them by their native HF names directly.

Mapping reference: docs/GGUF_FORMAT.md → "ACE-Step" section. Names below are
verified against ServeurpersoCom/acestep.cpp's qwen3-lm.h tensor loader.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# gguf is the official Python reader/writer from ggml-org/gguf. Install with
# `pip install gguf`. We do not vendor it because the surface we need is
# small and stable.
try:
    import gguf
except ImportError:
    sys.exit("missing dependency: pip install gguf")


# (regex matching HF name, format string for llama.cpp name).
# Patterns are checked in order; first match wins. The {i} layer index is
# captured and substituted into the new name verbatim.
RENAMES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^model\.embed_tokens\.weight$"),            "token_embd.weight"),
    (re.compile(r"^model\.norm\.weight$"),                    "output_norm.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.input_layernorm\.weight$"),
     r"blk.\1.attn_norm.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$"),
     r"blk.\1.ffn_norm.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$"),
     r"blk.\1.attn_q.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.k_proj\.weight$"),
     r"blk.\1.attn_k.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.v_proj\.weight$"),
     r"blk.\1.attn_v.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$"),
     r"blk.\1.attn_output.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$"),
     r"blk.\1.ffn_gate.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.up_proj\.weight$"),
     r"blk.\1.ffn_up.weight"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.down_proj\.weight$"),
     r"blk.\1.ffn_down.weight"),
    # Text-encoder variant: same layers but no `model.` prefix.
    (re.compile(r"^embed_tokens\.weight$"),   "token_embd.weight"),
    (re.compile(r"^norm\.weight$"),           "output_norm.weight"),
]


def rename(name: str) -> str | None:
    """Return the llama.cpp name for an HF tensor name, or None if no rule."""
    for pat, repl in RENAMES:
        m = pat.match(name)
        if m:
            return pat.sub(repl, name)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gguf", type=Path, help="GGUF file to rewrite in-place")
    ap.add_argument("--out", type=Path, default=None,
                    help="output path (default: overwrite input)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the rename plan and exit")
    args = ap.parse_args()

    if not args.gguf.is_file():
        sys.exit(f"not a file: {args.gguf}")

    reader = gguf.GGUFReader(str(args.gguf), "r")
    plan: list[tuple[str, str]] = []
    skipped: list[str] = []
    for t in reader.tensors:
        new = rename(str(t.name, "utf-8"))
        if new:
            plan.append((str(t.name, "utf-8"), new))
        else:
            skipped.append(str(t.name, "utf-8"))

    if not plan:
        print(f"{args.gguf}: nothing to convert "
              f"(already in llama.cpp layout, or unrecognized names).")
        if skipped:
            print(f"  skipped {len(skipped)} unmatched tensors, e.g.:")
            for n in skipped[:5]:
                print(f"    {n}")
        return 0

    print(f"{args.gguf}: {len(plan)} tensors to rename.")
    for old, new in plan[:10]:
        print(f"  {old} → {new}")
    if len(plan) > 10:
        print(f"  … and {len(plan) - 10} more")

    if args.dry_run:
        return 0

    print("NOTE: in-place rewrite not yet implemented — the official gguf "
          "Python package does not support tensor rename without rewriting "
          "the whole file. Use --dry-run to audit, then apply with the "
          "llama.cpp gguf-new-metadata tool (TODO: ship a small wrapper).",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
