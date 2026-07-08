#!/usr/bin/env python3
"""Convert Silero VAD weights from safetensors to GGUF.

Usage:
    python tools/convert_silero_vad.py /path/to/silero_vad_16k.safetensors \\
        [/path/to/silero_vad.gguf]

The output GGUF uses prefix "silero_vad." and stores all weights as F32
(Silero is 1.8 MB — quantization is pointless).

Source model: https://huggingface.co/snakers4/silero-vad
"""

import sys
from pathlib import Path

# gguf-py ships in third_party/llama.cpp/gguf-py
_GGUF_PY = Path(__file__).resolve().parent.parent / "third_party" / "llama.cpp" / "gguf-py"
if str(_GGUF_PY) not in sys.path:
    sys.path.insert(0, str(_GGUF_PY))

import gguf          # noqa: E402
import numpy as np   # noqa: E402
import safetensors.torch  # noqa: E402

PREFIX = "silero_vad."


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".gguf")

    if not src.exists():
        print(f"input not found: {src}", file=sys.stderr)
        return 2

    tensors = safetensors.torch.load_file(str(src))
    print(f"[convert_silero_vad] loaded {len(tensors)} tensors from {src}")

    writer = gguf.GGUFWriter(path=str(dst), arch="silero_vad")

    # Metadata consumed by the runtime on load.
    writer.add_string("silero_vad.architecture", "silero-v6.2")
    writer.add_string("silero_vad.source", "snakers4/silero-vad")
    writer.add_uint32("silero_vad.sample_rate", 16000)
    writer.add_uint32("silero_vad.chunk_samples", 512)    # 32 ms at 16 kHz
    writer.add_uint32("silero_vad.fft_n", 512)
    writer.add_uint32("silero_vad.stft_bins", 257)
    writer.add_uint32("silero_vad.hidden_size", 128)

    for name, tensor in tensors.items():
        arr = tensor.detach().cpu().numpy()
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        gguf_name = PREFIX + name
        writer.add_tensor(gguf_name, arr)
        print(f"  {gguf_name}: shape={list(arr.shape)}, dtype={arr.dtype}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"[convert_silero_vad] wrote {dst} ({dst.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
