#!/usr/bin/env python3
"""Evaluate audio quality via Cloud VLM (MiMo-V2.5).

DEPRECATED — this is now a thin CLI wrapper around `tools/audio_vlm.py`,
which is the faithful, stdlib-only port of the media-analysis-mcp `cloud_vlm`
tool. New code should import from `audio_vlm` directly:

    from audio_vlm import describe_audio
    verdict = describe_audio("/tmp/out.wav")

Usage:
    python3 tools/audio_vlm_test.py test_mse2_output.wav
    python3 tools/audio_vlm_test.py /path/to/audio.wav --prompt "Describe the music quality"
"""

import sys
from pathlib import Path

# Make sibling audio_vlm.py importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio_vlm import describe_audio_raw, DEFAULT_PROMPT  # noqa: E402


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", help="Path to WAV/MP3/OGG/FLAC file")
    parser.add_argument("--prompt", "-p", default=None,
                        help="Evaluation prompt (default: structured quality audit)")
    parser.add_argument("--json", action="store_true",
                        help="Emit raw result JSON")
    args = parser.parse_args()

    res = describe_audio_raw(args.audio, args.prompt)
    if args.json:
        import json
        print(json.dumps(res, indent=2))
        return 0 if res.get("success") else 1

    if not res.get("success"):
        print(f"ERROR: {res.get('error')}", file=sys.stderr)
        return 1

    print("\n" + "=" * 60)
    print("VLM ANALYSIS")
    print("=" * 60)
    print(res["response"])
    usage = res.get("usage") or {}
    if usage:
        print(f"\nTokens: {usage.get('total_tokens', '?')} total "
              f"({usage.get('prompt_tokens', '?')} prompt, "
              f"{usage.get('completion_tokens', '?')} completion) "
              f"[model={res.get('model', '?')}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
