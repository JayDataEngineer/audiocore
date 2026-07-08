#!/usr/bin/env python3
"""Verify audio output with a cloud multimodal VLM (Xiaomi MiMo-V2.5).

Extracted from the local `media-analysis-mcp` server's `cloud_vlm` tool at
`~/Documents/programs/mcp/media-mcp/src/services/cloud_vlm_service.py`.

This is a faithful, self-contained port — stdlib only, no dependency on the
MCP server being up, no httpx/install required. Same model, same endpoint,
same auth resolution, same multimodal content-block schema as the source.

Auth resolution (first non-empty wins):
  1. MEDIA_CLOUD_VLM_API_KEY env var
  2. $OPENCODE_AUTH_FILE
  3. $OPENCODE_AUTH_DIR/auth.json
  4. ~/.local/share/opencode/auth.json  → "opencode-go" entry, then "opencode"
  5. ~/.opencode/auth.json              (legacy opencode location)

The key file lives OUTSIDE this repo, so it is never committed. See .gitignore
for the defensive block that catches any accidental local copy.

Default backend (override via env):
  MEDIA_CLOUD_VLM_BASE_URL  (default https://opencode.ai/zen/go/v1)
  MEDIA_CLOUD_VLM_MODEL     (default mimo-v2.5)
  MEDIA_CLOUD_VLM_TIMEOUT   (default 120s)

Public API:
  describe_audio(path, prompt=None, max_tokens=2048, temperature=0.4) -> str
      Returns the model's text verdict on the audio file. Raises
      AudioVlmError on any failure (missing key, network, upstream error,
      malformed response). Use this when you want to FAIL FAST on
      verification problems — which is what you want for CI / before
      declaring a generation "done".

  describe_audio_raw(path, prompt=None, ...) -> dict
      Returns the full result dict: {success, response, model, usage} on
      success, or {success: False, error} on failure (no raise).

  verify_audio(path, max_tokens=2048, temperature=0.4) -> str
      STRICT MODE — second VLM function with a HARDCODED adversarial prompt.
      There is NO prompt parameter. The question ("Is this white noise, or
      screeching. Is this clean audio? Can you describe the audio in
      detail.") is fixed in source and cannot be overridden by the caller.
      Use this — NOT describe_audio — for any gate that must actually block
      bad output, so a caller cannot soften the prompt to manufacture a PASS.

  verify_audio_raw(path, ...) -> dict
      Same as verify_audio but returns a result dict instead of raising.

  read_api_key() -> str
      Exposes the auth-resolution chain. Raises AudioVlmError if no key.

CLI:
  python3 tools/audio_vlm.py path/to/audio.wav                # strict (default)
  python3 tools/audio_vlm.py path/to/audio.wav --verify       # strict (explicit)
  python3 tools/audio_vlm.py path/to/audio.wav --describe ... # custom prompt
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Optional

# ─── config (env-overridable, mirrors media-mcp settings.py) ────────────────

DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1"
DEFAULT_MODEL = "mimo-v2.5"
DEFAULT_TIMEOUT = 120.0
MAX_OUTPUT_TOKENS_CEILING = 4096  # hard ceiling from media-mcp settings

# OpenAI input_audio spec accepts only "wav" or "mp3". Anything else is mapped.
_AUDIO_FORMAT_FROM_MIME = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/ogg": "mp3",    # ogg not in spec; mp3 container is closest fallback
    "audio/flac": "wav",   # FLAC not in spec; client should pre-convert
    "audio/x-m4a": "mp3",
    "audio/aac": "mp3",
}
_AUDIO_FORMAT_FROM_EXT = {
    ".wav": "wav", ".mp3": "mp3", ".ogg": "mp3", ".flac": "wav",
    ".m4a": "mp3", ".aac": "mp3",
}

_DATA_URI_MIME_RE = re.compile(r"^data:([\w/+.-]+);base64,(.+)$", re.DOTALL)

# Default verification prompt — deliberately adversarial. Asks for a 1-10
# score AND a pass/fail, so automated callers can grep the verdict.
DEFAULT_PROMPT = (
    "Analyze this audio carefully and HONESTLY. Do not be polite — if it is "
    "garbage, say so. Report:\n"
    "1) What you hear (speech? music? noise? static? silence?).\n"
    "2) If speech: is it intelligible? Transcribe the first phrase.\n"
    "3) If music: is there rhythm, melody, harmony? Name instruments if any.\n"
    "4) Artifacts: distortion, robotic/stutter/industrial noise, dropouts,\n"
    "   wrong sample rate, clipping, phase issues.\n"
    "5) Production quality 1-10 (10 = studio reference, 1 = unusable).\n"
    "End with a single line in EXACTLY this format:\n"
    "VERDICT: PASS  | reason\n"
    "or\n"
    "VERDICT: FAIL  | reason\n"
    "PASS only if the audio is fit for its apparent purpose."
)

# ─── STRICT MODE — hardcoded anti-cheat prompt ─────────────────────────────
# verify_audio() exposes NO prompt parameter, by design. The question below is
# fixed in source. Do NOT soften it. If you change it, change it HERE only —
# never let a caller pass their own. The point is to stop a downstream agent
# from running the VLM with an easy prompt ("is this audio?") to manufacture a
# PASS on garbage output. This prompt specifically forces the model to answer
# for white noise, screeching, and cleanliness — the failure modes that
# actually matter for TTS/music generation.
VERIFY_PROMPT = (
    "Is this white noise, or screeching. Is this clean audio? "
    "Can you describe the audio in detail.\n\n"
    "Be brutal. If it is white noise, static, screeching, industrial "
    "machinery, robotic stuttering, buzzing, clicking, or any other kind of "
    "broken or unusable output, say so explicitly and do not hedge.\n\n"
    "After your description, end with a single line in EXACTLY this format:\n"
    "VERDICT: PASS  | reason\n"
    "or\n"
    "VERDICT: FAIL  | reason\n"
    "FAIL if there is ANY white noise, screeching, or corruption. PASS only "
    "if the audio is genuinely clean."
)


class AudioVlmError(RuntimeError):
    """Raised by describe_audio() on any failure (auth/network/upstream/parse)."""


# ─── auth resolution — faithful port of cloud_vlm_service._read_opencode_key ─

def _auth_file_candidates() -> list[Path]:
    cands: list[Path] = []
    env_file = os.environ.get("OPENCODE_AUTH_FILE")
    if env_file:
        cands.append(Path(env_file))
    env_dir = os.environ.get("OPENCODE_AUTH_DIR")
    if env_dir:
        cands.append(Path(env_dir) / "auth.json")
    cands.append(Path.home() / ".local" / "share" / "opencode" / "auth.json")
    cands.append(Path.home() / ".opencode" / "auth.json")
    return cands


def _read_opencode_key(service_id: str) -> str:
    """Best-effort read of one key from opencode's auth.json. Returns "" if missing."""
    for p in _auth_file_candidates():
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
            entry = data.get(service_id) or {}
            if isinstance(entry, dict):
                key = entry.get("key")
                if isinstance(key, str) and key:
                    return key
        except Exception:
            continue
    return ""


def read_api_key() -> str:
    """First non-empty of: MEDIA_CLOUD_VLM_API_KEY → opencode-go → opencode."""
    env_key = os.environ.get("MEDIA_CLOUD_VLM_API_KEY", "").strip()
    if env_key:
        return env_key
    for svc in ("opencode-go", "opencode"):
        k = _read_opencode_key(svc)
        if k:
            return k
    raise AudioVlmError(
        "No API key. Set MEDIA_CLOUD_VLM_API_KEY, or run `opencode auth login` "
        "so ~/.local/share/opencode/auth.json has an 'opencode-go' entry."
    )


# ─── audio → multimodal content block (port of _audio_block) ────────────────

def _audio_block_from_file(path: str) -> dict:
    """Read a local audio file → OpenAI input_audio block {data, format}."""
    p = Path(path)
    if not p.exists():
        raise AudioVlmError(f"Audio file not found: {path}")
    raw = p.read_bytes()
    if not raw:
        raise AudioVlmError(f"Audio file is empty: {path}")

    mime, _ = mimetypes.guess_type(str(p))
    mime = (mime or "audio/wav").lower()
    fmt = _AUDIO_FORMAT_FROM_MIME.get(mime)
    if fmt is None:
        ext = p.suffix.lower()
        fmt = _AUDIO_FORMAT_FROM_EXT.get(ext, "mp3")
    return {
        "type": "input_audio",
        "input_audio": {
            "data": base64.b64encode(raw).decode(),
            "format": fmt,  # "wav" | "mp3"
        },
    }


def _audio_block_from_source(source: str) -> dict:
    """Accept a local path, a data: URI, or raw base64. (URLs not downloaded
    in the sync port — pass a local file for verification.)"""
    if source.startswith(("http://", "https://")):
        raise AudioVlmError(
            "HTTP audio URLs are not supported by the sync port. Download the "
            "file first, then pass the local path."
        )
    m = _DATA_URI_MIME_RE.match(source)
    if m:
        mime = m.group(1).lower()
        data = base64.b64decode(m.group(2))
        fmt = _AUDIO_FORMAT_FROM_MIME.get(mime) or _AUDIO_FORMAT_FROM_EXT.get(
            "." + mime.split("/")[-1].split("-")[-1], "mp3"
        )
        return {"type": "input_audio", "input_audio": {
            "data": base64.b64encode(data).decode(), "format": fmt}}
    # Local path — also catches raw-base64 falling through to file-not-found
    if os.path.exists(source):
        return _audio_block_from_file(source)
    # Last resort: treat as raw base64, assume mp3
    try:
        data = base64.b64decode(source, validate=True)
    except Exception as e:
        raise AudioVlmError(f"Unrecognized audio source (prefix={source[:40]!r}): {e}")
    return {"type": "input_audio", "input_audio": {
        "data": base64.b64encode(data).decode(), "format": "mp3"}}


# ─── core: call the OpenAI-compatible chat completions endpoint ─────────────

def _chat_completions(
    api_key: str,
    prompt: str,
    audio_block: dict,
    base_url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> dict:
    payload = json.dumps({
        "model": model,
        "max_tokens": max(1, min(int(max_tokens), MAX_OUTPUT_TOKENS_CEILING)),
        "temperature": float(temperature),
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            audio_block,
        ]}],
    }).encode()

    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "audiocore-audio-vlm/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body)


def describe_audio_raw(
    path: str,
    prompt: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.4,
    *,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
) -> dict:
    """Verify an audio file with the cloud VLM. Returns a result dict.

    On success: {"success": True, "response": str, "model": str, "usage": dict}
    On failure: {"success": False, "error": str}
    """
    base_url = base_url or os.environ.get(
        "MEDIA_CLOUD_VLM_BASE_URL", DEFAULT_BASE_URL)
    model = model or os.environ.get(
        "MEDIA_CLOUD_VLM_MODEL", DEFAULT_MODEL)
    timeout = timeout or float(os.environ.get(
        "MEDIA_CLOUD_VLM_TIMEOUT", DEFAULT_TIMEOUT))

    try:
        api_key = read_api_key()
    except AudioVlmError as e:
        return {"success": False, "error": str(e)}

    if not prompt:
        prompt = DEFAULT_PROMPT

    try:
        audio_block = _audio_block_from_source(path)
    except AudioVlmError as e:
        return {"success": False, "error": str(e)}

    try:
        data = _chat_completions(
            api_key, prompt, audio_block, base_url, model,
            max_tokens, temperature, timeout)
    except urllib.error.HTTPError as e:
        snippet = ""
        try:
            snippet = e.read().decode()[:500]
        except Exception:
            pass
        return {"success": False, "error": f"Upstream HTTP {e.code}: {snippet}",
                "status_code": e.code}
    except urllib.error.URLError as e:
        return {"success": False, "error": f"Network error: {e.reason}"}
    except Exception as e:
        return {"success": False, "error": f"Request failed: {type(e).__name__}: {e}"}

    if "error" in data:
        return {"success": False, "error": f"Upstream: {data['error']}"}

    try:
        text = data["choices"][0]["message"]["content"]
    except Exception as e:
        return {"success": False,
                "error": f"Malformed upstream response: {type(e).__name__}: {e}"}

    return {
        "success": True,
        "response": text,
        "model": data.get("model", model),
        "usage": data.get("usage"),
    }


def describe_audio(
    path: str,
    prompt: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.4,
    *,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
) -> str:
    """Verify an audio file with the cloud VLM. Returns the verdict text.

    Raises AudioVlmError on any failure (no key, network, upstream error,
    malformed response). Use this when you want to FAIL FAST — which is
    what you want before declaring a generation complete.
    """
    res = describe_audio_raw(
        path, prompt, max_tokens, temperature,
        base_url=base_url, model=model, timeout=timeout)
    if not res.get("success"):
        raise AudioVlmError(res.get("error", "Unknown VLM failure"))
    return res["response"]


def verify_audio_raw(
    path: str,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    *,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
) -> dict:
    """STRICT MODE VLM check. Hardcoded prompt — NO prompt parameter.

    This is the anti-cheat path. The question ("Is this white noise, or
    screeching. Is this clean audio? Can you describe the audio in detail.")
    is fixed in source (see VERIFY_PROMPT) and cannot be overridden. Use this
    — NOT describe_audio_raw — for any gate that must actually block bad
    output. Returns the same result dict shape as describe_audio_raw.

    Default temperature is 0.0 (greedy) for determinism — a verification gate
    must give the same verdict on the same input across runs. Override with
    the `temperature` arg only if you know why.
    """
    return describe_audio_raw(
        path, VERIFY_PROMPT, max_tokens, temperature,
        base_url=base_url, model=model, timeout=timeout)


def verify_audio(
    path: str,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    *,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
) -> str:
    """STRICT MODE VLM check. Hardcoded prompt — NO prompt parameter.

    Raises AudioVlmError on any failure. Returns the verdict text. Use this
    for any pre-commit / pre-complete gate. Pair with an assertion on the
    trailing VERDICT line:

        from tools.audio_vlm import verify_audio
        assert "VERDICT: PASS" in verify_audio("/tmp/out.wav")

    Default temperature is 0.0 (greedy) for determinism.
    """
    res = verify_audio_raw(
        path, max_tokens, temperature,
        base_url=base_url, model=model, timeout=timeout)
    if not res.get("success"):
        raise AudioVlmError(res.get("error", "Unknown VLM failure"))
    return res["response"]


# ─── CLI (thin wrapper; the existing tools/audio_vlm_test.py also wraps this) ─

def _main():
    parser = argparse.ArgumentParser(
        description="Verify audio output via cloud VLM (MiMo-V2.5). "
                    "Extracted from media-analysis-mcp cloud_vlm.")
    parser.add_argument("audio", help="Path to WAV/MP3/OGG/FLAC/M4A file")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true", default=True,
                      help="STRICT MODE (default): use the hardcoded anti-cheat "
                           "prompt. No override possible.")
    mode.add_argument("--describe", "-d", dest="describe", action="store_true",
                      help="FLEXIBLE MODE: use --prompt (or the default "
                           "structured audit prompt).")
    parser.add_argument("--prompt", "-p", default=None,
                        help="Custom prompt (only valid with --describe)")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", "-t", type=float, default=None,
                        help="Sampling temperature. Strict mode defaults to "
                             "0.0 (greedy/deterministic); --describe defaults "
                             "to 0.4.")
    parser.add_argument("--json", action="store_true",
                        help="Emit the full result dict as JSON (no banner)")
    args = parser.parse_args()

    if args.prompt and not args.describe:
        parser.error("--prompt requires --describe (strict mode has no "
                     "overrideable prompt by design).")

    # Mode-aware temperature default: strict = 0.0 (deterministic gate),
    # describe = 0.4 (more natural prose).
    temp = args.temperature if args.temperature is not None else (
        0.4 if args.describe else 0.0)

    if args.describe:
        res = describe_audio_raw(
            args.audio, args.prompt, args.max_tokens, temp)
    else:
        res = verify_audio_raw(
            args.audio, args.max_tokens, temp)

    if args.json:
        print(json.dumps(res, indent=2))
        return 0 if res.get("success") else 1

    if not res.get("success"):
        print(f"ERROR: {res.get('error')}", file=sys.stderr)
        return 1

    print("=" * 60, file=sys.stderr)
    print("VLM VERDICT", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(res["response"])
    usage = res.get("usage") or {}
    if usage:
        print(f"\nTokens: {usage.get('total_tokens', '?')} total "
              f"({usage.get('prompt_tokens', '?')} prompt, "
              f"{usage.get('completion_tokens', '?')} completion) "
              f"[model={res.get('model', '?')}", file=sys.stderr)
    # Exit code: PASS → 0, FAIL → 2, missing verdict → 1.
    # A missing VERDICT line is a FAILURE — the model dodging ("I can't tell",
    # "I don't want to fake an analysis", truncated output, etc.) must not pass
    # the gate. Only an explicit VERDICT: PASS counts as success.
    text = res["response"].upper()
    if "VERDICT: PASS" in text:
        return 0
    if "VERDICT: FAIL" in text:
        return 2
    print("ERROR: no VERDICT line in response — model dodged or truncated. "
          "Treating as failure.", file=sys.stderr)
    print("(full response printed above — investigate before re-running)",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(_main())
