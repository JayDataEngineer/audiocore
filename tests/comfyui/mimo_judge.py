"""Mimo-2.5 multimodal judge for audio cleanliness verification.

Mimo (mimo-v2.5 via the OpenAI-compatible endpoint) listens to a clip and
returns a structured JSON verdict:

    {
        "speech_present": bool,
        "intelligible":   bool,
        "noise_level":    "clean" | "moderate" | "noisy",
        "clipping":       bool,
        "quality_score":  int  (1..10),
        "description":    str   (free-form)
    }

This is the only judge we have for "does the audio actually sound right"
without a human in the loop. It catches regressions that byte-size + format
+ duration checks can't (e.g. moss_tts producing 30s of mechanical buzzing
that's technically a valid FLAC but sounds terrible).

OPSEC: API key is read from the MIMO_API_KEY env var ONLY. No hardcoded
fallback — that would leak a credential into git history. Tests that need
Mimo should pytest.skip() when the env var is unset.

Endpoint + API shape come from /tmp/mimo_e2e_test.py (the throwaway script).
"""
from __future__ import annotations

import base64
import json
import os
from typing import Any

import requests

MIMO_URL = "https://opencode.ai/zen/go/v1/chat/completions"
MIMO_MODEL = "mimo-v2.5"

# We deliberately do NOT ship a default API key. Set MIMO_API_KEY in the env.
_DEFAULT_TIMEOUT = 120


def mimo_available() -> bool:
    """True iff MIMO_API_KEY is set AND the endpoint answers a 1-token ping."""
    key = os.environ.get("MIMO_API_KEY")
    if not key:
        return False
    try:
        r = requests.post(
            MIMO_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0",
            },
            json={
                "model": MIMO_MODEL,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            },
            timeout=15,
        )
        return r.status_code == 200
    except Exception:
        return False


def _audio_format(filename: str) -> str:
    """Map a filename extension to the OpenAI input_audio format tag."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else "wav"
    return {
        "wav": "wav",
        "mp3": "mp3",
        "flac": "flac",
        "ogg": "ogg",
        "m4a": "mp4",
        "aac": "aac",
    }.get(ext, "wav")


def judge_audio_bytes(blob: bytes, *, filename: str = "audio.flac",
                      question: str | None = None,
                      timeout: int = _DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Send audio bytes to Mimo and return the parsed JSON verdict.

    Raises RuntimeError if Mimo is unavailable, returns a non-200, or
    returns content that isn't valid JSON.
    """
    key = os.environ.get("MIMO_API_KEY")
    if not key:
        raise RuntimeError(
            "MIMO_API_KEY env var is not set — Mimo judge unavailable"
        )

    if question is None:
        question = (
            "You are an audio quality expert. Listen carefully and judge "
            "this clip. Respond with ONLY a JSON object — no prose, no "
            "markdown fences — with this exact shape: "
            "{\"speech_present\": true|false, "
            "\"intelligible\": true|false, "
            "\"noise_level\": \"clean\"|\"moderate\"|\"noisy\", "
            "\"clipping\": true|false, "
            "\"quality_score\": <int 1-10>, "
            "\"description\": \"<one sentence>\"}."
        )

    audio_b64 = base64.b64encode(blob).decode()
    content = [
        {"type": "text", "text": question},
        {
            "type": "input_audio",
            "input_audio": {
                "data": audio_b64,
                "format": _audio_format(filename),
            },
        },
    ]

    r = requests.post(
        MIMO_URL,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
        },
        json={
            "model": MIMO_MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 400,
            "temperature": 0.0,
        },
        timeout=timeout,
    )
    if r.status_code != 200:
        raise RuntimeError(
            f"Mimo returned HTTP {r.status_code}: {r.text[:300]}"
        )
    body = r.json()
    # Defensive: choices[] may be missing or content may be null (refusal,
    # rate-limit, content-filter, or a transient upstream issue).
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError(
            f"Mimo returned no choices. Body: {json.dumps(body)[:400]}"
        )
    message = choices[0].get("message") or {}
    text = message.get("content") or ""
    # reasoning_content is sometimes populated when content is empty.
    if not text and message.get("reasoning_content"):
        text = message["reasoning_content"]
    text = text.strip()
    if not text:
        raise RuntimeError(
            f"Mimo returned empty content. Body: {json.dumps(body)[:400]}"
        )
    # Strip markdown code fences if present.
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Mimo didn't return valid JSON: {e}. Raw: {text[:300]}"
        ) from e


# Convenience predicates for tests that want a single verdict.
def is_clean_audio(verdict: dict[str, Any], *, min_score: int = 6) -> bool:
    """True if Mimo judged the clip as non-degenerate speech/music.

    Defaults require: speech_present=true, noise_level in {clean, moderate},
    no clipping, quality_score >= 6. Tests can override `min_score` for
    stricter or laxer thresholds.
    """
    if not verdict:
        return False
    return (
        verdict.get("speech_present") is True
        and verdict.get("noise_level") in ("clean", "moderate")
        and verdict.get("clipping") is not True
        and int(verdict.get("quality_score", 0)) >= min_score
    )
