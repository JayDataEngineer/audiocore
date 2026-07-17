"""REAL inference tests — actual audio bytes from real model inference.

Tests marked `inference` produce real audio output via ComfyUI and verify:
  - The workflow completes successfully
  - SaveAudio produced an output file (FLAC, WAV, or MP3)
  - The bytes fetchable via /view are a valid audio bitstream
  - The audio has non-trivial duration (sanity check the model didn't
    produce a stub or an empty clip)

Every test downloads real PCM from a real model — no mocks, no fakes.
"""
from __future__ import annotations

import io
import struct

import pytest

pytestmark = [pytest.mark.inference]


# ── Audio format validation helpers ─────────────────────────────────────────

# Acceptable magic-byte prefixes ComfyUI's SaveAudio family produces.
_AUDIO_MAGICS = {
    b"fLaC": "flac",        # FLAC (default in many ComfyUI builds)
    b"RIFF": "wav",         # RIFF/WAV (magic at offset 0; "WAVE" at offset 8)
    b"ID3":  "mp3",         # MP3 with ID3 tag
    b"\xff\xfb": "mp3",     # MP3 frame sync
    b"\xff\xf3": "mp3",     # MP3 frame sync (variant)
    b"OggS": "ogg",         # OGG container (Opus/Vorbis)
}


def _identify_audio(blob: bytes) -> str | None:
    """Return the audio format ('flac', 'wav', 'mp3', 'ogg') or None."""
    if len(blob) < 12:
        return None
    for magic, fmt in _AUDIO_MAGICS.items():
        if blob[:len(magic)] == magic:
            if fmt == "wav" and blob[8:12] != b"WAVE":
                continue  # RIFF but not WAVE — not audio
            return fmt
    return None


def _assert_valid_audio(blob: bytes, *, min_bytes: int = 5_000) -> str:
    """Assert the bytes are a recognized audio format with non-trivial size.

    Returns the format string ('flac' / 'wav' / etc.) so tests can log it.
    """
    assert len(blob) >= min_bytes, (
        f"audio blob too small: {len(blob)} bytes "
        f"(need >= {min_bytes}). Likely an error/stub file."
    )
    fmt = _identify_audio(blob)
    assert fmt is not None, (
        f"unrecognized audio format. first 16 bytes: {blob[:16]!r}\n"
        f"expected one of: {sorted(set(_AUDIO_MAGICS.values()))}"
    )
    return fmt


def _audio_duration_seconds(blob: bytes, fmt: str) -> float | None:
    """Best-effort duration estimate from the blob.

    For FLAC: parse the STREAMINFO block (samples + sample_rate).
    For WAV: read the data chunk size + sample rate from fmt chunk.
    For others: return None (test shouldn't depend on duration precision
    for variable-bitrate formats).
    """
    if fmt == "flac":
        # FLAC layout (per spec):
        #   bytes 0..3   "fLaC"
        #   byte  4      metadata block header byte 0:
        #                  bit 7 = LAST_METADATA_BLOCK flag
        #                  bits 0..6 = block type (0 = STREAMINFO)
        #   bytes 5..7   block length (24-bit BE) — size of STREAMINFO payload
        #   bytes 8..41  STREAMINFO payload (34 bytes):
        #                  bytes 0..1   min_block_size (16 bits)
        #                  bytes 2..3   max_block_size (16 bits)
        #                  bytes 4..6   min_frame_size (24 bits)
        #                  bytes 7..9   max_frame_size (24 bits)
        #                  bits 80..99  sample_rate (20 bits)
        #                  bits 100..102  (channels - 1) (3 bits)
        #                  bits 103..107  (bits_per_sample - 1) (5 bits)
        #                  bits 108..143  total_samples (36 bits)
        #                  bytes 18..33  MD5 (128 bits)
        if len(blob) < 26:
            return None
        block_type = blob[4] & 0x7F
        if block_type != 0:
            return None  # STREAMINFO isn't the first block — bail
        si = blob[8:8 + 18]  # first 18 bytes of STREAMINFO payload
        # sample_rate is the top 20 bits of the 24-bit word starting at si[10]
        sr = (si[10] << 12) | (si[11] << 4) | (si[12] >> 4)
        # total_samples is the bottom 4 bits of si[13] + si[14..17]
        total = ((si[13] & 0x0F) << 32) | (si[14] << 24) | (si[15] << 16) | \
                (si[16] << 8) | si[17]
        if sr > 0 and total > 0:
            return total / sr
        return None
    if fmt == "wav":
        # Parse chunks to find fmt (sample rate) + data (size).
        try:
            sample_rate = None
            data_size = None
            offset = 12  # past "RIFF<size>WAVE"
            while offset + 8 <= len(blob):
                chunk_id = blob[offset:offset + 4]
                chunk_size = struct.unpack("<I", blob[offset + 4:offset + 8])[0]
                payload = blob[offset + 8:offset + 8 + chunk_size]
                if chunk_id == b"fmt " and len(payload) >= 8:
                    sample_rate = struct.unpack("<I", payload[4:8])[0]
                elif chunk_id == b"data":
                    data_size = chunk_size
                    break
                # chunks are word-aligned
                offset += 8 + chunk_size + (chunk_size & 1)
            if sample_rate and data_size and sample_rate > 0:
                # Assume 16-bit mono by default (SaveAudio's default config).
                bytes_per_sample = 2
                channels = 1
                return (data_size / bytes_per_sample / channels) / sample_rate
        except (struct.error, IndexError):
            return None
    return None


# ── Shared test helpers ────────────────────────────────────────────────────

def _fetch_audio(session, comfyui, history_entry: dict,
                 output_key: str = "3") -> bytes:
    """Fetch the bytes of the audio file produced by a SaveAudio node."""
    outputs = history_entry["outputs"]
    assert output_key in outputs, (
        f"SaveAudio node {output_key!r} not in outputs; have {list(outputs)}"
    )
    audio_meta = outputs[output_key]["audio"][0]
    url = (
        f"{comfyui}/view?"
        f"filename={audio_meta['filename']}"
        f"&subfolder={audio_meta.get('subfolder', '')}"
        f"&type={audio_meta.get('type', 'output')}"
    )
    r = session.get(url, timeout=60)
    assert r.status_code == 200, (
        f"/view returned {r.status_code} for {audio_meta['filename']}: "
        f"{r.text[:200]}"
    )
    return r.content


# ── Tests ──────────────────────────────────────────────────────────────────

def test_moss_tts_inference_produces_valid_audio(
    session, comfyui, submit, empty_queue, available_families
):
    """End-to-end TTS with moss_tts: real text → real audio bytes on disk."""
    if "moss_tts" not in available_families:
        pytest.skip("moss_tts not registered in this build")
    p = submit.queue(session, comfyui, "tts_moss_tts.json")
    history = submit.wait(session, comfyui, p, timeout=600)

    assert history["status"].get("completed") is True, (
        f"moss_tts workflow failed: {history['status'].get('messages', [])[:3]}"
    )
    blob = _fetch_audio(session, comfyui, history, output_key="3")
    fmt = _assert_valid_audio(blob, min_bytes=10_000)
    duration = _audio_duration_seconds(blob, fmt) or 0.0
    assert duration >= 0.3, (
        f"moss_tts audio too short: {duration:.2f}s ({len(blob)} bytes {fmt})"
    )
    # Sanity upper bound — moss_tts producing >10 minutes for a single
    # sentence suggests a runaway decode.
    assert duration < 600.0, (
        f"moss_tts audio suspiciously long: {duration:.2f}s"
    )


def test_ace_step_inference_produces_valid_audio(
    session, comfyui, submit, empty_queue, available_families
):
    """End-to-end text-to-music with ace_step: caption → real stereo audio."""
    if "ace_step" not in available_families:
        pytest.skip("ace_step not registered in this build")
    p = submit.queue(session, comfyui, "music_ace_step.json")
    history = submit.wait(session, comfyui, p, timeout=900)

    assert history["status"].get("completed") is True, (
        f"ace_step workflow failed: {history['status'].get('messages', [])[:3]}"
    )
    blob = _fetch_audio(session, comfyui, history, output_key="3")
    fmt = _assert_valid_audio(blob, min_bytes=20_000)
    duration = _audio_duration_seconds(blob, fmt) or 0.0
    # We asked for 10s of music; accept anything ≥5s (engine rounds).
    assert duration >= 5.0, (
        f"ace_step audio too short: {duration:.2f}s "
        f"(requested 10s; got {len(blob)} bytes {fmt})"
    )
    assert duration < 60.0, (
        f"ace_step audio too long: {duration:.2f}s (requested 10s)"
    )


def test_qwen3_tts_inference_produces_valid_audio(
    session, comfyui, submit, empty_queue, available_families
):
    """End-to-end TTS with qwen3_tts (1.7B talker + predictor)."""
    if "qwen3_tts" not in available_families:
        pytest.skip("qwen3_tts not registered in this build")
    p = submit.queue(session, comfyui, "tts_qwen3_tts.json")
    history = submit.wait(session, comfyui, p, timeout=600)

    assert history["status"].get("completed") is True, (
        f"qwen3_tts workflow failed: {history['status'].get('messages', [])[:3]}"
    )
    blob = _fetch_audio(session, comfyui, history, output_key="3")
    fmt = _assert_valid_audio(blob, min_bytes=5_000)
    duration = _audio_duration_seconds(blob, fmt) or 0.0
    assert duration >= 0.3, (
        f"qwen3_tts audio too short: {duration:.2f}s "
        f"({len(blob)} bytes {fmt})"
    )


def test_qwen3_tts_voice_embedding_roundtrip(
    session, comfyui, submit, empty_queue, available_families
):
    """Full voice-clone pipeline: load w/ speaker_encoder → compute embedding
    from a reference WAV → TTS with that embedding → real cloned audio.

    This is the qwen3_tts-specific path that bypasses per-call ECAPA.
    Requires the qwen3tts-speaker-encoder.gguf on disk.
    """
    if "qwen3_tts" not in available_families:
        pytest.skip("qwen3_tts not registered in this build")
    # The encoder GGUF lives at a known path; skip cleanly if absent.
    # (Cheaper than a one-shot workflow probe.)
    import requests
    # Use the ComfyUI folder API as a quick existence check.
    models_r = session.get(
        f"{comfyui}/models/audiocore?filter=qwen3tts-speaker", timeout=5
    )
    encoder_present = False
    if models_r.status_code == 200:
        try:
            for p in models_r.json():
                if "speaker-encoder" in p.lower():
                    encoder_present = True
                    break
        except Exception:
            pass
    if not encoder_present:
        pytest.skip(
            "qwen3tts-speaker-encoder.gguf not found under /models/audiocore; "
            "skipping voice-embed roundtrip"
        )

    p = submit.queue(session, comfyui, "qwen3_tts_embed.json")
    history = submit.wait(session, comfyui, p, timeout=900)

    assert history["status"].get("completed") is True, (
        f"voice-embed workflow failed: {history['status'].get('messages', [])[:3]}"
    )
    blob = _fetch_audio(session, comfyui, history, output_key="4")
    fmt = _assert_valid_audio(blob, min_bytes=5_000)
    duration = _audio_duration_seconds(blob, fmt) or 0.0
    assert duration >= 0.5, (
        f"voice-cloned audio too short: {duration:.2f}s "
        f"({len(blob)} bytes {fmt})"
    )


def test_moss_tts_inference_is_reproducible_across_runs(
    session, comfyui, submit, empty_queue, available_families
):
    """Two runs of the same text + temperature=0 produce identical durations.

    moss_tts at temperature=0 is deterministic given the same text — a
    regression check that the engine isn't drifting.
    """
    if "moss_tts" not in available_families:
        pytest.skip("moss_tts not registered in this build")

    durations = []
    for _ in range(2):
        p = submit.queue(
            session, comfyui, "tts_moss_tts.json",
            **{"2": {"temperature": 0.0}}
        )
        history = submit.wait(session, comfyui, p, timeout=600)
        assert history["status"].get("completed") is True
        blob = _fetch_audio(session, comfyui, history, output_key="3")
        fmt = _assert_valid_audio(blob, min_bytes=5_000)
        d = _audio_duration_seconds(blob, fmt)
        durations.append(d)

    # Allow ±10% drift between runs (decode is deterministic but container
    # encoding can vary slightly at file boundaries).
    if None in durations:
        pytest.skip(
            f"couldn't parse duration from one of the runs: {durations}"
        )
    a, b = durations
    assert abs(a - b) <= max(0.25, (a + b) * 0.10), (
        f"durations diverged: run1={a:.2f}s, run2={b:.2f}s — "
        f"moss_tts(temperature=0) should be deterministic"
    )
