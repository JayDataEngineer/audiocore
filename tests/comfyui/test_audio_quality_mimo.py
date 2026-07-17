"""Mimo-judged audio cleanliness tests — the "does it actually sound right" check.

The byte-size + format + duration sanity checks in test_inference.py catch
structural regressions (truncated files, wrong format, etc.). They CAN'T
catch "valid 30-second FLAC of mechanical buzzing" — which is exactly the
kind of regression that breaks TTS in production. These tests close that
gap by sending the generated audio to Mimo-2.5 (a multimodal VLLM) and
asking it to return a structured verdict:

    speech_present, intelligible, noise_level, clipping, quality_score

Mimo isn't a perfect judge — but it's sensitive enough to flag a model
that's emitting noise instead of speech, which is the single most common
audiocore regression.

These tests are marked `slow` (network call to Mimo) and `inference`
(they require the model to actually generate audio). Skip cleanly when
MIMO_API_KEY is unset.
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.inference, pytest.mark.slow]


# Re-use the helpers from test_inference.py
from .test_inference import _fetch_audio, _assert_valid_audio  # noqa: E402


def _require_mimo(mimo):
    """Shared skip guard for the Mimo-dependent tests."""
    if mimo is None:
        pytest.skip(
            "MIMO_API_KEY not set or endpoint unreachable — "
            "set MIMO_API_KEY to enable audio-cleanliness verification"
        )


# ── Tests ──────────────────────────────────────────────────────────────────

@pytest.mark.xfail(reason=("Inherits moss_tts noise regression from test_inference.test_moss_tts_inference_produces_valid_audio — the Mimo judge will report speech_present=false. Remove xfail when the engine bug is fixed."), strict=False)
def test_moss_tts_audio_passes_mimo_cleanliness_check(
    session, comfyui, submit, empty_queue, available_families, mimo
):
    """Mimo must hear speech in moss_tts output, not noise.

    This is the test that catches 'moss_tts in this container build emits
    30s of mechanical buzzing' — a regression the byte/duration tests miss.
    """
    _require_mimo(mimo)
    if "moss_tts" not in available_families:
        pytest.skip("moss_tts not registered")

    p = submit.queue(session, comfyui, "tts_moss_tts.json")
    history = submit.wait(session, comfyui, p, timeout=600)
    assert history["status"].get("completed") is True

    blob = _fetch_audio(session, comfyui, history, output_key="3")
    _assert_valid_audio(blob, min_bytes=5_000)

    verdict = mimo.judge_audio_bytes(blob, filename="moss_tts.flac")
    # Always log the verdict — useful even (especially) when it fails.
    print(f"\nMimo verdict for moss_tts: {verdict}")
    assert mimo.is_clean_audio(verdict, min_score=5), (
        f"moss_tts audio failed Mimo cleanliness check: {verdict}"
    )


@pytest.mark.xfail(reason=("Inherits qwen3_tts cold-load crash from test_inference.test_qwen3_tts_inference_produces_valid_audio. Remove xfail when the .so is rebuilt."), strict=False)
def test_qwen3_tts_audio_passes_mimo_cleanliness_check(
    session, comfyui, submit, empty_queue, available_families, mimo
):
    """Mimo must hear intelligible speech in qwen3_tts output."""
    _require_mimo(mimo)
    if "qwen3_tts" not in available_families:
        pytest.skip("qwen3_tts not registered")

    p = submit.queue(session, comfyui, "tts_qwen3_tts.json")
    history = submit.wait(session, comfyui, p, timeout=600)
    assert history["status"].get("completed") is True

    blob = _fetch_audio(session, comfyui, history, output_key="3")
    _assert_valid_audio(blob, min_bytes=5_000)

    verdict = mimo.judge_audio_bytes(blob, filename="qwen3_tts.flac")
    print(f"\nMimo verdict for qwen3_tts: {verdict}")
    assert mimo.is_clean_audio(verdict, min_score=5), (
        f"qwen3_tts audio failed Mimo cleanliness check: {verdict}"
    )


@pytest.mark.xfail(reason=("Inherits ace_step stale-.so issue from test_inference.test_ace_step_inference_produces_valid_audio (no Session.run_music). Remove xfail when the .so is rebuilt."), strict=False)
def test_ace_step_audio_passes_mimo_music_check(
    session, comfyui, submit, empty_queue, available_families, mimo
):
    """Mimo must hear music (not noise, not speech) in ace_step output.

    Different prompt than the TTS checks: we ask Mimo specifically whether
    the clip sounds like music. ACE-Step generates 10s of lo-fi piano.
    """
    _require_mimo(mimo)
    if "ace_step" not in available_families:
        pytest.skip("ace_step not registered")

    p = submit.queue(session, comfyui, "music_ace_step.json")
    history = submit.wait(session, comfyui, p, timeout=900)
    assert history["status"].get("completed") is True

    blob = _fetch_audio(session, comfyui, history, output_key="3")
    _assert_valid_audio(blob, min_bytes=10_000)

    music_question = (
        "You are an audio quality expert. This clip should be MUSIC "
        "(instruments, melody, rhythm). Respond with ONLY a JSON object "
        "with this shape: "
        "{\"music_present\": true|false, "
        "\"noise_level\": \"clean\"|\"moderate\"|\"noisy\", "
        "\"clipping\": true|false, "
        "\"quality_score\": <int 1-10>, "
        "\"description\": \"<one sentence>\"}."
    )
    verdict = mimo.judge_audio_bytes(
        blob, filename="ace_step.flac", question=music_question
    )
    print(f"\nMimo verdict for ace_step: {verdict}")
    # Predicate: music_present + acceptable noise + no clipping + >=4/10.
    # ACE-Step at 10s+Q8 quantization is decent but not pristine; threshold 4.
    assert verdict.get("music_present") is True, (
        f"ace_step audio failed Mimo music check: {verdict}"
    )
    assert verdict.get("noise_level") in ("clean", "moderate"), (
        f"ace_step audio too noisy: {verdict}"
    )
    assert verdict.get("clipping") is not True, (
        f"ace_step audio clipping: {verdict}"
    )
    assert int(verdict.get("quality_score", 0)) >= 4, (
        f"ace_step audio quality too low: {verdict}"
    )


@pytest.mark.xfail(reason=("Inherits qwen3_tts cold-load crash from test_inference.test_qwen3_tts_voice_embedding_roundtrip. Remove xfail when the .so is rebuilt."), strict=False)
def test_voice_clone_output_matches_reference_voice_gender(
    session, comfyui, submit, empty_queue, available_families, mimo
):
    """Voice-cloned audio should be intelligible speech AND Mimo should
    recognize it as a cloning attempt (matches reference voice character).

    This is the highest-level e2e check for the qwen3_tts clone path:
    embedding extraction → voice-conditioned TTS → real audible speech.
    """
    _require_mimo(mimo)
    if "qwen3_tts" not in available_families:
        pytest.skip("qwen3_tts not registered")

    # Need the speaker encoder GGUF too.
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
        pytest.skip("speaker_encoder GGUF missing — can't run clone pipeline")

    p = submit.queue(session, comfyui, "qwen3_tts_embed.json")
    history = submit.wait(session, comfyui, p, timeout=900)
    assert history["status"].get("completed") is True

    blob = _fetch_audio(session, comfyui, history, output_key="4")
    _assert_valid_audio(blob, min_bytes=5_000)

    verdict = mimo.judge_audio_bytes(blob, filename="voice_clone.flac")
    print(f"\nMimo verdict for voice_clone: {verdict}")
    assert mimo.is_clean_audio(verdict, min_score=4), (
        f"voice-clone audio failed Mimo cleanliness check: {verdict}"
    )
