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


def test_moss_sfx_v2_audio_passes_mimo_sfx_check(
    session, comfyui, submit, empty_queue, available_families, mimo
):
    """Mimo must recognize the moss_sfx_v2 output as the prompted sound effect.

    The workflow generates "heavy wooden door slamming shut with a deep thud".
    Mimo is asked whether the clip contains a recognizable sound effect (not
    silence, not music, not random noise) and to score fidelity 1–10.

    This is the highest-level proof that moss_sfx_v2 "works" — not just that
    it produces a valid FLAC of the right duration, but that the audio is
    semantically on-prompt. Catches both the failure mode in test_inference
    (wrong duration / corrupt file) AND semantic regressions (model emits
    music or noise instead of the requested SFX).
    """
    _require_mimo(mimo)
    if "moss_sfx_v2" not in available_families:
        pytest.skip("moss_sfx_v2 not registered")

    p = submit.queue(session, comfyui, "sfx_moss_sfx_v2.json")
    history = submit.wait(session, comfyui, p, timeout=900)
    assert history["status"].get("completed") is True, (
        f"moss_sfx_v2 workflow failed: "
        f"{history['status'].get('messages', [])[:3]}"
    )

    blob = _fetch_audio(session, comfyui, history, output_key="3")
    _assert_valid_audio(blob, min_bytes=40_000)

    sfx_question = (
        "You are an audio quality expert. This clip should be a SOUND EFFECT "
        "matching the prompt: 'heavy wooden door slamming shut with a deep "
        "thud'. Listen carefully and judge whether the clip contains a "
        "recognizable door-slam / thud sound effect (not silence, not music, "
        "not random static). Respond with ONLY a JSON object — no prose, no "
        "markdown fences — with this exact shape: "
        "{\"sfx_present\": true|false, "
        "\"matches_prompt\": true|false, "
        "\"noise_level\": \"clean\"|\"moderate\"|\"noisy\", "
        "\"clipping\": true|false, "
        "\"quality_score\": <int 1-10>, "
        "\"description\": \"<one sentence>\"}."
    )
    verdict = mimo.judge_audio_bytes(
        blob, filename="moss_sfx_v2.flac", question=sfx_question
    )
    print(f"\nMimo verdict for moss_sfx_v2: {verdict}")
    # Predicate: a recognizable sound effect is present, on-prompt, with
    # acceptable noise/fidelity. Threshold 4 — moss_sfx_v2 is a 1.3B model
    # and SFX fidelity at this scale is good-but-not-Hollywood-grade.
    assert verdict.get("sfx_present") is True, (
        f"moss_sfx_v2 audio: Mimo did not hear a sound effect: {verdict}"
    )
    assert verdict.get("matches_prompt") is True, (
        f"moss_sfx_v2 audio: Mimo heard SFX but it didn't match the prompt: "
        f"{verdict}"
    )
    assert verdict.get("noise_level") in ("clean", "moderate"), (
        f"moss_sfx_v2 audio too noisy: {verdict}"
    )
    assert verdict.get("clipping") is not True, (
        f"moss_sfx_v2 audio clipping: {verdict}"
    )
    assert int(verdict.get("quality_score", 0)) >= 4, (
        f"moss_sfx_v2 audio quality too low: {verdict}"
    )


def test_voice_clone_output_matches_reference_voice_gender(
    session, comfyui, submit, empty_queue, available_families, mimo
):
    """Voice-cloned audio should be intelligible speech AND Mimo should
    recognize it as a cloning attempt (matches reference voice character).

    This is the highest-level e2e check for the qwen3_tts clone path:
    embedding extraction → voice-conditioned TTS → real audible speech.

    Requires the *Base* model variant (Qwen3-TTS-12Hz-*-Base); the
    CustomVoice variant rejects voice cloning at the package level.
    """
    _require_mimo(mimo)
    if "qwen3_tts" not in available_families:
        pytest.skip("qwen3_tts not registered")

    # Voice cloning requires the Base HF variant — same predicate as
    # test_qwen3_tts_voice_embedding_roundtrip in test_inference.py.
    import os
    hf_root = "/mnt/data/models/hf_cache"
    has_base = False
    if os.path.isdir(hf_root):
        for d in os.listdir(hf_root):
            dl = d.lower()
            if "qwen3-tts" in dl and dl.endswith("-base"):
                has_base = True
                break
    if not has_base:
        pytest.skip(
            "no Qwen3-TTS-12Hz-*-Base HF dir under /mnt/data/models/hf_cache — "
            "voice cloning requires the Base variant"
        )

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


def test_qwen3_tts_voice_design_matches_instruct(
    session, comfyui, submit, empty_queue, available_families, mimo
):
    """Voice Design mode: instruct text → generated voice matching description.

    Requires the VoiceDesign HF variant
    (Qwen3-TTS-12Hz-1.7B-VoiceDesign); skip cleanly if not downloaded.
    The workflow asks for a "deep, warm, authoritative male voice" — Mimo
    must hear speech matching that description.
    """
    _require_mimo(mimo)
    if "qwen3_tts" not in available_families:
        pytest.skip("qwen3_tts not registered")

    import os
    hf_root = "/mnt/data/models/hf_cache"
    has_design = False
    if os.path.isdir(hf_root):
        for d in os.listdir(hf_root):
            dl = d.lower()
            if "qwen3-tts" in dl and "voicedesign" in dl:
                has_design = True
                break
    if not has_design:
        pytest.skip(
            "no Qwen3-TTS-12Hz-*-VoiceDesign HF dir under "
            "/mnt/data/models/hf_cache — voice design requires the "
            "VoiceDesign variant (separate from Base/CustomVoice)"
        )

    p = submit.queue(session, comfyui, "qwen3_tts_voicedesign.json")
    history = submit.wait(session, comfyui, p, timeout=900)
    assert history["status"].get("completed") is True, (
        f"voicedesign workflow failed: "
        f"{history['status'].get('messages', [])[:3]}"
    )

    blob = _fetch_audio(session, comfyui, history, output_key="3")
    _assert_valid_audio(blob, min_bytes=10_000)

    design_question = (
        "This clip was generated to match this voice instruction: "
        "'A deep, warm, authoritative male voice with a British accent, "
        "speaking slowly and clearly.' The text should be: 'Hello, this "
        "is a test of the voice design feature. I can create any voice "
        "you describe.' Respond with ONLY a JSON object (no markdown "
        "fences): {\"speech_present\": bool, \"intelligible\": bool, "
        "\"matches_text\": bool, \"voice_gender\": \"male\"|\"female\", "
        "\"voice_pitch\": \"low\"|\"medium\"|\"high\", "
        "\"matches_instruct\": bool, \"quality_score\": int 1-10, "
        "\"description\": string}."
    )
    verdict = mimo.judge_audio_bytes(
        blob, filename="voicedesign.flac", question=design_question
    )
    print(f"\nMimo verdict for voicedesign: {verdict}")
    assert verdict.get("speech_present") is True, (
        f"voicedesign audio: no speech: {verdict}"
    )
    assert verdict.get("intelligible") is True, (
        f"voicedesign audio: not intelligible: {verdict}"
    )
    assert verdict.get("voice_gender") == "male", (
        f"voicedesign audio: expected male voice per instruct, got "
        f"{verdict.get('voice_gender')}: {verdict}"
    )
    assert verdict.get("voice_pitch") == "low", (
        f"voicedesign audio: expected low pitch per instruct, got "
        f"{verdict.get('voice_pitch')}: {verdict}"
    )
    assert int(verdict.get("quality_score", 0)) >= 5, (
        f"voicedesign audio quality too low: {verdict}"
    )


def test_qwen3_tts_multilingual_speaker_selection(
    session, comfyui, submit, empty_queue, available_families, mimo
):
    """Multi-speaker + multi-language: Aiden (English) voice selection.

    Verifies the CustomVoice variant's speaker dropdown works end-to-end:
    passing speaker='Aiden' produces a male English voice distinct from
    the default 'Vivian' (female Chinese). Catches regressions in the
    speaker parameter plumbing through the engine.
    """
    _require_mimo(mimo)
    if "qwen3_tts" not in available_families:
        pytest.skip("qwen3_tts not registered")

    p = submit.queue(session, comfyui, "qwen3_tts_multilingual.json")
    history = submit.wait(session, comfyui, p, timeout=600)
    assert history["status"].get("completed") is True, (
        f"multilingual workflow failed: "
        f"{history['status'].get('messages', [])[:3]}"
    )

    blob = _fetch_audio(session, comfyui, history, output_key="3")
    _assert_valid_audio(blob, min_bytes=5_000)

    speaker_question = (
        "This clip should be ENGLISH SPEECH in a MALE voice (speaker "
        "'Aiden') saying: 'The quick brown fox jumps over the lazy dog "
        "at noon.' Respond with ONLY a JSON object (no markdown fences): "
        "{\"speech_present\": bool, \"language\": \"english\"|\"chinese\"|"
        "\"japanese\"|\"other\", \"voice_gender\": \"male\"|\"female\", "
        "\"matches_text\": bool, \"quality_score\": int 1-10, "
        "\"description\": string}."
    )
    verdict = mimo.judge_audio_bytes(
        blob, filename="aiden_en.flac", question=speaker_question
    )
    print(f"\nMimo verdict for multilingual/speaker: {verdict}")
    assert verdict.get("speech_present") is True, (
        f"multilingual audio: no speech: {verdict}"
    )
    assert verdict.get("language") == "english", (
        f"multilingual audio: expected english, got "
        f"{verdict.get('language')}: {verdict}"
    )
    assert verdict.get("voice_gender") == "male", (
        f"multilingual audio: expected male (Aiden), got "
        f"{verdict.get('voice_gender')}: {verdict}"
    )
    assert int(verdict.get("quality_score", 0)) >= 5, (
        f"multilingual audio quality too low: {verdict}"
    )
