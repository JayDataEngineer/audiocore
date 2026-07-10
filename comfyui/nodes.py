"""ComfyUI node classes for the audiocore audio engine.

Exposes the full audiocore Python binding surface:
  - LoadAudiocoreModel  — load a family (moss_tts / ace_step / qwen3_tts / moss_sfx_v2)
  - AudiocoreTTS        — full TTS (voice clone, design, multilingual)
  - AudiocoreMusic      — text-to-music (ACE-Step)
  - AudiocoreVoiceEmbedding — ECAPA voice-caching (compute once, reuse)
  - UnloadAudiocoreModel — release VRAM
  - AudiocoreFamilyInfo — list registered families

Model handles (AUDIOCORE_MODEL) pass through the graph, so multiple
families can coexist.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import torch

from .core import ManagedModel, init

logger = logging.getLogger("audiocore-nodes")

# ── Family metadata ──────────────────────────────────────────────────────────

FAMILY_NAMES = {
    "moss_tts": "MOSS-TTS (8B)",
    "qwen3_tts": "Qwen3-TTS (1.7B)",
    "ace_step": "ACE-Step (music)",
    "moss_sfx_v2": "MOSS-SFX v2 (sound effects)",
}

# Conventional model directories.
_DEFAULT_MODEL_PATH = {
    "moss_tts": "/models/moss-tts",
    "qwen3_tts": "/models/qwen3-tts",
    "ace_step": "/models/acestep",
    "moss_sfx_v2": "/models/moss-sfx-v2",
}


# ── Node: Load Audiocore Model ───────────────────────────────────────────────

class LoadAudiocoreModel:
    """Load an audiocore model session for a given family.

    The returned AUDIOCORE_MODEL handle is passed to downstream TTS /
    embedding nodes. Multiple models can be loaded simultaneously.

    Advanced: ``extras`` accepts a JSON object for family-specific sub-model
    paths. qwen3_tts accepts: talker_path, predictor_path, codec_path,
    speaker_encoder_path.
    """

    TITLE = "Load Audiocore Model"
    CATEGORY = "audio/audiocore"
    RETURN_TYPES = ("AUDIOCORE_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "family": (list(FAMILY_NAMES.keys()),
                           {"default": "moss_tts"}),
                "model_path": ("STRING",
                               {"default": "/models/moss-tts"}),
                "backend": (["ggml_cuda", "ggml_cpu"],
                            {"default": "ggml_cuda"}),
            },
            "optional": {
                "extras": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": '{"talker_path": "...", "codec_path": "..."}',
                }),
            },
        }

    def load(self, family: str, model_path: str,
             backend: str = "ggml_cuda", extras: str = ""):
        extras_dict: dict[str, str] = {}
        if extras and extras.strip():
            try:
                extras_dict = json.loads(extras)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"extras must be valid JSON: {e}") from e

        m = ManagedModel(family, model_path, backend)
        if not m.load(**extras_dict):
            raise RuntimeError(
                f"Failed to load {family} from {model_path}"
            )
        return (m,)


# ── Node: Audiocore TTS ──────────────────────────────────────────────────────

class AudiocoreTTS:
    """Text-to-speech with the full audiocore parameter surface.

    Modes (family-dependent):
      tts    — plain text-to-speech
      clone  — zero-shot voice cloning (needs reference_audio + reference_text)
      design — instruction-following voice design (needs instruct)

    Voice caching: connect an AUDIOCORE_EMBEDDING from
    AudiocoreVoiceEmbedding to speaker_embedding to bypass per-call
    ECAPA computation (qwen3_tts only).
    """

    TITLE = "Audiocore TTS"
    CATEGORY = "audio/audiocore"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "synthesize"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("AUDIOCORE_MODEL",),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello world.",
                }),
            },
            "optional": {
                "mode": (["tts", "clone", "design"],
                         {"default": "tts"}),
                "voice": ("STRING", {"default": ""}),
                "language": ("STRING", {
                    "default": "",
                    "placeholder": "en, zh, auto...",
                }),
                "temperature": ("FLOAT",
                                {"default": 0.8, "min": 0.0, "max": 2.0,
                                 "step": 0.05}),
                "top_p": ("FLOAT",
                          {"default": 0.9, "min": 0.0, "max": 1.0,
                           "step": 0.01}),
                "speed": ("FLOAT",
                          {"default": 1.0, "min": 0.5, "max": 2.0,
                           "step": 0.1}),
                "repetition_penalty": ("FLOAT",
                                       {"default": 1.05, "min": 0.8,
                                        "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0,
                                 "max": 2147483647}),
                "reference_audio": ("STRING", {
                    "default": "",
                    "placeholder": "/path/to/clone_ref.wav (mode=clone)",
                }),
                "reference_text": ("STRING", {
                    "default": "",
                    "placeholder": "transcript of reference_audio",
                }),
                "speaker_name": ("STRING", {"default": ""}),
                "instruct": ("STRING", {
                    "default": "",
                    "placeholder": "voice description (mode=design)",
                    "multiline": True,
                }),
                "speaker_embedding": ("AUDIOCORE_EMBEDDING",),
            },
        }

    def synthesize(self, model: ManagedModel, text: str, **kwargs):
        if not hasattr(model, "run_tts"):
            raise RuntimeError("invalid model reference")

        call_kwargs = {
            k: v for k, v in kwargs.items()
            if v is not None and v != ""
        }

        # ── Mode alias mapping ──
        # The node UI exposes "clone" / "design" for readability, but the
        # C++ engine (moss_tts session.cpp) expects "voice_clone" for the
        # zero-shot cloning mode. Map it here so users don't hit a runtime
        # error from the engine's mode validation.
        mode = call_kwargs.get("mode", "tts")
        if mode == "clone":
            call_kwargs["mode"] = "voice_clone"

        pcm, sr = model.run_tts(text, **call_kwargs)
        audio_np = np.clip(np.array(pcm, dtype=np.float32), -1.0, 1.0)
        waveform = torch.from_numpy(audio_np).reshape(1, 1, -1)
        return ({"waveform": waveform, "sample_rate": sr},)


# ── Node: Audiocore Voice Embedding ──────────────────────────────────────────

class AudiocoreVoiceEmbedding:
    """Compute a speaker embedding from a WAV file (voice caching).

    qwen3_tts only (requires speaker_encoder GGUF loaded). Compute once,
    then pass the AUDIOCORE_EMBEDDING to AudiocoreTTS(speaker_embedding=)
    on every subsequent call — bypasses reference_audio load + ECAPA.
    """

    TITLE = "Audiocore Voice Embedding"
    CATEGORY = "audio/audiocore"
    RETURN_TYPES = ("AUDIOCORE_EMBEDDING",)
    RETURN_NAMES = ("embedding",)
    FUNCTION = "compute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("AUDIOCORE_MODEL",),
                "wav_path": ("STRING", {
                    "default": "",
                    "placeholder": "/path/to/voice.wav",
                }),
            },
        }

    def compute(self, model: ManagedModel, wav_path: str):
        if not wav_path:
            raise RuntimeError("wav_path is required")
        emb = model.compute_embedding(wav_path)
        if not emb:
            raise RuntimeError(
                "compute_embedding returned empty — "
                "only qwen3_tts with a loaded speaker_encoder GGUF "
                "supports this call"
            )
        return ({"vector": torch.tensor(emb, dtype=torch.float32)},)


# ── Node: Audiocore Music ────────────────────────────────────────────────────

class AudiocoreMusic:
    """Text-to-music generation via ACE-Step.

    Generates stereo audio at 48 kHz from a text caption. Optional lyrics,
    duration, seed, diffusion steps, and musical metadata (bpm, key, time
    signature) for deterministic CoT YAML injection.
    """

    TITLE = "Audiocore Music"
    CATEGORY = "audio/audiocore"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("AUDIOCORE_MODEL",),
                "caption": ("STRING", {
                    "multiline": True,
                    "default": "lo-fi ambient piano",
                }),
            },
            "optional": {
                "lyrics": ("STRING", {"multiline": True, "default": ""}),
                "duration": ("FLOAT",
                             {"default": 30.0, "min": 1.0, "max": 300.0,
                              "step": 1.0}),
                "seed": ("INT", {"default": 0, "min": 0,
                                 "max": 2147483647}),
                "guidance_scale": ("FLOAT",
                                   {"default": 1.0, "min": 0.1, "max": 10.0,
                                    "step": 0.1}),
                "n_diffusion_steps": ("INT",
                                      {"default": 0, "min": 0, "max": 200}),
                "temperature": ("FLOAT",
                                {"default": 0.85, "min": 0.0, "max": 2.0,
                                 "step": 0.05}),
                "top_p": ("FLOAT",
                          {"default": 0.9, "min": 0.0, "max": 1.0,
                           "step": 0.01}),
                "lm_cfg_scale": ("FLOAT",
                                 {"default": 2.0, "min": 0.0, "max": 10.0,
                                  "step": 0.1}),
            },
        }

    def generate(self, model: ManagedModel, caption: str, **kwargs):
        if not hasattr(model, "run_music"):
            raise RuntimeError("invalid model reference")

        call_kwargs = {
            k: v for k, v in kwargs.items()
            if v is not None and v != ""
        }

        pcm, sr, channels = model.run_music(caption, **call_kwargs)
        audio_t = torch.tensor(pcm, dtype=torch.float32).clamp(-1.0, 1.0)
        waveform = audio_t.reshape(-1, channels).T.unsqueeze(0).contiguous()
        return ({"waveform": waveform, "sample_rate": sr},)


# ── Node: Unload Audiocore Model ─────────────────────────────────────────────

class UnloadAudiocoreModel:
    """Release a model's VRAM. Wire after the last consumer in the graph."""

    TITLE = "Unload Audiocore Model"
    CATEGORY = "audio/audiocore"
    RETURN_TYPES = ()
    FUNCTION = "unload"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("AUDIOCORE_MODEL",),
            },
        }

    def unload(self, model: ManagedModel):
        model.unload()
        return ()


# ── Node: Audiocore Family Info ──────────────────────────────────────────────

class AudiocoreFamilyInfo:
    """List registered audiocore families (for debugging / discovery)."""

    TITLE = "Audiocore Family Info"
    CATEGORY = "audio/audiocore"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "info"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def info(self):
        try:
            init()
        except Exception as e:
            return (f"audiocore not available: {e}",)
        import audiocore
        families = audiocore.list_families()
        lines = [f"Registered families: {', '.join(families) or '(none)'}"]
        for f in families:
            display = FAMILY_NAMES.get(f, f)
            lines.append(f"  {f} -> {display}")
        return ("\n".join(lines),)


# ── Mappings ─────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "LoadAudiocoreModel": LoadAudiocoreModel,
    "AudiocoreTTS": AudiocoreTTS,
    "AudiocoreMusic": AudiocoreMusic,
    "AudiocoreVoiceEmbedding": AudiocoreVoiceEmbedding,
    "UnloadAudiocoreModel": UnloadAudiocoreModel,
    "AudiocoreFamilyInfo": AudiocoreFamilyInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAudiocoreModel": "Load Audiocore Model",
    "AudiocoreTTS": "Audiocore TTS",
    "AudiocoreMusic": "Audiocore Music",
    "AudiocoreVoiceEmbedding": "Audiocore Voice Embedding",
    "UnloadAudiocoreModel": "Unload Audiocore Model",
    "AudiocoreFamilyInfo": "Audiocore Family Info",
}
