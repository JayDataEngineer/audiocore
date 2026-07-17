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
import os
from typing import Any

import numpy as np
import torch

from .core import ManagedModel, _AUDIOCORE_MODELS_DIR, init

logger = logging.getLogger("audiocore-nodes")

# ── Family metadata ──────────────────────────────────────────────────────────

FAMILY_NAMES = {
    "moss_tts": "MOSS-TTS (8B)",
    "qwen3_tts": "Qwen3-TTS (1.7B)",
    "ace_step": "ACE-Step (music)",
    "moss_sfx_v2": "MOSS-SFX v2 (sound effects)",
}

# Conventional model directory NAMES (relative to AUDIOCORE_MODELS_DIR).
# These are the subdirectory names that contain the .gguf weights.
_DEFAULT_MODEL_DIR = {
    "moss_tts": "moss-tts",
    "qwen3_tts": "qwen3-tts",
    "ace_step": "acestep-cpp-converted",
    "moss_sfx_v2": "moss-sfx-v2",
}


def _list_audiocore_models() -> list[str]:
    """List available audiocore model subdirectories for the combo dropdown.

    Scans AUDIOCORE_MODELS_DIR for immediate subdirectories. Each one is a
    model family directory containing .gguf weight files. This populates
    the LoadAudiocoreModel dropdown so users can pick models by name
    instead of typing absolute paths.

    Called by INPUT_TYPES (refreshed each time ComfyUI rebuilds the node
    menu), so newly downloaded models appear without a restart.
    """
    try:
        entries = sorted(os.listdir(_AUDIOCORE_MODELS_DIR))
        return [
            e for e in entries
            if os.path.isdir(os.path.join(_AUDIOCORE_MODELS_DIR, e))
        ]
    except OSError:
        return []


def _resolve_model_path(model_path: str) -> str:
    """Resolve a model_path to an absolute filesystem path.

    Accepts both:
      - Absolute paths (e.g. /mnt/data/models/audio/moss-tts) — used by
        the ray server's catalog and programmatic callers.
      - Relative names (e.g. "moss-tts") — used by the ComfyUI UI dropdown.
        Resolved via folder_paths.get_full_path("audiocore", name) with a
        fallback to AUDIOCORE_MODELS_DIR/name.

    This dual-mode keeps backwards compatibility (absolute paths from the
    builder) while making the UI dropdown work (relative names from the
    combo widget).
    """
    if os.path.isabs(model_path):
        return model_path
    # Relative — try folder_paths first (registered in __init__.py)
    try:
        import folder_paths
        resolved = folder_paths.get_full_path("audiocore", model_path)
        if resolved and os.path.exists(resolved):
            return resolved
    except ImportError:
        pass
    # Fallback: prepend AUDIOCORE_MODELS_DIR
    return os.path.join(_AUDIOCORE_MODELS_DIR, model_path)


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
        # Layer 3: model_path is a combo dropdown populated by scanning
        # AUDIOCORE_MODELS_DIR for subdirectories. Each subdirectory is a
        # model family (moss-tts/, moss-sfx-v2/, ...). The user picks one;
        # load() resolves the relative name to an absolute path.
        models = _list_audiocore_models()
        default_family = "moss_tts"
        default_model = _DEFAULT_MODEL_DIR.get(default_family, "")
        if default_model not in models and models:
            default_model = models[0]
        return {
            "required": {
                "family": (list(FAMILY_NAMES.keys()),
                           {"default": default_family}),
                "model_path": (models,
                               {"default": default_model}),
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

        # Layer 3: resolve relative names (from the dropdown) to absolute
        # paths. Absolute paths (from the ray builder) pass through.
        resolved_path = _resolve_model_path(model_path)

        m = ManagedModel(family, resolved_path, backend)
        if not m.load(**extras_dict):
            raise RuntimeError(
                f"Failed to load {family} from {resolved_path}"
            )
        return (m,)


# ── Node: Audiocore TTS ──────────────────────────────────────────────────────

class AudiocoreTTS:
    """Text-to-speech with the full audiocore parameter surface.

    Modes (family-dependent):
      tts    — plain text-to-speech
      clone  — zero-shot voice cloning (needs reference_audio + reference_text)
      design — instruction-following voice design (needs instruct)

    Combined mode (critical): reference_audio/voice_file AND instruct can be
    passed simultaneously. The C++ engine's voice_clone path accepts a speaker
    embedding (voice identity) AND an instruct string (emotion/style direction)
    at the same time — this is the "custom voice + emotion" pipeline that
    Qwen3-TTS cannot do natively. The service layer auto-escalates to clone
    mode when a voice source is present, while still forwarding instruct.

    Voice files: a .voice file is a pre-computed speaker embedding (ECAPA or
    PCA-steered). Pass voice_file="/path/to/voice.voice" to load it instead
    of running ECAPA on a reference WAV. Optionally pass voice_pca_strengths
    as a JSON string like '{"pca_pc1.dir": 0.5, "pca_pc2.dir": -0.3}' to
    steer the embedding along principal component directions.

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
                    "placeholder": "emotion/style direction (works with ANY mode)",
                    "multiline": True,
                }),
                "speaker_embedding": ("AUDIOCORE_EMBEDDING",),
                "voice_file": ("STRING", {
                    "default": "",
                    "placeholder": "/path/to/voice.voice (pre-computed speaker embedding)",
                }),
                "voice_pca_strengths": ("STRING", {
                    "default": "",
                    "placeholder": '{"pca_pc1.dir": 0.5, "pca_pc2.dir": -0.3}',
                    "multiline": True,
                }),
            },
        }

    def synthesize(self, model: ManagedModel, text: str, **kwargs):
        if not hasattr(model, "run_tts"):
            raise RuntimeError("invalid model reference")

        call_kwargs = {
            k: v for k, v in kwargs.items()
            if v is not None and v != ""
        }

        # ── Voice file loading + PCA steering ──
        # A .voice file is a pre-computed speaker embedding (QWEN3VOICE
        # header + float32 vector). When provided, load it and pass as
        # speaker_embedding — bypasses ECAPA extraction from a WAV ref.
        # Optionally apply PCA steering: each .dir file is a principal
        # component direction; voice_pca_strengths is JSON mapping
        # direction file names to strength multipliers.
        voice_file = call_kwargs.pop("voice_file", "")
        pca_json = call_kwargs.pop("voice_pca_strengths", "")

        if voice_file:
            import json as _json
            import struct as _struct
            import numpy as _np

            with open(voice_file, "rb") as f:
                data = f.read()
            MAGIC = b"QWEN3VOICE"
            if len(data) >= 36 and data[:len(MAGIC)] == MAGIC:
                dim = _struct.unpack_from("<I", data, 20)[0]
                emb = _np.frombuffer(data, dtype=_np.float32,
                                     count=dim, offset=32)
            else:
                emb = _np.frombuffer(data, dtype=_np.float32)
            emb = _np.array(emb, dtype=_np.float32)

            # Apply PCA steering if requested.
            if pca_json:
                import os.path as _osp
                voices_dir = _osp.dirname(voice_file)
                strengths = _json.loads(pca_json)
                for dir_name, strength in strengths.items():
                    dir_path = _osp.join(voices_dir, dir_name)
                    if not _osp.exists(dir_path):
                        continue
                    with open(dir_path, "rb") as f:
                        ddata = f.read()
                    if len(ddata) >= 36 and ddata[:len(MAGIC)] == MAGIC:
                        ddim = _struct.unpack_from("<I", ddata, 20)[0]
                        direction = _np.frombuffer(ddata, dtype=_np.float32,
                                                   count=ddim, offset=32)
                    else:
                        direction = _np.frombuffer(ddata, dtype=_np.float32)
                    direction = _np.array(direction, dtype=_np.float32)
                    if len(direction) == len(emb):
                        emb = emb + direction * float(strength)

            # Override speaker_embedding with the loaded+steered vector.
            # Remove any pre-existing speaker_embedding (from graph input)
            # so our computed one takes precedence.
            call_kwargs.pop("speaker_embedding", None)
            call_kwargs["speaker_embedding"] = emb.tolist()

        # ── Mode alias mapping ──
        # The node UI exposes "clone" / "design" for readability, but the
        # C++ engine (moss_tts session.cpp) expects "voice_clone" for the
        # zero-shot cloning mode. Map it here so users don't hit a runtime
        # error from the engine's mode validation.
        #
        # Critical: voice_clone mode accepts BOTH reference_audio AND
        # instruct simultaneously. If a voice source is present, always
        # use voice_clone so the combined path activates.
        mode = call_kwargs.get("mode", "tts")
        has_voice = bool(call_kwargs.get("reference_audio")
                         or call_kwargs.get("speaker_embedding")
                         or call_kwargs.get("voice_path"))
        if mode == "clone" or (has_voice and mode in ("tts", "design", "")):
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
    """List registered audiocore families (for debugging / discovery).

    When wired downstream of LoadAudiocoreModel (via the optional ``model``
    input), the report also includes the currently-active session's family,
    path, and backend — useful as a load verification sink in workflows
    that don't otherwise produce an output artifact.
    """

    TITLE = "Audiocore Family Info"
    CATEGORY = "audio/audiocore"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "info"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                # Accepting the model handle makes this node a valid sink for
                # load-only graphs (ComfyUI requires an OUTPUT_NODE terminator).
                "model": ("AUDIOCORE_MODEL",),
            },
        }

    def info(self, model=None):
        try:
            init()
        except Exception as e:
            text = f"audiocore not available: {e}"
            return {"ui": {"text": [text]}, "result": (text,)}
        import audiocore
        families = audiocore.list_families()
        lines = [f"Registered families: {', '.join(families) or '(none)'}"]
        for f in families:
            display = FAMILY_NAMES.get(f, f)
            lines.append(f"  {f} -> {display}")
        # When chained after LoadAudiocoreModel, surface what's actually live.
        if model is not None and getattr(model, "loaded", False):
            lines.append("")
            lines.append(
                f"Active session: family={model.family} "
                f"path={model.path} backend={model.backend}"
            )
        text = "\n".join(lines)
        # OUTPUT_NODE convention: {"ui": {...}, "result": (...)} so the text
        # appears in /history outputs as node_output["text"].
        return {"ui": {"text": [text]}, "result": (text,)}


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
