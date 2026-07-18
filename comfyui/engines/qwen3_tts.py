"""qwen3_tts — text-to-speech via the upstream qwen-tts Python package.

Proxies the official `qwen_tts` package (pip-installed) which wraps
Qwen3TTSForConditionalGeneration + Qwen3TTSProcessor. No GGUF, no
conversion. The model is loaded directly from the HF source dir
(config.json + model.safetensors + speech_tokenizer/ + tokenizer bits).

Three call modes match the audiocore surface:
  - tts         -> generate_custom_voice(text, speaker, language)
  - voice_clone -> generate_voice_clone(text, ref_audio, ref_text)
                   (needs a "base" model variant, not "custom_voice")
  - design      -> generate_voice_design(text, instruct)
                   (needs a "voicedesign" model variant)

Hyperparameters (temperature, top_p, top_k, repetition_penalty, seed)
are forwarded into the underlying transformers generate() call.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("audiocore.qwen3_tts")


# Default hyperparameters — match the audiocore node defaults where possible.
_DEFAULTS = {
    "speaker": "vivian",
    "language": "english",
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.05,
}


class Qwen3TtsEngine:
    """Wraps qwen_tts.Qwen3TTSModel to match audiocore's session contract."""

    def __init__(self) -> None:
        self.model: Any = None          # qwen_tts.Qwen3TTSModel
        self._model_dir: str | None = None

    # ── audiocore session contract ──────────────────────────────────────

    def load(self, path: str, **extras: Any) -> bool:
        """Load Qwen3TTSModel from `path` (HF source dir with config.json).

        Accepts the legacy GGUF dir names and remaps to the HF source dir
        (qwen3-tts GGUF dir → Qwen3-TTS-12Hz-0.6B-CustomVoice in the HF
        cache). The CustomVoice variant is the most capable single-pkg
        model: it supports generate_custom_voice for plain TTS with
        predefined speakers.
        """
        import contextlib
        import io

        hf_dir = self._resolve_model_dir(path)
        if hf_dir is None:
            raise RuntimeError(
                f"qwen3_tts: could not resolve an HF source dir from {path!r}. "
                f"Point model_path at a Qwen3-TTS HF dir (containing "
                f"config.json + model.safetensors)."
            )

        # The qwen_tts package prints a flash-attn warning to stdout on
        # import — silence it so it doesn't pollute ComfyUI logs.
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                from qwen_tts import Qwen3TTSModel
            except ImportError as e:
                raise RuntimeError(
                    f"qwen3_tts: qwen-tts Python package not installed: {e}"
                ) from e

        import torch

        logger.info("qwen3_tts: loading Qwen3TTSModel from %s", hf_dir)
        self.model = Qwen3TTSModel.from_pretrained(
            hf_dir,
            dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        self._model_dir = hf_dir
        logger.info(
            "qwen3_tts: model ready (device=%s, speakers=%s)",
            self.model.device,
            (self.model.get_supported_speakers() or [])[:5],
        )
        return True

    def run_tts(self, text: str, **kwargs: Any) -> tuple[list[float], int]:
        """Generate speech.

        Dispatches based on `mode`:
          - "voice_clone" (or any mode + reference_audio)  → clone path
          - "design" + instruct                            → design path
          - otherwise                                      → custom_voice
        """
        if self.model is None:
            raise RuntimeError("qwen3_tts: engine not loaded")

        # Pop audiocore control kwargs that don't map to qwen_tts.
        mode = str(kwargs.get("mode", "tts")).lower()
        # `voice` is audiocore's speaker alias.
        speaker = str(kwargs.get("voice", kwargs.get("speaker", _DEFAULTS["speaker"])))
        language = self._coerce_language(kwargs.get("language", _DEFAULTS["language"]))
        instruct = kwargs.get("instruct", "") or ""
        seed = self._coerce_seed(kwargs.get("seed", 0))

        # Build gen kwargs forwarded to transformers generate().
        gen_kwargs: dict[str, Any] = {}
        for k in ("temperature", "top_p", "top_k", "repetition_penalty"):
            if k in kwargs and kwargs[k] is not None and kwargs[k] != "":
                try:
                    gen_kwargs[k] = float(kwargs[k]) if k != "top_k" else int(kwargs[k])
                except (TypeError, ValueError):
                    pass
        if seed > 0:
            import torch
            torch.manual_seed(seed)
            gen_kwargs["seed"] = seed

        # Dispatch.
        ref_audio = kwargs.get("reference_audio", "") or ""
        ref_text = kwargs.get("reference_text", "") or ""
        if mode in ("voice_clone", "clone") or ref_audio:
            return self._voice_clone(text, language, ref_audio, ref_text,
                                     kwargs, gen_kwargs)
        if mode in ("design", "voice_design") and instruct:
            return self._voice_design(text, language, instruct, gen_kwargs)

        # Default: plain TTS via custom_voice.
        return self._custom_voice(text, speaker, language, instruct, gen_kwargs)

    def run_music(self, caption: str, **kwargs: Any) -> tuple[list[float], int, int]:
        raise RuntimeError("qwen3_tts does not support music generation")

    def compute_embedding(self, wav_path: str) -> list[float]:
        """Compute a speaker embedding from a WAV via the loaded model.

        Uses qwen_tts' create_voice_clone_prompt to extract the x-vector
        embedding from the reference audio. Returns it as a flat list so
        the AudiocoreVoiceEmbedding node can cache + reuse it.
        """
        if self.model is None:
            raise RuntimeError("qwen3_tts: engine not loaded")
        if not wav_path or not os.path.isfile(wav_path):
            raise RuntimeError(
                f"qwen3_tts: compute_embedding needs a valid wav_path, "
                f"got {wav_path!r}")

        # create_voice_clone_prompt builds a VoiceClonePromptItem carrying
        # the x-vector embedding extracted from the reference audio.
        prompt_item = self.model.create_voice_clone_prompt(
            ref_audio=wav_path,
            ref_text=None,
            x_vector_only_mode=False,
        )
        # Pull the x-vector out of the prompt dict / dataclass.
        emb = None
        if hasattr(prompt_item, "x_vector"):
            emb = getattr(prompt_item, "x_vector", None)
        elif isinstance(prompt_item, dict):
            emb = prompt_item.get("x_vector") or prompt_item.get("embedding")
        if emb is None:
            raise RuntimeError(
                "qwen3_tts: could not extract x-vector from voice-clone "
                "prompt; the loaded model may not support speaker-encoder "
                "embedding extraction."
            )
        import torch
        if isinstance(emb, torch.Tensor):
            return emb.detach().cpu().float().view(-1).tolist()
        import numpy as np
        if isinstance(emb, np.ndarray):
            return emb.astype(np.float32).reshape(-1).tolist()
        return list(emb)

    def unload(self) -> None:
        if self.model is not None:
            # Drop the underlying nn.Module + processor references.
            del self.model
            self.model = None
            try:
                import torch
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass

    # Alias so ManagedModel lifecycle (which calls .close() on the C++
    # session) works uniformly on either kind of _session.
    def close(self) -> None:
        self.unload()

    # ── dispatch helpers ────────────────────────────────────────────────

    def _custom_voice(self, text: str, speaker: str, language: str,
                      instruct: str, gen_kwargs: dict[str, Any]
                      ) -> tuple[list[float], int]:
        # The 0.6B CustomVoice model ignores instruct.
        audios, sr = self.model.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            instruct=instruct or None,
            **gen_kwargs,
        )
        return self._to_pcm(audios, sr)

    def _voice_clone(self, text: str, language: str,
                     ref_audio: str, ref_text: str,
                     kwargs: dict[str, Any], gen_kwargs: dict[str, Any]
                     ) -> tuple[list[float], int]:
        # Prefer a pre-computed speaker_embedding (from AudiocoreVoiceEmbedding).
        emb = kwargs.get("speaker_embedding")
        if emb is not None:
            # Build a VoiceClonePromptItem from the embedding so we skip
            # re-extracting from a reference WAV.
            try:
                prompt_item = self._build_prompt_from_embedding(emb)
                audios, sr = self.model.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=prompt_item,
                    **gen_kwargs,
                )
                return self._to_pcm(audios, sr)
            except Exception as e:
                logger.warning(
                    "qwen3_tts: embedding-based clone failed (%s); "
                    "falling back to ref_audio path", e)

        if not ref_audio:
            raise RuntimeError(
                "qwen3_tts voice_clone requires either speaker_embedding "
                "(from AudiocoreVoiceEmbedding) or reference_audio path."
            )
        audios, sr = self.model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text or None,
            **gen_kwargs,
        )
        return self._to_pcm(audios, sr)

    def _voice_design(self, text: str, language: str, instruct: str,
                      gen_kwargs: dict[str, Any]
                      ) -> tuple[list[float], int]:
        audios, sr = self.model.generate_voice_design(
            text=text,
            language=language,
            instruct=instruct,
            **gen_kwargs,
        )
        return self._to_pcm(audios, sr)

    def _build_prompt_from_embedding(self, emb: Any) -> Any:
        """Wrap a pre-computed x-vector into a VoiceClonePromptItem."""
        from qwen_tts import VoiceClonePromptItem
        import torch
        import numpy as np
        if isinstance(emb, list):
            emb = torch.tensor(emb, dtype=torch.float32)
        elif isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb).float()
        return VoiceClonePromptItem(x_vector=emb)

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _to_pcm(audios: list, sr: int) -> tuple[list[float], int]:
        """Convert qwen_tts' List[np.ndarray] → (list[float], int)."""
        import numpy as np
        if not audios:
            return [], int(sr)
        a = audios[0]
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        a = np.clip(a, -1.0, 1.0)
        return a.tolist(), int(sr)

    @staticmethod
    def _coerce_seed(seed: Any) -> int:
        """audiocore convention: seed=0 → randomize."""
        try:
            s = int(seed)
        except (TypeError, ValueError):
            s = 0
        if s <= 0:
            import random
            return random.randint(1, 2_147_483_647)
        return s

    @staticmethod
    def _coerce_language(lang: Any) -> str:
        """Map short codes → full names expected by qwen_tts.

        qwen_tts accepts: auto, chinese, english, french, german, italian,
        japanese, korean, portuguese, russian, spanish. Map common short
        forms; pass through unknown values (qwen_tts will validate).
        """
        if not lang:
            return "auto"
        s = str(lang).strip().lower()
        return {
            "en": "english", "zh": "chinese", "cn": "chinese",
            "fr": "french", "de": "german", "it": "italian",
            "ja": "japanese", "jp": "japanese",
            "ko": "korean", "kr": "korean",
            "pt": "portuguese", "br": "portuguese",
            "ru": "russian", "es": "spanish",
        }.get(s, s)

    @staticmethod
    def _resolve_model_dir(path: str) -> str | None:
        """Resolve `path` to an HF source dir containing config.json.

        Handles four inputs:
          1. Absolute HF dir → return as-is if it has config.json.
          2. Absolute GGUF dir (qwen3-tts) → look for HF source nearby.
          3. Relative name "qwen3-tts" / "qwen3_tts" → search candidates.
          4. Relative HF dir name → resolve via AUDIOCORE_MODELS_DIR.
        """
        roots = [
            os.environ.get("AUDIOCORE_MODELS_DIR", "/mnt/data/models/audio"),
            os.path.join(os.environ.get("AUDIOCORE_MODELS_DIR",
                                        "/mnt/data/models/audio")),
            "/mnt/data/models/hf_cache",
        ]

        candidates: list[str] = []
        if os.path.isabs(path):
            candidates.append(path)
        else:
            for root in roots:
                candidates.append(os.path.join(root, path))

        # Always try these remap candidates (legacy GGUF dir → HF source):
        # CustomVoice 0.6B is the best general-purpose variant for TTS.
        remaps = [
            # legacy "qwen3-tts" or "qwen3_tts" relative name
            os.path.join(roots[0], "qwen3_tts", "0.6b-customvoice"),
            os.path.join(roots[0], "qwen3_tts", "1.7b-customvoice"),
            os.path.join(roots[1], "qwen3_tts", "0.6b-base"),
            # HF cache layouts
            os.path.join(roots[2], "Qwen3-TTS-12Hz-0.6B-CustomVoice"),
            os.path.join(roots[2], "Qwen3-TTS-12Hz-1.7B-CustomVoice"),
            os.path.join(roots[2], "Qwen3-TTS-12Hz-0.6B-Base"),
            os.path.join(roots[2], "Qwen3-TTS-12Hz-1.7B-Base"),
        ]
        for c in candidates:
            if os.path.isfile(os.path.join(c, "config.json")):
                return c
        for c in remaps:
            if os.path.isfile(os.path.join(c, "config.json")):
                return c
        return None
