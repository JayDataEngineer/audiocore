"""moss_sfx_v2 — text-to-sound-effect via upstream MossSoundEffectPipeline.

Proxies OpenMOSS/MOSS-TTS/moss_soundeffect_v2 directly. No GGUF, no
conversion. Loads the diffusers-style pipeline from the HF source dir
(model_index.json + transformer/ + vae/ + text_encoder/ + tokenizer/) and
runs flow-matching DiT inference.

The pipeline lives at /opt/MOSS-TTS/moss_soundeffect_v2 (cloned from
github.com/OpenMOSS/MOSS-TTS). The HF weights live at
${MODELS_DIR}/moss-soundeffect-v2/ (safetensors, downloaded via
huggingface).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("audiocore.moss_sfx_v2")

# Upstream pipeline lives here (cloned once at container build time).
_UPSTREAM_DIR = "/opt/MOSS-TTS"
_UPSTREAM_LOADED = False


def _ensure_upstream_on_path() -> None:
    """Insert the upstream MOSS-TTS repo on sys.path exactly once."""
    global _UPSTREAM_LOADED
    if _UPSTREAM_LOADED:
        return
    if _UPSTREAM_DIR not in sys.path:
        sys.path.insert(0, _UPSTREAM_DIR)
    _UPSTREAM_LOADED = True


def _disable_torch_compile() -> None:
    """Make torch.compile a no-op BEFORE the upstream pipeline imports.

    The upstream `model_fn_wan_video` is decorated with
    `@torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)`.
    On torch 2.11 + CUDA 12.8 the cudagraphs path calls
    `torch._C._cuda_checkPoolLiveAllocations`, which is incompatible with
    `cudaMallocAsync` (the default CUDA allocator backend) and crashes with:

        RuntimeError: cudaMallocAsync does not yet support
        checkPoolLiveAllocations.

    Setting `torch._dynamo.config.disable = True` BEFORE the upstream
    module is imported turns the `@torch.compile(...)` decorator into a
    no-op, so `model_fn_wan_video` runs eagerly. Inference is still fast
    (~10 it/s on a 4090) and numerically identical to the compiled path.

    Must run before `from moss_soundeffect_v2 import MossSoundEffectPipeline`
    so the decorator is a no-op at function-definition time.
    """
    try:
        import torch._dynamo
        if not torch._dynamo.config.disable:
            torch._dynamo.config.disable = True
            logger.info(
                "moss_sfx_v2: disabled torch.compile globally "
                "(cudaMallocAsync + cudagraphs checkPoolLiveAllocations "
                "incompatibility in this container); running eagerly")
    except Exception:
        logger.debug("moss_sfx_v2: could not set torch._dynamo.config.disable")


def _patch_sage_attention_off() -> None:
    """Disable SageAttention in the upstream DiT module.

    The upstream `flash_attention` dispatcher picks SageAttention when
    installed, but its `q.dtype == k.dtype == v.dtype` assert trips on
    the bf16 rope-applied tensors in this container build (flash_attn 2/3
    are not installed here, so we can't fall through to those branches).
    Forcing `SAGE_ATTN_AVAILABLE = False` makes the dispatch land on the
    PyTorch SDPA fallback, which handles bf16 correctly and produces
    numerically equivalent output.

    Idempotent: safe to call before or after the DiT module is imported.
    """
    try:
        import moss_soundeffect_v2.diffsynth.models.wan_video_dit as dit
        if getattr(dit, "SAGE_ATTN_AVAILABLE", False):
            dit.SAGE_ATTN_AVAILABLE = False
            logger.info(
                "moss_sfx_v2: disabled SageAttention in wan_video_dit "
                "(dtype-assert incompatibility); using torch SDPA fallback")
    except Exception:
        logger.debug(
            "moss_sfx_v2: wan_video_dit not yet imported; "
            "will retry patch on next load")


# Default hyperparameters from upstream README. Caller can override via kwargs.
_DEFAULTS = {
    "num_inference_steps": 100,   # upstream default
    "cfg_scale": 4.0,
    "sigma_shift": 5.0,
    "seconds": 10.0,
}


class MossSfxV2Engine:
    """Wraps MossSoundEffectPipeline to match audiocore's session contract."""

    def __init__(self) -> None:
        self.pipe: Any = None
        self._model_dir: str | None = None

    # ── audiocore session contract ──────────────────────────────────────

    def load(self, path: str, **extras: Any) -> bool:
        """Load the pipeline from `path` (HF source dir with model_index.json).

        Accepts either the HF source dir directly, or the legacy GGUF dir
        name ("moss-sfx-v2") — the latter is remapped to the HF dir because
        that's where the safetensors live.
        """
        import torch

        hf_dir = self._resolve_model_dir(path)
        if hf_dir is None:
            raise RuntimeError(
                f"moss_sfx_v2: could not resolve a HF source dir from {path!r}. "
                f"Point model_path at the moss-soundeffect-v2 HF dir."
            )

        _ensure_upstream_on_path()
        # Disable torch.compile BEFORE importing the upstream pipeline so
        # its @torch.compile decorator becomes a no-op. See
        # _disable_torch_compile for the cudaMallocAsync rationale.
        _disable_torch_compile()

        try:
            from moss_soundeffect_v2 import MossSoundEffectPipeline
        except ImportError as e:
            raise RuntimeError(
                f"moss_sfx_v2: upstream pipeline not available "
                f"(looked at {_UPSTREAM_DIR}): {e}"
            ) from e

        # Apply the SageAttention compatibility patch AFTER the import so
        # the DiT module is definitely cached in sys.modules.
        _patch_sage_attention_off()

        logger.info("moss_sfx_v2: loading MossSoundEffectPipeline from %s", hf_dir)
        self.pipe = MossSoundEffectPipeline.from_pretrained(
            hf_dir,
            torch_dtype=torch.bfloat16,
            device="cuda",
        )
        self._model_dir = hf_dir
        logger.info("moss_sfx_v2: pipeline ready (sample_rate=%d)",
                    self.pipe.sample_rate)
        return True

    def run_tts(self, text: str, **kwargs: Any) -> tuple[list[float], int]:
        """Generate sound-effect audio.

        `text` is the SFX prompt (e.g. "heavy wooden door slamming shut").
        kwargs map to pipeline hyperparameters; unrecognized kwargs are
        dropped silently (the C++ engine accepted many moss_tts-style
        kwargs that don't apply here).
        """
        if self.pipe is None:
            raise RuntimeError("moss_sfx_v2: engine not loaded")

        seed = self._coerce_seed(kwargs.get("seed", 0))
        seconds = float(kwargs.get("duration", _DEFAULTS["seconds"]))
        steps = int(kwargs.get("num_inference_steps",
                               kwargs.get("steps", _DEFAULTS["num_inference_steps"])))
        cfg = float(kwargs.get("cfg_scale", _DEFAULTS["cfg_scale"]))
        sigma_shift = float(kwargs.get("sigma_shift", _DEFAULTS["sigma_shift"]))

        # 'speed' / 'temperature' / 'top_p' / 'voice' / etc. are TTS params
        # that don't apply to flow-matching SFX; silently drop them.
        logger.info(
            "moss_sfx_v2: prompt=%r seconds=%.1f steps=%d cfg=%.1f seed=%d",
            text[:80], seconds, steps, cfg, seed,
        )
        audio = self.pipe(
            prompt=text,
            seconds=seconds,
            num_inference_steps=steps,
            cfg_scale=cfg,
            sigma_shift=sigma_shift,
            seed=seed,
        )
        # audio is a tuple/list of tensors (one per batch). We take [0],
        # squeeze any channel dim, and flatten to a mono float list.
        import torch
        wav = audio[0].detach().cpu().float()
        if wav.dim() > 1:
            wav = wav.view(-1)
        pcm = wav.clamp(-1.0, 1.0).tolist()
        return pcm, int(self.pipe.sample_rate)

    def run_music(self, caption: str, **kwargs: Any) -> tuple[list[float], int, int]:
        raise RuntimeError("moss_sfx_v2 does not support music generation")

    def compute_embedding(self, wav_path: str) -> list[float]:
        raise RuntimeError(
            "moss_sfx_v2 does not support speaker embedding computation"
        )

    def unload(self) -> None:
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    # Alias so the ManagedModel lifecycle code (which calls `.close()` on the
    # C++ session) works uniformly on either kind of `_session`.
    def close(self) -> None:
        self.unload()

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_model_dir(path: str) -> str | None:
        """Resolve `path` to a HF source dir containing model_index.json.

        Handles four inputs (nodes.py resolves relative names to absolute
        paths before calling ManagedModel, so we must remap for both):
          1. Absolute path to the HF dir (moss-soundeffect-v2) → as-is.
          2. Absolute path to the legacy GGUF dir (moss-sfx-v2) → remap
             sibling → moss-soundeffect-v2.
          3. Relative name "moss-soundeffect-v2" → AUDIOCORE_MODELS_DIR/name.
          4. Relative name "moss-sfx-v2" → remap to moss-soundeffect-v2.
        """
        candidates: list[str] = []
        if os.path.isabs(path):
            candidates.append(path)
            # Legacy GGUF-dir remap: try sibling "moss-soundeffect-v2"
            # next to the requested "moss-sfx-v2" directory.
            if path.rstrip("/").endswith("moss-sfx-v2"):
                candidates.append(
                    os.path.join(os.path.dirname(path.rstrip("/")),
                                 "moss-soundeffect-v2"))
        else:
            root = os.environ.get(
                "AUDIOCORE_MODELS_DIR", "/mnt/data/models/audio"
            )
            candidates.append(os.path.join(root, path))
            if path.rstrip("/").endswith("moss-sfx-v2"):
                candidates.append(os.path.join(root, "moss-soundeffect-v2"))

        for c in candidates:
            if os.path.isfile(os.path.join(c, "model_index.json")):
                return c
        return None

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
