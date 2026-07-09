"""audiocore Python bindings loader for ComfyUI.

Loads the audiocore pybind11 module (audiocore.cpython-*.so) and provides
a ManagedModel wrapper that passes through the ComfyUI graph.

Env vars:
    AUDIOCORE_BUILD_DIR  — build output root (default /opt/audiocore/build-py)
    AUDIOCORE_SO_DIR     — .so directory (default $BUILD_DIR/python)
    AUDIOCORE_LIB_DIR    — shared lib directory (default $BUILD_DIR/bin)
"""
from __future__ import annotations

import ctypes
import logging
import os
import sys

logger = logging.getLogger("audiocore-nodes")

# ── Paths (all env-var driven) ───────────────────────────────────────────────
_BUILD_DIR = os.environ.get("AUDIOCORE_BUILD_DIR", "/opt/audiocore/build-py")
_AUDIOCORE_SO_DIR = os.environ.get(
    "AUDIOCORE_SO_DIR",
    os.path.join(_BUILD_DIR, "python"),
)
_AUDIOCORE_LIB_DIR = os.environ.get(
    "AUDIOCORE_LIB_DIR",
    os.path.join(_BUILD_DIR, "bin"),
)

_initialized = False


def init():
    """Initialise audiocore (idempotent). Loads the .so + GGML shared libs."""
    global _initialized
    if _initialized:
        return

    # Find the pybind11 module for the running Python version.
    import sysconfig
    ext = sysconfig.get_config_var("EXT_SUFFIX")
    so_path = os.path.join(_AUDIOCORE_SO_DIR, f"audiocore{ext}")
    if not os.path.exists(so_path):
        # Fall back to scanning for any audiocore.cpython-*.so
        import glob
        candidates = sorted(
            glob.glob(os.path.join(_AUDIOCORE_SO_DIR, "audiocore.cpython-*.so")),
            reverse=True,
        )
        if not candidates:
            raise RuntimeError(
                f"audiocore .so not found in {_AUDIOCORE_SO_DIR}. "
                "Build it: cmake -S . -B build-py "
                "-DENGINE_BUILD_PYTHON=ON -DGGML_CUDA=ON && "
                "cmake --build build-py"
            )
        so_path = candidates[0]

    # Preload GGML shared libs so the pybind11 module can resolve symbols.
    for lib_name in ("libggml-cuda.so.0", "libllama.so.0"):
        lib_path = os.path.join(_AUDIOCORE_LIB_DIR, lib_name)
        if os.path.exists(lib_path):
            ctypes.CDLL(lib_path, ctypes.RTLD_GLOBAL)

    if _AUDIOCORE_SO_DIR not in sys.path:
        sys.path.insert(0, _AUDIOCORE_SO_DIR)
    import audiocore
    audiocore.init()
    logger.info("audiocore initialized: families=%s", audiocore.list_families())
    _initialized = True


class ManagedModel:
    """A loaded audiocore model session, passed through the ComfyUI graph.

    Each LoadAudiocoreModel node creates one ManagedModel. Downstream TTS /
    embedding nodes receive it via the AUDIOCORE_MODEL type. Multiple
    ManagedModels (different families or paths) can coexist in one graph.
    """

    def __init__(self, family: str, path: str, backend: str = "ggml_cuda"):
        self.family = family
        self.path = path
        self.backend = backend
        self._session = None

    def load(self, **extras) -> bool:
        """Load model weights. Returns True on success."""
        if self._session is not None:
            return True
        init()
        import audiocore
        try:
            s = audiocore.Session.create(self.family)
            kwargs: dict = {"backend": self.backend}
            load_kwargs = {k: v for k, v in extras.items() if v}
            s.load(self.path, **kwargs, **load_kwargs)
            self._session = s
            logger.info("loaded %s from %s", self.family, self.path)
            return True
        except Exception as e:
            logger.error("load failed for %s: %s", self.family, e)
            return False

    def unload(self):
        """Release the model session (frees VRAM)."""
        self._session = None
        import gc
        gc.collect()

    def run_tts(self, text: str, **kwargs) -> tuple[list[float], int]:
        """Run TTS inference. Returns (pcm_float32, sample_rate)."""
        if self._session is None:
            raise RuntimeError("model not loaded")
        return self._session.run_tts(text, **kwargs)

    def run_music(self, caption: str, **kwargs) -> tuple[list[float], int, int]:
        """Run music generation (ACE-Step). Returns (pcm_stereo, sample_rate, channels)."""
        if self._session is None:
            raise RuntimeError("model not loaded")
        return self._session.run_music(caption, **kwargs)

    def compute_embedding(self, wav_path: str) -> list[float]:
        """Compute a speaker embedding (qwen3_tts with speaker_encoder only)."""
        if self._session is None:
            raise RuntimeError("model not loaded")
        return self._session.compute_embedding(wav_path)

    @property
    def loaded(self) -> bool:
        return self._session is not None
