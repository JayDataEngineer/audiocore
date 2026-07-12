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
import importlib
import importlib.util
import logging
import os
import sys
import sysconfig

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
_ext = None  # type: ignore[assignment]  # The C++ pybind11 module, cached after init().


def _load_ext():
    """Load the audiocore C++ pybind11 .so and return the module.

    The custom_nodes/audiocore/ Python package shadows the .so of the same
    name in sys.modules.  We evict the cached package, load the .so directly
    via importlib, then restore the package so ComfyUI node imports survive.
    """
    ext = sysconfig.get_config_var("EXT_SUFFIX")
    so_path = os.path.join(_AUDIOCORE_SO_DIR, f"audiocore{ext}")
    if not os.path.exists(so_path):
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

    # ── Evict the Python package from sys.modules so the .so loads ──
    saved_pkg = sys.modules.pop("audiocore", None)
    saved_sub = {}
    for key in list(sys.modules):
        if key.startswith("audiocore."):
            saved_sub[key] = sys.modules.pop(key)

    try:
        # importlib.util.spec_from_file_location with the name "audiocore"
        # is the only way to load a pybind11 .so whose PyInit is PyInit_audiocore.
        spec = importlib.util.spec_from_file_location("audiocore", so_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["audiocore"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        # Restore the Python package so ComfyUI node imports still resolve.
        if saved_pkg is not None:
            sys.modules["audiocore"] = saved_pkg
        for k, v in saved_sub.items():
            sys.modules[k] = v


def init():
    """Initialise audiocore (idempotent). Loads the .so + GGML shared libs."""
    global _initialized, _ext
    if _initialized:
        return

    _ext = _load_ext()
    _ext.init()
    logger.info("audiocore initialized: families=%s", _ext.list_families())
    _initialized = True


class ManagedModel:
    """A loaded audiocore model session, passed through the ComfyUI graph.

    Each LoadAudiocoreModel node creates one ManagedModel. Downstream TTS /
    embedding nodes receive it via the AUDIOCORE_MODEL type. Multiple
    ManagedModels (different families or paths) can coexist in one graph.

    VRAM management: a global registry tracks all live sessions. When a new
    model is loaded, the registry unloads previous sessions first to free
    VRAM — preventing OOM-induced "tensor count mismatch" errors that occur
    when GGML can't allocate weight buffers.
    """

    # Global registry: only ONE model resident in VRAM at a time (singleton).
    # GGML models are large (8B MOSS-TTS Q8 ≈ 9 GB). On a 24 GB 4090, loading
    # a second model without freeing the first causes silent weight corruption.
    _active_model: "ManagedModel | None" = None

    def __init__(self, family: str, path: str, backend: str = "ggml_cuda"):
        self.family = family
        self.path = path
        self.backend = backend
        self._session = None

    def load(self, **extras) -> bool:
        """Load model weights. Returns True on success."""
        if self._session is not None:
            return True

        # ── Free VRAM from any previously loaded model ──
        prev = ManagedModel._active_model
        if prev is not None and prev is not self:
            logger.info("unloading %s from VRAM before loading %s",
                        prev.family, self.family)
            prev._cleanup()

        init()
        try:
            s = _ext.Session.create(self.family)
            # The C++ Session.load() accepts: (model_path, backend, device,
            # threads, extras: dict[str,str]).  Family-specific keys like
            # te_path, talker_path, etc. must be wrapped inside `extras`.
            load_extras = {k: str(v) for k, v in extras.items() if v}
            s.load(self.path, backend=self.backend, extras=load_extras)
            self._session = s
            ManagedModel._active_model = self
            logger.info("loaded %s from %s", self.family, self.path)
            return True
        except Exception as e:
            logger.error("load failed for %s: %s", self.family, e)
            return False

    def _cleanup(self):
        """Actually free the C++ session and its VRAM."""
        if self._session is not None:
            # Call the C++ close() — destroys the Session, which destroys
            # the Backend, which calls ggml_backend_free → cudaFree.
            close_fn = getattr(self._session, "close", None)
            if close_fn:
                try:
                    close_fn()
                except Exception:
                    pass
            self._session = None
        if ManagedModel._active_model is self:
            ManagedModel._active_model = None
        import gc
        gc.collect()
        # Force CUDA synchronisation + cache clear so GGML's cudaFree
        # allocations are visible to the next model load.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def unload(self):
        """Release the model session (frees VRAM)."""
        self._cleanup()

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
