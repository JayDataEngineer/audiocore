"""audiocore Python bindings loader for ComfyUI.

Loads the audiocore pybind11 module (audiocore.cpython-*.so) and provides
a ManagedModel wrapper that passes through the ComfyUI graph.

Env vars:
    AUDIOCORE_BUILD_DIR   — build output root (default /opt/audiocore/build-py)
    AUDIOCORE_SO_DIR      — .so directory (default $BUILD_DIR/python)
    AUDIOCORE_LIB_DIR     — shared lib directory (default $BUILD_DIR/bin)
    AUDIOCORE_MODELS_DIR  — model families root (default /mnt/data/models/audio)
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
_AUDIOCORE_MODELS_DIR = os.environ.get(
    "AUDIOCORE_MODELS_DIR", "/mnt/data/models/audio",
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


def _estimate_vram(model_path: str) -> int:
    """Estimate resident VRAM by summing .gguf file sizes in the model dir.

    This is an UPPER BOUND — GGML may memory-map weights and only touch a
    subset on CUDA. But it's the right number for model_management's
    accounting: if ComfyUI thinks audiocore occupies 9 GB, it won't try to
    simultaneously load a 20 GB image model that would OOM the GPU.
    """
    total = 0
    try:
        for f in os.listdir(model_path):
            if f.endswith(".gguf"):
                total += os.path.getsize(os.path.join(model_path, f))
    except OSError:
        pass
    return total


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
        self._estimated_vram: int = 0
        self._loaded_model_wrapper: "AudiocoreLoadedModel | None" = None

    def load(self, **extras) -> bool:
        """Load model weights. Returns True on success.

        Caching: if the previously-loaded model (tracked by the
        ``_active_model`` class variable) matches this model's family +
        path, REUSE its session instead of re-loading from disk. This is
        the hot path that drops back-to-back TTS from ~20s (cold GGUF
        load) to ~2s (cached). Each ComfyUI graph creates a fresh
        ``ManagedModel`` instance, so without this reuse check the cache
        would never hit.

        model_management: on successful COLD load, the session is
        registered in ``comfy.model_management.current_loaded_models`` via
        an ``AudiocoreLoadedModel`` wrapper. This makes audiocore's VRAM
        visible to ComfyUI's accounting and lets ``unload_all_models()``
        evict it cooperatively.
        """
        if self._session is not None:
            return True  # this instance already loaded

        # ── Hot path: reuse the active session if compatible ──
        active = ManagedModel._active_model
        if (active is not None
                and active is not self
                and active.family == self.family
                and active.path == self.path
                and active._session is not None):
            self._session = active._session
            self._estimated_vram = active._estimated_vram
            # The wrapper stays on the original (active) ManagedModel —
            # model_unload() on it would evict the shared session.
            logger.info("reusing cached %s session (path=%s)",
                        self.family, self.path)
            return True

        # ── Cold path: free any incompatible active model, then load ──
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
            self._estimated_vram = _estimate_vram(self.path)
            ManagedModel._active_model = self
            self._register_in_model_management()
            logger.info("loaded %s from %s (%d MB estimated VRAM)",
                        self.family, self.path,
                        self._estimated_vram // (1024 * 1024))
            return True
        except Exception as e:
            logger.error("load failed for %s: %s", self.family, e)
            return False

    def _register_in_model_management(self):
        """Register this session in ComfyUI's current_loaded_models.

        Layer 4 integration: makes audiocore visible to ComfyUI's VRAM
        accounting (/system_stats), lets the "Free Models" button evict
        it, and lets load_models_gpu plan around audiocore's allocation.
        """
        try:
            from comfy import model_management
        except ImportError:
            return  # not running inside ComfyUI (e.g. standalone test)
        # Remove any stale wrapper for this ManagedModel
        if self._loaded_model_wrapper is not None:
            try:
                model_management.current_loaded_models.remove(
                    self._loaded_model_wrapper)
            except ValueError:
                pass
        self._loaded_model_wrapper = AudiocoreLoadedModel(self)
        model_management.current_loaded_models.insert(
            0, self._loaded_model_wrapper)

    def _unregister_from_model_management(self):
        """Remove this session's wrapper from current_loaded_models."""
        if self._loaded_model_wrapper is None:
            return
        try:
            from comfy import model_management
            model_management.current_loaded_models.remove(
                self._loaded_model_wrapper)
        except (ValueError, ImportError):
            pass
        self._loaded_model_wrapper = None

    def _free_session(self):
        """Free the C++ session WITHOUT touching current_loaded_models.

        This is the path for ComfyUI-initiated evictions (model_unload).
        ComfyUI's free_memory() iterates current_loaded_models by index and
        pops successful unloads ITSELF (pop(i) after the loop). If we remove
        from the list inside model_unload(), free_memory's subsequent
        ``current_loaded_models[i]`` access hits IndexError — killing the
        prompt_worker thread and stalling the entire queue.

        So: free the C++ session, clear _active_model, invalidate the stale
        wrapper reference — but leave the wrapper in current_loaded_models
        for ComfyUI to pop. is_dead() returns True (session is None), so
        free_memory's ``not shift_model.is_dead()`` guard excludes us on
        future passes.
        """
        if self._session is not None:
            close_fn = getattr(self._session, "close", None)
            if close_fn:
                try:
                    close_fn()
                except Exception:
                    pass
            self._session = None
        if ManagedModel._active_model is self:
            ManagedModel._active_model = None
        self._loaded_model_wrapper = None  # stale — ComfyUI will pop it
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def _cleanup(self):
        """Actually free the C++ session and its VRAM.

        Full lifecycle teardown for OUR code paths (/audiocore/free,
        unload(), cold-path eviction). Unregisters from model_management
        AND frees the C++ session. NOT used inside model_unload() — see
        _free_session() for why.
        """
        # Remove from model_management BEFORE freeing — so is_dead()
        # returns True and free_memory doesn't try to double-evict.
        self._unregister_from_model_management()
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
        self._ensure_loaded()
        return self._session.run_tts(text, **kwargs)

    def run_music(self, caption: str, **kwargs) -> tuple[list[float], int, int]:
        """Run music generation (ACE-Step). Returns (pcm_stereo, sample_rate, channels)."""
        self._ensure_loaded()
        return self._session.run_music(caption, **kwargs)

    def compute_embedding(self, wav_path: str) -> list[float]:
        """Compute a speaker embedding (qwen3_tts with speaker_encoder only)."""
        self._ensure_loaded()
        return self._session.compute_embedding(wav_path)

    def _ensure_loaded(self):
        """Auto-reload if the session was evicted (cache-staleness guard).

        ComfyUI's output cache may reuse a ManagedModel whose C++ session was
        freed by unload_all_models() between prompts. Transparently reload
        via the hot path (if the same family is still _active_model) or the
        cold path (full GGUF reload).
        """
        if self._session is not None:
            return
        active = ManagedModel._active_model
        if (active is not None
                and active.family == self.family
                and active.path == self.path
                and active._session is not None):
            self._session = active._session
            self._estimated_vram = active._estimated_vram
            logger.info("auto-recovered %s session from active cache", self.family)
            return
        logger.info("session was evicted — cold-reloading %s", self.family)
        if not self.load():
            raise RuntimeError(
                f"model not loaded and auto-reload failed for {self.family}"
            )

    @property
    def loaded(self) -> bool:
        return self._session is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 4: ComfyUI model_management integration
# ═══════════════════════════════════════════════════════════════════════════════
# These classes let audiocore GGML sessions participate in ComfyUI's unified
# model lifecycle (comfy.model_management.current_loaded_models) so that:
#
#   - /system_stats VRAM accounting is honest (audiocore's ~9 GB is visible)
#   - ComfyUI's "Free Models" button (unload_all_models) evicts audiocore
#   - load_models_gpu can see audiocore and plan around its allocation
#   - soft_empty_cache cooperates with audiocore's CUDA pool
#
# The integration is READ-HONEST: audiocore reports its real VRAM footprint
# and complies with full-unload requests. But GGML sessions are atomic — they
# can't be partially offloaded to CPU like PyTorch ModelPatchers. So small
# free_memory requests (< model_memory) are REFUSED, keeping the session hot
# for back-to-back calls. Full-unload requests (>= model_memory, or None)
# comply immediately.

class _AudiocorePatcherStub:
    """Minimal stub that quacks like a comfy.model_patcher.ModelPatcher.

    comfy.model_management.free_memory() touches these on each LoadedModel's
    ``.model`` property:

        .is_dynamic()               → bool  (False: GGML sessions aren't dynamic)
        .model_size()               → int   (total weight bytes)
        .loaded_size()              → int   (currently-resident bytes)
        .model.__class__.__name__   → str   (for the "Unloading X" log line)

    ``sys.getrefcount()`` is also called on ``.model`` — so the property must
    return a STABLE object (self), not a fresh wrapper on each access.
    """

    __slots__ = ("_managed",)

    def __init__(self, managed: ManagedModel):
        self._managed = managed

    # .model is accessed as .model.model.__class__.__name__ in free_memory's
    # logging. Point it at self so the chain resolves to our class name.
    @property
    def model(self):
        return self

    def is_dynamic(self) -> bool:
        return False

    def model_size(self) -> int:
        return self._managed._estimated_vram

    def loaded_size(self) -> int:
        if self._managed._session is not None:
            return self._managed._estimated_vram
        return 0

    def current_loaded_device(self):
        """Return the device the session is currently on (for memory_required)."""
        import torch
        if self._managed._session is not None and torch.cuda.is_available():
            return torch.device("cuda", 0)
        return None


class AudiocoreLoadedModel:
    """LoadedModel-shaped wrapper for audiocore GGML sessions.

    Registered in ``comfy.model_management.current_loaded_models`` so the
    session participates in ComfyUI's model lifecycle. See the Layer 4
    block comment above for the full design rationale.

    Eviction policy — all-or-nothing (GGML sessions are atomic):
        memory_to_free is None or >= model_memory → comply (full unload)
        memory_to_free < model_memory              → refuse (can't partially
                                                     offload; ComfyUI looks
                                                     for smaller models instead)
    """

    def __init__(self, managed: ManagedModel):
        self._managed = managed
        try:
            import torch
            self.device = torch.device("cuda", 0) if torch.cuda.is_available() else None
        except ImportError:
            self.device = None
        self.currently_used = True
        self.real_model = None
        self._patcher = _AudiocorePatcherStub(managed)

    # ── LoadedModel interface ───────────────────────────────────────────
    @property
    def model(self):
        # free_memory accesses .model.is_dynamic(), .model.model_size(),
        # .model.loaded_size(), sys.getrefcount(.model), .model.model.__class__
        return self._patcher

    def model_memory(self) -> int:
        return self._managed._estimated_vram

    def model_loaded_memory(self) -> int:
        return self.model_memory()

    def model_offloaded_memory(self) -> int:
        return 0  # always fully loaded or fully unloaded (no partial offload)

    def model_memory_required(self, device) -> int:
        return self.model_memory()

    def model_load(self, lowvram_model_memory=0, force_patch_weights=False):
        return self  # already loaded — nothing to do

    def should_reload_model(self, force_patch_weights=False) -> bool:
        return False

    def model_unload(self, memory_to_free=None, unpatch_weights=True) -> bool:
        """Evict the GGML session. Returns True if fully unloaded.

        See class docstring for the all-or-nothing policy.

        CRITICAL: calls _free_session(), NOT _cleanup(). _cleanup() would
        remove from current_loaded_models while ComfyUI's free_memory() is
        iterating that list by index → IndexError → dead prompt_worker.
        _free_session() leaves the wrapper in the list for ComfyUI to pop.
        """
        if memory_to_free is not None and memory_to_free < self.model_memory():
            return False  # refuse — can't partially offload a GGML session
        self._managed._free_session()
        return True

    def model_use_more_vram(self, extra_memory, force_patch_weights=False):
        pass  # no-op — always fully loaded

    def __eq__(self, other):
        return isinstance(other, AudiocoreLoadedModel) \
            and self._managed is other._managed

    def __hash__(self):
        return id(self._managed)

    def is_dead(self) -> bool:
        return self._managed._session is None

    def real_model(self):
        return None
