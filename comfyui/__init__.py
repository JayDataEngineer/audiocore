"""audiocore — ComfyUI custom nodes for the audiocore C++ audio engine.

Wraps the audiocore GGML/CUDA audio engine (Python bindings built with
ENGINE_BUILD_PYTHON=ON). Provides in-process TTS, voice cloning, voice
design, music generation, SFX, and voice-embedding caching inside ComfyUI
workflows.

Families: moss_tts, qwen3_tts, ace_step, moss_sfx_v2

Env vars (set by the Docker image or the user):
    AUDIOCORE_BUILD_DIR   — build output root (default /opt/audiocore/build-py)
    AUDIOCORE_SO_DIR      — .so directory (default $BUILD_DIR/python)
    AUDIOCORE_LIB_DIR     — shared lib directory (default $BUILD_DIR/bin)
    AUDIOCORE_MODELS_DIR  — model families root (default /mnt/data/models/audio)

Layer 3 — folder_paths:
    The ``audiocore`` folder is registered with ComfyUI's folder_paths.
    LoadAudiocoreModel's ``model_path`` input becomes a dropdown listing
    the subdirectories of AUDIOCORE_MODELS_DIR. The /models/audiocore API
    works; ComfyUI's model browser shows audiocore models.

Layer 4 — model_management:
    Each loaded session is registered in
    comfy.model_management.current_loaded_models via an
    AudiocoreLoadedModel wrapper. This makes audiocore's VRAM visible to
    /system_stats, lets unload_all_models() evict it, and lets
    load_models_gpu plan around audiocore's allocation.

Server extension:
    POST /audiocore/free   — unload the active GGML session (frees VRAM)
    GET  /audiocore/status — report active family, path, and VRAM estimate

The free endpoint lets image/video builders explicitly yield VRAM before
heavy GPU jobs, WITHOUT making audiocore reload on every TTS call. This is
the integration point that lets audiocore cooperate with ComfyUI's model
management instead of fighting it.
"""
import os

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
WEB_DIRECTORY = "./web"


# ── Layer 3: folder_paths registration ──────────────────────────────────────
def _register_folder_paths():
    """Register AUDIOCORE_MODELS_DIR as ComfyUI's ``audiocore`` folder.

    This makes the model directory discoverable via:
      - folder_paths.get_folder_paths("audiocore")
      - The /models/audiocore API endpoint
      - ComfyUI's model browser UI

    LoadAudiocoreModel's INPUT_TYPES scans this folder for subdirectories
    (one per family: moss-tts/, moss-sfx-v2/, ...) and presents them as a
    combo dropdown. The node resolves relative names to absolute paths via
    folder_paths.get_full_path() at load time.
    """
    try:
        import folder_paths
    except ImportError:
        return  # not running inside ComfyUI
    models_dir = os.environ.get(
        "AUDIOCORE_MODELS_DIR", "/mnt/data/models/audio")
    if os.path.isdir(models_dir):
        folder_paths.add_model_folder_path("audiocore", models_dir)


# ── Server extension: register HTTP routes on ComfyUI's PromptServer ───────
def _register_routes():
    """Add /audiocore/free + /audiocore/status to ComfyUI's HTTP API.

    Called once at custom-node load time. Idempotent — guards against
    double-registration under ComfyUI's hot reload.
    """
    try:
        from server import PromptServer  # type: ignore[import-not-found]
        from aiohttp import web
    except Exception:
        # ComfyUI not available (e.g. running tests outside the container).
        return

    routes = PromptServer.instance.routes
    # Guard against double-registration.
    existing = {getattr(r, "path", None) for r in routes}
    if "/audiocore/free" in existing:
        return

    @routes.post("/audiocore/free")
    async def post_audiocore_free(request):
        """Unload the active audiocore session. Idempotent.

        Body (optional JSON):
            {"force": true}  — also call gc.collect() + torch.cuda.empty_cache()
        """
        try:
            from .core import ManagedModel
            force = False
            try:
                body = await request.json()
                force = bool(body.get("force", False))
            except Exception:
                pass
            active = ManagedModel._active_model
            family = active.family if active else None
            if active is not None:
                active._cleanup()
            if force:
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
            return web.json_response({
                "status": "ok",
                "unloaded": family,
                "had_model": family is not None,
            })
        except Exception as e:
            return web.json_response(
                {"status": "error", "error": str(e)}, status=500,
            )

    @routes.get("/audiocore/status")
    async def get_audiocore_status(request):
        """Report the active audiocore session, if any."""
        try:
            from .core import ManagedModel
            active = ManagedModel._active_model
            if active is None:
                return web.json_response({"loaded": False})
            return web.json_response({
                "loaded": True,
                "family": active.family,
                "model_path": active.path,
                "backend": active.backend,
                "estimated_vram_mb": active._estimated_vram // (1024 * 1024),
            })
        except Exception as e:
            return web.json_response(
                {"status": "error", "error": str(e)}, status=500,
            )


# ── Layer 4: model_management integration ────────────────────────────────────
# audiocore sessions ARE now registered in current_loaded_models (see
# AudiocoreLoadedModel in core.py), so unload_all_models() CAN evict them.
# The wrapper's model_unload() uses an all-or-nothing policy:
#   - Full-unload requests (memory_to_free >= model_memory, or None) comply.
#   - Partial requests (memory_to_free < model_memory) are REFUSED, because
#     GGML sessions can't be partially offloaded like PyTorch ModelPatchers.
#
# This means ComfyUI's "Free Models" button works (sends unload_models=True
# → free_memory(1e30) → model_unload(1e30) → comply). But load_models_gpu's
# small free_memory requests won't evict audiocore — keeping the session hot
# for back-to-back calls.
#
# The explicit /audiocore/free endpoint above remains for the ray server's
# COLD path (non-audio job): it calls _cleanup() directly, which removes
# the wrapper and frees the C++ session synchronously.


# ── ComfyUI safety patch: cleanup_models_gc ──────────────────────────────────
def _patch_cleanup_models_gc():
    """Patch cleanup_models_gc to also evict stale LoadedModel entries.

    ComfyUI's cleanup_models_gc() (called at the top of free_memory) checks
    is_dead(), which returns ``real_model() is not None and model is None``.
    When BOTH real_model() and model are None (ModelPatcher fully GC'd after
    the execution cache is cleared by /free(free_memory=True)), is_dead()
    returns False — so the stale LoadedModel survives into free_memory's
    first loop, where model_offloaded_memory() calls self.model.model_size()
    on None and crashes with AttributeError, killing the prompt_worker thread
    and stalling the entire queue.

    The fix: also call cleanup_models() (which checks ``real_model() is None``
    and pops stale entries). This is a no-op when there are no stale models.
    """
    try:
        from comfy import model_management
        _original = model_management.cleanup_models_gc

        def _patched():
            _original()
            try:
                model_management.cleanup_models()
            except Exception:
                pass

        model_management.cleanup_models_gc = _patched
    except (ImportError, AttributeError):
        pass  # not running inside ComfyUI


_register_folder_paths()
_register_routes()
_patch_cleanup_models_gc()
