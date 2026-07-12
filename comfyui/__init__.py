"""audiocore — ComfyUI custom nodes for the audiocore C++ audio engine.

Wraps the audiocore GGML/CUDA audio engine (Python bindings built with
ENGINE_BUILD_PYTHON=ON). Provides in-process TTS, voice cloning, voice
design, music generation, SFX, and voice-embedding caching inside ComfyUI
workflows.

Families: moss_tts, qwen3_tts, ace_step, moss_sfx_v2

Env vars (set by the Docker image or the user):
    AUDIOCORE_BUILD_DIR  — build output root (default /opt/audiocore/build-py)
    AUDIOCORE_SO_DIR     — .so directory (default $BUILD_DIR/python)
    AUDIOCORE_LIB_DIR    — shared lib directory (default $BUILD_DIR/bin)

Server extension:
    POST /audiocore/free   — unload the active GGML session (frees VRAM)
    GET  /audiocore/status — report active family, path, and VRAM estimate

The free endpoint lets image/video builders explicitly yield VRAM before
heavy GPU jobs, WITHOUT making audiocore reload on every TTS call. This is
the integration point that lets audiocore cooperate with ComfyUI's model
management instead of fighting it.
"""
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
WEB_DIRECTORY = "./web"


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
            })
        except Exception as e:
            return web.json_response(
                {"status": "error", "error": str(e)}, status=500,
            )


# ── Hook into ComfyUI's existing /free endpoint ────────────────────────────
# DELIBERATELY NOT DONE: monkey-patching comfy.model_management.unload_all_models
# to also free audiocore would fire on EVERY /free call, defeating the caching
# policy. Instead, the explicit /audiocore/free endpoint above is the ONLY way
# to evict the audiocore GGML session — it's called by server.comfyui_client
# when image/video jobs need VRAM, and skipped for back-to-back audio calls.
#
# This keeps audiocore under explicit control: same-family audio calls stay
# hot (~2s), cross-domain switches yield VRAM via the explicit endpoint.


_register_routes()
