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
"""
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
WEB_DIRECTORY = "./web"
