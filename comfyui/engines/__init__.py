"""Family engine registry — one Python class per audiocore family.

Engines are loaded lazily on first use to avoid importing heavy deps
(torch, diffusers) when ComfyUI just wants to enumerate families.
"""
from __future__ import annotations
from typing import Any


_ENGINES: dict[str, str] = {
    # family_id → "module.ClassName" inside this package
    "moss_sfx_v2": "moss_sfx_v2.MossSfxV2Engine",
    # moss_tts, qwen3_tts, ace_step: still on C++ for now.
    # Port these in Phase 2 once moss_sfx_v2 pattern is proven in prod.
}


def get_engine_class(family: str) -> Any | None:
    """Return the engine class for `family`, or None if no Python engine
    is registered (caller should fall back to C++ .so)."""
    spec = _ENGINES.get(family)
    if spec is None:
        return None
    module_name, cls_name = spec.split(".")
    import importlib
    mod = importlib.import_module(f".{module_name}", package=__name__)
    return getattr(mod, cls_name)


def python_families() -> set[str]:
    """Families that have a Python engine."""
    return set(_ENGINES.keys())
