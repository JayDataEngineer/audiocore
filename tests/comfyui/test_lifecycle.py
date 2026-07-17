"""ComfyUI lifecycle tests — verify the runtime surface is healthy.

Tests marked `lifecycle` exercise:
  - ComfyUI itself is up and reports sane system_stats
  - All 6 audiocore custom nodes are registered with expected inputs
  - /audiocore/status + /audiocore/free endpoints (added by comfyui/__init__.py)
  - /audiocore/free is idempotent (safe to call when nothing is loaded)
  - The /audiocore/family-info node returns the 4 production families
  - /models/audiocore folder_paths registration succeeds
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.lifecycle


def test_comfyui_system_stats_reports_gpu(session, comfyui):
    """GET /system_stats returns 200 + a non-trivial GPU device."""
    r = session.get(f"{comfyui}/system_stats", timeout=5)
    assert r.status_code == 200
    stats = r.json()
    sysinfo = stats["system"]
    # ComfyUI always populates these.
    assert "comfyui_version" in sysinfo
    assert sysinfo["ram_total"] > 1_000_000_000  # >1 GB
    devices = stats.get("devices", [])
    assert devices, "no CUDA devices reported"
    assert devices[0]["name"], "device[0].name is empty"
    # Free VRAM at idle should be a meaningful fraction of total.
    vram_total = int(devices[0]["vram_total"])
    vram_free = int(devices[0]["vram_free"])
    if vram_total > 0:
        assert vram_free > 0, "no free VRAM — ComfyUI is saturated"


def test_all_six_audiocore_nodes_registered(object_info):
    """All 6 audiocore node classes appear in /object_info."""
    expected = {
        "LoadAudiocoreModel",
        "UnloadAudiocoreModel",
        "AudiocoreTTS",
        "AudiocoreMusic",
        "AudiocoreVoiceEmbedding",
        "AudiocoreFamilyInfo",
    }
    found = {name for name in object_info if "audiocore" in name.lower()}
    missing = expected - found
    assert not missing, f"missing audiocore nodes: {missing}"


def test_load_node_inputs_match_contract(object_info):
    """LoadAudiocoreModel exposes family + model_path + backend inputs."""
    node = object_info["LoadAudiocoreModel"]["input"]
    required = node["required"]
    assert "family" in required, "LoadAudiocoreModel missing required 'family'"
    assert "model_path" in required
    assert "backend" in required
    # family must be a combo listing the 4 production families
    family_combo = required["family"][0]
    assert isinstance(family_combo, list)
    assert {"moss_tts", "qwen3_tts", "ace_step", "moss_sfx_v2"}.issubset(set(family_combo)), (
        f"family dropdown missing production families; got {family_combo}"
    )
    # backend is ggml_cuda | ggml_cpu
    backend_combo = required["backend"][0]
    assert "ggml_cuda" in backend_combo
    assert "ggml_cpu" in backend_combo


def test_tts_node_inputs_match_contract(object_info):
    """AudiocoreTTS exposes model + text + voice + standard sampling inputs."""
    required = object_info["AudiocoreTTS"]["input"]["required"]
    optional = object_info["AudiocoreTTS"]["input"].get("optional", {})
    assert "model" in required
    assert "text" in required
    # voice_file + voice_pca_strengths are the combined voice+emotion steering surface
    assert "voice_file" in optional
    assert "voice_pca_strengths" in optional


def test_audiocore_status_idle_reports_unloaded(session, comfyui, empty_queue):
    """/audiocore/status returns {loaded: false} when no model is active."""
    r = session.get(f"{comfyui}/audiocore/status", timeout=5)
    assert r.status_code == 200
    body = r.json()
    assert body.get("loaded") is False, (
        f"expected loaded=false at idle, got: {body}"
    )


def test_audiocore_free_is_idempotent_when_idle(session, comfyui, empty_queue):
    """POST /audiocore/free is safe to call when nothing is loaded.

    Returns {status: ok, had_model: false} — must NOT error.
    """
    r = session.post(f"{comfyui}/audiocore/free", json={}, timeout=5)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["had_model"] is False
    assert body["unloaded"] is None


def test_audiocore_free_with_force_does_not_crash_idle_server(
    session, comfyui, empty_queue
):
    """force=true asks for torch.cuda.empty_cache() — must still succeed at idle."""
    r = session.post(
        f"{comfyui}/audiocore/free", json={"force": True}, timeout=15
    )
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_models_folder_registered(session, comfyui):
    """/models/audiocore returns the model directory listing (Layer 3)."""
    r = session.get(f"{comfyui}/models/audiocore", timeout=10)
    assert r.status_code == 200
    # Response is a flat JSON list of relative paths.
    listing = r.json() if "json" in r.headers.get("content-type", "") else r.json()
    assert isinstance(listing, list), f"expected list, got {type(listing)}"
    # Production has hundreds of files; sanity-check at least one family dir.
    flat = "\n".join(listing)
    assert "moss-tts" in flat or "moss_tts" in flat, (
        "neither 'moss-tts/' nor 'moss_tts/' appears in /models/audiocore — "
        "folder_paths registration failed"
    )


def test_queue_endpoint_returns_running_and_pending_lists(session, comfyui):
    """/queue always returns queue_running + queue_pending arrays."""
    r = session.get(f"{comfyui}/queue", timeout=5)
    assert r.status_code == 200
    body = r.json()
    assert "queue_running" in body
    assert "queue_pending" in body
    assert isinstance(body["queue_running"], list)
    assert isinstance(body["queue_pending"], list)
