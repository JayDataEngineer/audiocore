"""Shared pytest fixtures for the audiocore ComfyUI e2e suite.

All tests hit a REAL ComfyUI instance via HTTP. No mocks. Configure via env:

    COMFYUI_HOST        default 10.100.17.3   (Docker network IP of inference-comfyui)
    COMFYUI_PORT        default 18465        (NOT 8188 — that one is not bound)
    COMFYUI_TIMEOUT     default 120          (seconds, per inference request)
    COMFYUI_SKIP_INFERENCE=1                 skip tests that produce audio
                                                (use for fast smoke runs)

If ComfyUI is unreachable, every test in the suite is SKIPPED (not failed)
with a clear reason — see `pytest_collection_modifyitems`.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import pytest
import requests

COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "10.100.17.3")
COMFYUI_PORT = int(os.environ.get("COMFYUI_PORT", "18465"))
COMFYUI_BASE = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"
COMFYUI_TIMEOUT = int(os.environ.get("COMFYUI_TIMEOUT", "120"))
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = Path(__file__).resolve().parent / "workflows"


def _comfyui_reachable() -> bool:
    try:
        r = requests.get(f"{COMFYUI_BASE}/system_stats", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


COMFYUI_UP = _comfyui_reachable()


def pytest_collection_modifyitems(config, items):
    """Skip every test in this dir if ComfyUI is not reachable."""
    if not COMFYUI_UP:
        skip = pytest.mark.skip(
            reason=f"ComfyUI not reachable at {COMFYUI_BASE} — "
                   "set COMFYUI_HOST/COMFYUI_PORT to point at a live instance."
        )
        for item in items:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def comfyui() -> str:
    """Return the ComfyUI base URL. Session-scoped — one server per run."""
    assert COMFYUI_UP, f"ComfyUI not reachable at {COMFYUI_BASE}"
    return COMFYUI_BASE


@pytest.fixture(scope="session")
def session() -> requests.Session:
    """A requests.Session with a Connection-pooling adapter."""
    s = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=8)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


@pytest.fixture(scope="session")
def object_info(session, comfyui) -> dict[str, Any]:
    """GET /object_info once per session (~700 nodes, ~1MB)."""
    r = session.get(f"{comfyui}/object_info", timeout=30)
    assert r.status_code == 200, f"/object_info failed: {r.status_code}"
    return r.json()


# ── ComfyUI prompt-submission helpers ─────────────────────────────────────

def _load_workflow(name: str) -> dict:
    """Load a ComfyUI workflow JSON from tests/comfyui/workflows/<name>."""
    path = WORKFLOWS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"workflow {name!r} not found at {path}")
    with path.open() as f:
        return json.load(f)


def _queue_prompt(session: Session, comfyui: str, workflow: dict,
                  client_id: str | None = None) -> str:
    """POST /prompt with a workflow graph. Returns the assigned prompt_id."""
    client_id = client_id or str(uuid.uuid4())
    payload = {"prompt": workflow, "client_id": client_id}
    r = session.post(f"{comfyui}/prompt", json=payload, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(
            f"/prompt rejected workflow ({r.status_code}): {r.text[:500]}"
        )
    body = r.json()
    if "error" in body:
        raise RuntimeError(f"workflow error: {body}")
    return body["prompt_id"]


def _poll_history(session: Session, comfyui: str, prompt_id: str,
                  timeout: float = 120.0, interval: float = 0.5) -> dict:
    """Poll /history until the prompt has executed. Returns the status entry.

    Raises on execution error or timeout.

    Per-poll read timeout is intentionally larger than `interval` — ComfyUI's
    HTTP thread can be momentarily unresponsive while a 9GB GGUF is mid-load
    (we've observed 5-15s stalls). The wall-clock budget is bounded by the
    outer `timeout` arg, not the per-request timeout.
    """
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            r = session.get(f"{comfyui}/history/{prompt_id}", timeout=30)
        except requests.exceptions.ReadTimeout:
            # Server busy (likely mid-load). Continue polling against the
            # outer deadline.
            time.sleep(interval)
            continue
        if r.status_code == 200:
            data = r.json()
            if prompt_id in data:
                entry = data[prompt_id]
                status = entry.get("status", {})
                if status.get("completed", False):
                    return entry
                if status.get("status_str") == "error":
                    # Surface the actual error messages from the executor.
                    msgs = status.get("messages", [])
                    raise RuntimeError(
                        f"prompt {prompt_id} failed: "
                        f"{json.dumps(msgs, indent=2)[:1000]}"
                    )
                last = status
        time.sleep(interval)
    raise TimeoutError(
        f"prompt {prompt_id} did not finish within {timeout}s. last status: {last}"
    )


def _fetch_output_bytes(session: Session, comfyui: str,
                        history_entry: dict, output_key: str = "0") -> bytes:
    """Fetch the binary output (WAV/PT) referenced by a history entry.

    ComfyUI's /history returns links under outputs[*].*.{filename,subfolder,type}.
    The 'output' type maps to /output; 'temp' to /temp.
    """
    outputs = history_entry.get("outputs", {})
    if output_key not in outputs:
        raise RuntimeError(
            f"history entry missing output[{output_key}]. have: {list(outputs)}"
        )
    node_out = outputs[output_key]
    # Most audio-producing audiocore nodes return a 'audio' or 'wav' field.
    candidates = ["audio", "wav", "result", "waveform"]
    for kind in candidates:
        if kind in node_out:
            entry = node_out[kind]
            # Newer ComfyUI wraps as a list of dicts.
            if isinstance(entry, list) and entry:
                entry = entry[0]
            fname = entry["filename"]
            subfolder = entry.get("subfolder", "")
            ftype = entry.get("type", "output")
            base = {"output": "/output", "temp": "/temp"}[ftype]
            url = f"{comfyui}{base}/{subfolder}/{fname}".replace("//", "/")
            url = url.replace(f"{comfyui}/", f"{comfyui}")  # tidy
            url = f"{comfyui}/view?filename={fname}&subfolder={subfolder}&type={ftype}"
            r = session.get(url, timeout=30)
            if r.status_code != 200:
                raise RuntimeError(
                    f"/view for {fname} returned {r.status_code}: {r.text[:200]}"
                )
            return r.content
    raise RuntimeError(
        f"no recognized output field in node output: {json.dumps(node_out)[:400]}"
    )


# Re-export for test modules
from requests import Session  # noqa: E402

@pytest.fixture
def submit():
    """Returns an object with .queue() and .wait() and .fetch_output()."""
    class _Submit:
        def queue(self, session: Session, comfyui: str, workflow_name: str,
                  **overrides) -> str:
            wf = _load_workflow(workflow_name)
            # Allow callers to patch values into the graph before queuing.
            for node_id, patch in overrides.items():
                if node_id not in wf:
                    raise KeyError(f"workflow has no node {node_id!r}")
                for k, v in patch.items():
                    wf[node_id]["inputs"][k] = v
            return _queue_prompt(session, comfyui, wf)

        def wait(self, session: Session, comfyui: str, prompt_id: str,
                 timeout: float = 120.0) -> dict:
            return _poll_history(session, comfyui, prompt_id, timeout=timeout)

        def fetch_output(self, session: Session, comfyui: str,
                         history_entry: dict, output_key: str = "0") -> bytes:
            return _fetch_output_bytes(session, comfyui, history_entry, output_key)

    return _Submit()


@pytest.fixture
def empty_queue(session, comfyui):
    """Clear ComfyUI's queue + history + caches so each test runs cold.

    - POST /queue {clear:true}   nuke pending+running
    - DELETE /history            wipe completed entries
    - POST /free {unload_models:true, free_memory:true}
                               set worker flags that free all VRAM AND
                               invalidate the per-node output cache on the
                               next prompt dispatch (otherwise identical
                               workflows short-circuit via execution_cached
                               and LoadAudiocoreModel never re-executes)
    Teardown: also POST /audiocore/free so the next test starts at
    /audiocore/status == {loaded: false}.

    If ComfyUI is mid-restart (a previous test crashed the prompt_worker
    and the container's `unless-stopped` policy is bringing it back),
    we wait up to 60s for /system_stats to answer 200 before proceeding.
    """
    # Health-check — wait if ComfyUI just crashed and is auto-restarting.
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            r = session.get(f"{comfyui}/system_stats", timeout=3)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)

    session.post(f"{comfyui}/queue", json={"clear": True}, timeout=5)
    session.delete(f"{comfyui}/history", timeout=5)
    # Set worker flags — they're consumed at the top of the next prompt's
    # dispatch, so the very next submit.queue() in the test body will run
    # cold. The 200 response just acknowledges the flags were set.
    session.post(
        f"{comfyui}/free",
        json={"unload_models": True, "free_memory": True},
        timeout=15,
    )
    yield
    # Reset audiocore session state on teardown.
    session.post(f"{comfyui}/audiocore/free", json={}, timeout=10)


@pytest.fixture(scope="session")
def manifest() -> dict:
    """Load models/manifest.json (describes what auto-download supports)."""
    p = REPO_ROOT / "models" / "manifest.json"
    if not p.exists():
        pytest.skip(f"manifest not found at {p}")
    return json.loads(p.read_text())


@pytest.fixture(scope="session")
def available_families(session, comfyui) -> set[str]:
    """The set of families the loaded audiocore .so actually registered.

    Queries `audiocore.list_families()` via a one-shot FamilyInfo workflow
    at session start. Tests that target a specific family (e.g. moss_sfx_v2)
    should `pytest.skip()` if their family isn't in this set — the live .so
    may lag the source tree (a stale build won't have moss_sfx_v2).
    """
    wf = {"1": {"class_type": "AudiocoreFamilyInfo", "inputs": {}}}
    payload = {"prompt": wf, "client_id": "family-probe"}
    r = session.post(f"{comfyui}/prompt", json=payload, timeout=10)
    if r.status_code != 200:
        pytest.skip(
            f"couldn't probe families via /prompt: {r.status_code} {r.text[:200]}"
        )
    pid = r.json()["prompt_id"]
    entry = _poll_history(session, comfyui, pid, timeout=60)
    text = entry["outputs"].get("1", {}).get("text", [""])[0]
    # The first line looks like:
    #   "Registered families: ace_step, moss_tts, qwen3_tts"
    first = text.splitlines()[0] if text else ""
    if ":" not in first:
        return set()
    csv = first.split(":", 1)[1].strip()
    if csv == "(none)":
        return set()
    return {f.strip() for f in csv.split(",")}


def require_family(available: set[str], family: str) -> None:
    """Skip the calling test if `family` isn't in `available`."""
    if family not in available:
        pytest.skip(
            f"family {family!r} not registered in the live audiocore build "
            f"(available: {sorted(available) or 'none'}). Rebuild the .so "
            f"to enable this test."
        )


# ── Mimo audio judge ───────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def mimo():
    """Return the mimo_judge module IF MIMO_API_KEY is set AND the endpoint
    answers. Otherwise None — tests depending on Mimo should pytest.skip
    when this fixture returns None.

    OPSEC: the API key is read from env ONLY. No hardcoded fallbacks.
    """
    import os
    from .mimo_judge import mimo_available
    if not os.environ.get("MIMO_API_KEY"):
        return None
    if not mimo_available():
        return None
    from . import mimo_judge
    return mimo_judge
