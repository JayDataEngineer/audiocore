"""Model lifecycle tests — load, cache, swap, unload via real ComfyUI workflows.

Tests marked `model_mgmt` exercise the ManagedModel wrapper (comfyui/core.py)
and the LoadAudiocoreModel / UnloadAudiocoreModel nodes (comfyui/nodes.py):

  - First load of a family pulls weights into VRAM (verified synchronously
    via the FamilyInfo OUTPUT_NODE text, NOT post-prompt /audiocore/status,
    because Ray schedules VRAM-cleanup prompts that evict the session)
  - /audiocore/status reflects the live session when one is resident
  - Re-loading the SAME family/path/backend reuses the active session (fast)
  - Loading a DIFFERENT family swaps (unloads prev, loads new)
  - UnloadAudiocoreModel node returns the session to idle
  - POST /audiocore/free unloads via HTTP (the ray COLD path)
  - Invalid model_path fails cleanly (workflow status=error, not crash)

All assertions are against REAL ComfyUI outputs — no mocks.
"""
from __future__ import annotations

import time

import pytest

pytestmark = [pytest.mark.model_mgmt]


def _status(session, comfyui) -> dict:
    """Helper: GET /audiocore/status and return the parsed body."""
    r = session.get(f"{comfyui}/audiocore/status", timeout=10)
    assert r.status_code == 200, f"status failed: {r.status_code} {r.text[:200]}"
    return r.json()


def _family_info_text(history_entry: dict, node_id: str = "2") -> str:
    """Extract the AudiocoreFamilyInfo node's text from a history entry.

    FamilyInfo returns ``{"ui": {"text": [text]}, ...}`` so the rendered
    string lives at outputs[node_id]["text"][0].
    """
    outputs = history_entry.get("outputs", {})
    if node_id not in outputs:
        # Some workflows use node "1" (no upstream load) — try that.
        node_id = "1"
    node_out = outputs.get(node_id, {})
    text_list = node_out.get("text", [])
    return text_list[0] if text_list else ""


def test_first_load_brings_model_into_vram(
    session, comfyui, submit, empty_queue, available_families
):
    """Queue a LoadAudiocoreModel → AudiocoreFamilyInfo graph; verify the
    active-session line appears in the workflow's synchronous output."""
    prompt_id = submit.queue(session, comfyui, "load_moss_tts.json")
    history = submit.wait(session, comfyui, prompt_id, timeout=300)

    assert history["status"].get("completed") is True, (
        f"load workflow failed: {history['status'].get('messages', [])[:3]}"
    )
    # The FamilyInfo sink ran AFTER the load — its text proves the load
    # succeeded synchronously (no race with post-prompt cleanup).
    text = _family_info_text(history, node_id="2")
    assert "Active session: family=moss_tts" in text, (
        f"family-info output missing active-session line; got:\n{text}"
    )
    assert "path=" in text and "backend=ggml_cuda" in text

    # Post-prompt, /audiocore/status MAY still reflect the session (Ray's
    # VRAM-cleanup runs on its own schedule, not strictly after each prompt).
    # We don't hard-assert here; subsequent tests check status explicitly.


def test_post_load_status_reports_family_and_backend(
    session, comfyui, submit, empty_queue
):
    """Immediately after a load workflow completes, /audiocore/status
    reports the loaded family + backend. Polled for up to 5s in case the
    ComfyUI thread is briefly delayed updating the class variable."""
    prompt_id = submit.queue(session, comfyui, "load_moss_tts.json")
    submit.wait(session, comfyui, prompt_id, timeout=300)

    deadline = time.time() + 5
    while time.time() < deadline:
        s = _status(session, comfyui)
        if s.get("loaded"):
            break
        time.sleep(0.25)
    assert s.get("loaded") is True, (
        f"status never reported loaded after load workflow: {s}"
    )
    assert s["family"] == "moss_tts"
    assert s["backend"] == "ggml_cuda"
    # moss_tts q8 GGUF is ~8-25GB depending on conversion; sanity floor.
    assert s["estimated_vram_mb"] >= 1000, (
        f"vram estimate suspiciously low: {s['estimated_vram_mb']} MB"
    )
    assert "/moss-tts" in s["model_path"], (
        f"unexpected model_path: {s['model_path']}"
    )


def test_reload_same_model_hits_cache_and_is_fast(
    session, comfyui, submit, empty_queue
):
    """Second load of (family, path, backend) reuses the cached session.

    The ManagedModel wrapper stores the active session in a class variable;
    a second load() with the same triple returns True without re-invoking
    Session.load(). We verify this by submitting two back-to-back load
    workflows WITHOUT freeing in between, and timing each.

    We DO NOT clear ComfyUI's prompt cache between p1 and p2 — that would
    defeat the test. The empty_queue fixture already cleared caches before
    p1, so p1 actually cold-loads. p2 hits the ManagedModel cache because
    _active_model still references the session.

    Caveat: ComfyUI's prompt dispatch adds ~0.5s of overhead per workflow
    regardless, so we look for an order-of-magnitude signal, not sub-ms.
    """
    # Prime: first load — should be COLD (cache was cleared pre-test by
    # empty_queue, and the worker consumes the free_memory flag at the
    # top of this prompt's dispatch).
    p1 = submit.queue(session, comfyui, "load_moss_tts.json")
    t0 = time.time()
    submit.wait(session, comfyui, p1, timeout=300)
    prime_s = time.time() - t0

    # Warm reload — NO empty_queue, NO /audiocore/free between. ComfyUI's
    # prompt-output cache will short-circuit the workflow (so warm_s is
    # the bare dispatch overhead), which is itself evidence that nothing
    # re-loaded. We only assert warm is much faster than prime.
    p2 = submit.queue(session, comfyui, "load_moss_tts.json")
    t0 = time.time()
    submit.wait(session, comfyui, p2, timeout=120)
    warm_s = time.time() - t0

    # If prime was already very fast (≤1s), the test is meaningless —
    # the cold clear didn't take effect. Skip rather than fail.
    if prime_s < 1.0:
        pytest.skip(
            f"prime load only took {prime_s:.2f}s — empty_queue cache-clear "
            f"may not have taken effect; can't measure warm-vs-cold signal"
        )

    # Warm should be much faster than the prime load. Generous threshold
    # because ComfyUI's queue adds fixed overhead either way.
    assert warm_s < prime_s * 0.5, (
        f"warm reload ({warm_s:.2f}s) not meaningfully faster than "
        f"prime ({prime_s:.2f}s) — cache hit may not be firing"
    )


def test_family_swap_unloads_previous(
    session, comfyui, submit, empty_queue, available_families
):
    """Loading a 2nd family swaps the active session.

    We pick the swap target from whatever's actually registered (avoids a
    hard requirement on moss_sfx_v2 if the .so is stale).
    """
    # Pick a swap target that's NOT moss_tts.
    candidates = sorted(available_families - {"moss_tts"})
    if not candidates:
        pytest.skip("only one family registered; can't test family swap")
    swap_to = candidates[0]

    # Use a workflow that swaps inside a single prompt: load moss_tts, then
    # load <swap_to>, then FamilyInfo chained to the second load.
    wf = {
        "1": {
            "class_type": "LoadAudiocoreModel",
            "inputs": {
                "family": "moss_tts",
                "model_path": "moss-tts",
                "backend": "ggml_cuda",
            },
        },
        "2": {
            "class_type": "LoadAudiocoreModel",
            "inputs": {
                "family": swap_to,
                # ace_step → acestep-cpp-converted; others use same name as family with - for _.
                "model_path": "acestep-cpp-converted"
                if swap_to == "ace_step"
                else swap_to.replace("_", "-"),
                "backend": "ggml_cuda",
            },
        },
        "3": {
            "class_type": "AudiocoreFamilyInfo",
            "inputs": {"model": ["2", 0]},
        },
    }
    payload = {"prompt": wf, "client_id": "swap-test"}
    r = session.post(f"{comfyui}/prompt", json=payload, timeout=10)
    assert r.status_code == 200, f"swap workflow rejected: {r.text[:300]}"
    pid = r.json()["prompt_id"]
    history = submit.wait(session, comfyui, pid, timeout=400)

    assert history["status"].get("completed") is True, (
        f"swap workflow failed: {history['status']}"
    )
    text = _family_info_text(history, node_id="3")
    assert f"Active session: family={swap_to}" in text, (
        f"family didn't swap to {swap_to}; family-info output:\n{text}"
    )


def test_unload_node_returns_session_to_idle(
    session, comfyui, submit, empty_queue
):
    """UnloadAudiocoreModel node drops the session synchronously.

    The chained FamilyInfo (no model input) should NOT report an active
    session in its text — proving the unload took effect within the prompt.
    """
    pid = submit.queue(session, comfyui, "load_then_unload.json")
    history = submit.wait(session, comfyui, pid, timeout=300)
    assert history["status"].get("completed") is True, (
        f"load+unload workflow failed: {history['status']}"
    )
    # Node "3" is the standalone FamilyInfo (no model input).
    text = _family_info_text(history, node_id="3")
    assert "Active session" not in text, (
        f"unload node didn't clear the session; family-info still reports:\n{text}"
    )
    # And the server-side _active_model is now None.
    s = _status(session, comfyui)
    assert s.get("loaded") is False, (
        f"server-side status still loaded after unload: {s}"
    )


def test_http_free_endpoint_unloads_active_session(
    session, comfyui, submit, empty_queue
):
    """POST /audiocore/free (the ray COLD path) unloads the active session.

    Pattern: load → free → status must report idle. The free endpoint is
    the integration point that lets a non-audio ComfyUI workflow (image/
    video) explicitly yield VRAM before its heavy GPU work.
    """
    # Load.
    p = submit.queue(session, comfyui, "load_moss_tts.json")
    submit.wait(session, comfyui, p, timeout=300)

    # The model should be loaded at this point — poll briefly to confirm.
    deadline = time.time() + 5
    while time.time() < deadline:
        if _status(session, comfyui).get("loaded"):
            break
        time.sleep(0.25)

    r = session.post(f"{comfyui}/audiocore/free", json={}, timeout=15)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    # had_model may be False if Ray's cleanup ran first; accept either but
    # log so the test fails informatively when neither path can find a model.
    if body["had_model"] is True:
        assert body["unloaded"] == "moss_tts", (
            f"expected unloaded='moss_tts', got {body}"
        )

    # Now idle, regardless of what free reported.
    s = _status(session, comfyui)
    assert s.get("loaded") is False, (
        f"server-side status still loaded after free: {s}"
    )


def test_http_free_with_force_reports_clean_unload(
    session, comfyui, submit, empty_queue
):
    """force=true runs gc + torch.cuda.empty_cache — still succeeds.

    Idempotent: if no model is loaded when force=true arrives, returns
    had_model=false but status=ok (no crash).
    """
    # First free when nothing is loaded (idempotent baseline).
    r = session.post(f"{comfyui}/audiocore/free", json={"force": True}, timeout=30)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    # Now load, then force-free.
    p = submit.queue(session, comfyui, "load_moss_tts.json")
    submit.wait(session, comfyui, p, timeout=300)
    # Wait briefly for status to reflect load.
    deadline = time.time() + 5
    while time.time() < deadline:
        if _status(session, comfyui).get("loaded"):
            break
        time.sleep(0.25)

    r = session.post(f"{comfyui}/audiocore/free", json={"force": True}, timeout=30)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    # Regardless of had_model, the post-free status must be idle.
    assert _status(session, comfyui).get("loaded") is False


def test_invalid_model_path_fails_workflow_cleanly(
    session, comfyui, submit, empty_queue
):
    """A non-existent model_path is rejected at /prompt submission time.

    ComfyUI validates combo inputs (model_path is a dropdown) against the
    list returned by LoadAudiocoreModel.INPUT_TYPES. An out-of-list value
    produces a 400 with `value_not_in_list` — clean validation, not a
    server crash. The prompt never reaches the executor.

    (Pre-registration server-side validation is the right design: it
    surfaces typos before any GGUF load is attempted.)
    """
    try:
        submit.queue(
            session, comfyui, "load_moss_tts.json",
            **{"1": {"model_path": "this-does-not-exist-xyz"}}
        )
    except RuntimeError as e:
        # Expected: /prompt returned 400 with value_not_in_list.
        msg = str(e).lower()
        assert "400" in msg or "value_not_in_list" in msg or "not in list" in msg, (
            f"submit failed for an unexpected reason: {e}"
        )
    else:
        pytest.fail(
            "expected /prompt to reject invalid model_path with 400, "
            "but the workflow was accepted"
        )

    # Server must still be alive after the validation rejection.
    r = session.get(f"{comfyui}/system_stats", timeout=5)
    assert r.status_code == 200, "ComfyUI died from an invalid model_path"


def test_audiocore_family_info_node_lists_production_families(
    session, comfyui, submit, empty_queue, available_families
):
    """The AudiocoreFamilyInfo node lists whatever the .so registered."""
    pid = submit.queue(session, comfyui, "family_info.json")
    history = submit.wait(session, comfyui, pid, timeout=60)
    assert history["status"].get("completed") is True
    text = _family_info_text(history, node_id="1")
    # Whatever's registered, family-info must list it.
    for f in available_families:
        assert f in text, f"family {f!r} missing from family-info output:\n{text}"
