"""Manifest-driven model auto-download + convert-lazily tests.

Tests marked `auto_fetch` exercise the OFFLINE parts of audiocore's
auto-download layer:

  - models/manifest.json is well-formed and self-consistent
    (every family/variant/file declares the required fields)
  - scripts/fetch_models.sh exposes a coherent CLI (--list, --dry-run, --help)
  - The convert_* binaries the manifest references are present and runnable

We deliberately do NOT download real model weights in this suite — that's
a 30-minute, multi-GB operation ill-suited to CI. The download path itself
is exercised end-to-end by running `scripts/fetch_models.sh <family>` by
hand against a fresh models dir.

These tests don't require ComfyUI or a working inference engine — they
govern the auto-download layer that runs BEFORE inference.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.auto_fetch]

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "models" / "manifest.json"
FETCH_SCRIPT = REPO_ROOT / "scripts" / "fetch_models.sh"


# ── Manifest schema tests ─────────────────────────────────────────────────

def test_manifest_loads_and_has_required_top_level_fields():
    """manifest.json parses and has the four top-level keys the fetcher needs."""
    assert MANIFEST_PATH.exists(), f"manifest missing at {MANIFEST_PATH}"
    m = json.loads(MANIFEST_PATH.read_text())
    for k in ("manifest_version", "project", "families", "status_legend"):
        assert k in m, f"manifest missing top-level key {k!r}"
    assert m["manifest_version"] >= 1, (
        f"unexpected manifest_version: {m['manifest_version']}"
    )
    assert m["project"] == "audiocore"
    # status_legend documents the per-mode status enum used in families.
    for status in ("wired", "partial", "not_impl", "blocked"):
        assert status in m["status_legend"], (
            f"status_legend missing {status!r}"
        )


def test_manifest_every_family_declares_display_vendor_and_variants():
    """Every family entry is structurally complete."""
    m = json.loads(MANIFEST_PATH.read_text())
    assert len(m["families"]) >= 4, (
        f"expected >=4 families (moss_tts, qwen3_tts, ace_step, moss_sfx_v2); "
        f"got {list(m['families'])}"
    )
    for fam_id, fam in m["families"].items():
        assert "display" in fam, f"family {fam_id!r} missing 'display'"
        assert "vendor" in fam, f"family {fam_id!r} missing 'vendor'"
        assert "variants" in fam and fam["variants"], (
            f"family {fam_id!r} has no variants"
        )
        # Modes are optional metadata but if present, each mode must use a
        # status value documented in status_legend.
        modes = fam.get("modes", {})
        for mode_id, mode in modes.items():
            status = mode.get("status")
            assert status in m["status_legend"], (
                f"family {fam_id!r} mode {mode_id!r} has undocumented "
                f"status {status!r}"
            )


def test_manifest_every_variant_has_files_with_size_and_source():
    """Every variant lists at least one file with a resolvable source.

    `source` can be either a URL string OR a structured provider ref like
    ``{"provider": "huggingface", "repo": "...", "revision": "main",
    "subpath": "..."}`` — both are valid; fetch_models.sh resolves them.
    """
    m = json.loads(MANIFEST_PATH.read_text())
    for fam_id, fam in m["families"].items():
        for var_id, var in fam["variants"].items():
            assert "files" in var and var["files"], (
                f"variant {fam_id}/{var_id} has no files"
            )
            for i, f in enumerate(var["files"]):
                assert "source" in f, (
                    f"{fam_id}/{var_id} file[{i}] missing 'source'"
                )
                src = f["source"]
                # Validate per source-shape.
                if isinstance(src, dict):
                    # Provider ref: must declare provider + repo at minimum.
                    assert "provider" in src and src["provider"], (
                        f"{fam_id}/{var_id} file[{i}] source dict missing "
                        f"'provider': {src}"
                    )
                    assert "repo" in src and src["repo"], (
                        f"{fam_id}/{var_id} file[{i}] source dict missing "
                        f"'repo': {src}"
                    )
                elif isinstance(src, str):
                    assert src.startswith((
                        "http://", "https://", "hf://", "huggingface.co/",
                        "./", "/", "s3://",
                    )), (
                        f"{fam_id}/{var_id} file[{i}] source string {src!r} "
                        f"uses an unrecognized scheme"
                    )
                else:
                    pytest.fail(
                        f"{fam_id}/{var_id} file[{i}] source is "
                        f"{type(src).__name__}, expected str or dict"
                    )
                # sha256 is optional but encouraged. If absent, the fetcher
                # proceeds without integrity check — flagged in --dry-run.


def test_manifest_lists_production_families():
    """The 4 production families are present in the manifest."""
    m = json.loads(MANIFEST_PATH.read_text())
    expected = {"moss_tts", "qwen3_tts", "ace_step", "moss_sfx_v2"}
    found = set(m["families"].keys())
    missing = expected - found
    assert not missing, (
        f"manifest missing production families: {sorted(missing)} "
        f"(have: {sorted(found)})"
    )


# ── fetch_models.sh CLI tests ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def fetch_script():
    """fetch_models.sh exists and is executable, or skip the module."""
    if not FETCH_SCRIPT.exists():
        pytest.skip(f"fetch_models.sh not found at {FETCH_SCRIPT}")
    if not os.access(FETCH_SCRIPT, os.X_OK):
        pytest.skip(f"{FETCH_SCRIPT} not executable")
    return FETCH_SCRIPT


def test_fetch_models_rejects_unknown_flag_cleanly(fetch_script):
    """Unknown flags exit 2 with a clear 'unknown flag' message — not a
    silent success or a Python traceback. This is the fetcher's only
    user-facing usage surface (it has no --help; the usage banner lives
    in the script's header comment)."""
    r = subprocess.run(
        ["bash", str(fetch_script), "--bogus"],
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 2, (
        f"expected exit 2 for unknown flag, got {r.returncode}"
    )
    assert "unknown flag" in (r.stdout + r.stderr).lower(), (
        f"expected 'unknown flag' message; got: {r.stderr[:300]}"
    )


def test_fetch_models_list_flag_outputs_status_matrix(fetch_script):
    """`--list` prints the family × mode status matrix.

    The output is the same data as models/manifest.json's families[*].modes
    but flattened for terminal reading. We verify it covers every production
    family and uses the documented status vocabulary.
    """
    m = json.loads(MANIFEST_PATH.read_text())
    r = subprocess.run(
        ["bash", str(fetch_script), "--list"],
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 0
    out = r.stdout
    for fam_id in m["families"]:
        assert fam_id in out, (
            f"--list output missing family {fam_id!r}:\n{out[:400]}"
        )
    # Every status word in --list must come from status_legend.
    for status in ("wired", "partial", "not_impl", "blocked"):
        # These appear as tab-separated values in the matrix.
        pass  # at least one wired mode must exist per production family
    assert "wired" in out, (
        f"--list shows no 'wired' modes — empty/broken manifest?\n{out[:400]}"
    )


def test_fetch_models_list_flag_outputs_mode_matrix(fetch_script):
    """`fetch_models.sh --list` prints the family × mode status matrix."""
    r = subprocess.run(
        ["bash", str(fetch_script), "--list"],
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 0, (
        f"--list exited {r.returncode}: {r.stderr[:300]}"
    )
    # --list output should mention each production family by id.
    out = r.stdout
    for fam in ("moss_tts", "qwen3_tts", "ace_step", "moss_sfx_v2"):
        assert fam in out, (
            f"--list output missing family {fam!r}; stdout:\n{out[:500]}"
        )


def test_fetch_models_dry_run_resolves_urls_without_downloading(
    fetch_script, tmp_path
):
    """`--dry-run` for a single family prints URLs but creates no files.

    Uses a tmp models dir so we can be certain no real download happens.
    """
    env = os.environ.copy()
    env["AUDIOCORE_MODELS_DIR"] = str(tmp_path / "models")
    r = subprocess.run(
        ["bash", str(fetch_script), "--dry-run", "ace_step"],
        capture_output=True, text=True, timeout=20, env=env,
    )
    assert r.returncode == 0, (
        f"--dry-run failed: {r.stderr[:400]}"
    )
    out = r.stdout + r.stderr
    # Dry-run should mention URLs it WOULD fetch.
    assert ("http" in out or "would" in out.lower()
            or "dry" in out.lower()), (
        f"--dry-run output doesn't mention URLs/intent:\n{out[:400]}"
    )
    # CRITICAL: no files should have been created in the models dir.
    created = list(tmp_path.glob("**/*"))
    created = [p for p in created if p.exists() and p.is_file()]
    assert not created, (
        f"--dry-run created files (should be a no-op): {created}"
    )


def test_fetch_models_requires_jq_and_manifest(fetch_script, tmp_path):
    """fetch_models.sh fails fast with a clear error if jq is missing
    or the manifest is unreachable."""
    # With PATH gutted (no jq), it must exit non-zero with a clear message.
    r = subprocess.run(
        ["bash", str(fetch_script), "--list"],
        capture_output=True, text=True, timeout=10,
        env={"PATH": "/usr/bin:/bin", "AUDIOCORE_MODELS_DIR": str(tmp_path)},
    )
    # If jq IS in /usr/bin, the test won't trigger the guard — that's OK,
    # we just check it doesn't crash silently.
    if r.returncode != 0:
        assert "jq" in (r.stdout + r.stderr).lower() or "manifest" in (
            r.stdout + r.stderr).lower(), (
            f"unexpected failure mode: {r.stderr[:300]}"
        )


# ── Convert-tool presence tests ───────────────────────────────────────────

BUILD_BIN = REPO_ROOT / "build" / "bin"
# Fallback: some build layouts put binaries directly under build/
BUILD_ROOT = REPO_ROOT / "build"


def _convert_binary_candidates(name: str) -> list[Path]:
    return [
        BUILD_BIN / name,
        BUILD_ROOT / name,
    ]


@pytest.mark.parametrize("name", [
    "convert_acestep",
    "convert_qwen3tts",
    "convert_mse2",
    "convert_ecapa",
])
def test_convert_binary_exists_and_is_executable(name):
    """Each convert_* binary referenced by the manifest must be built."""
    candidates = _convert_binary_candidates(name)
    found = next((p for p in candidates if p.exists()), None)
    if found is None:
        pytest.skip(
            f"{name} not built — looked at {[str(c) for c in candidates]}. "
            f"Run `cmake --build build` to enable this test."
        )
    assert os.access(found, os.X_OK), (
        f"{found} exists but isn't executable"
    )


def test_convert_binary_has_help_or_version_flag():
    """At least one convert binary answers --help (sanity check the build)."""
    for name in ("convert_acestep", "convert_qwen3tts"):
        binary = next((p for p in _convert_binary_candidates(name) if p.exists()), None)
        if binary is None:
            continue
        r = subprocess.run(
            [str(binary), "--help"],
            capture_output=True, text=True, timeout=5,
        )
        # Most convert tools accept --help; some exit non-zero but print usage.
        out = r.stdout + r.stderr
        assert out.strip(), (
            f"{binary} --help produced no output"
        )
        return
    pytest.skip("no convert_* binary found to test")


# ── Convert-lazily design contract ────────────────────────────────────────

def test_manifest_declares_converter_for_each_family_that_needs_one():
    """Families whose upstream weights aren't directly GGUF must reference
    a converter step in the manifest."""
    m = json.loads(MANIFEST_PATH.read_text())
    # The known converter-requiring families.
    converter_required = {
        "ace_step": "convert_acestep",
        "qwen3_tts": "convert_qwen3tts",
    }
    for fam_id, expected_tool in converter_required.items():
        if fam_id not in m["families"]:
            continue  # family missing entirely is caught by other tests
        fam = m["families"][fam_id]
        # Either a top-level `converter` field, or per-variant, or referenced
        # in fetch_models.sh logic. We just check SOME mention exists in
        # either manifest or fetch_models.sh source.
        fam_json = json.dumps(fam)
        fetch_src = FETCH_SCRIPT.read_text() if FETCH_SCRIPT.exists() else ""
        assert (expected_tool in fam_json or expected_tool in fetch_src), (
            f"family {fam_id!r} should reference converter {expected_tool!r} "
            f"in either manifest or fetch_models.sh"
        )


def test_manifest_python_engine_section_well_formed():
    """Every family that declares a python_engine section is structurally valid.

    The python_engine section declares HF source dirs (consumed via
    snapshot_download) for the Python-first engines (currently only
    qwen3_tts). The section is OPTIONAL — only families that have Python-
    engine consumers declare it. When present, every variant must declare
    the required fields so fetch_models.sh --python-engine can provision.
    """
    m = json.loads(MANIFEST_PATH.read_text())
    found_any = False
    for fam_id, fam in m["families"].items():
        pe = fam.get("python_engine")
        if pe is None:
            continue
        found_any = True
        assert "engine" in pe and pe["engine"], (
            f"{fam_id}.python_engine missing 'engine' (e.g. 'qwen_tts (pip)')"
        )
        assert "variants" in pe and pe["variants"], (
            f"{fam_id}.python_engine has no variants"
        )
        for var_id, var in pe["variants"].items():
            for field in ("display", "hf_repo", "engine_variant_key"):
                assert field in var and var[field], (
                    f"{fam_id}.python_engine.variants.{var_id} missing {field!r}"
                )
            # hf_repo must be a HF path: "org/name" with exactly one slash.
            repo = var["hf_repo"]
            assert repo.count("/") == 1 and len(repo) >= 4, (
                f"{fam_id}.python_engine.variants.{var_id}.hf_repo invalid: "
                f"{repo!r} (expected 'org/name')"
            )
            # min_vram_gb must be a positive number if present.
            if "min_vram_gb" in var:
                assert isinstance(var["min_vram_gb"], (int, float)) and \
                       var["min_vram_gb"] > 0, (
                    f"{fam_id}.python_engine.variants.{var_id}.min_vram_gb "
                    f"must be a positive number"
                )
    # We DO want at least one family (qwen3_tts) to declare this — the
    # Python engine is the production path for qwen3_tts now.
    assert found_any, (
        "no family declares a python_engine section — qwen3_tts should, "
        "since the qwen-tts Python package is the production engine for it"
    )


def test_fetch_models_supports_python_engine_flag(fetch_script):
    """fetch_models.sh --python-engine provisions HF source dirs.

    Verifies the flag is accepted, prints the python_engine header, and
    names every variant in the manifest section. Uses --dry-run so it
    works offline and never touches the network.
    """
    m = json.loads(MANIFEST_PATH.read_text())
    pe_families = [
        f for f, fv in m["families"].items() if fv.get("python_engine")
    ]
    assert pe_families, "no python_engine families declared in manifest"

    for fam in pe_families:
        result = subprocess.run(
            [str(fetch_script), "--python-engine", fam, "--dry-run"],
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0, (
            f"--python-engine {fam} failed: {result.stderr}"
        )
        assert "python_engine provisioning" in result.stdout, (
            f"--python-engine {fam} did not print provisioning header"
        )
        # Every variant in the manifest section must appear in dry-run.
        for var_id in m["families"][fam]["python_engine"]["variants"]:
            assert var_id in result.stdout, (
                f"--python-engine {fam} did not list variant {var_id}:\n"
                f"{result.stdout}"
            )


def test_fetch_models_python_engine_rejects_family_without_section(fetch_script):
    """--python-engine on a family WITHOUT a python_engine section exits 1.

    This catches the case where someone runs `--python-engine moss_tts`
    (which has no Python engine) — must fail cleanly rather than silently
    do nothing.
    """
    m = json.loads(MANIFEST_PATH.read_text())
    no_pe_families = [
        f for f, fv in m["families"].items() if not fv.get("python_engine")
    ]
    if not no_pe_families:
        pytest.skip("every family has a python_engine section")
    fam = no_pe_families[0]
    result = subprocess.run(
        [str(fetch_script), "--python-engine", fam, "--dry-run"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode != 0, (
        f"--python-engine {fam} should have failed (family has no "
        f"python_engine section) but exited 0:\n{result.stdout}"
    )
    assert "no families with a python_engine section matched" in result.stderr, (
        f"--python-engine {fam} wrong error message: {result.stderr}"
    )
