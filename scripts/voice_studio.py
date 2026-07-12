#!/usr/bin/env python3
"""══════════════════════════════════════════════════════════════════════════════
VOICE STUDIO — SOTA Control Suite for Qwen3-TTS
══════════════════════════════════════════════════════════════════════════════

DaVinci Resolve for voice. Full parameter matrix, multi-take audition,
script composition, voice baking, and profile management.

COMMANDS
  generate    Single generation with full parameter control
  audition    Multi-seed takes → save all, compare, pick the best
  compose     Multi-line script with per-line emotion/voice → one WAV
  bake        Generate emotional audio → clone as new voice (needs Base model)
  voices      Voice management (list / info / delete)
  profile     Parameter preset management (save / load / list)
  describe    Print the full parameter reference card
  status      Server health, available voices, models, exprs

EXAMPLES
  # Generate with every knob
  voice-studio generate --voice default \\
    --text "Welcome home." --instruct "warm, gentle" \\
    --icl-frames 10 --temperature 0.8 --seed 42 --rate 0.95

  # Audition 10 takes
  voice-studio audition --voice default \\
    --text "I missed you." --instruct "sad, longing" \\
    --seeds 42-51 -o audition/

  # Compose a scene
  voice-studio compose --script scene.json -o scene.wav

  # Bake a joyful voice
  voice-studio bake --source default_wdelta --save-as narrator_joy \\
    --text "I'm so happy!" --instruct "warm joy" --seed 42
══════════════════════════════════════════════════════════════════════════════"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
import urllib.error
import wave
import io
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_ENDPOINT = os.environ.get(
    "VOICE_STUDIO_ENDPOINT", "http://TAILSCALE_HOST:39517"
)
PROFILE_DIR = Path.home() / ".voice_studio_profiles"
CACHE_DIR = Path.home() / ".voice_studio_cache"

# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER SCHEMA — the full control matrix
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Param:
    flags: list[str]          # CLI flags (e.g. ["--voice-strength", "--vs"])
    json_name: str            # API field name
    ptype: type               # float, int, str, bool
    default: Any = None
    min_val: float | None = None
    max_val: float | None = None
    choices: list[str] | None = None
    help: str = ""
    node: str = ""            # grouping for describe command

PARAMS: list[Param] = [
    # ── VOICE IDENTITY NODE ──────────────────────────────────────────────
    Param(["--voice", "-v"], "voice", str, "default",
          help="Voice name (.qvoice filename without extension)", node="VOICE IDENTITY"),
    Param(["--voice-strength", "--vs"], "voice_strength", float, 1.0,
          min_val=0.0, max_val=3.0,
          help="WDELTA blend. 1.0=full identity, <1=more emotion room, >1=exaggerate. "
               "NO-OP on lite graft voices (only works on heavy WDELTA >100MB)",
          node="VOICE IDENTITY"),
    Param(["--icl-frames", "--icl"], "icl_frames", int, None,
          min_val=0, max_val=200,
          help="Cap ICL reference frames. Lower=less prosody anchor=more emotion freedom. "
               "0=all (default), 6-12=lighter anchor, good for freeing emotion",
          node="VOICE IDENTITY"),
    Param(["--graft"], "graft", bool, False,
          help="Drop .qvoice ref_codes, use x-vector only. More emotion, less identity fidelity",
          node="VOICE IDENTITY"),

    # ── EMOTION NODE ─────────────────────────────────────────────────────
    Param(["--emotion", "-e"], "emotion", str, None,
          choices=["sad","joy","anger","fear","disgust","surprise",
                   "contempt","awe","nostalgia","disapproval","remorse","outrage","despair"],
          help="STEER emotion preset. WARNING: trained on different speaker — can artifact. "
               "Prefer --instruct for cleaner emotion",
          node="EMOTION"),
    Param(["--instruct", "-i"], "instruct", str, None,
          help="Natural language style instruction. Softer than --emotion, fewer artifacts. "
               'e.g. "Speak with warm affection"',
          node="EMOTION"),
    Param(["--language", "-l"], "language", str, None,
          choices=["en","ja","ko","zh","de","fr","it"],
          help="Language hint", node="EMOTION"),

    # ── EXPRESSIVITY NODE ────────────────────────────────────────────────
    Param(["--expr"], "expr", str, None,
          help="Expressivity .expr file (italian_csp_topk6, german_csp_k6, french_csp_k6)",
          node="EXPRESSIVITY"),
    Param(["--expr-weight", "--ew"], "expr_weight", float, 1.0,
          min_val=0.0, max_val=3.0,
          help=".expr dose. 0.6=subtler, 1.0=as trained, 1.5=stronger", node="EXPRESSIVITY"),
    Param(["--roughness"], "roughness", float, None,
          min_val=0.0, max_val=1.0,
          help="Texture/roughness knob (q2-down blend on Code Predictor)", node="EXPRESSIVITY"),

    # ── SAMPLING NODE ────────────────────────────────────────────────────
    Param(["--temperature", "-T"], "temperature", float, None,
          min_val=0.1, max_val=2.0,
          help="Sampling temperature. Lower=more consistent. Default: 0.9", node="SAMPLING"),
    Param(["--top-p", "-p"], "top_p", float, None,
          min_val=0.1, max_val=1.0, help="Top-p nucleus sampling. Default: 1.0", node="SAMPLING"),
    Param(["--top-k", "-k"], "top_k", int, None,
          min_val=1, max_val=500, help="Top-k sampling. Default: 50", node="SAMPLING"),
    Param(["--rep-penalty"], "repetition_penalty", float, None,
          min_val=1.0, max_val=2.0, help="Repetition penalty. Default: 1.05", node="SAMPLING"),
    Param(["--seed", "-s"], "seed", int, None,
          help="Random seed. -1=random, >0=reproducible (same seed+params=same output)",
          node="SAMPLING"),

    # ── SHAPING NODE ─────────────────────────────────────────────────────
    Param(["--volume"], "volume", float, None,
          min_val=0.0, max_val=3.0,
          help="Output gain. 1.0=unchanged, 1.1=+10%% louder", node="SHAPING"),
    Param(["--rate"], "rate", float, None,
          min_val=0.25, max_val=4.0,
          help="Speaking rate (pitch-preserving). >1=faster, <1=slower. Default: 1.0",
          node="SHAPING"),
    Param(["--compose-pause"], "compose_pause", float, None,
          min_val=0.0, max_val=5.0,
          help="Gap between inline [emotion] spans in seconds. Default: 0.12",
          node="SHAPING"),

    # ── LIMITS NODE ──────────────────────────────────────────────────────
    Param(["--max-duration"], "max_duration", int, None,
          help="Max audio duration in seconds", node="LIMITS"),
    Param(["--max-tokens"], "max_tokens", int, None,
          help="Max generation tokens", node="LIMITS"),
]

PARAM_BY_FLAG: dict[str, Param] = {}
for p in PARAMS:
    for f in p.flags:
        PARAM_BY_FLAG[f] = p


# ═══════════════════════════════════════════════════════════════════════════════
# API CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

class VoiceEngine:
    """HTTP client for the audiocore TTS server."""

    def __init__(self, endpoint: str = DEFAULT_ENDPOINT):
        self.endpoint = endpoint.rstrip("/")

    def _get(self, path: str) -> dict:
        try:
            with urllib.request.urlopen(f"{self.endpoint}{path}", timeout=15) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:300]
            raise RuntimeError(f"HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Cannot reach server at {self.endpoint}: {e.reason}") from e

    def _post(self, path: str, body: dict, timeout: float = 600) -> tuple[bytes, dict]:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.endpoint}{path}", data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                wav = r.read()
                elapsed = time.time() - t0
                headers = dict(r.headers)
                status = r.status
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:500]
            raise RuntimeError(f"HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Cannot reach server at {self.endpoint}: {e.reason}") from e
        return wav, {"elapsed_s": round(elapsed, 1), "headers": headers,
                     "status": status}

    def _delete(self, path: str) -> dict:
        req = urllib.request.Request(f"{self.endpoint}{path}", method="DELETE")
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:300]
            raise RuntimeError(f"HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Cannot reach server at {self.endpoint}: {e.reason}") from e

    def health(self) -> dict:
        return self._get("/health")

    def voices(self) -> dict:
        return self._get("/v1/voices")

    def exprs(self) -> dict:
        return self._get("/v1/exprs")

    def generate(self, text: str, voice: str = "default", **params) -> tuple[bytes, dict]:
        body = {"input": text, "voice": voice}
        body.update(params)
        return self._post("/v1/audio/speech", body)

    def bake(self, source: str, save_as: str, text: str, **params) -> dict:
        body = {"source_voice": source, "save_as": save_as, "text": text}
        body.update(params)
        try:
            with urllib.request.urlopen(
                urllib.request.Request(
                    f"{self.endpoint}/v1/voices/bake",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                ), timeout=900,
            ) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            return {"error": f"HTTP {e.code}", "detail": e.read().decode()[:500]}

    def delete_voice(self, name: str):
        """Delete a voice via the server DELETE endpoint."""
        self._delete(f"/v1/voices/{name}")
        print(f"✓ Deleted voice: {name}")


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def wav_duration(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def wav_info(wav_bytes: bytes) -> dict:
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return {
            "duration_s": round(w.getnframes() / float(w.getframerate()), 2),
            "sample_rate": w.getframerate(),
            "channels": w.getnchannels(),
            "sample_width": w.getsampwidth(),
        }


def concatenate_wavs(wav_chunks: list[bytes], gaps_s: list[float] | None = None) -> bytes:
    """Concatenate WAV byte arrays into one continuous WAV."""
    if not wav_chunks:
        return b""
    if gaps_s is None:
        gaps_s = [0.0] * (len(wav_chunks) - 1)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        params_set = False
        for i, chunk in enumerate(wav_chunks):
            with wave.open(io.BytesIO(chunk), "rb") as w:
                if not params_set:
                    out.setnchannels(w.getnchannels())
                    out.setsampwidth(w.getsampwidth())
                    out.setframerate(w.getframerate())
                    params_set = True
                out.writeframes(w.readframes(w.getnframes()))
            # Insert silence gap between clips
            if i < len(gaps_s) and gaps_s[i] > 0:
                with wave.open(io.BytesIO(chunk), "rb") as w:
                    silence_bytes = int(gaps_s[i] * w.getframerate()) * w.getsampwidth() * w.getnchannels()
                    out.writeframes(b"\x00" * silence_bytes)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# CACHE
# ═══════════════════════════════════════════════════════════════════════════════

def cache_key(text: str, voice: str, params: dict, endpoint: str = "") -> str:
    blob = json.dumps({"t": text, "v": voice, "p": params, "e": endpoint}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def cache_get(key: str) -> bytes | None:
    path = CACHE_DIR / f"{key}.wav"
    if path.exists():
        return path.read_bytes()
    return None


def cache_put(key: str, wav: bytes):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.wav").write_bytes(wav)


# ═══════════════════════════════════════════════════════════════════════════════
# PROFILE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def profile_save(name: str, params: dict):
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    (PROFILE_DIR / f"{name}.json").write_text(json.dumps(params, indent=2))


def profile_load(name: str) -> dict:
    path = PROFILE_DIR / f"{name}.json"
    if not path.exists():
        print(f"✗ Profile '{name}' not found", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text())


def profile_list() -> list[str]:
    if not PROFILE_DIR.exists():
        return []
    return sorted(f.stem for f in PROFILE_DIR.glob("*.json"))


# ═══════════════════════════════════════════════════════════════════════════════
# CLI COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

def add_param_args(parser: argparse.ArgumentParser):
    """Add all generation parameters to an argparse parser."""
    for p in PARAMS:
        kwargs: dict[str, Any] = {"help": p.help, "default": None, "dest": p.json_name}
        if p.ptype is float:
            kwargs["type"] = float
        elif p.ptype is int:
            kwargs["type"] = int
        elif p.ptype is bool:
            kwargs["action"] = "store_true"
            kwargs.pop("default")
            kwargs["default"] = False
        parser.add_argument(*p.flags, **kwargs)


def collect_params(args: argparse.Namespace, exclude: set[str] | None = None) -> dict:
    """Extract non-None params from parsed args, applying defaults and validation."""
    exclude = exclude or set()
    result = {}
    for p in PARAMS:
        if p.json_name in exclude:
            continue
        val = getattr(args, p.json_name, None)
        if val is None or val is False:
            continue
        if val is True:
            result[p.json_name] = True
            continue
        # Clamp to range
        if p.min_val is not None and isinstance(val, (int, float)):
            val = max(p.min_val, val)
        if p.max_val is not None and isinstance(val, (int, float)):
            val = min(p.max_val, val)
        result[p.json_name] = val
    return result


def fmt_duration(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    return f"{int(s//60)}m{int(s%60)}s"


# ── GENERATE ────────────────────────────────────────────────────────────────

def cmd_generate(args):
    engine = VoiceEngine(args.endpoint)
    params = collect_params(args, exclude={"voice"})
    voice = args.voice or "default"
    text = args.text

    # Profile merge
    if args.profile:
        prof = profile_load(args.profile)
        prof.update(params)
        params = prof

    # Cache
    key = cache_key(text, voice, params, args.endpoint)
    if not args.no_cache:
        cached = cache_get(key)
        if cached:
            info = wav_info(cached)
            print(f"✓ CACHE HIT — {info['duration_s']}s "
                  f"(skipped generation, --no-cache to disable)")
            Path(args.output).write_bytes(cached)
            print(f"  → {args.output}")
            return

    print(f"▶ Generating: voice={voice} seed={params.get('seed', 'random')}")
    print(f"  params: {json.dumps(params, indent=2)}")
    print(f"  endpoint: {args.endpoint}")

    try:
        wav, meta = engine.generate(text, voice, **params)
    except Exception as e:
        print(f"✗ Generation failed: {e}", file=sys.stderr)
        sys.exit(1)

    info = wav_info(wav)
    print(f"✓ {info['duration_s']}s audio in {fmt_duration(meta['elapsed_s'])} "
          f"({len(wav)} bytes)")
    print(f"  voice_type: {meta['headers'].get('X-Voice-Type', '?')}")

    Path(args.output).write_bytes(wav)
    cache_put(key, wav)

    # Write sidecar metadata
    meta_path = Path(args.output).with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "text": text, "voice": voice, "params": params,
        "wav_info": info, "generation_time_s": meta["elapsed_s"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    print(f"  → {args.output}")
    print(f"  → {meta_path}")


# ── AUDITION ────────────────────────────────────────────────────────────────

def cmd_audition(args):
    engine = VoiceEngine(args.endpoint)
    params = collect_params(args, exclude={"voice", "seed"})
    voice = args.voice or "default"
    text = args.text

    # Parse seed range
    seeds = parse_seed_range(args.seeds)

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "text": text, "voice": voice, "base_params": params,
        "seeds": seeds, "takes": [],
    }

    print(f"▶ AUDITION: {len(seeds)} takes of voice={voice}")
    print(f"  text: \"{text[:80]}{'...' if len(text)>80 else ''}\"")
    print(f"  params: {json.dumps(params)}")
    print(f"  output: {outdir}/")
    print()

    for i, seed in enumerate(seeds):
        seed_params = dict(params)
        seed_params["seed"] = seed

        wav_path = outdir / f"seed_{seed:04d}.wav"
        key = cache_key(text, voice, seed_params, args.endpoint)

        # Cache check
        if not args.no_cache:
            cached = cache_get(key)
            if cached:
                info = wav_info(cached)
                wav_path.write_bytes(cached)
                take = {"seed": seed, "path": str(wav_path),
                        "duration_s": info["duration_s"],
                        "generation_time_s": 0.0, "size_bytes": len(cached),
                        "cached": True}
                manifest["takes"].append(take)
                print(f"  [{i+1}/{len(seeds)}] seed={seed} CACHE HIT "
                      f"{info['duration_s']}s → {wav_path.name}")
                continue

        print(f"  [{i+1}/{len(seeds)}] seed={seed}...", end=" ", flush=True)
        t0 = time.time()

        try:
            wav, meta = engine.generate(text, voice, **seed_params)
            elapsed = time.time() - t0
            info = wav_info(wav)
            wav_path.write_bytes(wav)
            cache_put(key, wav)

            take = {"seed": seed, "path": str(wav_path), "duration_s": info["duration_s"],
                    "generation_time_s": round(elapsed, 1), "size_bytes": len(wav)}
            manifest["takes"].append(take)

            print(f"{info['duration_s']}s in {fmt_duration(elapsed)} → {wav_path.name}")
        except Exception as e:
            print(f"FAILED: {e}")
            manifest["takes"].append({"seed": seed, "error": str(e)})

    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Summary
    succeeded = [t for t in manifest["takes"] if "error" not in t]
    cached_n = sum(1 for t in succeeded if t.get("cached"))
    total_time = sum(t.get("generation_time_s", 0) for t in succeeded)
    gen_n = len(succeeded) - cached_n
    print(f"\n✓ {len(succeeded)}/{len(seeds)} takes "
          f"({gen_n} generated, {cached_n} cached) in {fmt_duration(total_time)}")
    print(f"  manifest: {manifest_path}")
    print(f"\n  Listen and pick the best:")
    print(f"  for f in {outdir}/seed_*.wav; do ffplay \"$f\"; done")
    print(f"\n  Then bake the winner:")
    best = succeeded[0]["seed"] if succeeded else "SEED"
    print(f"  voice-studio bake --source {voice} --save-as {voice}_best "
          f"--text '{text[:40]}...' --seed {best}")


def parse_seed_range(spec: str) -> list[int]:
    """Parse '42-51' or '42,43,45' or '42' into a list of seeds."""
    if "-" in spec and "," not in spec:
        parts = spec.split("-")
        return list(range(int(parts[0]), int(parts[-1]) + 1))
    return [int(x.strip()) for x in spec.split(",")]


# ── COMPOSE ─────────────────────────────────────────────────────────────────

def cmd_compose(args):
    engine = VoiceEngine(args.endpoint)
    script_path = Path(args.script)
    if not script_path.exists():
        print(f"✗ Script not found: {script_path}", file=sys.stderr)
        sys.exit(1)

    script = json.loads(script_path.read_text())
    lines = script.get("lines", script if isinstance(script, list) else [])

    print(f"▶ COMPOSE: {len(lines)} lines")
    chunks = []
    gaps = []          # gaps[i] = silence between chunks[i] and chunks[i+1]
    line_metas = []
    pending_pause = None

    for i, line in enumerate(lines):
        if "pause" in line:
            pending_pause = float(line["pause"])
            print(f"  [{i+1}] pause {line['pause']}s")
            continue

        text = line["input"] or line.get("text", "")
        voice = line.get("voice", script.get("default_voice", "default"))
        params = {k: v for k, v in line.items()
                  if k not in ("input", "text", "voice", "pause") and v is not None}

        print(f"  [{i+1}] voice={voice} \"{text[:60]}{'...' if len(text)>60 else ''}\"")

        try:
            wav, meta = engine.generate(text, voice, **params)
            info = wav_info(wav)
            # Gap before this chunk (not for the first)
            if chunks:
                gap = pending_pause if pending_pause is not None else script.get("default_pause", 0.3)
                gaps.append(gap)
            chunks.append(wav)
            line_metas.append({"text": text, "voice": voice, "params": params,
                               "duration_s": info["duration_s"],
                               "gen_time_s": meta["elapsed_s"]})
            pending_pause = None
            print(f"       → {info['duration_s']}s in {fmt_duration(meta['elapsed_s'])}")
        except Exception as e:
            print(f"       ✗ FAILED: {e}", file=sys.stderr)
            sys.exit(1)

    # Concatenate — gaps has exactly len(chunks)-1 entries
    output = concatenate_wavs(chunks, gaps)
    Path(args.output).write_bytes(output)

    total_dur = sum(m["duration_s"] for m in line_metas)
    total_gap = sum(gaps)
    total_gen = sum(m["gen_time_s"] for m in line_metas)

    print(f"\n✓ Composed: {len(chunks)} clips, {total_dur:.1f}s audio "
          f"({total_gap:.1f}s pauses), {fmt_duration(total_gen)} total gen time")
    print(f"  → {args.output}")

    # Sidecar
    meta_path = Path(args.output).with_suffix(".meta.json")
    meta_path.write_text(json.dumps({
        "lines": line_metas, "gaps": gaps,
        "total_duration_s": total_dur + total_gap,
        "total_generation_s": round(total_gen, 1),
    }, indent=2))


# ── BAKE ────────────────────────────────────────────────────────────────────

def cmd_bake(args):
    engine = VoiceEngine(args.endpoint)
    params = collect_params(args, exclude={"voice"})

    print(f"▶ BAKE: source={args.source} → save_as={args.save_as}")
    print(f"  text: \"{args.text[:80]}\"")
    print(f"  params: {json.dumps(params)}")

    result = engine.bake(args.source, args.save_as, args.text, **params)

    if "error" in result:
        print(f"\n✗ Bake failed: {result['error']}", file=sys.stderr)
        if "detail" in result:
            print(f"  {result['detail'][:300]}", file=sys.stderr)
        if "not available" in result.get("detail", "").lower():
            print("\n  The bake pipeline requires the Base model on the server.", file=sys.stderr)
            print("  Deploy it with:", file=sys.stderr)
            print(f"    rsync -a --partial weights/qwen3_tts/qwen3-tts-1.7b-base/ "
                  f"ubuntu@TAILSCALE_HOST:/opt/audiocore/weights/qwen3_tts/qwen3-tts-1.7b-base/",
                  file=sys.stderr)
        sys.exit(1)

    print(f"\n✓ Voice baked: {result.get('voice_name', args.save_as)}")
    print(f"  size: {result.get('size_mb', '?')} MB")
    print(f"  type: {result.get('type', '?')}")
    print(f"\n  Generate with it:")
    print(f"  voice-studio generate --voice {args.save_as} --text '...' --seed 42")


# ── VOICES ──────────────────────────────────────────────────────────────────

def cmd_voices(args):
    engine = VoiceEngine(args.endpoint)

    if args.subcmd == "list" or args.subcmd is None:
        data = engine.voices()
        voices = data.get("voices", [])
        if not voices:
            print("(no voices found)")
            return
        print(f"{'NAME':<30} {'TYPE':<8} {'SIZE':>10}")
        print("─" * 52)
        for v in voices:
            size = v.get("size_mb", v.get("size", 0) / 1048576)
            print(f"{v['name']:<30} {v.get('type', '?'):<8} {size:>8.1f}MB")
        print(f"\n{len(voices)} voice(s)")

    elif args.subcmd == "info":
        data = engine.voices()
        match = [v for v in data.get("voices", []) if v["name"] == args.name]
        if not match:
            print(f"✗ Voice '{args.name}' not found", file=sys.stderr)
            sys.exit(1)
        v = match[0]
        print(json.dumps(v, indent=2))

    elif args.subcmd == "delete":
        engine.delete_voice(args.name)


# ── PROFILE ─────────────────────────────────────────────────────────────────

def cmd_profile(args):
    if args.subcmd == "save":
        params = collect_params(args, exclude={"voice"})
        if args.voice:
            params["voice"] = args.voice
        profile_save(args.name, params)
        print(f"✓ Profile saved: {args.name}")
        print(f"  {json.dumps(params, indent=2)}")

    elif args.subcmd == "load":
        prof = profile_load(args.name)
        print(json.dumps(prof, indent=2))
        print(f"\n  Use with: voice-studio generate --profile {args.name} --text '...'")

    elif args.subcmd == "list" or args.subcmd is None:
        names = profile_list()
        if not names:
            print("(no profiles saved)")
            return
        for n in names:
            prof = json.loads((PROFILE_DIR / f"{n}.json").read_text())
            keys = ", ".join(f"{k}={v}" for k, v in prof.items() if k != "voice")
            voice = prof.get("voice", "?")
            print(f"  {n:<25} voice={voice:<15} {keys}")


# ── DESCRIBE ────────────────────────────────────────────────────────────────

def cmd_describe(args):
    current_node = None
    for p in PARAMS:
        if p.node != current_node:
            current_node = p.node
            print(f"\n  ╔══ {current_node} {'═' * (50 - len(current_node))}╗")
            print(f"  ║{'':>72}║")

        flags = "/".join(p.flags)
        default = f" [default: {p.default}]" if p.default is not None else ""
        rng = ""
        if p.min_val is not None and p.max_val is not None:
            rng = f" [{p.min_val}-{p.max_val}]"
        elif p.choices:
            rng = f" {{{'|'.join(p.choices[:5])}{'...' if len(p.choices)>5 else ''}}}"

        print(f"  ║  {flags:<28} {p.help}")
        if default or rng:
            print(f"  ║{'':>12}{default}{rng}")
        print(f"  ║{'':>72}║")

    print(f"  ╚{'═'*74}╝")
    print(f"\n  INLINE EMOTION TAGS (in --text):")
    print(f"    [happy] [sad] [excited] [annoyed] [neutral] ...")
    print(f"    [sigh] [huff] [pause:400ms] [pause:1s]")
    print(f"    Example: \"Hi! [happy] Welcome! [pause:300ms] [sad] I missed you.\"")


# ── STATUS ──────────────────────────────────────────────────────────────────

def cmd_status(args):
    engine = VoiceEngine(args.endpoint)
    try:
        health = engine.health()
        voices = engine.voices()
        exprs = engine.exprs()
    except Exception as e:
        print(f"✗ Cannot reach server at {args.endpoint}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"╔══ VOICE STUDIO STATUS ══════════════════════════════════════════════╗")
    print(f"║  Endpoint:  {args.endpoint:<58}║")
    print(f"║  Service:   {health.get('service', '?'):<58}║")
    print(f"║  Version:   {health.get('version', '?'):<58}║")
    print(f"╠══════════════════════════════════════════════════════════════════════╣")
    vlist = voices.get("voices", [])
    print(f"║  Voices:    {len(vlist):<58}║")
    for v in vlist:
        tag = f"  • {v['name']} ({v.get('type','?')}, {v.get('size_mb',0):.0f}MB)"
        print(f"║{'':>2}{tag:<72}║")
    print(f"╠══════════════════════════════════════════════════════════════════════╣")
    elist = exprs.get("exprs", [])
    print(f"║  Exprs:     {', '.join(elist):<58}║")
    print(f"╚══════════════════════════════════════════════════════════════════════╝")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        prog="voice-studio",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                    help=f"Server URL (default: {DEFAULT_ENDPOINT})")
    sub = ap.add_subparsers(dest="command")

    # ── generate ───────────────────────────────────────────────────────
    gen = sub.add_parser("generate", help="Single generation with full parameter control")
    gen.add_argument("--text", "-t", required=True)
    gen.add_argument("--output", "-o", default="output.wav")
    gen.add_argument("--profile", help="Load saved parameter profile")
    gen.add_argument("--no-cache", action="store_true", help="Bypass local cache")
    add_param_args(gen)

    # ── audition ───────────────────────────────────────────────────────
    aud = sub.add_parser("audition", help="Multi-seed takes → save all, compare, pick best")
    aud.add_argument("--text", "-t", required=True)
    aud.add_argument("--seeds", default="42-51", help="Seed range: '42-51' or '42,43,45'")
    aud.add_argument("--output", "-o", default="audition")
    aud.add_argument("--no-cache", action="store_true", help="Bypass local cache")
    add_param_args(aud)

    # ── compose ────────────────────────────────────────────────────────
    comp = sub.add_parser("compose", help="Multi-line script with per-line emotion → one WAV")
    comp.add_argument("--script", "-s", required=True, help="JSON script file")
    comp.add_argument("--output", "-o", default="composition.wav")

    # ── bake ───────────────────────────────────────────────────────────
    bake_p = sub.add_parser("bake", help="Generate emotional audio → clone as new voice")
    bake_p.add_argument("--source", required=True, help="Source voice name")
    bake_p.add_argument("--save-as", required=True, help="New voice name")
    bake_p.add_argument("--text", "-t", required=True)
    add_param_args(bake_p)

    # ── voices ─────────────────────────────────────────────────────────
    voc = sub.add_parser("voices", help="Voice management")
    voc.add_argument("subcmd", nargs="?", choices=["list", "info", "delete"])
    voc.add_argument("name", nargs="?")

    # ── profile ────────────────────────────────────────────────────────
    prof = sub.add_parser("profile", help="Parameter preset management")
    prof.add_argument("subcmd", choices=["save", "load", "list"])
    prof.add_argument("name", nargs="?")
    prof.add_argument("--text", "-t")
    add_param_args(prof)

    # ── describe ───────────────────────────────────────────────────────
    sub.add_parser("describe", help="Print full parameter reference card")

    # ── status ─────────────────────────────────────────────────────────
    sub.add_parser("status", help="Server health and available resources")

    args = ap.parse_args()

    if not args.command:
        ap.print_help()
        sys.exit(0)

    try:
        {
            "generate": cmd_generate,
            "audition": cmd_audition,
            "compose": cmd_compose,
            "bake": cmd_bake,
            "voices": cmd_voices,
            "profile": cmd_profile,
            "describe": cmd_describe,
            "status": cmd_status,
        }[args.command](args)
    except RuntimeError as e:
        print(f"✗ {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
