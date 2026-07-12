#!/usr/bin/env python3
"""Audiocore TTS HTTP server — v3 (SOTA control suite engine).

Full-parameter generation + voice creation/baking pipeline + voice management.

Endpoints:
  GET  /health                → server status
  GET  /v1/voices             → list voices with metadata
  GET  /v1/voices/<name>      → single voice info
  GET  /v1/exprs              → list .expr expressivity files
  GET  /v1/models             → list available models and their status
  POST /v1/audio/speech       → OpenAI-compatible TTS generation
  POST /v1/voices/bake        → generate emotional audio → clone as new voice
  POST /v1/voices/clone       → clone voice from a server-local WAV file
  DELETE /v1/voices/<name>    → delete a voice
"""
from __future__ import annotations
import glob, json, os, pathlib, subprocess, sys, tempfile

from http.server import HTTPServer, BaseHTTPRequestHandler

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

BASE       = pathlib.Path("/opt/audiocore")
QWEN_TTS   = str(BASE / "ref-qwen3tts-src" / "qwen_tts")
VOICES_DIR = str(BASE / "voices")
EXPR_DIR   = str(BASE / "ref-qwen3tts-src" / "presets" / "expr")
CLIPS_DIR  = str(BASE / "clips")
PORT       = 39517
TIMEOUT    = 900  # 15 min per generation (bake = 2 generations)

HEAVY_THRESHOLD = 100 * 1024 * 1024  # voices > 100MB are heavy WDELTA

# CUDA backend: only enabled if the binary was compiled with CUDA support.
# Override with env var TTS_USE_CUDA=1 (or =0 to force off).
USE_CUDA = os.environ.get("TTS_USE_CUDA", "0") == "1"

# Model directories — qwen_tts -d <path>
MODEL_DIRS = {
    "customvoice": str(BASE / "weights/qwen3_tts/qwen3-tts-1.7b-customvoice/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
    "base":        str(BASE / "weights/qwen3_tts/qwen3-tts-1.7b-base/Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
    "voicedesign": str(BASE / "weights/qwen3_tts/qwen3-tts-1.7b-voicedesign/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
}


def model_available(name: str) -> bool:
    """Check if a model directory exists and has model files."""
    path = MODEL_DIRS.get(name, "")
    return bool(path and os.path.exists(os.path.join(path, "config.json")))


# ═══════════════════════════════════════════════════════════════════════════════
# HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write(f"[tts] {self.client_address[0]} {fmt % args}\n")

    def _json(self, code: int, obj: dict):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self) -> dict:
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        return json.loads(raw)

    def _voice_info(self, qvoice_path: str) -> dict:
        name = os.path.basename(qvoice_path).replace(".qvoice", "")
        size = os.path.getsize(qvoice_path)
        return {
            "name": name,
            "size": size,
            "size_mb": round(size / 1048576, 1),
            "heavy": size > HEAVY_THRESHOLD,
            "type": "WDELTA" if size > HEAVY_THRESHOLD else "graft",
        }

    # ── GET ─────────────────────────────────────────────────────────────
    def do_GET(self):
        if self.path == "/health":
            models = {k: model_available(k) for k in MODEL_DIRS}
            self._json(200, {
                "status": "ok", "service": "audiocore-tts", "version": "v3",
                "models": models,
            })
            return

        if self.path == "/v1/voices":
            voices = [self._voice_info(f) for f in
                      sorted(glob.glob(os.path.join(VOICES_DIR, "*.qvoice")))]
            self._json(200, {"voices": voices})
            return

        if self.path.startswith("/v1/voices/"):
            name = self.path[len("/v1/voices/"):]
            qvoice = os.path.join(VOICES_DIR, f"{name}.qvoice")
            if not os.path.exists(qvoice):
                self._json(404, {"error": f"voice '{name}' not found"})
                return
            self._json(200, self._voice_info(qvoice))
            return

        if self.path == "/v1/exprs":
            exprs = [os.path.basename(f).replace(".expr", "")
                     for f in sorted(glob.glob(os.path.join(EXPR_DIR, "*.expr")))]
            self._json(200, {"exprs": exprs})
            return

        if self.path == "/v1/models":
            models = {}
            for name, path in MODEL_DIRS.items():
                models[name] = {
                    "path": path,
                    "available": model_available(name),
                    "has_config": os.path.exists(os.path.join(path, "config.json")),
                }
            self._json(200, {"models": models})
            return

        self.send_error(404)

    # ── DELETE ──────────────────────────────────────────────────────────
    def do_DELETE(self):
        if self.path.startswith("/v1/voices/"):
            name = self.path[len("/v1/voices/"):]
            qvoice = os.path.join(VOICES_DIR, f"{name}.qvoice")
            if not os.path.exists(qvoice):
                self._json(404, {"error": f"voice '{name}' not found"})
                return
            os.unlink(qvoice)
            self.log_message("deleted voice: %s", name)
            self._json(200, {"deleted": name})
            return
        self.send_error(404)

    # ── POST ────────────────────────────────────────────────────────────
    def do_POST(self):
        if self.path == "/v1/audio/speech":
            return self._handle_generate()
        if self.path == "/v1/voices/bake":
            return self._handle_bake()
        if self.path == "/v1/voices/clone":
            return self._handle_clone()
        self.send_error(404)

    # ── POST /v1/audio/speech ───────────────────────────────────────────
    def _handle_generate(self):
        try:
            body = self._read_body()
        except Exception as e:
            self._json(400, {"error": f"invalid json: {e}"})
            return

        text = body.get("input") or body.get("text", "")
        if not text:
            self._json(400, {"error": "missing 'input' field"})
            return

        voice_name = body.get("voice") or body.get("speaker", "default")
        qvoice = os.path.join(VOICES_DIR, f"{voice_name}.qvoice")
        if not os.path.exists(qvoice):
            self._json(404, {"error": f"voice '{voice_name}' not found",
                            "available": [os.path.basename(f).replace(".qvoice", "")
                                          for f in glob.glob(os.path.join(VOICES_DIR, "*.qvoice"))]})
            return

        is_heavy = os.path.getsize(qvoice) > HEAVY_THRESHOLD
        model_dir = MODEL_DIRS["customvoice"]

        argv = self._build_gen_argv(model_dir, qvoice, is_heavy, text, body)
        wav_bytes = self._run_tts(argv, body.get("debug", False))
        if wav_bytes is None:
            return  # error already sent

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav_bytes)))
        self.send_header("X-Voice", voice_name)
        self.send_header("X-Voice-Type", "WDELTA" if is_heavy else "graft")
        self.end_headers()
        self.wfile.write(wav_bytes)
        self.log_message("→ %d bytes voice=%s", len(wav_bytes), voice_name)

    # ── POST /v1/voices/bake ────────────────────────────────────────────
    def _handle_bake(self):
        """The bake pipeline: generate emotional audio → clone as new voice."""
        try:
            body = self._read_body()
        except Exception as e:
            self._json(400, {"error": f"invalid json: {e}"})
            return

        source = body.get("source_voice") or body.get("source", "default")
        save_as = body.get("save_as", "")
        text = body.get("text") or body.get("input", "")
        wdelta = body.get("wdelta", False)  # Heavy WDELTA voice (~3.6GB)

        if not save_as or not text:
            self._json(400, {"error": "missing 'save_as' or 'text'"})
            return

        source_qvoice = os.path.join(VOICES_DIR, f"{source}.qvoice")
        if not os.path.exists(source_qvoice):
            self._json(404, {"error": f"source voice '{source}' not found"})
            return

        if not model_available("base"):
            self._json(503, {"error": "Base model not available on server",
                            "detail": "The bake pipeline requires the Base model. "
                                      "Deploy it first: rsync weights/qwen3_tts/qwen3-tts-1.7b-base/ "
                                      "to the remote."})
            return

        # Check name collision
        out_qvoice = os.path.join(VOICES_DIR, f"{save_as}.qvoice")
        if os.path.exists(out_qvoice):
            # Auto-number
            n = 2
            while os.path.exists(os.path.join(VOICES_DIR, f"{save_as}_{n}.qvoice")):
                n += 1
            save_as = f"{save_as}_{n}"
            out_qvoice = os.path.join(VOICES_DIR, f"{save_as}.qvoice")

        is_heavy_source = os.path.getsize(source_qvoice) > HEAVY_THRESHOLD

        # ── Step 1: Generate emotional audio with CustomVoice ──────────
        self.log_message("BAKE step 1: generating with %s → emotional audio", source)
        gen_argv = self._build_gen_argv(
            MODEL_DIRS["customvoice"], source_qvoice, is_heavy_source, text, body)

        # Use a temp file for the intermediate audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=CLIPS_DIR) as tf:
            bake_wav = tf.name
        gen_argv += ["-o", bake_wav]

        proc = subprocess.run(gen_argv, capture_output=True, text=True, timeout=TIMEOUT)
        if proc.returncode != 0 or not os.path.exists(bake_wav):
            self._json(500, {"error": "Step 1 (generation) failed",
                            "detail": (proc.stderr or "")[-800:]})
            if os.path.exists(bake_wav): os.unlink(bake_wav)
            return

        self.log_message("BAKE step 1 done: %s", bake_wav)

        # ── Step 2: Clone with Base model ──────────────────────────────
        # Default: ~24MB graft voice (x-vector + metadata + saved source weights).
        # With wdelta=True: ~3.6GB WDELTA voice (full weight delta, exact identity).
        self.log_message("BAKE step 2: cloning with Base model → %s.qvoice%s",
                         save_as, " (WDELTA)" if wdelta else "")
        clone_argv = [
            QWEN_TTS, "-d", MODEL_DIRS["base"],
            "--ref-audio", bake_wav,
            "--save-voice", out_qvoice,
        ]
        if wdelta:
            clone_argv.append("--target-cv")

        proc = subprocess.run(clone_argv, capture_output=True, text=True, timeout=TIMEOUT)
        if proc.returncode != 0 or not os.path.exists(out_qvoice):
            self._json(500, {"error": "Step 2 (cloning) failed",
                            "detail": (proc.stderr or "")[-800:]})
            if os.path.exists(bake_wav): os.unlink(bake_wav)
            if os.path.exists(out_qvoice): os.unlink(out_qvoice)
            return

        # Cleanup intermediate
        os.unlink(bake_wav)

        info = self._voice_info(out_qvoice)
        self.log_message("BAKE complete: %s (%.1fMB %s)",
                         save_as, info["size_mb"], info["type"])
        self._json(200, {
            "status": "baked",
            "voice_name": save_as,
            "source": source,
            "size_mb": info["size_mb"],
            "type": info["type"],
            "seed": body.get("seed"),
            "text": text[:200],
        })

    # ── POST /v1/voices/clone ───────────────────────────────────────────
    def _handle_clone(self):
        """Clone a voice from a server-local WAV file."""
        try:
            body = self._read_body()
        except Exception as e:
            self._json(400, {"error": f"invalid json: {e}"})
            return

        ref_audio = body.get("ref_audio", "")
        save_as = body.get("save_as", "")
        target_cv = body.get("target_cv", True)

        if not ref_audio or not save_as:
            self._json(400, {"error": "missing 'ref_audio' or 'save_as'"})
            return
        if not os.path.exists(ref_audio):
            self._json(404, {"error": f"ref_audio not found: {ref_audio}"})
            return
        if not model_available("base"):
            self._json(503, {"error": "Base model not available"})
            return

        out_qvoice = os.path.join(VOICES_DIR, f"{save_as}.qvoice")
        argv = [QWEN_TTS, "-d", MODEL_DIRS["base"],
                "--ref-audio", ref_audio, "--save-voice", out_qvoice]
        if target_cv:
            argv.append("--target-cv")

        proc = subprocess.run(argv, capture_output=True, text=True, timeout=TIMEOUT)
        if proc.returncode != 0 or not os.path.exists(out_qvoice):
            self._json(500, {"error": "Clone failed",
                            "detail": (proc.stderr or "")[-800:]})
            return

        info = self._voice_info(out_qvoice)
        self._json(200, {"status": "cloned", **info})

    # ── CLI argv builder ────────────────────────────────────────────────
    def _build_gen_argv(self, model_dir: str, qvoice: str, is_heavy: bool,
                        text: str, body: dict) -> list[str]:
        """Build the qwen_tts CLI argv from request body parameters."""
        argv = [QWEN_TTS, "-d", model_dir, "--load-voice", qvoice]
        if not is_heavy:
            argv.append("--icl-only")
        argv += ["--text", text]

        # Sampling
        for flag, key, cast in [
            ("-T", "temperature", float), ("-p", "top_p", float),
            ("-k", "top_k", int), ("--rep-penalty", "repetition_penalty", float)]:
            v = body.get(key)
            if v is not None:
                argv += [flag, str(cast(v))]

        seed = body.get("seed")
        if seed is not None and int(seed) > 0:
            argv += ["--seed", str(int(seed))]

        # Identity/Expression
        strength = body.get("voice_strength")
        if strength is not None and abs(float(strength) - 1.0) > 0.001:
            argv += ["--voice-strength", str(strength)]
        icl = body.get("icl_frames")
        if icl is not None:
            argv += ["--icl-frames", str(int(icl))]
        if body.get("graft"):
            argv.append("--graft")

        # Emotion
        if body.get("emotion"):
            argv += ["--emotion", str(body["emotion"])]
        if body.get("instruct"):
            argv += ["--instruct", str(body["instruct"])]
        if body.get("language"):
            argv += ["-l", str(body["language"])]

        # Expressivity
        expr = body.get("expr") or body.get("expr_file")
        if expr:
            resolved = self._resolve_expr(expr)
            if resolved:
                argv += ["--expr", resolved]
                ew = body.get("expr_weight")
                if ew is not None:
                    argv += ["--expr-weight", str(ew)]

        # Audio shaping
        for flag, key in [("--volume", "volume"), ("--rate", "rate"),
                          ("--roughness", "roughness"),
                          ("--compose-pause", "compose_pause")]:
            v = body.get(key)
            if v is not None:
                argv += [flag, str(v)]

        # Limits
        if body.get("max_duration") is not None:
            argv += ["--max-duration", str(int(body["max_duration"]))]
        if body.get("max_tokens") is not None:
            argv += ["-m", str(int(body["max_tokens"]))]

        # GPU backend (only if binary compiled with CUDA)
        if USE_CUDA:
            argv += ["--backend", "cuda"]

        if body.get("debug"):
            argv.append("-D")

        return argv

    def _run_tts(self, argv: list[str], debug: bool = False) -> bytes | None:
        """Run qwen_tts and return WAV bytes, or None on error."""
        # Find the output path from argv
        out_wav = None
        if "-o" in argv:
            out_wav = argv[argv.index("-o") + 1]
        else:
            tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="/tmp")
            tf.close()
            out_wav = tf.name
            argv += ["-o", out_wav]

        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            self._json(504, {"error": "TTS generation timed out", "timeout_s": TIMEOUT})
            return None

        if proc.returncode != 0 or not os.path.exists(out_wav) or os.path.getsize(out_wav) == 0:
            if os.path.exists(out_wav): os.unlink(out_wav)
            self._json(500, {"error": "TTS generation failed",
                            "exit": proc.returncode,
                            "detail": (proc.stderr or "")[-800:]})
            return None

        with open(out_wav, "rb") as f:
            wav = f.read()
        os.unlink(out_wav)
        return wav

    def _resolve_expr(self, name: str) -> str | None:
        if os.path.isabs(name) and os.path.exists(name):
            return name
        for c in [os.path.join(EXPR_DIR, f"{name}.expr"),
                  os.path.join(EXPR_DIR, name), name]:
            if os.path.exists(c):
                return c
        return None


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs(CLIPS_DIR, exist_ok=True)
    print(f"[tts] v3 — SOTA control suite engine")
    print(f"[tts] port:   {PORT}")
    print(f"[tts] voices: {VOICES_DIR}")
    print(f"[tts] models: {json.dumps({k: model_available(k) for k in MODEL_DIRS})}")
    print(f"[tts] cuda:   {'ON' if USE_CUDA else 'OFF (set TTS_USE_CUDA=1 to enable)'}")
    print(f"[tts] binary: {QWEN_TTS}")
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    server.serve_forever()
