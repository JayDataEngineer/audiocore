#!/usr/bin/env bash
# scripts/run_webapp.sh — one-shot launcher for the audiocore Voice Studio.
#
# Builds the server (if missing or stale), auto-generates a server.json from
# the weights/ tree, picks a free port, and starts listening. Prints the URL
# to open. Designed for local dev; for production use the Dockerfile or your
# own server.json (see examples/server.json).
#
# Usage:
#   scripts/run_webapp.sh                 # auto: build + config + run
#   scripts/run_webapp.sh --no-build      # skip the build step
#   scripts/run_webapp.sh --port 8000     # pick the port
#   scripts/run_webapp.sh --config FILE   # use an explicit server.json
#   scripts/run_webapp.sh --weights DIR   # override weights/ location
#
# Environment:
#   AUDIOCORE_PORT          same as --port
#   AUDIOCORE_WEIGHTS_DIR   same as --weights (default: ./weights)
#   AUDIOCORE_CLIPS_DIR     clips storage (default: ./webapp/clips)
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
BIN="$ROOT/build/audiocore_server"
DEFAULT_WEIGHTS="$ROOT/weights"
CLIPS_DIR="${AUDIOCORE_CLIPS_DIR:-$ROOT/webapp/clips}"
DO_BUILD=1
PORT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-build) DO_BUILD=0; shift ;;
        --port)     PORT="$2"; shift 2 ;;
        --config)   CONFIG="$2"; shift 2 ;;
        --weights)  DEFAULT_WEIGHTS="$2"; shift 2 ;;
        -h|--help)
            cat <<'HELP'
scripts/run_webapp.sh — one-shot launcher for the audiocore Voice Studio.

Builds the server (if missing or stale), auto-generates a server.json from
the weights/ tree, picks a free port, and starts listening. Prints the URL
to open. Designed for local dev; for production use the Dockerfile or your
own server.json (see examples/server.json).

Usage:
  scripts/run_webapp.sh                 # auto: build + config + run
  scripts/run_webapp.sh --no-build      # skip the build step
  scripts/run_webapp.sh --port 8000     # pick the port
  scripts/run_webapp.sh --config FILE   # use an explicit server.json
  scripts/run_webapp.sh --weights DIR   # override weights/ location

Environment:
  AUDIOCORE_PORT          same as --port
  AUDIOCORE_WEIGHTS_DIR   same as --weights (default: ./weights)
  AUDIOCORE_CLIPS_DIR     clips storage (default: ./webapp/clips)
HELP
            exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

WEIGHTS_DIR="${AUDIOCORE_WEIGHTS_DIR:-$DEFAULT_WEIGHTS}"

# ── 1. Build the server if missing or older than the source ────────────────
if [[ "$DO_BUILD" == "1" ]]; then
    need_build=0
    if [[ ! -x "$BIN" ]]; then
        need_build=1
    else
        # Rebuild if any source is newer than the binary.
        newest_src=$(find "$ROOT/src" "$ROOT/include" "$ROOT/webapp/public" \
                         "$ROOT/CMakeLists.txt" \
                         -type f -newer "$BIN" -print -quit 2>/dev/null || true)
        [[ -n "$newest_src" ]] && need_build=1
    fi
    if [[ "$need_build" == "1" ]]; then
        echo "▶ building audiocore_server (first run or sources changed)…" >&2
        cmake -S . -B build -DENGINE_ENABLE_CUDA=ON \
              -DENGINE_BUILD_SERVER=ON -DENGINE_BUILD_CLI=ON \
              -DENGINE_BUILD_WEBAPP=ON >/dev/null
        cmake --build build --target audiocore_server -j"$(nproc)" >&2
    fi
fi
[[ -x "$BIN" ]] || { echo "audiocore_server binary missing — run with --no-build off or build manually." >&2; exit 1; }

# ── 2. Pick a free port if none specified ──────────────────────────────────
pick_free_port() {
    python3 - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
}
PORT="${PORT:-${AUDIOCORE_PORT:-$(pick_free_port)}}"
# (Belt-and-braces: prefer an explicit --port/AUDIOCORE_PORT over the random
#  pick so URLs in logs stay stable across restarts.)

# ── 3. Generate a server.json from the weights tree if not given ───────────
if [[ -z "${CONFIG:-}" ]]; then
    CONFIG="$ROOT/build/webapp.server.json"
    python3 - "$WEIGHTS_DIR" "$CLIPS_DIR" "$PORT" > "$CONFIG" <<'PY'
import json, os, sys, glob
weights, clips, port = sys.argv[1:4]

def exists(*ps): return any(os.path.exists(os.path.join(weights, p)) for p in ps)

models = []

# ACE-Step — register every variant directory found under weights/ace_step/.
ace_dir = os.path.join(weights, "ace_step")
if os.path.isdir(ace_dir):
    for entry in sorted(os.listdir(ace_dir)):
        d = os.path.join(ace_dir, entry)
        if not os.path.isdir(d): continue
        lid = entry.lower()
        if "turbo" in lid:       tag = "turbo"
        elif "base"  in lid:     tag = "base"
        elif "sft"   in lid:     tag = "sft"
        else:                    tag = "turbo"
        lm = "0.6B" if "0.6b" in lid else ("4B" if "xl" in lid else "1.7B")
        extras = {"dit_variant": tag, "lm_variant": lm, "n_gpu_layers": "99"}
        if "scrag" in lid:
            extras["vae_variant"] = "scrag"
        models.append({"id": entry, "family": "ace_step", "path": d + "/",
                       "backend": "ggml_cuda", "extras": extras})

# Qwen3-TTS — scan both qwen3_tts/ and qwen3_tts_gguf_fresh/ for variants.
for wdir in ["qwen3_tts", "qwen3_tts_gguf_fresh"]:
    base = os.path.join(weights, wdir)
    if not os.path.isdir(base): continue
    for entry in sorted(os.listdir(base)):
        d = os.path.join(base, entry)
        if not os.path.isdir(d): continue
        talker = glob.glob(os.path.join(d, "*talker*.gguf"))
        pred   = glob.glob(os.path.join(d, "*predictor*.gguf"))
        codec  = glob.glob(os.path.join(d, "*tokenizer*.gguf")) or \
                 glob.glob(os.path.join(d, "*codec*.gguf"))
        if not (talker and pred): continue
        # Generate a stable model id from the directory name.
        mid = entry.replace("_", "-")
        if wdir == "qwen3_tts_gguf_fresh" and not mid.startswith("qwen3-tts"):
            mid = "qwen3-tts-" + mid
        extras = {"talker_path": os.path.basename(talker[0]),
                  "predictor_path": os.path.basename(pred[0]),
                  "n_gpu_layers": "99"}
        if codec: extras["codec_path"] = os.path.basename(codec[0])
        # Speaker encoder is optional (only some dirs ship it).
        spk = glob.glob(os.path.join(d, "*speaker*encoder*.gguf"))
        if spk: extras["speaker_encoder_path"] = os.path.basename(spk[0])
        models.append({"id": mid, "family": "qwen3_tts", "path": d + "/",
                       "backend": "ggml_cuda", "extras": extras})

# MOSS-TTS — scan moss_tts/ for backbone + optional codec sidecar.
moss_base = os.path.join(weights, "moss_tts")
if os.path.isdir(moss_base):
    for entry in sorted(os.listdir(moss_base)):
        d = os.path.join(moss_base, entry)
        if not os.path.isdir(d): continue
        ggufs = [f for f in sorted(os.listdir(d))
                 if f.endswith(".gguf") and "extras" not in f]
        if not ggufs: continue
        backbone = os.path.join(d, ggufs[0])
        sidecar = backbone.replace(".gguf", ".extras.gguf")
        extras = {"n_gpu_layers": "99"}
        if os.path.exists(sidecar): extras["extras_gguf_path"] = sidecar
        models.append({"id": entry, "family": "moss_tts",
                       "path": backbone, "backend": "ggml_cuda", "extras": extras})

# MOSS-SFX-v2 — optional sidecar GGUFs (DiT + VAE + TE).
mse2_dit = os.path.join(weights, "moss_sfx_v2", "mse2-dit.gguf")
if os.path.exists(mse2_dit):
    extras = {"n_gpu_layers": "99"}
    for tag, fn in [("vae_path", "mse2-vae.gguf"), ("te_path", "mse2-te.gguf")]:
        p = os.path.join(weights, "moss_sfx_v2", fn)
        if os.path.exists(p): extras[tag] = p
    models.append({"id": "moss-sfx-v2", "family": "moss_sfx_v2",
                   "path": mse2_dit, "backend": "ggml_cuda", "extras": extras})

cfg = {"host": "0.0.0.0", "port": int(port), "device": 0, "threads": 4,
       "clips_dir": clips, "models": models}
print(json.dumps(cfg, indent=2))
PY
    echo "▶ generated $CONFIG ($(python3 -c 'import json,sys;print(len(json.load(open(sys.argv[1]))["models"]), "models")' "$CONFIG"))" >&2
fi

mkdir -p "$CLIPS_DIR"

# ── 4. Launch ──────────────────────────────────────────────────────────────
# Prefer an accessible IP for the printed URL (Tailscale if present, else
# the first non-loopback IPv4).
print_url() {
    local port="$1"
    local ip
    ip="$(ip -4 -o addr show scope global 2>/dev/null \
          | awk '{print $4}' | cut -d/ -f1 | head -1 || true)"
    [[ -z "$ip" ]] && ip="localhost"
    if command -v tailscale >/dev/null 2>&1; then
        ts="$(tailscale ip -4 2>/dev/null | head -1 || true)"
        [[ -n "$ts" ]] && ip="$ts"
    fi
    echo "  🎙️  Voice Studio: http://$ip:$port"
}

echo
print_url "$PORT"
echo "  config: $CONFIG"
echo "  (Ctrl-C to stop)"
echo
exec "$BIN" --config "$CONFIG"
