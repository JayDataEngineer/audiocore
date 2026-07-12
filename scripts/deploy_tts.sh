#!/usr/bin/env bash
# =============================================================================
# deploy_tts.sh — Deploy audiocore TTS to ubuntu-desktop (TAILSCALE_HOST) over Tailscale.
#
# v3 (2026-07-11): native build on remote (fixes GLIBC 2.39 vs 2.43 mismatch).
#   - Copies ref-qwen3tts SOURCE and builds with gcc on the remote
#   - Uses a Python HTTP wrapper (tts_server.py) instead of audiocore_server
#   - The wrapper calls qwen_tts CLI per request with --load-voice
#   - systemd service for auto-restart
# =============================================================================
set -euo pipefail

AUDIOCORE="<AUDIOCORE_ROOT>"
REMOTE="ubuntu@TAILSCALE_HOST"
RDIR="/opt/audiocore"
PORT=39517

# ─── 1. Create remote directory tree ──────────────────────────────────────
echo "▶ [1/7] Creating remote directory tree..."
ssh "$REMOTE" "mkdir -p $RDIR/voices $RDIR/ref-qwen3tts-src $RDIR/weights/qwen3_tts $RDIR/clips"

# ─── 2. Copy qwen_tts source + build natively on remote ───────────────────
echo "▶ [2/7] Copying qwen_tts source + building on remote (native, no GLIBC issue)..."
rsync -a --exclude='*.o' --exclude='qwen_tts' --exclude='.git' \
  "$AUDIOCORE/ref-qwen3tts/" "$REMOTE:$RDIR/ref-qwen3tts-src/"
ssh "$REMOTE" "cd $RDIR/ref-qwen3tts-src && make -j4 \
  CFLAGS_BASE='-Wall -Wextra -O3 -mavx2 -mfma -ffast-math -Ivendor' \
  LDLIBS='-lm -lpthread -lopenblas' qwen_tts 2>&1 | tail -3"
ssh "$REMOTE" "$RDIR/ref-qwen3tts-src/qwen_tts --help >/dev/null 2>&1 && echo '  binary OK' || echo '  BINARY FAILED'"

# ─── 3. Copy default voice ─────────────────────────────────────────────────
echo "▶ [3/7] Copying default.qvoice (25MB lite graft)..."
scp -q "$AUDIOCORE/voices/default.qvoice" "$REMOTE:$RDIR/voices/"

# ─── 4. Rsync model weights (~13GB, resumable via --partial) ──────────────
echo "▶ [4/7] Rsyncing model weights (~13GB, resumable)..."
rsync -a --partial --info=progress2 \
  "$AUDIOCORE/weights/qwen3_tts/qwen3-tts-1.7b-customvoice/" \
  "$REMOTE:$RDIR/weights/qwen3_tts/qwen3-tts-1.7b-customvoice/"

# ─── 5. Copy Python HTTP wrapper ──────────────────────────────────────────
echo "▶ [5/7] Copying Python TTS wrapper..."
scp -q "$AUDIOCORE/scripts/tts_server_remote.py" "$REMOTE:$RDIR/tts_server.py"

# ─── 6. Install systemd service ───────────────────────────────────────────
echo "▶ [6/7] Installing systemd service..."
ssh "$REMOTE" "sudo tee /etc/systemd/system/audiocore-tts.service" << 'UNIT'
[Unit]
Description=Audiocore TTS Server (voice generation via qwen_tts CLI)
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/audiocore
ExecStart=/usr/bin/python3 /opt/audiocore/tts_server.py
Restart=on-failure
RestartSec=5
User=ubuntu
Group=ubuntu
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNIT
ssh "$REMOTE" "sudo systemctl daemon-reload && sudo systemctl enable audiocore-tts"

# ─── 7. Start server ──────────────────────────────────────────────────────
echo "▶ [7/7] Starting server..."
ssh "$REMOTE" "sudo systemctl restart audiocore-tts"
sleep 3
ssh "$REMOTE" "sudo systemctl is-active audiocore-tts"

echo ""
echo "================================================================"
echo " Deploy complete. Endpoint: http://TAILSCALE_HOST:$PORT"
echo ""
echo " Test:"
echo "   curl http://TAILSCALE_HOST:$PORT/v1/audio/speech \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"input\":\"Hello!\",\"voice\":\"default\"}' -o test.wav"
echo ""
echo " Voices:  curl http://TAILSCALE_HOST:$PORT/v1/voices"
echo " Health:  curl http://TAILSCALE_HOST:$PORT/health"
echo " Logs:    ssh $REMOTE 'sudo journalctl -u audiocore-tts -f'"
echo "================================================================"
