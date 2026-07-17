#!/usr/bin/env bash
# =============================================================================
# deploy_tts.sh — Deploy audiocore TTS to ubuntu-desktop (TAILSCALE_HOST) over Tailscale.
#
# v4 (2026-07-12): CUDA acceleration + VoiceDesign + --target-cv support.
#   - Builds qwen_tts with CUDA backend (make cuda, sm_75 for RTX 2070 SUPER)
#   - Rsyncs all 3 model dirs (customvoice, base, voicedesign)
#   - systemd service includes TTS_USE_CUDA=1 + LD_LIBRARY_PATH for cuBLAS
#   - ~25x speedup vs CPU-only (8s vs 210s per generation)
#
# v3 (2026-07-11): native build on remote (fixes GLIBC 2.39 vs 2.43 mismatch).
#   - Copies ref-qwen3tts SOURCE and builds with gcc on the remote
#   - Uses a Python HTTP wrapper (tts_server.py) instead of audiocore_server
#   - The wrapper calls qwen_tts CLI per request with --load-voice
#   - systemd service for auto-restart
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AUDIOCORE="$(dirname "$SCRIPT_DIR")"
REMOTE="ubuntu@TAILSCALE_HOST"
RDIR="/opt/audiocore"
PORT=39517

# ─── 1. Create remote directory tree ──────────────────────────────────────
echo "▶ [1/9] Creating remote directory tree..."
ssh "$REMOTE" "mkdir -p $RDIR/voices $RDIR/ref-qwen3tts-src $RDIR/weights/qwen3_tts $RDIR/clips"

# ─── 2. Copy qwen_tts source + build with CUDA on remote ──────────────────
echo "▶ [2/9] Copying qwen_tts source + building with CUDA on remote..."
rsync -a --exclude='*.o' --exclude='qwen_tts' --exclude='.git' \
  "$AUDIOCORE/ref-qwen3tts/" "$REMOTE:$RDIR/ref-qwen3tts-src/"
ssh "$REMOTE" "cd $RDIR/ref-qwen3tts-src && export PATH=/usr/local/cuda/bin:\$PATH && \
  make clean && make cuda NVCC_ARCH='-arch=sm_75' -j4 2>&1 | tail -5"
ssh "$REMOTE" "$RDIR/ref-qwen3tts-src/qwen_tts --help >/dev/null 2>&1 && echo '  binary OK' || echo '  BINARY FAILED'"

# ─── 3. Copy default voice ─────────────────────────────────────────────────
echo "▶ [3/9] Copying default.qvoice (25MB lite graft)..."
scp -q "$AUDIOCORE/voices/default.qvoice" "$REMOTE:$RDIR/voices/"

# ─── 4. Rsync CustomVoice model (~4.3GB, resumable via --partial) ──────────
echo "▶ [4/9] Rsyncing CustomVoice model weights (~4.3GB, resumable)..."
rsync -a --partial --info=progress2 \
  "$AUDIOCORE/weights/qwen3_tts/qwen3-tts-1.7b-customvoice/" \
  "$REMOTE:$RDIR/weights/qwen3_tts/qwen3-tts-1.7b-customvoice/"

# ─── 5. Rsync Base model (~4.3GB, needed for voice bake/clone) ─────────────
echo "▶ [5/9] Rsyncing Base model weights (~4.3GB, resumable)..."
rsync -a --partial --info=progress2 \
  "$AUDIOCORE/weights/qwen3_tts/qwen3-tts-1.7b-base/" \
  "$REMOTE:$RDIR/weights/qwen3_tts/qwen3-tts-1.7b-base/"

# ─── 6. Rsync VoiceDesign model (~4.3GB, needed for voice creation) ────────
echo "▶ [6/9] Rsyncing VoiceDesign model weights (~4.3GB, resumable)..."
rsync -a --partial --info=progress2 \
  "$AUDIOCORE/weights/qwen3_tts/qwen3-tts-1.7b-voicedesign/" \
  "$REMOTE:$RDIR/weights/qwen3_tts/qwen3-tts-1.7b-voicedesign/"

# ─── 7. Copy Python HTTP wrapper ──────────────────────────────────────────
echo "▶ [7/9] Copying Python TTS wrapper..."
scp -q "$AUDIOCORE/scripts/tts_server_remote.py" "$REMOTE:$RDIR/tts_server.py"

# ─── 8. Install systemd service (with CUDA env vars) ──────────────────────
echo "▶ [8/9] Installing systemd service..."
ssh "$REMOTE" "sudo tee /etc/systemd/system/audiocore-tts.service" << 'UNIT'
[Unit]
Description=Audiocore TTS Server (voice generation via qwen_tts CLI + CUDA)
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/audiocore
Environment=TTS_USE_CUDA=1
Environment=LD_LIBRARY_PATH=/usr/local/cuda/lib64
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

# ─── 9. Start server ──────────────────────────────────────────────────────
echo "▶ [9/9] Starting server..."
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
