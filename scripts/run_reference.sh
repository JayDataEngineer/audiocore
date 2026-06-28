#!/usr/bin/env bash
# Run the upstream MOSS-TTS Python pipeline as reference.
# Produces output.wav for parity testing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WEIGHTS="$PROJECT_DIR/weights"

# Source the HF token
if [ -f "$PROJECT_DIR/config/secrets.env" ]; then
    set -a; source "$PROJECT_DIR/config/secrets.env"; set +a
fi

# Point PYTHONPATH to the vendor (upstream code) + our shim
VENDOR_DIR="<RAY_ROOT>/vendor/moss-tts-v2"
export PYTHONPATH="$PROJECT_DIR:$VENDOR_DIR:$PYTHONPATH"

# Download the audio tokenizer ONNX if not already cached locally
HF_CACHE_ONNX="<HF_ONNX_CACHE>"
if [ ! -d "$WEIGHTS/MOSS-Audio-Tokenizer-ONNX" ]; then
    ln -sfn "$HF_CACHE_ONNX" "$WEIGHTS/MOSS-Audio-Tokenizer-ONNX"
fi

# Also ensure GGUF symlink
HF_CACHE_GGUF="<HF_GGUF_CACHE>"
if [ ! -d "$WEIGHTS/MOSS-TTS-GGUF" ]; then
    ln -sfn "$HF_CACHE_GGUF" "$WEIGHTS/MOSS-TTS-GGUF"
fi

echo "=== Running upstream MOSS-TTS reference pipeline ==="
echo "GGUF: $WEIGHTS/MOSS-TTS-GGUF/MOSS_TTS_Q4_K_M.gguf"
echo "Output: $WEIGHTS/reference_output.wav"
echo ""

cd "$PROJECT_DIR"
python3 -m moss_tts_delay.llama_cpp \
    --config "$SCRIPT_DIR/reference_config.yaml" \
    --text "Hello, this is a test of the MOSS TTS pipeline." \
    --output "$WEIGHTS/reference_output.wav" \
    --profile \
    2>&1

echo ""
echo "=== Reference output saved to $WEIGHTS/reference_output.wav ==="
