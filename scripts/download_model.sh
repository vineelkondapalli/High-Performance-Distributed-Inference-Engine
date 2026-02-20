#!/usr/bin/env bash
# Download TinyLlama-1.1B-Chat Q4_K_M GGUF from HuggingFace.
# The file is ~670MB and requires no authentication.
#
# Usage:
#   bash scripts/download_model.sh
#
# Output:
#   models/tinyllama-1.1b-q4.gguf

set -euo pipefail

MODEL_DIR="$(dirname "$0")/../models"
MODEL_FILE="$MODEL_DIR/tinyllama-1.1b-q4.gguf"
MODEL_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_FILE" ]; then
    echo "[download_model] Model already exists at $MODEL_FILE"
    exit 0
fi

echo "[download_model] Downloading TinyLlama-1.1B Q4_K_M (~670MB)..."
echo "[download_model] Source: $MODEL_URL"
echo "[download_model] Destination: $MODEL_FILE"
echo ""

if command -v wget &>/dev/null; then
    wget -O "$MODEL_FILE" --progress=bar:force "$MODEL_URL"
elif command -v curl &>/dev/null; then
    curl -L --progress-bar -o "$MODEL_FILE" "$MODEL_URL"
else
    echo "ERROR: Neither wget nor curl is available. Install one and retry."
    exit 1
fi

echo ""
echo "[download_model] Done. Model saved to $MODEL_FILE"
echo "[download_model] File size: $(du -sh "$MODEL_FILE" | cut -f1)"
