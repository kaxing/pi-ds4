#!/bin/sh
# audreyt/pi-ds4 model preparation.
#
# Replaces the upstream ./download_model.sh.  Downloads the cyberneurova
# DeepSeek-V4-Flash GGUF (resumable via curl -C -) and symlinks
# ./ds4flash.gguf to it.  No harmonization, no Python venv: audreyt/ds4
# main now loads and runs the unmodified stock-recipe Q8_0 file directly
# on M-series Metal (see the m5-support-q8_0-token-embd commit there).
#
# Idempotent: if the file is already present, just refreshes the symlink.
# Run from the ds4 support checkout (cwd = ~/.pi/ds4/support).
#
# Usage: prepare_model.sh <quant>
set -eu

QUANT="${1:-q2}"

if [ "$QUANT" != "q2" ]; then
    echo "audreyt/pi-ds4 only supports q2 (cyberneurova ships Q2_K)." >&2
    echo "Requested quant: $QUANT" >&2
    exit 1
fi

REPO="cyberneurova/CyberNeurova-DeepSeek-V4-Flash-abliterated-GGUF"
MODEL_FILE="cyberneurova-DeepSeek-V4-Flash-abliterated-Q2_K.gguf"

OUT_DIR="./gguf"
SRC_PATH="$OUT_DIR/$MODEL_FILE"
LINK_PATH="./ds4flash.gguf"

mkdir -p "$OUT_DIR"

if [ ! -s "$SRC_PATH" ]; then
    echo "ds4 prep: downloading cyberneurova GGUF (~99 GB, one-time, resumable)..."
    URL="https://huggingface.co/$REPO/resolve/main/$MODEL_FILE"
    if [ -n "${HF_TOKEN:-}" ]; then
        curl -fL --progress-meter -C - -H "Authorization: Bearer $HF_TOKEN" -o "$SRC_PATH.part" "$URL"
    else
        curl -fL --progress-meter -C - -o "$SRC_PATH.part" "$URL"
    fi
    mv "$SRC_PATH.part" "$SRC_PATH"
else
    echo "ds4 prep: GGUF already downloaded ($SRC_PATH)"
fi

ln -sfn "gguf/$MODEL_FILE" "$LINK_PATH"
echo "ds4 prep: ./ds4flash.gguf -> gguf/$MODEL_FILE"
