#!/bin/sh
# audreyt/pi-ds4 model preparation.
#
# Replaces the upstream ./download_model.sh flow.  Downloads the cyberneurova
# DeepSeek-V4-Flash GGUF, harmonizes it for the ds4 engine (Q8_0/F32 small
# tensors -> F16 so PR #15's MPP F16 prefill path works correctly on M5), and
# symlinks ./ds4flash.gguf to the result.
#
# Idempotent: if the harmonized output already exists, just refreshes the
# symlink.  Run from the ds4 support checkout (cwd = ~/.pi/ds4/support); pass
# the absolute path to harmonize_gguf.py from the pi-ds4 extension dir.
#
# Usage: prepare_model.sh <quant> <harmonize_script>
set -eu

QUANT="${1:-q2}"
HARMONIZE_SCRIPT="${2:-}"

if [ "$QUANT" != "q2" ]; then
    echo "audreyt/pi-ds4 only supports q2 (cyberneurova ships Q2_K)." >&2
    echo "Requested quant: $QUANT" >&2
    exit 1
fi

if [ -z "$HARMONIZE_SCRIPT" ] || [ ! -f "$HARMONIZE_SCRIPT" ]; then
    echo "harmonize_gguf.py not found at: $HARMONIZE_SCRIPT" >&2
    exit 1
fi

REPO="cyberneurova/CyberNeurova-DeepSeek-V4-Flash-abliterated-GGUF"
SRC_FILE="cyberneurova-DeepSeek-V4-Flash-abliterated-Q2_K.gguf"
HARMONIZED_FILE="cyberneurova-Q2_K-ds4-harmonized.gguf"

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
# When invoked by pi-ds4, cwd is the ds4 support checkout, not the script dir.
# Use cwd-relative paths so the layout matches upstream download_model.sh.
OUT_DIR="./gguf"
SRC_PATH="$OUT_DIR/$SRC_FILE"
DST_PATH="$OUT_DIR/$HARMONIZED_FILE"
LINK_PATH="./ds4flash.gguf"

mkdir -p "$OUT_DIR"

# ----- 1. ensure ./ds4flash.gguf points at the harmonized output -----------
relink() {
    ln -sfn "gguf/$HARMONIZED_FILE" "$LINK_PATH"
    echo "ds4 prep: ./ds4flash.gguf -> gguf/$HARMONIZED_FILE"
}

if [ -s "$DST_PATH" ]; then
    echo "ds4 prep: harmonized file already present ($DST_PATH)"
    relink
    exit 0
fi

# ----- 2. bootstrap a Python venv that has gguf-py -------------------------
VENV="./.prep-venv"
if [ ! -x "$VENV/bin/python" ]; then
    echo "ds4 prep: creating Python venv for gguf-py..."
    if ! command -v python3 >/dev/null 2>&1; then
        echo "python3 not on PATH; cannot bootstrap gguf-py venv." >&2
        exit 1
    fi
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --upgrade pip
    "$VENV/bin/pip" install --quiet gguf
fi

# ----- 3. download cyberneurova source GGUF (~99 GB, resumable) -------------
if [ ! -s "$SRC_PATH" ]; then
    echo "ds4 prep: downloading cyberneurova source GGUF (~99 GB, one-time)..."
    URL="https://huggingface.co/$REPO/resolve/main/$SRC_FILE"
    if [ -n "${HF_TOKEN:-}" ]; then
        curl -fL --progress-meter -C - -H "Authorization: Bearer $HF_TOKEN" -o "$SRC_PATH.part" "$URL"
    else
        curl -fL --progress-meter -C - -o "$SRC_PATH.part" "$URL"
    fi
    mv "$SRC_PATH.part" "$SRC_PATH"
else
    echo "ds4 prep: source GGUF already downloaded ($SRC_PATH)"
fi

# ----- 4. harmonize: convert Q8_0/F32 small tensors to F16 ------------------
echo "ds4 prep: harmonizing GGUF (one-time, ~30 minutes on M5 NVMe)..."
"$VENV/bin/python" "$HARMONIZE_SCRIPT" "$SRC_PATH" "$DST_PATH.part"
mv "$DST_PATH.part" "$DST_PATH"

# ----- 5. final symlink -----------------------------------------------------
relink

echo "ds4 prep: done."
echo "ds4 prep: source kept at $SRC_PATH (~99 GB)."
echo "ds4 prep: rm it once you trust the harmonized output and want disk back."
