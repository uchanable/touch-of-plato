#!/usr/bin/env bash
# Download AnyTouch checkpoint from Google Drive (Feng et al., ICLR 2025).
# Upstream: https://github.com/GeWu-Lab/AnyTouch
# Google Drive file ID published in upstream README.
#
# Usage:
#   bash scripts/download_anytouch.sh
#
# This is a thin shell wrapper around the gdown call also embedded in
# `python scripts/download_checkpoints.py`. Either works; the Python
# script is the canonical entry point.

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

TARGET_DIR="checkpoints/anytouch"
mkdir -p "$TARGET_DIR"

if ! python -c "import gdown" 2>/dev/null; then
    echo "[anytouch] gdown not installed. Run: pip install -r requirements.txt" >&2
    exit 1
fi

FILE_ID="1L4jGUjIHNBMzOiD33Rv0jxWYKHBORD1R"
echo "[anytouch] Downloading from Google Drive (file id: $FILE_ID) -> $TARGET_DIR/checkpoint.pth"
cd "$TARGET_DIR"
python -m gdown "https://drive.google.com/uc?id=$FILE_ID" -O checkpoint.pth || {
    echo "[anytouch] FAILED. Download manually from https://github.com/GeWu-Lab/AnyTouch" >&2
    exit 1
}
echo "[anytouch] Done."
ls -la "$REPO_ROOT/$TARGET_DIR"
