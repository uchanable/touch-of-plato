#!/usr/bin/env bash
# Download the public Touch-Vision-Language (TVL) dataset from HuggingFace
# into ./data/tvl/. Roughly 75 GB; expect ~10-20 hours on a typical
# residential link.
#
# Usage:
#   bash scripts/download_tvl.sh [target_dir]
#
# Defaults target_dir=./data/tvl. Skips any subset that already exists.

set -e

TARGET="${1:-data/tvl}"
mkdir -p "$TARGET"

if ! python -c "import huggingface_hub" 2>/dev/null; then
  echo "[download_tvl] huggingface_hub not installed. Run 'pip install -r requirements.txt' first." >&2
  exit 1
fi

echo "[download_tvl] Downloading mlfu7/Touch-Vision-Language-Dataset to $TARGET"
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='mlfu7/Touch-Vision-Language-Dataset',
    repo_type='dataset',
    local_dir='$TARGET',
    local_dir_use_symlinks=False,
)
"

echo "[download_tvl] Done. Verify N=43,502 with:"
echo "    python -c \"from src.datasets.tvl import TVLDataset; from pathlib import Path; print(len(TVLDataset(root=Path('$TARGET'), subset='all')))\""
