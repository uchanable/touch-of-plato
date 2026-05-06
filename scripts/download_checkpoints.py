"""Download pretrained encoder checkpoints for initial sanity experiments.

Downloads to checkpoints/ (gitignored). Run after env_setup.sh completes.
"""
import os
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = REPO_ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

MODELS = [
    # Vision encoders
    ("facebook/dinov2-base", "vision"),
    ("openai/clip-vit-large-patch14", "vision+language"),
    ("google/siglip-base-patch16-224", "vision+language"),
    # Tactile encoders
    ("facebook/sparsh-dino-base", "tactile"),
    ("facebook/sparsh-ijepa-base", "tactile"),
    # Language-only (using sentence-transformers to avoid gated LLaMA)
    ("sentence-transformers/all-mpnet-base-v2", "language"),
]

def main():
    for repo_id, role in MODELS:
        target = CKPT_DIR / repo_id.replace("/", "__")
        if target.exists() and any(target.iterdir()):
            print(f"[skip] {repo_id} already at {target}")
            continue
        print(f"[download] {repo_id} ({role}) -> {target}")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(target),
                local_dir_use_symlinks=False,
                ignore_patterns=["*.msgpack", "*.h5", "*.ot", "flax_model.*", "tf_model.*"],
            )
            print(f"[done]     {repo_id}")
        except Exception as e:
            print(f"[fail]     {repo_id}: {e}")

if __name__ == "__main__":
    main()
