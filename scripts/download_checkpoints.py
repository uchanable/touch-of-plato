"""Download pretrained encoder checkpoints for the 12-encoder pool.

Downloads to checkpoints/ (gitignored). Run after `pip install -r
requirements.txt` and `bash scripts/setup_third_party.sh` complete.

Coverage (Paper Table 1):
    HuggingFace snapshots:
        - facebook/dinov2-{small,base,large}        (vision)
        - openai/clip-vit-large-patch14             (vision + text)
        - google/siglip-base-patch16-224            (vision + text)
        - sentence-transformers/all-mpnet-base-v2   (text)
        - facebook/sparsh-dino-base                 (tactile)
        - facebook/sparsh-ijepa-base                (tactile)
        - mlfu7/Touch-Vision-Language-Models        (TVL-ViT-B encoder ckpt)
    Google Drive (via gdown):
        - GeWu-Lab/AnyTouch checkpoint.pth          (tactile)

After this script completes, $PLATONIC_TOUCH_CKPT_ROOT/<id>/ should
contain everything `src.encoders` needs.
"""
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = REPO_ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

# HuggingFace snapshot downloads.
HF_MODELS: list[tuple[str, str]] = [
    # Vision encoders
    ("facebook/dinov2-small", "vision (Fig. 2 scale curve)"),
    ("facebook/dinov2-base", "vision"),
    ("facebook/dinov2-large", "vision (Fig. 2 scale curve)"),
    ("openai/clip-vit-large-patch14", "vision+language"),
    ("google/siglip-base-patch16-224", "vision+language"),
    # Language-only
    ("sentence-transformers/all-mpnet-base-v2", "language"),
    # Tactile
    ("facebook/sparsh-dino-base", "tactile"),
    ("facebook/sparsh-ijepa-base", "tactile"),
    # TVL-ViT-B encoder checkpoint (NOT the bundled LLaMA model)
    ("mlfu7/Touch-Vision-Language-Models", "tactile (TVL-ViT-B encoder)"),
]

# AnyTouch is released only on Google Drive. The file ID below comes from
# the official upstream README (https://github.com/GeWu-Lab/AnyTouch).
ANYTOUCH_GDRIVE_ID = "1L4jGUjIHNBMzOiD33Rv0jxWYKHBORD1R"
ANYTOUCH_TARGET = CKPT_DIR / "anytouch" / "checkpoint.pth"


def download_hf_models() -> None:
    for repo_id, role in HF_MODELS:
        target = CKPT_DIR / repo_id.replace("/", "__")
        if target.exists() and any(target.iterdir()):
            print(f"[skip]     {repo_id} already at {target}")
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


def download_anytouch() -> None:
    """Fetch AnyTouch checkpoint from Google Drive via gdown."""
    if ANYTOUCH_TARGET.exists():
        print(f"[skip]     AnyTouch already at {ANYTOUCH_TARGET}")
        return
    try:
        import gdown
    except ImportError:
        print("[fail]     gdown not installed. Run `pip install -r requirements.txt`.")
        return
    ANYTOUCH_TARGET.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] AnyTouch (Google Drive id={ANYTOUCH_GDRIVE_ID}) -> {ANYTOUCH_TARGET}")
    try:
        gdown.download(
            f"https://drive.google.com/uc?id={ANYTOUCH_GDRIVE_ID}",
            str(ANYTOUCH_TARGET),
            quiet=False,
        )
        print(f"[done]     AnyTouch")
    except Exception as e:
        print(f"[fail]     AnyTouch: {e}")
        print(f"           Manually download from https://github.com/GeWu-Lab/AnyTouch")


def main():
    download_hf_models()
    download_anytouch()


if __name__ == "__main__":
    main()
