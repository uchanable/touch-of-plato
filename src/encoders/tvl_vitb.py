"""TVL standalone tactile encoder (Paper Tab.~``tab:encoders`` row #12).

This is the *trained tactile encoder* released alongside TVL-LLaMA
(Fu et al., ICML 2024, "A Touch, Vision, and Language Dataset for
Multimodal Alignment"), NOT the LLaMA-2 generative model itself. The
encoder was trained on the TVL dataset with a contrastive objective
that aligns tactile features into the CLIP-ViT-L/14 latent space — so
in our PlatonicTouch axis it represents the *enforce* end of
"observe (frozen) vs enforce (cross-modal-trained)" while remaining
shape-comparable (768d, ViT-Base) to Sparsh-DINO/IJEPA.

Encoder ID is ``tvl_vitb`` (named after the actual checkpoint
``ckpt/tvl_enc/tvl_enc_vitb.pth`` from
``mlfu7/Touch-Vision-Language-Models``) — explicitly NOT ``tvl_llama``,
to avoid confusion with the bundled LLaMA-2-7B generative VLM.

Architecture (from upstream `tvl_enc/tvl.py::TVL`):

    tactile_encoder = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=768,        # = self.clip.transformer.width (CLIP-L/14 text width)
        global_pool="avg",      # mean-pool over patch tokens
    )

The released ckpt (``tvl_enc_vitb.pth``) is a dict with keys
``{model, optimizer, epoch, scaler}``. ``model`` already filters out
all CLIP weights (see ``TVL.state_dict``) and only contains
``tactile_encoder.*`` plus ``logit_scale`` (and optional
``modality_heads.*`` if ``common_latent_dim`` was used during pre-train —
not the case for the released vitb ckpt, which trains directly into the
768-d head).

Forward: ``(B, 3, 224, 224) -> (B, 768)`` — raw projection-head output
(no L2 normalisation, to stay consistent with how all other encoders in
our pool report features).

Tactile preprocessing (faithful to upstream `tvl_enc/tacvis.py`):

    1. ``tac_padding``: pad shorter edge to make square, then rotate 90°
       (their dataset stores DIGIT frames in landscape; the model was
       trained on the square+rotated form).
    2. ``Resize(224)``: now-square image -> 224x224.
    3. ``ToTensor`` + ``Normalize(mean=TAC_MEAN, std=TAC_STD)`` with the
       channel statistics from the TVL training set (tac stats, NOT
       ImageNet).
"""
from __future__ import annotations
import torch
import torch.nn as nn
from pathlib import Path
import torchvision.transforms.functional as TF
from .base import LoadedEncoder, CKPT_ROOT


# Constants from upstream `tvl_enc/tacvis.py`
_TAC_MEAN = (0.29174602047139075, 0.2971325588927249, 0.2910404549605639)
_TAC_STD = (0.18764469044810236, 0.19467651810273057, 0.21871583397361818)


def _tac_padding(img):
    """Pad shorter edge to square, then rotate 90° (tvl_enc/tacvis.py:138)."""
    if hasattr(img, "size"):  # PIL.Image
        w, h = img.size
    else:  # torch.Tensor (C, H, W)
        h, w = img.shape[-2:]
    long_edge = max(h, w)
    hpad = (long_edge - h) // 2
    wpad = (long_edge - w) // 2
    img = TF.pad(img, [wpad, hpad])
    img = TF.rotate(img, 90)
    return img


def _build_tvl_vitb_preprocess():
    from torchvision import transforms

    base = transforms.Compose([
        transforms.Lambda(_tac_padding),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_TAC_MEAN, std=_TAC_STD),
    ])

    def preprocess(pil_image):
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        return base(pil_image).unsqueeze(0)   # (1, 3, 224, 224)

    return preprocess


def load_tvl_vitb() -> LoadedEncoder:
    """Paper Tab.~``tab:encoders`` row #12: TVL standalone tactile encoder (Fu et al., ICML 2024).

    Loads ``ckpt/tvl_enc/tvl_enc_vitb.pth`` into a timm
    ``vit_base_patch16_224`` with ``num_classes=768, global_pool="avg"``.
    """
    import timm

    ckpt_path = Path(f"{CKPT_ROOT}/tvl_vitb/tvl_enc_vitb.pth")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"TVL-ViT-B checkpoint not found: {ckpt_path}")

    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=768,
        global_pool="avg",
    )

    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw

    prefix = "tactile_encoder."
    te_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if not te_sd:
        raise RuntimeError(
            f"No keys with prefix {prefix!r} found in {ckpt_path}. "
            f"Top-level keys: {list(sd.keys())[:5]}..."
        )
    missing, unexpected = model.load_state_dict(te_sd, strict=False)
    if missing:
        print(f"[tvl_vitb] missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"[tvl_vitb] unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    base_preprocess = _build_tvl_vitb_preprocess()

    def preprocess(pil_image):
        return base_preprocess(pil_image).to(device)

    return LoadedEncoder(
        model=model,
        preprocess=preprocess,
        feature_dim=768,
        modality="tactile",
        name="tvl_vitb",
    )
