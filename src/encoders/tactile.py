"""Tactile encoders (Paper Tab.~``tab:encoders`` rows: tactile).

Sparsh checkpoints from facebook/sparsh-{dino,ijepa}-base.

Implementation note (verified during initial dataset prep):
    Sparsh uses a custom ViT-Base architecture from
    `facebookresearch/sparsh/tactile_ssl/model/vision_transformer.py`.
    It is NOT a HuggingFace AutoModel-compatible checkpoint.
    Key differences from a standard DINO ViT:

    - **6-channel input** (Sparsh concatenates two tactile frames along
      the channel axis for temporal context). For static TVL images we
      duplicate the single frame twice — see the paper's Limitations
      section.
    - **No CLS token**. `forward()` returns `x_norm_patchtokens`. We
      mean-pool the patch tokens to obtain a (B, 768) feature.
    - **Sinusoidal positional embedding** (not learned), 1 register token,
      layerscale enabled (`init_values=1`).

    The ViT construction arguments are reverse-engineered from the
    state_dict of `dino_vitbase.safetensors` (174 keys,
    `patch_embed.proj.weight.shape = (768, 6, 16, 16)`).
"""
from __future__ import annotations
import sys
import torch
import torch.nn as nn
from pathlib import Path
from .base import LoadedEncoder, CKPT_ROOT

# Add third_party/sparsh to sys.path so `tactile_ssl` is importable
_SPARSH_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "sparsh"
if str(_SPARSH_ROOT) not in sys.path:
    sys.path.insert(0, str(_SPARSH_ROOT))


class _SparshWrapper(nn.Module):
    """Wraps Sparsh VisionTransformer.

    Forward semantics:
        input  : (B, 6, 224, 224) — 6-channel tactile image
        output : (B, 768) — mean-pooled patch tokens (post layer-norm)

    Sparsh's native `forward()` returns `x_norm_patchtokens` of shape
    (B, num_patches, 768). We mean-pool over the patch dimension to
    obtain a single feature vector per image.
    """
    def __init__(self, vit: nn.Module):
        super().__init__()
        self.vit = vit

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.vit(pixel_values)  # (B, N_patches, 768)
        return patch_tokens.mean(dim=1)


def _build_sparsh_preprocess():
    """Standard ImageNet-normalized 224px transform, then duplicate to 6 channels.

    Divergence note: we use same-frame duplication `I_t || I_t -> 6ch` rather
    than the Sparsh-original temporal stride `I_t || I_{t-5}`. This is because
    TVL provides static tactile snapshots, not frame sequences. The resulting
    feature should still be a valid Sparsh representation for single-frame
    content, and we document this choice in the paper's Limitations.
    """
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def preprocess(pil_image):
        img_3ch = tf(pil_image).unsqueeze(0)              # (1, 3, 224, 224)
        img_6ch = torch.cat([img_3ch, img_3ch], dim=1)    # (1, 6, 224, 224)
        return img_6ch

    return preprocess


def _load_sparsh_vit(safetensors_path: Path) -> nn.Module:
    """Construct a Sparsh ViT-Base and load state_dict from safetensors."""
    from tactile_ssl.model.vision_transformer import vit_base
    from safetensors.torch import load_file

    model = vit_base(
        patch_size=16,
        in_chans=6,
        num_register_tokens=1,
        pos_embed_fn="sinusoidal",
        init_values=1,
    )
    state_dict = load_file(str(safetensors_path))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[sparsh] missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"[sparsh] unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
    model.eval()
    return model


def load_sparsh_dino_base() -> LoadedEncoder:
    """Paper Tab.~``tab:encoders`` row 5: Sparsh-DINO-Base (facebook/sparsh-dino-base). d=768."""
    path = Path(f"{CKPT_ROOT}/facebook__sparsh-dino-base/dino_vitbase.safetensors")
    model = _load_sparsh_vit(path)
    return LoadedEncoder(
        model=_SparshWrapper(model),
        preprocess=_build_sparsh_preprocess(),
        feature_dim=768,
        modality="tactile",
        name="sparsh_dino_base",
    )


def load_sparsh_ijepa_base() -> LoadedEncoder:
    """Paper Tab.~``tab:encoders`` row 6: Sparsh-IJEPA-Base (facebook/sparsh-ijepa-base). d=768."""
    path = Path(f"{CKPT_ROOT}/facebook__sparsh-ijepa-base/ijepa_vitbase.safetensors")
    model = _load_sparsh_vit(path)
    return LoadedEncoder(
        model=_SparshWrapper(model),
        preprocess=_build_sparsh_preprocess(),
        feature_dim=768,
        modality="tactile",
        name="sparsh_ijepa_base",
    )
