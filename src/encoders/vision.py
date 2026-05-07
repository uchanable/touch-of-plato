"""Vision encoders (Paper Tab.~``tab:encoders`` rows: vision tower).

Each loader returns a `LoadedEncoder` whose `model.forward(pixel_values)`
yields `Tensor[B, 768]`.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from .base import LoadedEncoder, CKPT_ROOT


class _DINOv2Wrapper(nn.Module):
    """Wraps HuggingFace DINOv2 to return CLS token only."""
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=pixel_values)
        return out.last_hidden_state[:, 0, :]


class _CLIPVisionWrapper(nn.Module):
    """Wraps CLIP-L/14 vision tower.

    In transformers >= 5.0, `CLIPModel.get_image_features()` returns a
    `BaseModelOutputWithPooling` whose `.pooler_output` is the projected
    (B, 768) image embedding.
    """
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.clip = clip_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.clip.get_image_features(pixel_values=pixel_values)
        return out.pooler_output


class _SigLIPVisionWrapper(nn.Module):
    """Wraps SigLIP-Base/16 vision tower.

    `SiglipModel.get_image_features()` returns a `BaseModelOutputWithPooling`
    whose `.pooler_output` is the (B, 768) image embedding.
    """
    def __init__(self, siglip_model: nn.Module):
        super().__init__()
        self.siglip = siglip_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.siglip.get_image_features(pixel_values=pixel_values)
        return out.pooler_output


def load_dinov2_base() -> LoadedEncoder:
    """Paper Tab.~``tab:encoders`` row 1: DINOv2-Base (facebook/dinov2-base).

    Feature: CLS token of last hidden layer, d=768.
    """
    from transformers import AutoModel, AutoImageProcessor
    path = f"{CKPT_ROOT}/facebook__dinov2-base"
    backbone = AutoModel.from_pretrained(path).eval()
    processor = AutoImageProcessor.from_pretrained(path)

    def preprocess(pil_image):
        return processor(images=pil_image, return_tensors="pt")["pixel_values"]

    return LoadedEncoder(
        model=_DINOv2Wrapper(backbone),
        preprocess=preprocess,
        feature_dim=768,
        modality="vision",
        name="dinov2_base",
    )


def load_clip_l_vision() -> LoadedEncoder:
    """Paper Tab.~``tab:encoders`` row 2 (vision side): CLIP ViT-L/14.

    Feature: `.pooler_output` from `get_image_features()`, d=768.
    """
    from transformers import CLIPModel, CLIPImageProcessor
    path = f"{CKPT_ROOT}/openai__clip-vit-large-patch14"
    clip = CLIPModel.from_pretrained(path).eval()
    processor = CLIPImageProcessor.from_pretrained(path)

    def preprocess(pil_image):
        return processor(images=pil_image, return_tensors="pt")["pixel_values"]

    return LoadedEncoder(
        model=_CLIPVisionWrapper(clip),
        preprocess=preprocess,
        feature_dim=768,
        modality="vision",
        name="clip_l_vision",
    )


def load_siglip_base_vision() -> LoadedEncoder:
    """Paper Tab.~``tab:encoders`` row 3 (vision side): SigLIP-Base/16-224.

    Feature: `.pooler_output` from `get_image_features()`, d=768.
    """
    from transformers import SiglipModel, SiglipImageProcessor
    path = f"{CKPT_ROOT}/google__siglip-base-patch16-224"
    siglip = SiglipModel.from_pretrained(path).eval()
    processor = SiglipImageProcessor.from_pretrained(path)

    def preprocess(pil_image):
        return processor(images=pil_image, return_tensors="pt")["pixel_values"]

    return LoadedEncoder(
        model=_SigLIPVisionWrapper(siglip),
        preprocess=preprocess,
        feature_dim=768,
        modality="vision",
        name="siglip_base_vision",
    )



def load_dinov2_small() -> LoadedEncoder:
    """Paper Tab.~``tab:encoders`` Fig. 2 reference: DINOv2-Small (facebook/dinov2-small).

    Used by §``sec:exp-fig2`` (scale curve) as the small vision-size reference point.
    Feature: CLS token of last hidden layer, d=384.
    """
    from transformers import AutoModel, AutoImageProcessor
    path = f"{CKPT_ROOT}/facebook__dinov2-small"
    backbone = AutoModel.from_pretrained(path).eval()
    processor = AutoImageProcessor.from_pretrained(path)

    def preprocess(pil_image):
        return processor(images=pil_image, return_tensors="pt")["pixel_values"]

    return LoadedEncoder(
        model=_DINOv2Wrapper(backbone),
        preprocess=preprocess,
        feature_dim=384,
        modality="vision",
        name="dinov2_small",
    )


def load_dinov2_large() -> LoadedEncoder:
    """Paper Tab.~``tab:encoders`` Fig. 2 reference: DINOv2-Large (facebook/dinov2-large).

    Used by §``sec:exp-fig2`` (scale curve) as the large vision-size reference point.
    Feature: CLS token of last hidden layer, d=1024.
    """
    from transformers import AutoModel, AutoImageProcessor
    path = f"{CKPT_ROOT}/facebook__dinov2-large"
    backbone = AutoModel.from_pretrained(path).eval()
    processor = AutoImageProcessor.from_pretrained(path)

    def preprocess(pil_image):
        return processor(images=pil_image, return_tensors="pt")["pixel_values"]

    return LoadedEncoder(
        model=_DINOv2Wrapper(backbone),
        preprocess=preprocess,
        feature_dim=1024,
        modality="vision",
        name="dinov2_large",
    )
