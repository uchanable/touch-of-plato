"""Language encoders (Paper Tab.~``tab:encoders`` rows: text tower).

Each loader returns a `LoadedEncoder` whose `model.forward(batch)` yields
`Tensor[B, 768]`.

Note: In transformers >= 5.0, `CLIPModel.get_text_features()` and
`SiglipModel.get_text_features()` return a `BaseModelOutputWithPooling`
whose `.pooler_output` is the (B, 768) projected text embedding.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from .base import LoadedEncoder, CKPT_ROOT


class _CLIPTextWrapper(nn.Module):
    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.clip = clip_model

    def forward(self, batch: dict) -> torch.Tensor:
        out = self.clip.get_text_features(**batch)
        return out.pooler_output


class _SigLIPTextWrapper(nn.Module):
    def __init__(self, siglip_model: nn.Module):
        super().__init__()
        self.siglip = siglip_model

    def forward(self, batch: dict) -> torch.Tensor:
        out = self.siglip.get_text_features(**batch)
        return out.pooler_output


class _MeanPoolWrapper(nn.Module):
    """Attention-masked mean pooling over token embeddings (for mpnet)."""
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, batch: dict) -> torch.Tensor:
        out = self.backbone(**batch)
        mask = batch["attention_mask"].unsqueeze(-1).float()
        summed = (out.last_hidden_state * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1e-9)
        return summed / count


def load_clip_l_text() -> LoadedEncoder:
    """Paper Tab.~``tab:encoders`` row 2 (language side): CLIP ViT-L/14 text tower. d=768."""
    from transformers import CLIPModel, CLIPTokenizer
    path = f"{CKPT_ROOT}/openai__clip-vit-large-patch14"
    clip = CLIPModel.from_pretrained(path).eval()
    tokenizer = CLIPTokenizer.from_pretrained(path)

    def preprocess(text):
        if isinstance(text, str):
            text = [text]
        return tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    return LoadedEncoder(
        model=_CLIPTextWrapper(clip),
        preprocess=preprocess,
        feature_dim=768,
        modality="language",
        name="clip_l_text",
    )


def load_siglip_base_text() -> LoadedEncoder:
    """Paper Tab.~``tab:encoders`` row 3 (language side): SigLIP-Base/16 text tower. d=768."""
    from transformers import SiglipModel, SiglipTokenizer
    path = f"{CKPT_ROOT}/google__siglip-base-patch16-224"
    siglip = SiglipModel.from_pretrained(path).eval()
    tokenizer = SiglipTokenizer.from_pretrained(path)

    def preprocess(text):
        if isinstance(text, str):
            text = [text]
        return tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")

    return LoadedEncoder(
        model=_SigLIPTextWrapper(siglip),
        preprocess=preprocess,
        feature_dim=768,
        modality="language",
        name="siglip_base_text",
    )


def load_mpnet() -> LoadedEncoder:
    """Paper Tab.~``tab:encoders`` row 4: sentence-transformers/all-mpnet-base-v2. d=768.

    Attention-masked mean-pooled sentence embedding.
    """
    from transformers import AutoModel, AutoTokenizer
    path = f"{CKPT_ROOT}/sentence-transformers__all-mpnet-base-v2"
    backbone = AutoModel.from_pretrained(path).eval()
    tokenizer = AutoTokenizer.from_pretrained(path)

    def preprocess(text):
        if isinstance(text, str):
            text = [text]
        return tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    return LoadedEncoder(
        model=_MeanPoolWrapper(backbone),
        preprocess=preprocess,
        feature_dim=768,
        modality="language",
        name="mpnet",
    )
