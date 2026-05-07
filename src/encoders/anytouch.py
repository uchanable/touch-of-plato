"""AnyTouch encoder loader (Paper Tab.~``tab:encoders`` row: tactile, AnyTouch).

Reference:
    Feng et al., "AnyTouch: Learning Unified Static-Dynamic Representation
    across Multiple Visuo-tactile Sensors", ICLR 2025.
    https://github.com/GeWu-Lab/AnyTouch

Architecture (reverse-engineered from `checkpoint.pth`, 1317 keys):

    - Backbone: HuggingFace `CLIPVisionTransformer`, ViT-L/14 (hidden=1024,
      24 layers, 16 heads, intermediate=4096, patch=14, image=224, num_patches=256).
      The full pretrained model (CLIP-ViT-L-14-DataComp.XL-s13B-b90K) is
      adapted; the touch tower has NO LoRA (only the vision tower does).
    - sensor_token: nn.Parameter(10, 5, 1024) — 10 sensor types, 5 tokens each.
    - touch_projection: Linear(1024, 768) — used by the contrastive head; we
      do NOT apply it (other tactile encoders return raw backbone features).

Forward pass (per upstream `linear_probe.py::TactileProbe.emb_forward`):

    [class_emb] + [class_pos]
    [sensor_token[sensor_type]]                (5 tokens, NO position emb)
    [patch_emb] + [patch_pos]                  (256 tokens)
    -> pre_layrnorm -> encoder -> post_layernorm
    -> CLS token at index 0 (used here as the (B, 1024) feature)

Sensor-type choice for TVL:
    TVL is collected with a GelSight DIGIT sensor. AnyTouch dataloader
    encodes TAG/GelSight = 0 (`TAGDataset`, `FeelDataset`). We default
    `sensor_type=0`. Configurable via env `ANYTOUCH_SENSOR_TYPE`.

Input: PIL.Image -> 224x224, ImageNet normalization (mean/std as in
upstream `dataloader/downstream_dataset.py`).

Output: (1, 1024) float32 tensor — CLS token after `post_layernorm`.
"""
from __future__ import annotations
import os
import torch
import torch.nn as nn
from pathlib import Path
from .base import LoadedEncoder, CKPT_ROOT


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------
class _AnyTouchWrapper(nn.Module):
    """Wraps a `CLIPVisionModel` + sensor_token for AnyTouch's touch encoder.

    Forward semantics:
        input  : (B, 3, 224, 224) — RGB tactile image, ImageNet-normalized
        output : (B, 1024) — CLS token after post_layernorm

    The forward path mirrors AnyTouch's `TactileProbe.emb_forward`:
        seq = [CLS] + [5 sensor tokens] + [256 patch tokens]   # length = 262
        seq = pre_layrnorm(seq)
        seq = encoder_layers(seq)
        seq = post_layernorm(seq)
        return seq[:, 0, :]   # CLS
    """
    def __init__(
        self,
        vision_model: nn.Module,   # transformers CLIPVisionTransformer (.vision_model)
        sensor_token: nn.Parameter,   # (10, 5, 1024)
        sensor_type: int = 0,
    ):
        super().__init__()
        self.vision_model = vision_model
        self.sensor_token = sensor_token
        self.sensor_type = sensor_type

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vm = self.vision_model
        emb = vm.embeddings   # CLIPVisionEmbeddings

        bsz = pixel_values.shape[0]
        # Patch embedding: (B, 1024, 16, 16) -> (B, 256, 1024)
        patch_embeds = emb.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # Class token + position embedding for [CLS, patches] (length 257)
        class_embeds = emb.class_embedding.expand(bsz, 1, -1)
        full = torch.cat([class_embeds, patch_embeds], dim=1)        # (B, 257, 1024)
        pos_ids = torch.arange(full.shape[1], device=full.device).unsqueeze(0)
        full = full + emb.position_embedding(pos_ids)                # add pos emb

        # Insert sensor tokens AFTER CLS, BEFORE patches (no positional emb on them).
        sensor = self.sensor_token[self.sensor_type].unsqueeze(0).expand(bsz, -1, -1).to(full.dtype)
        # full[:, :1] is CLS+pos, full[:, 1:] is patches+pos
        seq = torch.cat([full[:, :1, :], sensor, full[:, 1:, :]], dim=1)   # (B, 262, 1024)

        seq = vm.pre_layrnorm(seq)
        # CLIPEncoder accepts inputs_embeds via positional/keyword
        enc_out = vm.encoder(inputs_embeds=seq, return_dict=True)
        last_hidden = enc_out.last_hidden_state
        last_hidden = vm.post_layernorm(last_hidden)
        cls = last_hidden[:, 0, :]   # (B, 1024)
        return cls.float()


# ---------------------------------------------------------------------------
# Preprocess
# ---------------------------------------------------------------------------
def _build_anytouch_preprocess():
    """224x224 + ImageNet normalization (matches upstream dataloader)."""
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def preprocess(pil_image):
        # Ensure RGB; tactile images may load as RGBA / single-channel.
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        return tf(pil_image).unsqueeze(0)   # (1, 3, 224, 224)

    return preprocess


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_anytouch() -> LoadedEncoder:
    """Paper Tab.~``tab:encoders`` row: AnyTouch (Feng et al., ICLR 2025). d=1024."""
    from transformers import CLIPVisionConfig, CLIPVisionModel

    ckpt_path = Path(f"{CKPT_ROOT}/anytouch/checkpoint.pth")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"AnyTouch checkpoint not found: {ckpt_path}")

    # ViT-L/14 config matching CLIP-ViT-L-14-DataComp.XL-s13B-b90K.
    config = CLIPVisionConfig(
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_channels=3,
        image_size=224,
        patch_size=14,
        hidden_act="gelu",            # DataComp variant uses gelu
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
    )
    clip_vm = CLIPVisionModel(config)
    vision_model = clip_vm.vision_model   # CLIPVisionTransformer

    # ---- Load checkpoint ----
    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw

    # Strip the `touch_mae_model.touch_model.` prefix to match
    # `CLIPVisionTransformer` keys (e.g. `embeddings.class_embedding`,
    # `encoder.layers.0.self_attn.q_proj.weight`, `pre_layrnorm.weight`,
    # `post_layernorm.weight`).
    prefix = "touch_mae_model.touch_model."
    vm_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = vision_model.load_state_dict(vm_sd, strict=False)
    if missing:
        print(f"[anytouch] missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"[anytouch] unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")

    sensor_token = nn.Parameter(sd["touch_mae_model.sensor_token"].clone(), requires_grad=False)

    # Pick device: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    sensor_type = int(os.environ.get("ANYTOUCH_SENSOR_TYPE", "0"))
    if not (0 <= sensor_type < sensor_token.shape[0]):
        raise ValueError(
            f"ANYTOUCH_SENSOR_TYPE={sensor_type} out of range "
            f"[0, {sensor_token.shape[0]})"
        )

    wrapper = _AnyTouchWrapper(vision_model, sensor_token, sensor_type=sensor_type)
    wrapper = wrapper.to(device).eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)

    base_preprocess = _build_anytouch_preprocess()

    def preprocess(pil_image):
        return base_preprocess(pil_image).to(device)

    return LoadedEncoder(
        model=wrapper,
        preprocess=preprocess,
        feature_dim=1024,
        modality="tactile",
        name="anytouch",
    )
