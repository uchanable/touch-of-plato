"""TVL (Touch-Vision-Language) dataset loader.

Reference:
    Fu, Datta, Huang, Panitch, Drake, Ortiz, Mukadam, Lambeta, Calandra,
    Goldberg. "A Touch, Vision, and Language Dataset for Multimodal
    Alignment." ICML 2024. https://arxiv.org/abs/2402.13232

Dataset repo: https://huggingface.co/datasets/mlfu7/Touch-Vision-Language-Dataset

Actual structure (verified during initial dataset prep):

    data/tvl/tvl_dataset/
    ├── ssvtp/
    │   ├── train.csv
    │   ├── test.csv
    │   ├── images_tac/image_{id}_tac.jpg
    │   └── images_rgb/image_{id}_rgb.jpg
    └── hct/
        ├── data1/
        │   ├── train.csv   columns: url, tactile, tactile_background, caption
        │   ├── test.csv    (same columns)
        │   ├── finetune.json
        │   └── {run_timestamp}/
        │       ├── tactile/{frame_id}.jpg
        │       ├── vision/{frame_id}.jpg
        │       └── tactile_bg_latent.jpg
        ├── data2/ (same schema)
        └── data3/ (same schema)

    CSV sample row from hct/data1/train.csv:
        url: 0-1702507215.615537/vision/225-0.03534817695617676.jpg
        tactile: 0-1702507215.615537/tactile/225-0.03534817695617676.jpg
        tactile_background: 0-1702507215.615537/tactile_bg_latent.jpg
        caption: "smooth, reflective, hard, cool, sleek"

Paper binding: Section 3.3 "Encoders and Dataset" of the paper.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal
import csv
import os
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_TVL_ROOT = Path(os.environ.get("TVL_ROOT", "data/tvl"))
SubsetName = Literal["ssvtp", "hct", "all"]


@dataclass
class TVLItem:
    """One TVL sample: vision frame, tactile frame, text caption, subset tag."""
    vision_path: Path
    tactile_path: Path
    text: str
    subset: str

    def load_vision(self) -> Image.Image:
        return Image.open(self.vision_path).convert("RGB")

    def load_tactile(self) -> Image.Image:
        return Image.open(self.tactile_path).convert("RGB")


class TVLDataset(Dataset):
    """TVL vision-touch-text triple loader.

    Implements Section 3.3 ("Encoders and Dataset") of the paper.

    Args:
        root: Path to the `data/tvl` directory. Must contain `tvl_dataset/`
              after unzipping.
        subset: "ssvtp" | "hct" | "all".
        max_samples: Optional cap for sanity testing / initial sub-sampling
                     (used by Fig. 2 scale curves at data fractions).
        split: "train" | "test" | "all". Which CSV splits to include.

    Returns (via __getitem__):
        dict with keys 'vision' (PIL Image), 'tactile' (PIL Image),
        'text' (str), 'subset' (str).
    """

    def __init__(
        self,
        root: Path = DEFAULT_TVL_ROOT,
        subset: SubsetName = "all",
        max_samples: Optional[int] = None,
        split: Literal["train", "test", "all"] = "all",
    ):
        self.root = Path(root)
        self.subset = subset
        self.split = split
        self.max_samples = max_samples
        self._index: list[TVLItem] = self._build_index()

    def _build_index(self) -> list[TVLItem]:
        """Walk the TVL directory and build the sample index."""
        tvl_dataset_root = self.root / "tvl_dataset"
        if not tvl_dataset_root.exists():
            raise FileNotFoundError(
                f"TVL dataset not yet unzipped. Expected at {tvl_dataset_root}. "
                f"Run: cd {self.root} && zip -s0 tvl_dataset_sharded.zip "
                f"--out tvl_dataset_full.zip && unzip tvl_dataset_full.zip"
            )

        entries: list[TVLItem] = []
        if self.subset in ("ssvtp", "all"):
            ssvtp_root = tvl_dataset_root / "ssvtp"
            if ssvtp_root.exists():
                entries.extend(self._iter_ssvtp(ssvtp_root))
        if self.subset in ("hct", "all"):
            hct_root = tvl_dataset_root / "hct"
            if hct_root.exists():
                entries.extend(self._iter_hct(hct_root))

        if self.max_samples is not None:
            entries = entries[: self.max_samples]
        return entries

    def _splits_to_load(self) -> list[str]:
        return {
            "train": ["train.csv"],
            "test": ["test.csv"],
            "all": ["train.csv", "test.csv"],
        }[self.split]

    def _iter_ssvtp(self, ssvtp_root: Path) -> list[TVLItem]:
        """Parse SSVTP CSVs into TVLItem list.

        SSVTP column schema (verified during initial dataset prep):
            train.csv: url, tactile, caption  -> full vision-touch-text triples
            test.csv:  url, caption            -> NO tactile column; SKIPPED
                       (only ~500 samples, used by TVL-LLaMA for VQA eval)

        `url` is relative to ssvtp_root, e.g., `images_rgb/image_106_rgb.jpg`.
        `tactile` is relative to ssvtp_root, e.g., `images_tac/image_106_tac.jpg`.
        `caption` is a comma-separated adjective string, e.g., "soft, patterned, lined, fabric".
        """
        items: list[TVLItem] = []
        # Only train.csv has tactile column; test.csv is VQA-only, skipped.
        for split_name in [s for s in self._splits_to_load() if s == "train.csv"]:
            csv_path = ssvtp_root / split_name
            if not csv_path.exists():
                continue
            with csv_path.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    items.append(TVLItem(
                        vision_path=ssvtp_root / row["url"],
                        tactile_path=ssvtp_root / row["tactile"],
                        text=row["caption"],
                        subset="ssvtp",
                    ))
        return items

    def _iter_hct(self, hct_root: Path) -> list[TVLItem]:
        """Parse HCT data{1,2,3}/{train,test}.csv into TVLItem list.

        Confirmed columns: `url`, `tactile`, `tactile_background`, `caption`.
        """
        items: list[TVLItem] = []
        for data_dir in sorted(hct_root.glob("data*")):
            if not data_dir.is_dir():
                continue
            for split_name in self._splits_to_load():
                csv_path = data_dir / split_name
                if not csv_path.exists():
                    continue
                with csv_path.open() as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        vis_rel = row["url"]           # e.g., 0-170.../vision/225-0.035.jpg
                        tac_rel = row["tactile"]       # e.g., 0-170.../tactile/225-0.035.jpg
                        text = row["caption"]
                        items.append(TVLItem(
                            vision_path=data_dir / vis_rel,
                            tactile_path=data_dir / tac_rel,
                            text=text,
                            subset="hct",
                        ))
        return items

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        item = self._index[idx]
        return {
            "vision": item.load_vision(),
            "tactile": item.load_tactile(),
            "text": item.text,
            "subset": item.subset,
        }
