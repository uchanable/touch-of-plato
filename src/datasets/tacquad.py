"""TacQuad dataset loader — Stage 1.5-c cross-dataset replication.

Reference:
    Feng, Cui, Cao, Liu, Yu, Wang, Lan, Sun, Zhang, Zhao, Liu, Shao, Lu, Chen,
    Liu, Cui, Tang. "AnyTouch: Learning Unified Static-Dynamic Representation
    Across Multiple Visuo-tactile Sensors." ICLR 2025.
    Dataset: https://huggingface.co/datasets/Mai0313/TacQuad

Goal of Stage 1.5-c (NeurIPS 2026 §5.7 (i)-(c) rebuttal):
    Replicate the TVL T-V > V-L pattern on a different vision-touch-language
    dataset. If the same pattern emerges (T-V mutual-kNN >> V-L mutual-kNN)
    on TacQuad, then the gap is NOT a TVL-specific artifact.

Data layout (verified during initial dataset prep):

    /Volumes/SSD-MS/platonic-touch-data/tacquad/extracted/
    ├── contact_indoor.csv         # 100 rows (objects); 8 fields:
    │       obj_name, gel_start, gel_end, digit_start, digit_end,
    │       dura_start, dura_end, caption
    ├── contact_outdoor.csv        # 50 rows
    ├── data_indoor/<obj>/
    │   ├── digit/    <frame_id>.png      # 640x480 raw tactile (DIGIT)
    │   ├── img_digit/ <frame_id>.png     # 640x480 paired vision (camera)
    │   ├── gelsight/  <frame_id>.png     # 240x320 raw tactile (GelSight Mini)
    │   ├── img_gelsight/ <frame_id>.png  # 240x320 paired vision
    │   ├── duragel/   <frame_id>.png     # raw DuraGel
    │   ├── img_duragel/ <frame_id>.png
    │   ├── tac3d/                        # EMPTY (3D point cloud)
    │   ├── img_tac3d/                    # vision-only
    │   └── <sensor>.csv                  # frame_id, timestamp (no header)
    ├── data_outdoor/<obj>/                # same schema
    ├── tacquad_*_*.pt                     # CLIP-tokenized text only
    │                                      #   ((77,) int64, (77,) float32)
    │                                      # NOT raw images. Skip.
    └── text/obj_*.pt                      # same shape as above

    Per-object frame counts: ~100-150 frames (DIGIT, img_DIGIT often differ
    by ±1; we intersect frame_ids).

Caption format (much longer than TVL adjective lists):
    e.g. "The tactile sensor is touching the surface of a small rectangular
    object made of a white, smooth, plastic material created by 3Dprinter,
    characterized by even blocks-like grain, and moderate roughness..."

Choice — raw images vs preprocessed .pt:
    The .pt files contain ONLY tokenized text (CLIP context length 77).
    They do NOT contain image tensors. Therefore we MUST use raw PNGs.

Default sensor: DIGIT (most TVL-comparable optical gel sensor; 640x480 RGB).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal
import csv
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_TACQUAD_ROOT = Path(
    "/Volumes/SSD-MS/platonic-touch-data/tacquad/extracted"
)
SubsetName = Literal["indoor", "outdoor", "all"]
SensorName = Literal["digit", "gelsight", "duragel"]

# Index (in contact_*.csv 6-int columns) for each sensor's (start, end) pair.
# Field layout: obj_name, gel_s, gel_e, digit_s, digit_e, dura_s, dura_e, caption
_SENSOR_RANGE_COLS: dict[str, tuple[int, int]] = {
    "gelsight": (1, 2),
    "digit": (3, 4),
    "duragel": (5, 6),
}


@dataclass
class TacQuadItem:
    """One TacQuad sample: vision frame, tactile frame, caption, subset/sensor."""
    vision_path: Path
    tactile_path: Path
    text: str
    subset: str  # "indoor" or "outdoor"
    sensor: str
    object_name: str
    frame_id: int

    def load_vision(self) -> Image.Image:
        return Image.open(self.vision_path).convert("RGB")

    def load_tactile(self) -> Image.Image:
        return Image.open(self.tactile_path).convert("RGB")


class TacQuadDataset(Dataset):
    """TacQuad vision-touch-text triple loader for Stage 1.5-c.

    Same interface as TVLDataset:
        __getitem__(i) -> {"vision": PIL, "tactile": PIL, "text": str, ...}

    Args:
        root: Path to extracted TacQuad root containing data_indoor/, etc.
        subset: "indoor" | "outdoor" | "all".
        sensor: "digit" | "gelsight" | "duragel". DIGIT is default for
            TVL comparability (640x480 RGB optical gel).
        max_samples: Optional cap.
        per_object_max: Optional cap of frames-per-object (for balance).
    """

    def __init__(
        self,
        root: Path = DEFAULT_TACQUAD_ROOT,
        subset: SubsetName = "indoor",
        sensor: SensorName = "digit",
        max_samples: Optional[int] = None,
        per_object_max: Optional[int] = None,
    ):
        self.root = Path(root)
        if subset not in ("indoor", "outdoor", "all"):
            raise ValueError(f"subset must be indoor|outdoor|all, got {subset}")
        if sensor not in _SENSOR_RANGE_COLS:
            raise ValueError(
                f"sensor must be one of {list(_SENSOR_RANGE_COLS)}, got {sensor}"
            )
        self.subset = subset
        self.sensor = sensor
        self.max_samples = max_samples
        self.per_object_max = per_object_max
        self._index: list[TacQuadItem] = self._build_index()

    def _build_index(self) -> list[TacQuadItem]:
        if not self.root.exists():
            raise FileNotFoundError(f"TacQuad root not found: {self.root}")

        entries: list[TacQuadItem] = []
        if self.subset in ("indoor", "all"):
            entries.extend(self._iter_subset("indoor"))
        if self.subset in ("outdoor", "all"):
            entries.extend(self._iter_subset("outdoor"))

        if self.max_samples is not None:
            entries = entries[: self.max_samples]
        return entries

    def _iter_subset(self, subset: str) -> list[TacQuadItem]:
        contact_csv = self.root / f"contact_{subset}.csv"
        data_root = self.root / f"data_{subset}"
        if not contact_csv.exists():
            raise FileNotFoundError(f"missing {contact_csv}")
        if not data_root.exists():
            raise FileNotFoundError(f"missing {data_root}")

        items: list[TacQuadItem] = []
        s_col, e_col = _SENSOR_RANGE_COLS[self.sensor]
        tac_dir_name = self.sensor                  # "digit"
        vis_dir_name = f"img_{self.sensor}"         # "img_digit"

        with contact_csv.open() as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 8:
                    continue
                obj_name = row[0]
                try:
                    s = int(row[s_col])
                    e = int(row[e_col])
                except (ValueError, IndexError):
                    continue
                caption = row[7].strip()

                obj_dir = data_root / obj_name
                tac_dir = obj_dir / tac_dir_name
                vis_dir = obj_dir / vis_dir_name
                if not (tac_dir.is_dir() and vis_dir.is_dir()):
                    continue

                # Use only frame_ids that exist on BOTH sides of the pair,
                # constrained to the sensor's contact range [s, e].
                tac_ids = {int(p.stem) for p in tac_dir.glob("*.png")
                           if p.stem.isdigit()}
                vis_ids = {int(p.stem) for p in vis_dir.glob("*.png")
                           if p.stem.isdigit()}
                shared = sorted(tac_ids & vis_ids)
                shared = [fid for fid in shared if s <= fid <= e]

                if self.per_object_max is not None:
                    # Subsample uniformly within range
                    if len(shared) > self.per_object_max:
                        step = len(shared) / self.per_object_max
                        shared = [shared[int(i * step)]
                                  for i in range(self.per_object_max)]

                for fid in shared:
                    items.append(TacQuadItem(
                        vision_path=vis_dir / f"{fid}.png",
                        tactile_path=tac_dir / f"{fid}.png",
                        text=caption,
                        subset=subset,
                        sensor=self.sensor,
                        object_name=obj_name,
                        frame_id=fid,
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
            "sensor": item.sensor,
            "object": item.object_name,
            "frame_id": item.frame_id,
        }
