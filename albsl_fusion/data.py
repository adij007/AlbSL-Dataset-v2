"""
albsl_fusion.data
=================
Real dataset loader for the AlbSL training pipeline.

Sources loaded (all merged together):
  1. data/csv/videos/NN_LETTER.csv   — per-letter video frame keypoints (36 letters)
  2. data/csv/alfabeti_keypoints.csv — static annotated images (A/B/C/F + UNKNOWN)
  3. video_keypoints.csv             — output of part4_video_pipeline.py (if exists)

Each sample exposes:
  image        : Tensor[3, IMAGE_SIZE, IMAGE_SIZE]  — mean/std normalised RGB
  keypoints    : Tensor[SEQ_LEN, 123]               — window of hand features
  letter_index : Tensor[]  long                     — class index (0-based)
  bbox         : Tensor[4]  xyxy in pixel coords    — approximate hand bbox
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# ── Constants ─────────────────────────────────────────────────────────────────
# Full Albanian alphabet in canonical order (36 letters → indices 0-35)
ALBANIAN_LETTERS: list[str] = [
    "A", "B", "C", "Ç", "D", "Dh", "E", "Ë", "F", "G", "Gj",
    "H", "I", "J", "K", "L", "Ll", "M", "N", "Nj", "O", "P",
    "Q", "R", "Rr", "S", "Sh", "T", "Th", "U", "V", "X", "Xh",
    "Y", "Z", "Zh",
]
LETTER_TO_IDX: dict[str, int] = {l: i for i, l in enumerate(ALBANIAN_LETTERS)}
NUM_LETTERS = len(ALBANIAN_LETTERS)  # 36

# Canonical feature cols from data/csv/videos/*.csv
_LM_NAMES = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]
LM_COLS  = [f"{n}_{ax}" for n in _LM_NAMES for ax in ("x", "y", "z")]   # 63
BONE_COLS = [f"bone{i}_{f}" for i in range(20) for f in ("theta", "adj_prev", "adj_next")]  # 60
VIDEO_FEAT_COLS = LM_COLS + BONE_COLS  # 123

# Normalised-coord cols from alfabeti_keypoints.csv  (part2_extract.py output)
_ALFA_LM_COLS = (
    [f"lm{i}_x_norm" for i in range(21)] +
    [f"lm{i}_y_norm" for i in range(21)] +
    [f"lm{i}_z_norm" for i in range(21)]
)  # 63 — padded with zeros to 123

# Image normalisation constants (ImageNet)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Default paths (relative to repo root)
_ROOT = Path(__file__).resolve().parent.parent
VIDEO_CSV_DIR       = _ROOT / "datasets" / "processed" / "core_data" / "data" / "csv" / "videos"
ALFABETI_CSV        = _ROOT / "datasets" / "processed" / "core_data" / "data" / "csv" / "alfabeti_keypoints.csv"
PART4_CSV           = _ROOT / "datasets" / "processed" / "core_data" / "video_keypoints.csv"
ANNOTATED_IMAGE_DIR = _ROOT / "datasets" / "raw" / "images" / "Alfabeti_Annotated"
SPLIT_CLIPS_DIR     = _ROOT / "datasets" / "processed" / "clips" / "split_clips"

SEQ_LEN    = 16      # frames per sample window
IMAGE_SIZE = 224     # pixels (model expects 224×224)
STRIDE     = 8       # sliding window stride for video CSVs


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_image(path: Path | None, size: int) -> torch.Tensor:
    """Load and normalise an image → [3, size, size]. Returns zeros on failure."""
    if path is None or not path.exists():
        return torch.zeros(3, size, size)
    img = cv2.imread(str(path))
    if img is None:
        return torch.zeros(3, size, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return (t - _MEAN) / _STD


def _keypoints_bbox(kp: np.ndarray, img_w: int = 640, img_h: int = 640) -> list[float]:
    """Approximate xyxy bounding box from raw (x,y) pixel positions."""
    if kp.size == 0:
        return [0.0, 0.0, float(img_w), float(img_h)]
    xs = kp[:, 0]
    ys = kp[:, 1]
    pad = 20.0
    return [
        float(max(xs.min() - pad, 0)),
        float(max(ys.min() - pad, 0)),
        float(min(xs.max() + pad, img_w)),
        float(min(ys.max() + pad, img_h)),
    ]


def _letter_from_video_filename(name: str) -> str | None:
    """Extract letter from '01_A.csv', '04_Ç.csv', etc."""
    m = re.match(r"^\d+_(.+)\.csv$", name, re.IGNORECASE)
    if not m:
        return None
    token = m.group(1)
    for letter in ALBANIAN_LETTERS:
        if token.upper() == letter.upper():
            return letter
    return None


@dataclass
class DataSample:
    image: torch.Tensor
    keypoints: torch.Tensor
    letter_index: torch.Tensor
    bbox: torch.Tensor


# ── Dataset ───────────────────────────────────────────────────────────────────
class AlbslDataset(Dataset[DataSample]):
    """
    Loads all available AlbSL keypoint data and exposes fixed-length windows.

    Priority order of sources:
      1. data/csv/videos/NN_LETTER.csv  (all 36 letters, richest)
      2. data/csv/alfabeti_keypoints.csv (static images)
      3. video_keypoints.csv (part4 output — if present)
    """

    def __init__(
        self,
        seq_len: int = SEQ_LEN,
        image_size: int = IMAGE_SIZE,
        stride: int = STRIDE,
        split: str = "train",   # "train" | "val"  (90/10 split)
        seed: int = 42,
        video_csv_dir: Path = VIDEO_CSV_DIR,
        alfabeti_csv: Path = ALFABETI_CSV,
        part4_csv: Path = PART4_CSV,
    ) -> None:
        self.seq_len    = seq_len
        self.image_size = image_size
        self._samples: list[dict] = []

        self._load_video_csvs(video_csv_dir, stride)
        self._load_alfabeti_csv(alfabeti_csv)
        self._load_part4_csv(part4_csv, stride)

        # 90/10 train/val split (deterministic)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self._samples))
        cut = int(0.9 * len(idx))
        chosen = idx[:cut] if split == "train" else idx[cut:]
        self._samples = [self._samples[i] for i in chosen]

        if not self._samples:
            raise RuntimeError(
                "AlbslDataset: no samples loaded. "
                "Check that data/csv/videos/ and/or data/csv/alfabeti_keypoints.csv exist."
            )

        # Sorted unique labels present (for informational purposes)
        labels = sorted({s["letter"] for s in self._samples})
        print(
            f"[AlbslDataset] {split}: {len(self._samples)} samples, "
            f"{len(labels)} letters: {labels}"
        )

    # ── Source loaders ────────────────────────────────────────────────────────
    def _load_video_csvs(self, csv_dir: Path, stride: int) -> None:
        if not csv_dir.exists():
            return
        for csv_path in sorted(csv_dir.glob("*.csv")):
            letter = _letter_from_video_filename(csv_path.name)
            if letter is None or letter not in LETTER_TO_IDX:
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue

            # Build feature matrix (N_frames × 123)
            missing = [c for c in VIDEO_FEAT_COLS if c not in df.columns]
            if missing:
                continue
            feat = df[VIDEO_FEAT_COLS].fillna(0.0).to_numpy(dtype=np.float32)
            n = len(feat)

            # Representative image: look for letter subfolder in split_clips_annotated
            img_path = self._find_clip_frame(letter)

            # Sliding windows
            for start in range(0, max(n - self.seq_len + 1, 1), stride):
                window = feat[start: start + self.seq_len]
                if len(window) < self.seq_len:
                    window = np.vstack([
                        window,
                        np.zeros((self.seq_len - len(window), 123), dtype=np.float32),
                    ])
                self._samples.append({
                    "letter":   letter,
                    "features": window,          # (seq_len, 123)
                    "img_path": img_path,
                    "source":   "video_csv",
                })

    def _load_alfabeti_csv(self, csv_path: Path) -> None:
        if not csv_path.exists():
            return
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return

        for _, row in df.iterrows():
            letter = str(row.get("letter", "")).strip().upper()
            if letter not in LETTER_TO_IDX:
                continue  # skip UNKNOWN

            # Extract 63-d normalised xyz and zero-pad to 123
            try:
                xyz63 = np.array([row[c] for c in _ALFA_LM_COLS], dtype=np.float32)
            except KeyError:
                continue
            feat = np.zeros((self.seq_len, 123), dtype=np.float32)
            feat[:, :63] = xyz63  # broadcast same pose to all seq positions

            # Find source image
            img_file = str(row.get("image_file", ""))
            img_path = self._find_annotated_image(letter, img_file)

            self._samples.append({
                "letter":   letter,
                "features": feat,
                "img_path": img_path,
                "source":   "alfabeti",
            })

    def _load_part4_csv(self, csv_path: Path, stride: int) -> None:
        """Load video_keypoints.csv produced by part4_video_pipeline.py."""
        if not csv_path.exists():
            return
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return

        # Build 123-d from part4 columns.
        # Use whichever hand is detected (prefer right for consistency with v2 inference).
        right_norm_cols = [
            f"right_lm{i}_{ax}_norm"
            for i in range(21) for ax in ("x", "y", "z")
        ]
        right_bone_cols = [
            f"right_{bn}_{ax}"
            for bn in [f"bone_{p}_{c}" for p, c in [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),
                (0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20),
            ]]
            for ax in ("dx", "dy", "dz")
        ]
        left_norm_cols = [
            f"left_lm{i}_{ax}_norm"
            for i in range(21) for ax in ("x", "y", "z")
        ]
        left_bone_cols = [
            f"left_{bn}_{ax}"
            for bn in [f"bone_{p}_{c}" for p, c in [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),
                (0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20),
            ]]
            for ax in ("dx", "dy", "dz")
        ]
        p4_right = right_norm_cols + right_bone_cols
        p4_left = left_norm_cols + left_bone_cols
        missing = [c for c in p4_right if c not in df.columns] + [c for c in p4_left if c not in df.columns]
        if missing:
            return  # part4 CSV schema mismatch — skip silently

        for video_file, grp in df.groupby("video_file"):
            letter = str(grp["letter"].iloc[0]).strip().upper()
            if letter not in LETTER_TO_IDX:
                continue
            right_det = grp.get("hand_detected_right", pd.Series([True] * len(grp))).astype(bool).to_numpy()
            left_det = grp.get("hand_detected_left", pd.Series([False] * len(grp))).astype(bool).to_numpy()
            right_feat = grp[p4_right].fillna(0.0).to_numpy(dtype=np.float32)
            left_feat = grp[p4_left].fillna(0.0).to_numpy(dtype=np.float32)
            feat = np.where(right_det[:, None], right_feat, left_feat)
            feat = np.where((~right_det[:, None]) & (~left_det[:, None]), right_feat, feat)
            n = len(feat)
            img_path = self._find_clip_frame(letter)
            for start in range(0, max(n - self.seq_len + 1, 1), stride):
                window = feat[start: start + self.seq_len]
                if len(window) < self.seq_len:
                    window = np.vstack([
                        window,
                        np.zeros((self.seq_len - len(window), 123), dtype=np.float32),
                    ])
                self._samples.append({
                    "letter":   letter,
                    "features": window,
                    "img_path": img_path,
                    "source":   "part4",
                })

    # ── Image resolution helpers ──────────────────────────────────────────────
    def _find_annotated_image(self, letter: str, img_file: str) -> Path | None:
        """Look up annotated image in Alfabeti_Annotated/{letter}/."""
        for sub in [letter, letter.upper(), "_UNKNOWN", "Alfabeti"]:
            p = ANNOTATED_IMAGE_DIR / sub / img_file
            if p.exists():
                return p
        return None

    def _find_clip_frame(self, letter: str) -> Path | None:
        """Find first annotated frame image for a letter from split_clips_annotated."""
        ann_root = _ROOT / "datasets" / "processed" / "clips" / "split_clips_annotated" / letter
        if ann_root.exists():
            vids = sorted(ann_root.glob("*.mp4"))
            if vids:
                return vids[0]  # caller will handle video → frame extraction
        # Fallback: annotated image subdir
        for sub in [letter, letter.upper()]:
            d = ANNOTATED_IMAGE_DIR / sub
            if d.exists():
                imgs = sorted(d.glob("*.jpg")) + sorted(d.glob("*.jpeg")) + sorted(d.glob("*.png"))
                if imgs:
                    return imgs[0]
        return None

    # ── Dataset protocol ──────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> DataSample:
        s = self._samples[idx]
        letter  = s["letter"]
        letter_idx = LETTER_TO_IDX[letter]
        feats   = torch.from_numpy(s["features"])          # (seq_len, 123)
        img_path = s["img_path"]

        # ── Image ─────────────────────────────────────────────────────────
        image = _load_image(img_path, self.image_size)     # (3, H, W)

        # ── Approximate bbox from first-frame xy coords ────────────────────
        # Use WRIST + TIP landmarks (indices 0,4,8,12,16,20) from feat[0]
        # LM_COLS ordering: WRIST_x WRIST_y WRIST_z THUMB_CMC_x ...
        # Indices 0,1 = WRIST_x, WRIST_y; stride 3
        frame0 = s["features"][0]  # (123,)
        xy_raw = frame0[:63].reshape(21, 3)[:, :2] * self.image_size  # rough pixel scale
        bbox = _keypoints_bbox(xy_raw, self.image_size, self.image_size)
        bbox_t = torch.tensor(bbox, dtype=torch.float32)

        return DataSample(
            image=image,
            keypoints=feats,
            letter_index=torch.tensor(letter_idx, dtype=torch.long),
            bbox=bbox_t,
        )


# ── Collate & DataLoader ──────────────────────────────────────────────────────
def collate_batch(batch: list[DataSample]) -> dict[str, torch.Tensor]:
    return {
        "image":        torch.stack([b.image        for b in batch], dim=0),
        "keypoints":    torch.stack([b.keypoints    for b in batch], dim=0),
        "letter_index": torch.stack([b.letter_index for b in batch], dim=0),
        "bbox":         torch.stack([b.bbox         for b in batch], dim=0),
    }


def build_loader(
    batch_size: int = 4,
    length: int = 128,   # kept for API compat — ignored when real data exists
    split: str = "train",
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Build train or val DataLoader from all available real data."""
    ds = AlbslDataset(split=split, seed=seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
    )
