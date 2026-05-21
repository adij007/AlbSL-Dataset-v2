"""AlbSL live recognition app v2.

Single-file script with three subcommands:

  python albsl_app_v2.py
  python albsl_app_v2.py live --weights outputs/albsl_mlp.pt
  python albsl_app_v2.py diagnose
  python albsl_app_v2.py train --out outputs/albsl_mlp.pt --epochs 50

  Running with no subcommand is the same as ``live`` (webcam + MLP weights).

Key fixes vs. v1
---------------
- Identifies and reports class imbalance across ALL labeled sources.
- Trains a real classifier on ``data/keypoints/*.npz`` + ``alfabeti_keypoints.h5``
  using the exact same 123-d feature used at inference time (21*3 normalized xyz
  plus 20*3 dihedral features).
- Applies canonical hand normalization and chirality mirroring identically at
  training and inference, so left-hand signers are mapped into right-hand space.
- Live mode shows the 21-joint skeleton overlay (right=green, left=cyan), top-3
  predictions with confidence, a red frame border when no hand is detected, and
  says UNCERTAIN when top-1 is below ``--pred-min-conf`` and not clearly ahead of
  runner-up (see ``--pred-margin``).
- Press R to start a 3-second countdown then record 30 frames of keypoints into
  ``keypoints.h5`` under the currently selected label.

Controls inside the video window
--------------------------------
  L / K        cycle selected label forward/backward
  R            start 3s countdown, then record 30 frames
  SPACE        append predicted top-1 (same confidence rules as main ``pred=`` line)
  BACKSPACE    delete last letter
  ENTER        commit current word (printed to console, buffer cleared)
  C            clear word buffer
  Q            quit
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import threading
import time
from collections import Counter, deque
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Protocol, Sequence, Tuple

import cv2
import h5py
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from confirmed_csv_io import iter_confirmed_csv_records, iter_coordinates_csv_records

BaseOptions = mp_tasks.BaseOptions

DEFAULT_CONFIRMED_CSV = Path("datasets/csv_dataset/confirmed_labels.csv")

# --- Constants --------------------------------------------------------------

ALBANIAN_LETTERS: List[str] = [
    "A", "B", "C", "Ç", "D", "Dh", "E", "Ë", "F", "G", "Gj", "H", "I", "J", "K",
    "L", "Ll", "M", "N", "Nj", "O", "P", "Q", "R", "Rr", "S", "Sh", "T", "Th",
    "U", "V", "X", "Xh", "Y", "Z", "Zh",
]
NO_SIGN_LABEL = "No-Sign"
TEMPORAL_LABELS: List[str] = ALBANIAN_LETTERS + [NO_SIGN_LABEL]
LETTER_TO_IDX: Dict[str, int] = {l: i for i, l in enumerate(ALBANIAN_LETTERS)}
TEMPORAL_LETTER_TO_IDX: Dict[str, int] = {l: i for i, l in enumerate(TEMPORAL_LABELS)}

# Hand skeleton bones: (parent, child) for the 21-joint MediaPipe hand model.
HAND_DRAW: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
]

HAND_BONES: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

KEY_DIM = 21 * 3 + 20 * 3  # 123
LANDMARK_DIM = 21 * 3
FUSION_IMAGE_SIZE = 224
DEFAULT_SEQUENCE_LEN = 8
DEFAULT_SEQUENCE_STRIDE = 2
DEFAULT_MIN_VALID_FRAMES = 4
IDLE_MOTION_STD = 0.01


@dataclass
class HandLandmarkFrame:
    xyz: np.ndarray
    normalized_xyz: np.ndarray
    is_left: bool
    score: float
    side: str


@dataclass
class TemporalSample:
    sequence: np.ndarray
    label: str
    source: str

# --- Feature engineering (shared between training and inference) ------------


def _rodrigues(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
    a = from_vec / (np.linalg.norm(from_vec) + 1e-8)
    b = to_vec / (np.linalg.norm(to_vec) + 1e-8)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-8:
        return np.eye(3, dtype=np.float32)
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float32,
    )
    return np.eye(3, dtype=np.float32) + vx + (vx @ vx) * ((1.0 - c) / (s * s))


def normalize_hand_landmarks(
    xyz: np.ndarray,
    is_left: bool = False,
    align_orientation: bool = True,
) -> np.ndarray:
    """
    Normalize MediaPipe 21x3 hand landmarks for translation and scale invariance.

    1. Mirror left hands into right-hand space for a single canonical chirality.
    2. Center the wrist (landmark 0) at the origin.
    3. Scale by the landmark bounding-box diagonal.
    4. Optionally align MCP9 toward +Y for mild rotation stability.
    """
    if not np.any(xyz):
        return xyz.astype(np.float32, copy=True)
    out = xyz.astype(np.float32, copy=True)
    if is_left:
        out[:, 0] *= -1.0
    out -= out[0]
    diag = float(np.linalg.norm(out.max(axis=0) - out.min(axis=0)))
    if diag > 1e-8:
        out /= diag
    if align_orientation and np.linalg.norm(out[9]) > 1e-8:
        R = _rodrigues(out[9], np.array([0.0, 1.0, 0.0], dtype=np.float32))
        out = (R @ out.T).T
        signed = float(np.cross(out[5] - out[0], out[17] - out[0])[2])
        if signed < 0.0:
            out[:, 0] *= -1.0
    return out.astype(np.float32)


def canonical_normalize_hand(xyz: np.ndarray, is_left: bool = False) -> np.ndarray:
    """Backward-compatible alias used throughout the existing app."""
    return normalize_hand_landmarks(xyz, is_left=is_left, align_orientation=True)


def _bone_dirs(xyz: np.ndarray) -> np.ndarray:
    out = np.zeros((len(HAND_BONES), 3), dtype=np.float32)
    for i, (p, c) in enumerate(HAND_BONES):
        v = xyz[c] - xyz[p]
        out[i] = v / (np.linalg.norm(v) + 1e-8)
    return out


def dihedral_features(xyz: np.ndarray) -> np.ndarray:
    dirs = _bone_dirs(xyz)
    out = np.zeros((20, 3), dtype=np.float32)
    for i in range(1, len(dirs) - 1):
        n1 = np.cross(dirs[i - 1], dirs[i])
        n2 = np.cross(dirs[i], dirs[i + 1])
        n1 /= np.linalg.norm(n1) + 1e-8
        n2 /= np.linalg.norm(n2) + 1e-8
        theta = math.atan2(
            float(np.dot(np.cross(n1, n2), dirs[i])),
            float(np.dot(n1, n2)),
        )
        out[i] = np.array(
            [theta, float(np.dot(dirs[i - 1], dirs[i])), float(np.dot(dirs[i], dirs[i + 1]))],
            dtype=np.float32,
        )
    return out


def build_feature(xyz_21x3: np.ndarray, is_left: bool = False) -> np.ndarray:
    """Single 123-d feature used at BOTH training and inference time."""
    if not np.any(xyz_21x3):
        return np.zeros(KEY_DIM, dtype=np.float32)
    normalized = canonical_normalize_hand(xyz_21x3, is_left=is_left)
    normalized = np.clip(normalized, -1.0, 1.0)
    angles = dihedral_features(normalized)
    return np.concatenate([normalized.reshape(-1), angles.reshape(-1)], axis=0).astype(np.float32)


def augment_landmark_sequence(
    sequence: np.ndarray,
    noise_std: float = 0.01,
    max_rotation_deg: float = 15.0,
    scale_range: Tuple[float, float] = (0.95, 1.05),
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Runtime augmentation for normalized landmark sequences.

    Applies light 3D jitter, minor camera-angle rotations, and scale variation.
    The input is expected to be [T, 21, 3] or [21, 3].
    """
    if rng is None:
        rng = np.random.default_rng()
    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim == 2:
        seq = seq[None, ...]
    out = seq.astype(np.float32, copy=True)
    if not np.any(out):
        return out

    out += rng.normal(0.0, noise_std, size=out.shape).astype(np.float32)

    yaw = math.radians(float(rng.uniform(-max_rotation_deg, max_rotation_deg)))
    pitch = math.radians(float(rng.uniform(-0.5 * max_rotation_deg, 0.5 * max_rotation_deg)))
    roll = math.radians(float(rng.uniform(-0.5 * max_rotation_deg, 0.5 * max_rotation_deg)))

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    rot = rz @ ry @ rx
    out = np.einsum("ij,tkj->tki", rot, out, optimize=True)

    scale = float(rng.uniform(scale_range[0], scale_range[1]))
    out *= scale
    return out.astype(np.float32)


def _augment_landmark_batch(
    batch: torch.Tensor,
    enabled_mask: Optional[torch.Tensor] = None,
    noise_std: float = 0.01,
    max_rotation_deg: float = 15.0,
    scale_range: Tuple[float, float] = (0.95, 1.05),
) -> torch.Tensor:
    """Vectorized GPU-friendly augmentation for [B, T, 21, 3] landmark batches."""
    if batch.numel() == 0:
        return batch
    out = batch.clone()
    B = out.size(0)
    dev = out.device
    out = out + torch.randn_like(out) * noise_std

    yaw = (torch.rand(B, device=dev) * 2.0 - 1.0) * math.radians(max_rotation_deg)
    pitch = (torch.rand(B, device=dev) * 2.0 - 1.0) * math.radians(max_rotation_deg * 0.5)
    roll = (torch.rand(B, device=dev) * 2.0 - 1.0) * math.radians(max_rotation_deg * 0.5)

    cy, sy = yaw.cos(), yaw.sin()
    cp, sp = pitch.cos(), pitch.sin()
    cr, sr = roll.cos(), roll.sin()
    zeros = torch.zeros(B, device=dev)
    ones = torch.ones(B, device=dev)

    rz = torch.stack(
        [
            torch.stack([cy, -sy, zeros], dim=1),
            torch.stack([sy, cy, zeros], dim=1),
            torch.stack([zeros, zeros, ones], dim=1),
        ],
        dim=1,
    )
    ry = torch.stack(
        [
            torch.stack([cp, zeros, sp], dim=1),
            torch.stack([zeros, ones, zeros], dim=1),
            torch.stack([-sp, zeros, cp], dim=1),
        ],
        dim=1,
    )
    rx = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=1),
            torch.stack([zeros, cr, -sr], dim=1),
            torch.stack([zeros, sr, cr], dim=1),
        ],
        dim=1,
    )
    rot = torch.bmm(torch.bmm(rz, ry), rx)
    out = torch.einsum("bij,btnj->btni", rot, out)

    scale = (
        torch.rand(B, device=dev) * (scale_range[1] - scale_range[0]) + scale_range[0]
    ).view(B, 1, 1, 1)
    out = out * scale

    if enabled_mask is not None:
        return torch.where(enabled_mask.view(B, 1, 1, 1), out, batch)
    return out


class _HandLandmarkerVideoApi(Protocol):
    """Subset of MediaPipe :class:`HandLandmarker` used by this module."""

    def detect_for_video(self, image: Any, timestamp_ms: int) -> Any: ...


def extract_hand_landmarks_from_frame(
    frame_bgr: np.ndarray,
    hand_landmarker: _HandLandmarkerVideoApi,
    timestamp_ms: int,
) -> List[HandLandmarkFrame]:
    """
    Extract raw and normalized 21x3 MediaPipe hand landmarks from a single frame.

    This helper can be called for webcam frames, decoded video frames, or an
    image sequence. Each returned item includes translation/scale-invariant
    normalized landmarks ready for temporal training.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
    out: List[HandLandmarkFrame] = []
    hand_lms = result.hand_landmarks if result.hand_landmarks is not None else []
    for h_idx in range(len(hand_lms)):
        pts_raw = hand_lms[h_idx]
        hand_xyz = np.array([[p.x, p.y, p.z] for p in pts_raw], dtype=np.float32)
        hand_side = "right"
        hand_conf = 0.0
        if result.handedness and h_idx < len(result.handedness) and result.handedness[h_idx]:
            raw_side = result.handedness[h_idx][0].display_name.lower()
            hand_conf = float(result.handedness[h_idx][0].score)
            hand_side = "left" if raw_side == "right" else "right"
        is_left = hand_side == "left"
        out.append(
            HandLandmarkFrame(
                xyz=hand_xyz,
                normalized_xyz=normalize_hand_landmarks(hand_xyz, is_left=is_left),
                is_left=is_left,
                score=hand_conf,
                side=hand_side,
            )
        )
    return out


def extract_hand_landmarks_sequence(
    frames_bgr: Sequence[np.ndarray],
    hand_landmarker: _HandLandmarkerVideoApi,
    start_timestamp_ms: int = 0,
    frame_period_ms: int = 33,
) -> List[List[HandLandmarkFrame]]:
    """Convenience wrapper for processing a decoded image sequence into landmarks."""
    out: List[List[HandLandmarkFrame]] = []
    timestamp_ms = int(start_timestamp_ms)
    for frame in frames_bgr:
        out.append(extract_hand_landmarks_from_frame(frame, hand_landmarker, timestamp_ms))
        timestamp_ms += int(frame_period_ms)
    return out


def _is_plausible_hand_detection(
    hand_xyz: np.ndarray,
    score: float,
    min_score: float,
    min_area: float,
    max_area: float,
) -> bool:
    if hand_xyz.shape != (21, 3):
        return False
    if not np.isfinite(hand_xyz).all():
        return False
    if float(score) < float(min_score):
        return False
    xy = hand_xyz[:, :2]
    wh = xy.max(axis=0) - xy.min(axis=0)
    area = float(max(0.0, wh[0]) * max(0.0, wh[1]))
    if area < float(min_area) or area > float(max_area):
        return False
    tips = hand_xyz[[4, 8, 12, 16, 20], :2]
    wrist = hand_xyz[0, :2]
    spread = float(np.mean(np.linalg.norm(tips - wrist, axis=1)))
    if spread < 0.045:
        return False
    palm_len = float(np.linalg.norm(hand_xyz[9, :2] - hand_xyz[0, :2]))
    if palm_len < 0.07:
        return False
    w_dim = float(max(1e-6, wh[0]))
    h_dim = float(max(1e-6, wh[1]))
    aspect = max(w_dim, h_dim) / min(w_dim, h_dim)
    if aspect > 4.5:
        return False
    mcps = hand_xyz[[5, 9, 13, 17], :2]
    diffs = np.diff(mcps, axis=0)
    if not np.all(np.linalg.norm(diffs, axis=1) > 0.005):
        return False
    return True


def _is_face_like_cluster(
    hand_xyz_a: np.ndarray,
    hand_xyz_b: np.ndarray,
    overlap_iou_threshold: float = 0.55,
) -> bool:
    """Drop tightly overlapping duplicates (often face/skin double-fires)."""
    if hand_xyz_a.shape != (21, 3) or hand_xyz_b.shape != (21, 3):
        return False
    def _bbox(p: np.ndarray) -> Tuple[float, float, float, float]:
        xy = p[:, :2]
        return (
            float(xy[:, 0].min()),
            float(xy[:, 1].min()),
            float(xy[:, 0].max()),
            float(xy[:, 1].max()),
        )
    ax1, ay1, ax2, ay2 = _bbox(hand_xyz_a)
    bx1, by1, bx2, by2 = _bbox(hand_xyz_b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = max(1e-6, area_a + area_b - inter)
    iou = inter / union
    return float(iou) >= float(overlap_iou_threshold)


def _hand_xy_bbox_area(hand_xyz: np.ndarray) -> float:
    xy = hand_xyz[:, :2]
    wh = xy.max(axis=0) - xy.min(axis=0)
    return float(max(0.0, wh[0]) * max(0.0, wh[1]))


def _is_collapsed_landmark_cluster(hand_xyz: np.ndarray) -> bool:
    """Reject fingertip/skin blobs where all joints sit in a tiny cluster."""
    if hand_xyz.shape != (21, 3) or not np.isfinite(hand_xyz).all():
        return True
    xy = hand_xyz[:, :2]
    span = float(np.max(xy.max(axis=0) - xy.min(axis=0)))
    centroid = xy.mean(axis=0)
    max_r = float(np.max(np.linalg.norm(xy - centroid, axis=1)))
    palm = float(np.linalg.norm(hand_xyz[9, :2] - hand_xyz[0, :2]))
    return span < 0.055 or max_r < 0.045 or palm < 0.07


def _hands_are_duplicate_detections(
    hand_xyz_a: np.ndarray,
    hand_xyz_b: np.ndarray,
    *,
    overlap_iou_threshold: float = 0.38,
    centroid_dist: float = 0.14,
) -> bool:
    """True when two detections likely describe the same physical hand."""
    if _is_face_like_cluster(hand_xyz_a, hand_xyz_b, overlap_iou_threshold=overlap_iou_threshold):
        return True
    ca = hand_xyz_a[:, :2].mean(axis=0)
    cb = hand_xyz_b[:, :2].mean(axis=0)
    if float(np.linalg.norm(ca - cb)) < float(centroid_dist):
        return True
    area_a = _hand_xy_bbox_area(hand_xyz_a)
    area_b = _hand_xy_bbox_area(hand_xyz_b)
    if area_a <= 0.0 or area_b <= 0.0:
        return False
    ratio = min(area_a, area_b) / max(area_a, area_b)
    if ratio < 0.32 and float(np.linalg.norm(ca - cb)) < 0.22:
        return True
    return False


def filter_hand_landmark_frames(
    frames: Sequence[HandLandmarkFrame],
    *,
    min_score: float = 0.50,
    min_area: float = 0.006,
    max_area: float = 0.55,
) -> List[HandLandmarkFrame]:
    """Drop collapsed phantoms and overlapping duplicate hand hypotheses."""
    plausible: List[HandLandmarkFrame] = []
    for hand_frame in frames:
        if _is_collapsed_landmark_cluster(hand_frame.xyz):
            continue
        if not _is_plausible_hand_detection(
            hand_frame.xyz,
            hand_frame.score,
            min_score=min_score,
            min_area=min_area,
            max_area=max_area,
        ):
            continue
        plausible.append(hand_frame)
    if len(plausible) <= 1:
        return plausible
    kept: List[HandLandmarkFrame] = []
    for cand in sorted(plausible, key=lambda x: float(x.score), reverse=True):
        if any(_hands_are_duplicate_detections(cand.xyz, prev.xyz) for prev in kept):
            continue
        kept.append(cand)
    return kept


def _smooth_hand_landmarks(
    current_xyz: np.ndarray,
    prev_xyz: Optional[np.ndarray],
    alpha: float,
) -> np.ndarray:
    a = float(max(0.0, min(1.0, alpha)))
    if prev_xyz is None or prev_xyz.shape != current_xyz.shape:
        return current_xyz.astype(np.float32, copy=True)
    return (a * prev_xyz + (1.0 - a) * current_xyz).astype(np.float32)


# --- Data loading -----------------------------------------------------------


def _pick_hand_from_h5(right: np.ndarray, left: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Select the hand with non-zero data; returns (xyz, is_left)."""
    if np.any(right):
        return right, False
    if np.any(left):
        return left, True
    return right, False


def _h5_require_dataset(parent: h5py.Group, key: str) -> h5py.Dataset:
    """Resolve ``parent[key]`` as an :class:`h5py.Dataset` for static typing."""
    node = parent[key]
    if not isinstance(node, h5py.Dataset):
        raise TypeError(f"H5 {key!r}: expected Dataset, got {type(node).__name__}")
    return node


def load_labeled_samples(
    keypoints_dir: Path,
    alfabeti_h5: Path,
    legacy_h5: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, Counter[str]]:
    """Gather (features, labels, per-label counts) from all labeled sources."""
    features: List[np.ndarray] = []
    labels: List[str] = []

    # (1) Per-clip NPZ files from data/keypoints/
    if keypoints_dir.exists():
        for npz in sorted(keypoints_dir.glob("*.npz")):
            stem = npz.stem
            letter = stem.split("_", 1)[-1]
            if letter not in LETTER_TO_IDX:
                continue
            data = np.load(npz, allow_pickle=True)
            has_dual = "xyz_right" in data.files and "xyz_left" in data.files
            if has_dual:
                xyz_r = data["xyz_right"]
                xyz_l = data["xyz_left"]
                conf_r = data["conf_right"]
                conf_l = data["conf_left"]
                n_frames = xyz_r.shape[0]
                for t in range(n_frames):
                    r_ok = conf_r[t].max() >= 0.5
                    l_ok = conf_l[t].max() >= 0.5
                    if r_ok and conf_r[t].max() >= conf_l[t].max():
                        features.append(build_feature(xyz_r[t], is_left=False))
                        labels.append(letter)
                    elif l_ok:
                        features.append(build_feature(xyz_l[t], is_left=True))
                        labels.append(letter)
            else:
                xyz = data["xyz"]
                conf = data["conf"]
                n_frames = xyz.shape[0]
                for t in range(n_frames):
                    if conf[t].max() < 0.5:
                        continue
                    # Old-format clips already stored normalized xyz; re-extract anyway
                    # for consistency via build_feature (which re-normalizes).
                    features.append(build_feature(xyz[t], is_left=False))
                    labels.append(letter)

    # (2) Alfabeti per-image H5
    if alfabeti_h5.exists():
        with h5py.File(alfabeti_h5, "r") as f:
            xyz = _h5_require_dataset(f, "xyz")[...]
            det = _h5_require_dataset(f, "detected")[...]
            lbls = [x.decode() if isinstance(x, bytes) else x for x in _h5_require_dataset(f, "labels")[...]]
            for i in range(xyz.shape[0]):
                if not det[i]:
                    continue
                letter = str(lbls[i])
                if letter not in LETTER_TO_IDX:
                    continue
                features.append(build_feature(xyz[i], is_left=False))
                labels.append(letter)

    # (3) Legacy keypoints.h5 (only usable if its labels are per-letter).
    if legacy_h5 is not None and legacy_h5.exists():
        with h5py.File(legacy_h5, "r") as f:
            lbls = [x.decode() if isinstance(x, bytes) else x for x in _h5_require_dataset(f, "labels")[...]]
            right = _h5_require_dataset(f, "right_hand")[...]
            left = _h5_require_dataset(f, "left_hand")[...]
            unique = set(lbls)
            if unique.issubset(set(ALBANIAN_LETTERS)):
                for i in range(len(lbls)):
                    xyz, is_left = _pick_hand_from_h5(right[i], left[i])
                    if not np.any(xyz):
                        continue
                    features.append(build_feature(xyz, is_left=is_left))
                    labels.append(str(lbls[i]))
            else:
                logger.warning(
                    "legacy keypoints.h5 has non-letter labels {} — skipped for training.",
                    sorted(unique),
                )

    if not features:
        return np.zeros((0, KEY_DIM), dtype=np.float32), np.array([], dtype=object), Counter()

    X = np.stack(features, axis=0).astype(np.float32)
    Y = np.array(labels, dtype=object)
    return X, Y, Counter(labels)


def _select_device(preferred: str = "auto") -> torch.device:
    choice = preferred.lower()
    if torch.cuda.is_available() and choice in ("auto", "cuda"):
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available() and choice in ("auto", "xpu"):
        return torch.device("xpu")
    if choice == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")
    if choice == "xpu" and (not hasattr(torch, "xpu") or not torch.xpu.is_available()):
        logger.warning("XPU requested but not available; falling back to CPU.")
    return torch.device("cpu")


def _autocast_context(device: torch.device) -> Any:
    if device.type == "cuda":
        return torch.cuda.amp.autocast()
    return nullcontext()


def _configure_training_backend(device: torch.device) -> None:
    """Enable safe backend speed-ups for GPU training."""
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True


def _pad_sequence_window(seq: np.ndarray, target_len: int) -> np.ndarray:
    if seq.shape[0] == 0:
        return np.zeros((target_len, 21, 3), dtype=np.float32)
    if seq.shape[0] >= target_len:
        return seq[:target_len].astype(np.float32, copy=False)
    pad = np.repeat(seq[-1:,:,:], target_len - seq.shape[0], axis=0)
    return np.concatenate([seq, pad], axis=0).astype(np.float32)


def _valid_frame_count(seq: np.ndarray) -> int:
    flat = seq.reshape(seq.shape[0], -1)
    return int(np.count_nonzero(np.linalg.norm(flat, axis=1) > 1e-6))


def _append_windowed_samples(
    samples: List[TemporalSample],
    frames: np.ndarray,
    label: str,
    source: str,
    window_size: int,
    stride: int,
    min_valid_frames: int,
    relabel_sparse_as_idle: bool = True,
) -> None:
    if frames.size == 0:
        return
    max_start = max(0, frames.shape[0] - window_size)
    starts = list(range(0, max_start + 1, max(1, stride)))
    if not starts:
        starts = [0]
    if starts[-1] != max_start:
        starts.append(max_start)

    for start in starts:
        window = _pad_sequence_window(frames[start : start + window_size], window_size)
        valid_count = _valid_frame_count(window)
        out_label = label
        if relabel_sparse_as_idle and valid_count < min_valid_frames:
            out_label = NO_SIGN_LABEL
        samples.append(TemporalSample(sequence=window, label=out_label, source=source))


def _build_idle_sequences(count: int, window_size: int, rng: np.random.Generator) -> List[TemporalSample]:
    out: List[TemporalSample] = []
    for idx in range(max(0, count)):
        seq = np.zeros((window_size, 21, 3), dtype=np.float32)
        anchor = rng.normal(0.0, IDLE_MOTION_STD, size=(21, 3)).astype(np.float32)
        for t in range(window_size):
            jitter = rng.normal(0.0, IDLE_MOTION_STD, size=(21, 3)).astype(np.float32)
            seq[t] = (0.85 * seq[t - 1] if t > 0 else 0.0) + 0.35 * anchor + jitter
        seq[:, 0, :] = 0.0
        out.append(TemporalSample(sequence=seq, label=NO_SIGN_LABEL, source=f"idle-synth-{idx:05d}"))
    return out


def _normalize_feature63_frame(frame63: np.ndarray) -> np.ndarray:
    xyz = np.asarray(frame63, dtype=np.float32).reshape(21, 3)
    return normalize_hand_landmarks(xyz, is_left=False)


def load_temporal_training_samples(
    keypoints_dir: Path,
    alfabeti_h5: Path,
    legacy_h5: Optional[Path] = None,
    *,
    confirmed_csv: Optional[Path] = None,
    coordinates_csv: Optional[Path] = None,
    window_size: int = DEFAULT_SEQUENCE_LEN,
    stride: int = DEFAULT_SEQUENCE_STRIDE,
    min_valid_frames: int = DEFAULT_MIN_VALID_FRAMES,
    idle_ratio: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray, Counter[str]]:
    """
    Adapted temporal loader for the current repo layout.

    - Video clips in datasets/processed/core_data/data/keypoints/*.npz become
      sliding windows of normalized 21x3 landmarks.
    - Static H5 samples become pseudo-sequences by repeating the frame.
    - Sparse / missing-detection windows are routed to the explicit No-Sign class.
    """
    samples: List[TemporalSample] = []
    rng = np.random.default_rng(0)

    if keypoints_dir.exists():
        for npz in sorted(keypoints_dir.glob("*.npz")):
            letter = npz.stem.split("_", 1)[-1]
            if letter not in LETTER_TO_IDX:
                continue
            data = np.load(npz, allow_pickle=True)
            frames: List[np.ndarray] = []
            if "xyz_right" in data.files and "xyz_left" in data.files:
                xyz_r = data["xyz_right"]
                xyz_l = data["xyz_left"]
                conf_r = data["conf_right"]
                conf_l = data["conf_left"]
                for t in range(int(xyz_r.shape[0])):
                    r_score = float(np.max(conf_r[t])) if conf_r.size else 0.0
                    l_score = float(np.max(conf_l[t])) if conf_l.size else 0.0
                    if r_score < 0.5 and l_score < 0.5:
                        frames.append(np.zeros((21, 3), dtype=np.float32))
                        continue
                    if r_score >= l_score:
                        frames.append(normalize_hand_landmarks(xyz_r[t], is_left=False))
                    else:
                        frames.append(normalize_hand_landmarks(xyz_l[t], is_left=True))
            elif "xyz" in data.files:
                xyz = data["xyz"]
                conf = data["conf"] if "conf" in data.files else None
                for t in range(int(xyz.shape[0])):
                    score = float(np.max(conf[t])) if conf is not None else 1.0
                    if score < 0.5:
                        frames.append(np.zeros((21, 3), dtype=np.float32))
                    else:
                        frames.append(normalize_hand_landmarks(xyz[t], is_left=False))
            if frames:
                clip = np.stack(frames, axis=0).astype(np.float32)
                _append_windowed_samples(
                    samples,
                    clip,
                    label=letter,
                    source=npz.name,
                    window_size=window_size,
                    stride=stride,
                    min_valid_frames=min_valid_frames,
                    relabel_sparse_as_idle=True,
                )

    if alfabeti_h5.exists():
        with h5py.File(alfabeti_h5, "r") as f:
            xyz = _h5_require_dataset(f, "xyz")[...]
            det = _h5_require_dataset(f, "detected")[...]
            lbls = [x.decode() if isinstance(x, bytes) else x for x in _h5_require_dataset(f, "labels")[...]]
            for i in range(int(xyz.shape[0])):
                if not det[i]:
                    continue
                letter = str(lbls[i])
                if letter not in LETTER_TO_IDX:
                    continue
                frame = normalize_hand_landmarks(xyz[i], is_left=False)
                seq = np.repeat(frame[None, :, :], window_size, axis=0)
                samples.append(TemporalSample(sequence=seq, label=letter, source=f"alfabeti:{i}"))

    if legacy_h5 is not None and legacy_h5.exists():
        with h5py.File(legacy_h5, "r") as f:
            lbls = (
                [x.decode() if isinstance(x, bytes) else x for x in _h5_require_dataset(f, "labels")[...]]
                if "labels" in f
                else []
            )
            if "features" in f and lbls:
                feats = _h5_require_dataset(f, "features")[...]
                for i, letter in enumerate(lbls):
                    if letter not in LETTER_TO_IDX:
                        continue
                    feat_arr = np.asarray(feats[i], dtype=np.float32)
                    if feat_arr.ndim == 1 and feat_arr.size >= LANDMARK_DIM:
                        frame = _normalize_feature63_frame(feat_arr[:LANDMARK_DIM])
                        seq = np.repeat(frame[None, :, :], window_size, axis=0)
                        samples.append(TemporalSample(sequence=seq, label=str(letter), source=f"legacy-feat:{i}"))
                    elif feat_arr.ndim == 2 and feat_arr.shape[1] >= LANDMARK_DIM:
                        clip = np.stack(
                            [_normalize_feature63_frame(row[:LANDMARK_DIM]) for row in feat_arr],
                            axis=0,
                        ).astype(np.float32)
                        _append_windowed_samples(
                            samples,
                            clip,
                            label=str(letter),
                            source=f"legacy-feat-seq:{i}",
                            window_size=window_size,
                            stride=stride,
                            min_valid_frames=min_valid_frames,
                            relabel_sparse_as_idle=False,
                        )
            elif {"right_hand", "left_hand", "labels"}.issubset(set(f.keys())):
                right = _h5_require_dataset(f, "right_hand")[...]
                left = _h5_require_dataset(f, "left_hand")[...]
                for i, letter in enumerate(lbls):
                    if letter not in LETTER_TO_IDX:
                        continue
                    xyz, is_left = _pick_hand_from_h5(right[i], left[i])
                    if not np.any(xyz):
                        continue
                    frame = normalize_hand_landmarks(xyz, is_left=is_left)
                    seq = np.repeat(frame[None, :, :], window_size, axis=0)
                    samples.append(TemporalSample(sequence=seq, label=str(letter), source=f"legacy-hand:{i}"))

    if confirmed_csv is not None and confirmed_csv.exists():
        for i, rec in enumerate(iter_confirmed_csv_records(confirmed_csv)):
            if rec.label not in LETTER_TO_IDX:
                continue
            frame = normalize_hand_landmarks(rec.landmarks, is_left=False)
            seq = np.repeat(frame[None, :, :], window_size, axis=0)
            samples.append(TemporalSample(sequence=seq, label=rec.label, source=f"confirmed:{i}"))

    if coordinates_csv is not None and coordinates_csv.exists():
        for i, rec in enumerate(iter_coordinates_csv_records(coordinates_csv)):
            if rec.label not in LETTER_TO_IDX:
                continue
            frame = normalize_hand_landmarks(rec.landmarks, is_left=False)
            seq = np.repeat(frame[None, :, :], window_size, axis=0)
            samples.append(TemporalSample(sequence=seq, label=rec.label, source=f"coordinates:{i}"))

    real_samples = sum(1 for s in samples if s.label != NO_SIGN_LABEL)
    idle_samples = sum(1 for s in samples if s.label == NO_SIGN_LABEL)
    target_idle = max(idle_samples, int(round(real_samples * max(0.0, idle_ratio))))
    if idle_samples < target_idle:
        samples.extend(_build_idle_sequences(target_idle - idle_samples, window_size, rng))

    if not samples:
        return (
            np.zeros((0, window_size, 21, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            Counter(),
        )

    X = np.stack([s.sequence for s in samples], axis=0).astype(np.float32)
    Y_labels = np.array([s.label for s in samples], dtype=object)
    Y = np.array([TEMPORAL_LETTER_TO_IDX[str(label)] for label in Y_labels], dtype=np.int64)
    return X, Y, Counter(Y_labels.tolist())


# --- Model ------------------------------------------------------------------


class LetterMLP(nn.Module):
    def __init__(self, in_dim: int = KEY_DIM, hidden: int = 256, num_classes: int = len(ALBANIAN_LETTERS)) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalLandmarkLSTM(nn.Module):
    def __init__(
        self,
        frame_dim: int = LANDMARK_DIM,
        hidden_dim: int = 192,
        num_layers: int = 2,
        dropout: float = 0.25,
        num_classes: int = len(TEMPORAL_LABELS),
    ) -> None:
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Linear(frame_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.motion_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.reshape(x.size(0), x.size(1), -1)
        tokens = self.frame_encoder(x)
        temporal_out, _ = self.temporal(tokens)
        attn = torch.softmax(self.attn(temporal_out).squeeze(-1), dim=1)
        pooled = torch.sum(temporal_out * attn.unsqueeze(-1), dim=1)
        motion = self.motion_head(temporal_out[:, -1] - temporal_out[:, 0])
        fused = torch.cat([pooled, motion], dim=-1)
        return self.classifier(fused)


class LandmarkSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray) -> None:
        self.sequences = torch.from_numpy(sequences.astype(np.float32, copy=False))
        self.labels = torch.from_numpy(labels.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


def _stratified_train_val_split(
    labels: np.ndarray,
    train_fraction: float = 0.8,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(labels) <= 1:
        idx = np.arange(len(labels), dtype=np.int64)
        return idx, idx
    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        if len(idx) == 1:
            train_idx.extend(idx.tolist())
            continue
        cut = int(round(len(idx) * train_fraction))
        cut = min(max(1, cut), len(idx) - 1)
        train_idx.extend(idx[:cut].tolist())
        val_idx.extend(idx[cut:].tolist())
    if not val_idx:
        perm = rng.permutation(len(labels))
        cut = min(max(1, int(round(len(labels) * train_fraction))), len(labels) - 1)
        train_idx = perm[:cut].tolist()
        val_idx = perm[cut:].tolist()
    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)


def _build_class_weights(counts: Counter[str]) -> torch.Tensor:
    n_classes = len(TEMPORAL_LABELS)
    class_counts = np.ones(n_classes, dtype=np.float32)
    for idx, letter in enumerate(TEMPORAL_LABELS):
        class_counts[idx] = max(1.0, float(counts.get(letter, 0)))
    weights = class_counts.sum() / (n_classes * class_counts)
    weights[TEMPORAL_LETTER_TO_IDX[NO_SIGN_LABEL]] *= 0.75
    return torch.tensor(weights, dtype=torch.float32)


def _make_sequence_loader(
    sequences: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    workers: int,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    worker_count = max(0, int(workers))
    kwargs: Dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": shuffle,
        "num_workers": worker_count,
        "pin_memory": device.type == "cuda",
    }
    if worker_count > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(LandmarkSequenceDataset(sequences, labels), **kwargs)


# --- Subcommand: diagnose ---------------------------------------------------


def cmd_diagnose(args: argparse.Namespace) -> None:
    args.keypoints_dir = _resolve_model_path(args.keypoints_dir)
    args.alfabeti_h5 = _resolve_model_path(args.alfabeti_h5)
    if args.legacy_h5 is not None:
        args.legacy_h5 = _resolve_model_path(args.legacy_h5)
    print("=" * 72)
    print("AlbSL dataset diagnostics")
    print("=" * 72)

    if args.legacy_h5 and args.legacy_h5.exists():
        with h5py.File(args.legacy_h5, "r") as f:
            lbls = [x.decode() if isinstance(x, bytes) else x for x in _h5_require_dataset(f, "labels")[...]]
            c = Counter(lbls)
            print(f"\n[legacy] {args.legacy_h5}: {len(lbls)} samples, {len(c)} unique labels")
            for k, v in c.most_common(10):
                print(f"  {k:16s} {v}")
            if len(c) == 1:
                print("  !!  legacy file has only ONE unique label — USELESS for letter training")
    else:
        print(f"\n[legacy] {args.legacy_h5} not found — skipping")

    if args.alfabeti_h5.exists():
        with h5py.File(args.alfabeti_h5, "r") as f:
            lbls = [x.decode() if isinstance(x, bytes) else x for x in _h5_require_dataset(f, "labels")[...]]
            det = _h5_require_dataset(f, "detected")[...]
            c = Counter([lbls[i] for i in range(len(lbls)) if det[i]])
            print(f"\n[alfabeti] {args.alfabeti_h5}: {int(det.sum())}/{len(lbls)} detected, {len(c)} labels")
            for k, v in sorted(c.items()):
                print(f"  {k:16s} {v}")

    if args.keypoints_dir.exists():
        counts: Counter[str] = Counter()
        total_frames = 0
        for npz in sorted(args.keypoints_dir.glob("*.npz")):
            letter = npz.stem.split("_", 1)[-1]
            if letter not in LETTER_TO_IDX:
                continue
            data = np.load(npz, allow_pickle=True)
            n = int(data["xyz"].shape[0]) if "xyz" in data.files else 0
            if "conf" in data.files:
                n = int((data["conf"].max(axis=1) >= 0.5).sum())
            counts[letter] += n
            total_frames += n
        print(f"\n[clips] {args.keypoints_dir}: {total_frames} usable frames across {len(counts)} letters")
        for k in ALBANIAN_LETTERS:
            v = counts.get(k, 0)
            # Use ASCII to avoid cp1252 terminal encoding issues on Windows.
            bar = "#" * min(40, v // max(1, total_frames // 400))
            print(f"  {k:4s} {v:5d} {bar}")

    X, Y, c = load_labeled_samples(args.keypoints_dir, args.alfabeti_h5, legacy_h5=args.legacy_h5)
    print(f"\n[combined] training set: {X.shape[0]} samples, {len(c)} letters")
    missing = [l for l in ALBANIAN_LETTERS if l not in c]
    if missing:
        print(f"  !!  missing letters in training data: {missing}")

    # Sanity check: ensure feature vectors are finite and in-range.
    if X.size:
        print(
            f"  feature stats: min={X.min():+.3f} max={X.max():+.3f} "
            f"mean={X.mean():+.3f} std={X.std():+.3f}"
        )


# --- Subcommand: train ------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> None:
    args.keypoints_dir = _resolve_model_path(args.keypoints_dir)
    args.alfabeti_h5 = _resolve_model_path(args.alfabeti_h5)
    if args.legacy_h5 is not None:
        args.legacy_h5 = _resolve_model_path(args.legacy_h5)
    if getattr(args, "confirmed_csv", None) is not None:
        args.confirmed_csv = _resolve_model_path(args.confirmed_csv)
    if getattr(args, "coordinates_csv", None) is not None:
        args.coordinates_csv = _resolve_model_path(args.coordinates_csv)
    args.out = _resolve_model_path(args.out)
    window_size = int(max(5, min(10, args.sequence_len)))
    min_valid_frames = min(window_size, max(1, int(args.min_valid_frames)))
    X, y_idx, counts = load_temporal_training_samples(
        args.keypoints_dir,
        args.alfabeti_h5,
        legacy_h5=args.legacy_h5,
        confirmed_csv=getattr(args, "confirmed_csv", None),
        coordinates_csv=getattr(args, "coordinates_csv", None),
        window_size=window_size,
        stride=max(1, int(args.sequence_stride)),
        min_valid_frames=min_valid_frames,
        idle_ratio=float(args.idle_ratio),
    )
    if len(X) == 0:
        logger.error("No labeled samples found. Run extraction first.")
        return
    present = sorted(str(k) for k in counts.keys())
    device = _select_device(args.device)
    _configure_training_backend(device)
    if device.type == "cpu":
        logger.warning(
            "Training on CPU. For best temporal-model training speed use --device cuda on a CUDA-capable GPU."
        )
    logger.info(
        "training samples: {} windows across {} classes on device {}",
        len(X),
        len(present),
        device,
    )

    tr_idx, va_idx = _stratified_train_val_split(y_idx, train_fraction=0.8, seed=0)
    X_train, y_train = X[tr_idx], y_idx[tr_idx]
    X_val, y_val = X[va_idx], y_idx[va_idx]
    train_loader = _make_sequence_loader(
        X_train,
        y_train,
        batch_size=int(args.batch_size),
        shuffle=True,
        device=device,
        workers=int(args.workers),
    )
    val_loader = _make_sequence_loader(
        X_val,
        y_val,
        batch_size=int(args.batch_size),
        shuffle=False,
        device=device,
        workers=int(args.workers),
    )

    class_weights = _build_class_weights(counts).to(device)
    model = TemporalLandmarkLSTM(
        frame_dim=LANDMARK_DIM,
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.layers),
        dropout=float(args.dropout),
        num_classes=len(TEMPORAL_LABELS),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    no_sign_idx = TEMPORAL_LETTER_TO_IDX[NO_SIGN_LABEL]
    best_acc = 0.0
    best_state: Optional[dict[str, Any]] = None

    for epoch in range(int(args.epochs)):
        model.train()
        epoch_loss = 0.0
        total_train = 0
        total_correct = 0
        for xb, yb in train_loader:
            xb = xb.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            yb = yb.to(device=device, non_blocking=device.type == "cuda")
            if not args.no_augment:
                xb = _augment_landmark_batch(xb, enabled_mask=(yb != no_sign_idx))
            opt.zero_grad(set_to_none=True)
            with _autocast_context(device):
                logits = model(xb)
                loss = criterion(logits, yb)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            epoch_loss += float(loss.detach().cpu()) * int(yb.size(0))
            pred = logits.detach().argmax(dim=-1)
            total_train += int(yb.size(0))
            total_correct += int((pred == yb).sum().item())

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
                yb = yb.to(device=device, non_blocking=device.type == "cuda")
                logits_v = model(xb)
                pred_v = logits_v.argmax(dim=-1)
                val_total += int(yb.size(0))
                val_correct += int((pred_v == yb).sum().item())
        acc = float(val_correct / max(val_total, 1))
        train_acc = float(total_correct / max(total_train, 1))

        logger.info(
            "epoch {:3d}  loss={:.4f}  train_acc={:.3f}  val_acc={:.3f}",
            epoch + 1,
            epoch_loss / max(total_train, 1),
            train_acc,
            acc,
        )
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": best_state or model.state_dict(),
        "classes": TEMPORAL_LABELS,
        "model_type": "temporal_lstm",
        "sequence_len": window_size,
        "sequence_stride": int(args.sequence_stride),
        "frame_dim": LANDMARK_DIM,
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.layers),
        "dropout": float(args.dropout),
        "val_acc": best_acc,
        "normalization": "wrist_center_bbox_scale_canonical",
    }
    torch.save(payload, str(args.out))
    logger.info("saved best model to {} (val_acc={:.3f})", args.out, best_acc)


# --- Live app ---------------------------------------------------------------


def _ensure_hand_model(models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    dst = models_dir / "hand_landmarker.task"
    if not dst.exists():
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, dst)
    return dst


def _draw_hand(
    frame: np.ndarray,
    pts_px: List[Tuple[int, int]],
    color: Tuple[int, int, int],
    label: str = "",
    conf: float = 0.0,
) -> None:
    """Draw skeleton, red landmark dots, and joint index numbers (like the reference diagram)."""
    # Bones — coloured lines
    for a, b in HAND_DRAW:
        if a < len(pts_px) and b < len(pts_px):
            cv2.line(frame, pts_px[a], pts_px[b], color, 2, cv2.LINE_AA)
    # Landmark dots + index numbers
    for idx, (cx, cy) in enumerate(pts_px):
        cv2.circle(frame, (cx, cy), 5, (0, 0, 200), -1, cv2.LINE_AA)   # red dot
        cv2.circle(frame, (cx, cy), 5, (255, 255, 255), 1, cv2.LINE_AA)  # white outline
        cv2.putText(
            frame, str(idx), (cx + 5, cy - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 0), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, str(idx), (cx + 5, cy - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 255), 1, cv2.LINE_AA,
        )
    # Hand label + confidence near wrist (landmark 0)
    if pts_px and label:
        wx, wy = pts_px[0]
        tag = f"{label} {conf:.2f}"
        cv2.putText(frame, tag, (wx + 6, wy + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, tag, (wx + 6, wy + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _put_text(frame: np.ndarray, text: str, org: Tuple[int, int], color: Tuple[int, int, int], scale: float = 0.65) -> None:
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def _safe_letter(letter: str) -> str:
    return {"Ç": "Cc", "Ë": "Ee"}.get(letter, letter)


def _append_recording_to_h5(
    h5_path: Path,
    features: np.ndarray,
    label: str,
    source: str,
) -> None:
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    new_samples = features.shape[0]
    mode = "a" if h5_path.exists() else "w"
    with h5py.File(h5_path, mode) as f:
        if "features" not in f:
            f.create_dataset(
                "features",
                data=features,
                maxshape=(None, KEY_DIM),
                chunks=(min(64, new_samples), KEY_DIM),
                dtype=np.float32,
                compression="gzip",
                compression_opts=4,
            )
            dt = h5py.string_dtype(encoding="utf-8")
            f.create_dataset("labels", data=np.array([label] * new_samples, dtype=object), maxshape=(None,), dtype=dt)
            f.create_dataset("sources", data=np.array([source] * new_samples, dtype=object), maxshape=(None,), dtype=dt)
        else:
            feats_ds = _h5_require_dataset(f, "features")
            lbl_ds = _h5_require_dataset(f, "labels")
            src_ds = _h5_require_dataset(f, "sources")
            old = feats_ds.shape[0]
            feats_ds.resize(old + new_samples, axis=0)
            feats_ds[old:] = features
            lbl_ds.resize(old + new_samples, axis=0)
            lbl_ds[old:] = np.array([label] * new_samples, dtype=object)
            src_ds.resize(old + new_samples, axis=0)
            src_ds[old:] = np.array([source] * new_samples, dtype=object)


def _hand_bbox(frame_shape: Tuple[int, int], xyz_image: np.ndarray, pad: float = 0.2) -> Tuple[int, int, int, int]:
    H, W = frame_shape[:2]
    pts = xyz_image[:, :2]
    if not np.any(pts):
        cx, cy = W / 2.0, H / 2.0
        side = min(H, W) * 0.4
        return (
            int(cx - side / 2),
            int(cy - side / 2),
            int(cx + side / 2),
            int(cy + side / 2),
        )
    xs = pts[:, 0] * W
    ys = pts[:, 1] * H
    x1 = max(0, int(xs.min() - pad * (xs.max() - xs.min())))
    y1 = max(0, int(ys.min() - pad * (ys.max() - ys.min())))
    x2 = min(W - 1, int(xs.max() + pad * (xs.max() - xs.min())))
    y2 = min(H - 1, int(ys.max() + pad * (ys.max() - ys.min())))
    if x2 - x1 < 10 or y2 - y1 < 10:
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        side = 100
        x1, x2 = int(cx - side / 2), int(cx + side / 2)
        y1, y2 = int(cy - side / 2), int(cy + side / 2)
    return x1, y1, x2, y2


def _preprocess_crop(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = frame_bgr
    crop = cv2.resize(crop, (FUSION_IMAGE_SIZE, FUSION_IMAGE_SIZE))
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def _landmark_refs_from_json_root(raw: dict[str, Any]) -> Dict[str, np.ndarray]:
    """Parse flat ``albsl_landmarks.json`` or unified bundle ``letters.static``."""
    out: Dict[str, np.ndarray] = {}
    if not isinstance(raw, dict):
        return out
    letters_block = raw.get("letters")
    if isinstance(letters_block, dict):
        static = letters_block.get("static")
        if isinstance(static, dict):
            for k, v in static.items():
                try:
                    arr = np.array(v, dtype=np.float32)
                except Exception:
                    continue
                ks = str(k)
                if arr.shape == (21, 3) and ks in LETTER_TO_IDX:
                    out[ks] = arr
            if out:
                return out
    for k, v in raw.items():
        if str(k) in ("letters", "schema_version", "generated_at", "words"):
            continue
        try:
            arr = np.array(v, dtype=np.float32)
        except Exception:
            continue
        ks = str(k)
        if arr.shape == (21, 3) and ks in LETTER_TO_IDX:
            out[ks] = arr
    return out


def _load_landmark_refs(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return _landmark_refs_from_json_root(raw)


def _resolve_json_path(path_like: Path) -> Path:
    """Resolve JSON paths when cwd is repo root, Script/, or elsewhere."""
    candidates = [
        path_like,
        Path("..") / path_like,
        ROOT / path_like,
    ]
    for c in candidates:
        if c.exists():
            return c
    if path_like.suffix.lower() != ".json":
        alt = Path(str(path_like) + ".json")
        for base in (Path("."), Path(".."), ROOT):
            if (base / alt).exists():
                return base / alt
        cwd_alt = Path(path_like.name + ".json")
        for base in (Path("."), Path(".."), ROOT):
            if (base / cwd_alt).exists():
                return base / cwd_alt
    # Common fallback names for this app.
    if path_like.name.lower() in ("albsl_landmarks", "albsl_landmarks.json"):
        for rel in (
            Path("datasets/processed/assets/albsl_landmarks.json"),
            Path("datasets/json_dataset/coordinates.json"),
        ):
            for base in (Path("."), Path(".."), ROOT):
                p = base / rel
                if p.exists():
                    return p
    return path_like


def _resolve_model_path(path_like: Path) -> Path:
    """Resolve model path when launching from repo root or Script dir."""
    if path_like.is_absolute():
        return path_like
    if path_like.exists():
        return path_like.resolve()
    parent_alt = Path("..") / path_like
    if parent_alt.exists():
        return parent_alt.resolve()
    root_alt = ROOT / path_like
    if root_alt.exists():
        return root_alt.resolve()
    # Missing relative path: anchor to repo root so writes do not land under cwd (e.g. Script/).
    return (ROOT / path_like).resolve()


def _template_match_letter(
    live_xyz: np.ndarray,
    refs: Dict[str, np.ndarray],
    max_dist: float = 0.16,
) -> Tuple[Optional[str], float]:
    if not refs:
        return None, float("inf")
    best_letter: Optional[str] = None
    best_dist = float("inf")
    for letter, ref in refs.items():
        d = float(np.mean(np.linalg.norm(live_xyz - ref, axis=1)))
        if d < best_dist:
            best_dist = d
            best_letter = letter
    if best_letter is not None and best_dist <= max_dist:
        return best_letter, best_dist
    return None, best_dist


def _parse_dynamic_template_mapping(mapping: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for letter, payload in mapping.items():
        if not isinstance(payload, dict):
            continue
        if letter not in LETTER_TO_IDX:
            continue
        tmpl = payload.get("template")
        if tmpl is None:
            continue
        try:
            arr = np.array(tmpl, dtype=np.float32)
        except Exception:
            continue
        if arr.ndim == 2 and arr.shape[1] == 63:
            out[letter] = {
                "template": arr,
                "sequence_len": int(payload.get("sequence_len", arr.shape[0])),
                "max_dist": float(payload.get("max_dist", 0.12)),
                "motion_weight": float(payload.get("motion_weight", 0.25)),
            }
    return out


def _dynamic_templates_from_json_root(raw: dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Standalone ``albsl_dynamic_templates.json`` or unified ``letters.dynamic``."""
    if not isinstance(raw, dict):
        return {}
    letters_block = raw.get("letters")
    if isinstance(letters_block, dict):
        dyn = letters_block.get("dynamic")
        if isinstance(dyn, dict) and dyn:
            got = _parse_dynamic_template_mapping(dyn)
            if got:
                return got
    return _parse_dynamic_template_mapping(raw)


def _load_dynamic_templates(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _dynamic_templates_from_json_root(raw)


def _resample_seq(seq: np.ndarray, n: int) -> np.ndarray:
    if seq.shape[0] == n:
        return seq.astype(np.float32, copy=False)
    src = np.linspace(0.0, 1.0, seq.shape[0], dtype=np.float32)
    dst = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out = np.zeros((n, seq.shape[1]), dtype=np.float32)
    for j in range(seq.shape[1]):
        out[:, j] = np.interp(dst, src, seq[:, j])
    return out


def _dynamic_match_letter(
    feat_history: Deque[np.ndarray],
    templates: Dict[str, Dict[str, Any]],
    max_dist: float = 0.12,
) -> Tuple[Optional[str], float]:
    if not templates:
        return None, float("inf")
    if len(feat_history) < 8:
        return None, float("inf")
    best_letter: Optional[str] = None
    best_dist = float("inf")
    hist = np.stack(list(feat_history), axis=0).astype(np.float32)
    for letter, payload in templates.items():
        tmpl = payload.get("template")
        if not isinstance(tmpl, np.ndarray) or tmpl.ndim != 2:
            continue
        t = int(tmpl.shape[0])
        if hist.shape[0] < t:
            chunk = hist
        else:
            chunk = hist[-t:]
        cand = _resample_seq(chunk, t)
        # Support template feature dims different from runtime feature dims.
        # Current runtime uses 123-d (xyz+angles); templates may be 63-d (xyz only).
        d_dim = min(int(cand.shape[1]), int(tmpl.shape[1]))
        if d_dim <= 0:
            continue
        base_dist = float(np.mean(np.abs(cand[:, :d_dim] - tmpl[:, :d_dim])))
        # Add a weak trajectory term for dynamic signs (motion-sensitive letters).
        cand_delta = np.diff(cand[:, :d_dim], axis=0)
        tmpl_delta = np.diff(tmpl[:, :d_dim], axis=0)
        motion_dist = float(np.mean(np.abs(cand_delta - tmpl_delta))) if cand_delta.size else 0.0
        weight = float(payload.get("motion_weight", 0.25))
        d = (1.0 - weight) * base_dist + weight * motion_dist
        if d < best_dist:
            best_dist = d
            best_letter = letter
    if best_letter is not None:
        per_letter_max = float(templates[best_letter].get("max_dist", max_dist))
        if best_dist <= min(max_dist, per_letter_max):
            return best_letter, best_dist
    return None, best_dist


def _suggest_word_from_letters(letter_seq: List[str], words_dict: List[Dict[str, object]]) -> Optional[str]:
    if not letter_seq or not words_dict:
        return None
    best_word: Optional[str] = None
    best_score = -1.0
    n = len(letter_seq)
    for item in words_dict:
        letters = item.get("letters")
        word = item.get("word")
        if not isinstance(letters, list) or not isinstance(word, str):
            continue
        target = [str(x) for x in letters]
        if not target:
            continue
        prefix_len = 0
        for i in range(min(n, len(target))):
            if letter_seq[i] != target[i]:
                break
            prefix_len += 1
        # prefix quality + mild length penalty for smoother realtime suggestions.
        score = (prefix_len / max(1, n)) - 0.03 * abs(len(target) - n)
        if score > best_score and prefix_len > 0:
            best_score = score
            best_word = word
    return best_word


def _build_words_index(words_dict: List[Dict[str, object]]) -> Dict[str, object]:
    exact: Dict[Tuple[str, ...], str] = {}
    items: List[Tuple[List[str], str]] = []
    for item in words_dict:
        letters = item.get("letters")
        word = item.get("word")
        if not isinstance(letters, list) or not isinstance(word, str):
            continue
        seq = [str(x) for x in letters]
        if not seq:
            continue
        exact[tuple(seq)] = word
        items.append((seq, word))
    return {"exact": exact, "items": items}


def _suggest_word_from_index(letter_seq: List[str], index: Dict[str, object]) -> Optional[str]:
    items = index.get("items", [])
    if not letter_seq or not isinstance(items, list):
        return None
    best_word: Optional[str] = None
    best_score = -1.0
    n = len(letter_seq)
    for target, word in items:
        if not isinstance(target, list) or not isinstance(word, str):
            continue
        prefix_len = 0
        for i in range(min(n, len(target))):
            if letter_seq[i] != target[i]:
                break
            prefix_len += 1
        score = (prefix_len / max(1, n)) - 0.03 * abs(len(target) - n)
        if score > best_score and prefix_len > 0:
            best_score = score
            best_word = word
    return best_word


def _match_word_from_index(letter_seq: List[str], index: Dict[str, object]) -> Optional[str]:
    if not letter_seq:
        return None
    exact = index.get("exact", {})
    if isinstance(exact, dict):
        hit = exact.get(tuple(letter_seq))
        if isinstance(hit, str):
            return hit
    return _suggest_word_from_index(letter_seq, index)


def _append_confirmed_coordinates_csv(
    csv_path: Path,
    letter: str,
    xyz_norm: np.ndarray,
    confidence: float,
    source: str,
    *,
    session_id: Optional[str] = None,
    frame_ts_ms: Optional[int] = None,
    model_source: Optional[str] = None,
    top2_margin: Optional[float] = None,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "timestamp",
        "label",
        "confidence",
        "source",
        "session_id",
        "frame_ts_ms",
        "model_source",
        "top2_margin",
        "landmarks_63",
    ] + [f"lm{i}_{ax}" for i in range(21) for ax in ("x", "y", "z")]
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "label": letter,
        "confidence": round(float(confidence), 6),
        "source": source,
        "session_id": str(session_id or ""),
        "frame_ts_ms": int(frame_ts_ms) if frame_ts_ms is not None else "",
        "model_source": str(model_source or ""),
        "top2_margin": round(float(top2_margin), 6) if top2_margin is not None else "",
        "landmarks_63": json.dumps(xyz_norm.reshape(-1).astype(np.float32).tolist(), ensure_ascii=False),
    }
    flat = xyz_norm.reshape(-1)
    for i in range(21):
        base = i * 3
        row[f"lm{i}_x"] = round(float(flat[base + 0]), 6)
        row[f"lm{i}_y"] = round(float(flat[base + 1]), 6)
        row[f"lm{i}_z"] = round(float(flat[base + 2]), 6)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _flush_confirmed_csv_queue(csv_path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "timestamp",
        "label",
        "confidence",
        "source",
        "session_id",
        "frame_ts_ms",
        "model_source",
        "top2_margin",
        "landmarks_63",
    ] + [f"lm{i}_{ax}" for i in range(21) for ax in ("x", "y", "z")]
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    rows.clear()


def _auto_append_letter(
    stable_state: Dict[str, Any],
    candidate: Optional[str],
    confidence: float,
    threshold: float,
    repeat_cooldown_ms: int,
    now_ms: int,
) -> Optional[str]:
    if candidate is None or confidence < threshold:
        stable_state["candidate"] = None
        stable_state["count"] = 0
        return None
    prev = stable_state.get("candidate")
    if prev == candidate:
        stable_state["count"] = int(stable_state.get("count", 0)) + 1
    else:
        stable_state["candidate"] = candidate
        stable_state["count"] = 1
    hold_frames = int(stable_state.get("hold_frames", 6))
    if int(stable_state.get("count", 0)) < hold_frames:
        return None
    last_letter = str(stable_state.get("last_emit_letter", ""))
    last_ms = int(stable_state.get("last_emit_ms", -10**9))
    if last_letter == candidate and (now_ms - last_ms) < repeat_cooldown_ms:
        return None
    stable_state["last_emit_letter"] = candidate
    stable_state["last_emit_ms"] = now_ms
    stable_state["count"] = 0
    return candidate


def _dynamic_letter_set(templates: Dict[str, Dict[str, Any]]) -> set[str]:
    return {k for k in templates.keys()}


def _parse_letter_set(raw: str) -> set[str]:
    out: set[str] = set()
    for token in str(raw).split(","):
        t = token.strip()
        if not t:
            continue
        for letter in ALBANIAN_LETTERS:
            if letter.lower() == t.lower():
                out.add(letter)
                break
    return out


def _motion_energy_from_xyz_history(xyz_history: Deque[np.ndarray], frames: int = 8) -> float:
    if len(xyz_history) < 3:
        return 0.0
    arr = np.stack(list(xyz_history)[-max(3, int(frames)):], axis=0).astype(np.float32)
    delta = np.diff(arr, axis=0)
    return float(np.mean(np.linalg.norm(delta.reshape(delta.shape[0], -1), axis=1)))


def _boost_focus_letters(
    top3: List[Tuple[str, float]],
    focus_letters: set[str],
    motion_energy: float,
    base_boost: float,
    motion_boost: float,
) -> List[Tuple[str, float]]:
    if not top3 or not focus_letters:
        return top3
    boosted: List[Tuple[str, float]] = []
    for letter, prob in top3:
        p = float(prob)
        if letter in focus_letters:
            p += float(base_boost) + float(motion_boost) * float(motion_energy)
        boosted.append((letter, p))
    boosted.sort(key=lambda x: x[1], reverse=True)
    s = sum(p for _, p in boosted)
    if s > 1e-8:
        boosted = [(l, float(p / s)) for l, p in boosted]
    return boosted[:3]


def _match_word_from_letters(letter_seq: List[str], words_dict: List[Dict[str, object]]) -> Optional[str]:
    if not letter_seq:
        return None
    for item in words_dict:
        letters = item.get("letters")
        word = item.get("word")
        if isinstance(letters, list) and isinstance(word, str):
            if [str(x) for x in letters] == letter_seq:
                return word
    return _suggest_word_from_letters(letter_seq, words_dict)


def _load_words_dictionary(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(raw, dict):
        words = raw.get("words", [])
        if isinstance(words, list):
            return [w for w in words if isinstance(w, dict)]
    return []


def _sequence_from_history(history: Deque[np.ndarray], window_size: int) -> np.ndarray:
    if len(history) == 0:
        return np.zeros((window_size, 21, 3), dtype=np.float32)
    seq = np.stack(list(history)[-window_size:], axis=0).astype(np.float32)
    return _pad_sequence_window(seq, window_size)


def _load_primary_model_checkpoint(
    path: Path,
    device: torch.device,
    fallback_sequence_len: int,
) -> Tuple[Optional[torch.nn.Module], Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "model_type": "frame_mlp",
        "classes": ALBANIAN_LETTERS,
        "sequence_len": max(1, int(fallback_sequence_len)),
    }
    if not path.exists():
        return None, meta
    payload = torch.load(str(path), map_location=device, weights_only=False)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload

    if isinstance(payload, dict) and payload.get("model_type") == "temporal_lstm":
        classes = [str(x) for x in payload.get("classes", TEMPORAL_LABELS)]
        model = TemporalLandmarkLSTM(
            frame_dim=int(payload.get("frame_dim", LANDMARK_DIM)),
            hidden_dim=int(payload.get("hidden_dim", 192)),
            num_layers=int(payload.get("num_layers", 2)),
            dropout=float(payload.get("dropout", 0.25)),
            num_classes=len(classes),
        ).to(device)
        model.load_state_dict(state, strict=False)
        model.eval()
        meta = {
            "model_type": "temporal_lstm",
            "classes": classes,
            "sequence_len": int(payload.get("sequence_len", fallback_sequence_len)),
        }
        return model, meta

    model = LetterMLP().to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, meta


def _load_landmark_model_checkpoint(path: Path, device: torch.device) -> Tuple[Optional[torch.nn.Module], Optional[Dict[str, Any]]]:
    """
    Load model exported by train_albsl.py -> models/trained/albsl_model_final/model_full.pt.
    Returns (model, lmap_payload) or (None, None) on failure.
    """
    if not path.exists():
        return None, None
    try:
        payload = torch.load(str(path), map_location=device, weights_only=False)
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        lmap = payload.get("lmap", None) if isinstance(payload, dict) else None
        if not isinstance(lmap, dict) or "label_to_id" not in lmap:
            return None, None
        # Lazy import so v2 still runs even if train_albsl deps are unavailable.
        from train_albsl import SignLandmarkModel  # type: ignore

        n_cls = int(max(lmap["label_to_id"].values())) + 1
        lm_model = SignLandmarkModel(n_cls, use_4bit=False).to(device)
        lm_model.load_state_dict(state, strict=False)
        lm_model.eval()
        return lm_model, lmap
    except Exception:
        return None, None


def _top1_is_actionable(
    top3: List[Tuple[str, float]],
    pred_min_conf: float,
    pred_margin: float,
) -> bool:
    """Whether to show the predicted letter (vs UNCERTAIN) or allow SPACE append."""
    if not top3:
        return False
    if top3[0][1] >= pred_min_conf:
        return True
    if len(top3) >= 2:
        return (top3[0][1] - top3[1][1]) >= pred_margin
    return False


def _require_opencv_highgui() -> None:
    """Fail fast if OpenCV was built without HighGUI (typical when opencv-python-headless is installed)."""
    try:
        cv2.namedWindow("__albsl_gui_probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__albsl_gui_probe__")
    except cv2.error as exc:
        mod = getattr(cv2, "__file__", "")
        raise RuntimeError(
            "OpenCV has no GUI backend (cv2.imshow is not available).\n"
            "This usually means ``opencv-python-headless`` is installed, or two OpenCV wheels conflict.\n\n"
            "Fix (PowerShell):\n"
            "  pip uninstall -y opencv-python-headless opencv-contrib-python-headless "
            "opencv-python opencv-contrib-python\n"
            "  pip install \"opencv-python>=4.8,<5\"\n"
            "  pip list | findstr /i opencv\n\n"
            f"Loaded cv2 from: {mod}"
        ) from exc


def cmd_live(args: argparse.Namespace) -> None:
    args.weights = _resolve_model_path(args.weights)
    args.fused_weights = _resolve_model_path(args.fused_weights)
    args.albsl_model = _resolve_model_path(args.albsl_model)
    args.recordings_h5 = _resolve_model_path(args.recordings_h5)
    args.confirmed_csv = _resolve_model_path(args.confirmed_csv)
    _require_opencv_highgui()
    device = _select_device(args.device)
    primary_model: Optional[torch.nn.Module] = None
    primary_meta: Dict[str, Any] = {
        "model_type": "frame_mlp",
        "classes": ALBANIAN_LETTERS,
        "sequence_len": max(5, min(10, int(args.sequence_len))),
    }
    loaded = False
    try:
        primary_model, primary_meta = _load_primary_model_checkpoint(args.weights, device, int(args.sequence_len))
        loaded = primary_model is not None
        if loaded:
            logger.info(
                "loaded {} weights from {} (sequence_len={})",
                primary_meta.get("model_type", "frame_mlp"),
                args.weights,
                primary_meta.get("sequence_len", 1),
            )
        else:
            logger.warning("no trained weights loaded — run `train` first for meaningful predictions")
    except Exception as exc:
        logger.warning("failed to load weights: {} — using fallback behaviour", exc)
    if not loaded:
        primary_model = None

    primary_classes = [str(x) for x in primary_meta.get("classes", ALBANIAN_LETTERS)]
    primary_class_to_idx = {label: i for i, label in enumerate(primary_classes)}
    temporal_loaded = loaded and primary_meta.get("model_type") == "temporal_lstm"
    sequence_len = int(primary_meta.get("sequence_len", args.sequence_len))
    lm_model, lm_payload = _load_landmark_model_checkpoint(args.albsl_model, device)
    lm_loaded = lm_model is not None and lm_payload is not None
    lm_idx_to_letter: Dict[int, str] = {}
    if lm_loaded and isinstance(lm_payload, dict):
        lm_idx_to_letter = {int(v): str(k) for k, v in lm_payload.get("label_to_id", {}).items()}
        logger.info("loaded iterative landmark model from {}", args.albsl_model)

    # Optional fusion checkpoint (outputs/fused_phase3.pt) for stronger inference.
    fusion_model: Optional[torch.nn.Module] = None
    fusion_loaded = False
    if args.fused_weights.exists():
        try:
            from albsl_fusion.model import FusionBatch, build_model

            cfg = type("Cfg", (), {
                "model": type("M", (), {"hidden_dim": 1152, "fusion": type("F", (), {"num_heads": 4})()})(),
                "data": type("D", (), {"num_letters": len(ALBANIAN_LETTERS)})(),
            })()
            fusion_model = build_model(cfg).to(device)
            state_f = torch.load(str(args.fused_weights), map_location=device, weights_only=True)
            fusion_model.load_state_dict(state_f, strict=False)
            fusion_model.eval()
            fusion_loaded = True
            logger.info("loaded fused checkpoint from {}", args.fused_weights)
        except Exception as exc:
            fusion_model = None
            logger.warning("failed to load fused checkpoint {}: {}", args.fused_weights, exc)
    landmarks_json = _resolve_json_path(args.landmarks_json)
    dynamic_templates_json = _resolve_json_path(args.dynamic_templates_json)
    words_dict_json = _resolve_json_path(args.words_dict_json)
    landmark_refs = _load_landmark_refs(landmarks_json)
    dynamic_templates = _load_dynamic_templates(dynamic_templates_json)
    words_dict = _load_words_dictionary(words_dict_json)
    # Optional unified coordinates JSON: words override + static/dynamic templates if assets missing.
    unified_path = _resolve_json_path(args.unified_coords_json) if args.unified_coords_json else None
    unified_payload: Optional[dict[str, Any]] = None
    if unified_path and unified_path.exists():
        try:
            raw_unified = json.loads(unified_path.read_text(encoding="utf-8"))
            if isinstance(raw_unified, dict):
                unified_payload = raw_unified
                words_section = raw_unified.get("words", {})
                if isinstance(words_section, dict):
                    words_list = words_section.get("words", [])
                    if isinstance(words_list, list) and words_list:
                        words_dict = [w for w in words_list if isinstance(w, dict)]
                        logger.info(
                            "loaded {} words from unified coords {}",
                            len(words_dict),
                            unified_path,
                        )
        except Exception as exc:
            logger.warning("failed to load unified coords {}: {}", unified_path, exc)
    if not landmark_refs and isinstance(unified_payload, dict):
        merged = _landmark_refs_from_json_root(unified_payload)
        if merged:
            landmark_refs = merged
            logger.info(
                "loaded {} static letter refs from unified bundle {}",
                len(landmark_refs),
                unified_path,
            )
    if not dynamic_templates and isinstance(unified_payload, dict):
        merged_d = _dynamic_templates_from_json_root(unified_payload)
        if merged_d:
            dynamic_templates = merged_d
            logger.info(
                "loaded {} dynamic templates from unified bundle {}",
                len(dynamic_templates),
                unified_path,
            )
    if landmark_refs:
        logger.info("loaded {} static letter refs from {}", len(landmark_refs), landmarks_json)
    else:
        logger.warning("landmark refs not loaded from {}", landmarks_json)
    if dynamic_templates:
        logger.info("loaded {} dynamic templates from {}", len(dynamic_templates), dynamic_templates_json)
    else:
        logger.warning("dynamic templates not loaded from {}", dynamic_templates_json)
    if words_dict:
        logger.info("loaded {} words from {}", len(words_dict), words_dict_json)
    else:
        logger.warning("word dictionary not loaded from {}", words_dict_json)
    logger.info("confirmed captures will be saved to {}", args.confirmed_csv.resolve())
    startup_status = {
        "primary_model": bool(loaded and primary_model is not None),
        "iterative_model": bool(lm_loaded),
        "fused_model": bool(fusion_loaded),
        "landmark_refs": bool(len(landmark_refs) > 0),
        "dynamic_templates": bool(len(dynamic_templates) > 0),
        "words_dict": bool(len(words_dict) > 0),
    }
    logger.info(
        "startup-status primary={} iterative={} fused={} refs={} dynamic={} words={}",
        startup_status["primary_model"],
        startup_status["iterative_model"],
        startup_status["fused_model"],
        startup_status["landmark_refs"],
        startup_status["dynamic_templates"],
        startup_status["words_dict"],
    )
    if bool(args.strict_startup):
        if not startup_status["primary_model"]:
            raise RuntimeError("--strict-startup: primary model weights are required.")
        if not startup_status["words_dict"]:
            raise RuntimeError("--strict-startup: words dictionary could not be loaded.")
        if not startup_status["dynamic_templates"]:
            raise RuntimeError("--strict-startup: dynamic templates could not be loaded.")

    hand_model = _ensure_hand_model(args.models_dir)
    hand = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(hand_model)),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=float(args.min_hand_detection_confidence),
            min_hand_presence_confidence=float(args.min_hand_presence_confidence),
            min_tracking_confidence=float(args.min_hand_tracking_confidence),
        )
    )

    cap = cv2.VideoCapture(int(args.camera))
    if not cap.isOpened():
        logger.error("cannot open camera {}", args.camera)
        return

    word_buffer: List[str] = []
    selected_label_idx = 0
    last_ts_ms = 0
    t_start = time.time()

    recording_state: Dict[str, Any] = {"mode": "idle", "countdown_until": 0.0, "frames": [], "record_label": None}
    RECORD_FRAME_COUNT = 30
    COUNTDOWN_SECS = 3

    # Small EMA over logits for temporal stability in the UI.
    ema_probs: Optional[np.ndarray] = None
    ema_alpha = 0.7
    # Extra temporal stabilizers (logic only, no UI changes):
    # - keep primary hand assignment stable across frames
    # - require short consensus across recent top-1 predictions
    side_sticky: Optional[str] = None
    prev_center: Optional[np.ndarray] = None
    recent_top1: Deque[Tuple[str, float]] = deque(maxlen=7)  # (letter, prob)
    feat_history: Deque[np.ndarray] = deque(maxlen=24)
    xyz_history: Deque[np.ndarray] = deque(maxlen=max(24, sequence_len * 3))
    auto_state: Dict[str, Any] = {
        "candidate": None,
        "count": 0,
        "hold_frames": max(2, int(args.auto_hold_frames)),
        "last_emit_letter": "",
        "last_emit_ms": -10**9,
    }
    smoothed_by_side: Dict[str, np.ndarray] = {}
    dynamic_letters = _dynamic_letter_set(dynamic_templates)
    requested_focus = _parse_letter_set(args.dynamic_focus_letters)
    # Always include user-priority dynamic letters and discovered template letters.
    focus_letters = set(dynamic_letters) | requested_focus | {"Ë", "Ç", "Sh", "Zh", "Xh"}

    # Detection hysteresis to suppress flicker and reject transient face/skin triggers.
    detect_state: Dict[str, int] = {"on_count": 0, "off_count": 0, "active": 0}
    DETECT_ON_FRAMES = max(1, int(args.detect_on_frames))
    DETECT_OFF_FRAMES = max(1, int(args.detect_off_frames))
    last_detected_word: str = ""
    words_index = _build_words_index(words_dict)
    last_word_key: Optional[Tuple[str, ...]] = None
    cached_suggestion: Optional[str] = None
    cached_exact: Optional[str] = None
    capture_session_id = datetime.now(timezone.utc).strftime("live-%Y%m%d-%H%M%S")
    confirmed_rows_batch: List[Dict[str, object]] = []
    confirm_flush_every = max(1, int(args.confirm_flush_every))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # selfie view
            H, W = frame.shape[:2]

            ts_ms = max(last_ts_ms + 1, int((time.time() - t_start) * 1000))
            last_ts_ms = ts_ms

            # ── Detect up to 2 hands ─────────────────────────────────────────
            # Primary hand selection is temporally stable:
            # score = detection confidence + side stickiness - center-jump penalty.
            xyz = np.zeros((21, 3), dtype=np.float32)
            xyz_norm = np.zeros((21, 3), dtype=np.float32)
            is_left = False
            detected = False
            hand_score = 0.0

            # Per-hand draw info: list of (pts_px, color, side_label, conf)
            hands_draw: List[Tuple[List[Tuple[int, int]], Tuple[int,int,int], str, float]] = []
            best_pts_px: List[Tuple[int, int]] = []

            candidates: List[Tuple[float, HandLandmarkFrame, List[Tuple[int, int]]]] = []
            hand_frames = extract_hand_landmarks_from_frame(frame, hand, ts_ms)
            filtered_frames = filter_hand_landmark_frames(
                hand_frames,
                min_score=float(args.min_hand_score),
                min_area=float(args.min_hand_area),
                max_area=float(args.max_hand_area),
            )
            for hand_frame in filtered_frames:
                h_is_left = hand_frame.is_left
                h_color = (0, 200, 255) if h_is_left else (0, 255, 0)
                pts_px_h = [(int(p[0] * W), int(p[1] * H)) for p in hand_frame.xyz]
                center = hand_frame.xyz[:, :2].mean(axis=0)
                center_penalty = 0.0
                if prev_center is not None:
                    center_penalty = float(np.linalg.norm(center - prev_center))
                sticky_bonus = 0.15 if (side_sticky is not None and side_sticky == hand_frame.side) else 0.0
                side_bias = 0.05 if hand_frame.side == "right" else 0.0
                score = float(hand_frame.score) + sticky_bonus + side_bias - 0.35 * center_penalty
                candidates.append((score, hand_frame, pts_px_h))

            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                _, best_frame, best_pts_px = candidates[0]
                h_color = (0, 200, 255) if best_frame.is_left else (0, 255, 0)
                hands_draw = [(best_pts_px, h_color, best_frame.side.upper(), best_frame.score)]
                prev_side_xyz = smoothed_by_side.get(best_frame.side)
                xyz = _smooth_hand_landmarks(
                    best_frame.xyz,
                    prev_side_xyz,
                    alpha=float(args.landmark_smooth_alpha),
                )
                smoothed_by_side[best_frame.side] = xyz.copy()
                xyz_norm = normalize_hand_landmarks(xyz, is_left=best_frame.is_left)
                is_left = best_frame.is_left
                hand_score = best_frame.score
                # Hysteresis: count consecutive plausible frames before declaring detected.
                detect_state["on_count"] = int(detect_state.get("on_count", 0)) + 1
                detect_state["off_count"] = 0
                if int(detect_state.get("active", 0)) == 1 or detect_state["on_count"] >= DETECT_ON_FRAMES:
                    detect_state["active"] = 1
                    detected = True
                else:
                    detected = False
                side_sticky = best_frame.side
                prev_center = xyz[:, :2].mean(axis=0)
            else:
                detect_state["off_count"] = int(detect_state.get("off_count", 0)) + 1
                detect_state["on_count"] = 0
                if detect_state["off_count"] >= DETECT_OFF_FRAMES:
                    detect_state["active"] = 0
                detected = bool(detect_state.get("active", 0))
                side_sticky = None
                prev_center = None
                smoothed_by_side.clear()

            # --- Prediction --------------------------------------------------
            top3: List[Tuple[str, float]] = []
            idle_detected = False
            idle_prob = 0.0
            if detected and ((loaded and primary_model is not None) or lm_loaded):
                feat = build_feature(xyz, is_left=is_left)
                xyz_history.append(xyz_norm.copy())
                feat_history.append(feat.copy())
                with torch.no_grad():
                    probs_primary: Optional[np.ndarray] = None
                    probs: Optional[np.ndarray] = None
                    if loaded and primary_model is not None:
                        if temporal_loaded:
                            seq_np = _sequence_from_history(xyz_history, sequence_len)
                            x = torch.from_numpy(seq_np).unsqueeze(0).to(device=device, dtype=torch.float32)
                        else:
                            x = torch.from_numpy(feat).unsqueeze(0).to(device=device, dtype=torch.float32)
                        logits = primary_model(x)[0]
                        probs_primary = F.softmax(logits, dim=-1).float().cpu().numpy().astype(np.float32)
                        probs = probs_primary.copy()
                    # Blend MLP/temporal and fusion probabilities when available.
                    probs_lm: Optional[np.ndarray] = None
                    if lm_loaded and lm_model is not None:
                        # train_albsl model expects 63-dim (21*3), while v2 feature is 123-dim.
                        feat63 = feat[:63].astype(np.float32, copy=False)
                        logits_lm = lm_model(
                            torch.from_numpy(feat63).view(1, -1).to(device=device, dtype=torch.float32)
                        )[0]
                        probs_lm = F.softmax(logits_lm, dim=-1).float().cpu().numpy()
                    if fusion_loaded and fusion_model is not None:
                        from albsl_fusion.model import FusionBatch

                        bbox = _hand_bbox(frame.shape, xyz)
                        crop = _preprocess_crop(frame, bbox)
                        fb = FusionBatch(
                            image=torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32),
                            keypoints=torch.from_numpy(feat).view(1, 1, -1).to(device=device, dtype=torch.float32),
                            bbox=torch.tensor([[bbox[0], bbox[1], bbox[2], bbox[3]]], dtype=torch.float32, device=device),
                            letter_index=torch.zeros(1, dtype=torch.long, device=device),
                        )
                        logits_f = fusion_model(fb)["logits"][0]
                        probs_f = F.softmax(logits_f, dim=-1).float().cpu().numpy()
                        if temporal_loaded:
                            mapped_f = np.zeros(len(primary_classes), dtype=np.float32)
                            for idx_f, letter in enumerate(ALBANIAN_LETTERS):
                                dst_idx = primary_class_to_idx.get(letter)
                                if dst_idx is not None and idx_f < len(probs_f):
                                    mapped_f[dst_idx] += float(probs_f[idx_f])
                            if probs is None:
                                probs = mapped_f
                            else:
                                probs = 0.75 * probs + 0.25 * mapped_f
                        else:
                            if probs_primary is None:
                                mapped_f = np.zeros(len(primary_classes), dtype=np.float32)
                                for idx_f, letter in enumerate(ALBANIAN_LETTERS):
                                    dst_idx = primary_class_to_idx.get(letter)
                                    if dst_idx is not None and idx_f < len(probs_f):
                                        mapped_f[dst_idx] += float(probs_f[idx_f])
                                probs = mapped_f
                            else:
                                probs = 0.55 * probs_primary + 0.45 * probs_f
                        assert probs is not None
                        s = probs.sum()
                        if s > 1e-8:
                            probs = probs / s
                    if probs_lm is not None and len(lm_idx_to_letter) > 0:
                        # Map iterative model class order to v2 alphabet order.
                        mapped = np.zeros(len(primary_classes), dtype=np.float32)
                        for j, p_val in enumerate(probs_lm):
                            letter = lm_idx_to_letter.get(int(j))
                            if letter is None:
                                continue
                            idx = primary_class_to_idx.get(letter)
                            if idx is None:
                                continue
                            mapped[idx] += float(p_val)
                        ms = mapped.sum()
                        if ms > 1e-8:
                            mapped /= ms
                            blend = 0.35 if temporal_loaded else 0.50
                            if probs is None:
                                probs = mapped
                            else:
                                probs = (1.0 - blend) * probs + blend * mapped
                            assert probs is not None
                            ps = probs.sum()
                            if ps > 1e-8:
                                probs /= ps
                    if probs is None:
                        continue
                if ema_probs is None or len(ema_probs) != len(probs):
                    ema_probs = probs
                else:
                    ema_probs = ema_alpha * ema_probs + (1 - ema_alpha) * probs
                no_sign_idx = primary_class_to_idx.get(NO_SIGN_LABEL)
                if no_sign_idx is not None:
                    idle_prob = float(ema_probs[no_sign_idx])
                sorted_idx = np.argsort(-ema_probs)
                for idx_sorted in sorted_idx:
                    label = primary_classes[int(idx_sorted)]
                    if label == NO_SIGN_LABEL:
                        continue
                    top3.append((label, float(ema_probs[int(idx_sorted)])))
                    if len(top3) == 3:
                        break
                idle_detected = (
                    no_sign_idx is not None
                    and idle_prob >= float(args.no_sign_threshold)
                    and idle_prob >= (top3[0][1] if top3 else 0.0)
                )
                motion_energy = _motion_energy_from_xyz_history(
                    xyz_history, frames=max(5, int(args.sequence_len))
                )
                if idle_detected:
                    # For dynamic-focus letters, allow motion evidence to break out of idle.
                    if motion_energy >= float(args.dynamic_focus_motion_threshold):
                        dyn_letter, dyn_dist = _dynamic_match_letter(
                            feat_history, dynamic_templates, max_dist=args.dynamic_max_dist
                        )
                        if dyn_letter is not None and dyn_letter in focus_letters:
                            dyn_conf = float(max(0.58, min(0.92, 1.0 - dyn_dist)))
                            top3 = [(dyn_letter, dyn_conf)]
                            idle_detected = False
                    if idle_detected:
                        top3 = []
                        recent_top1.clear()
                # Consensus smoothing on recent confident predictions.
                if not idle_detected and len(top3) >= 2:
                    margin = top3[0][1] - top3[1][1]
                else:
                    margin = 0.0
                if (not idle_detected) and top3 and (top3[0][1] >= 0.60 or margin >= 0.15):
                    recent_top1.append((top3[0][0], top3[0][1]))
                if (not idle_detected) and len(recent_top1) >= 4:
                    vote = Counter([x[0] for x in recent_top1])
                    voted_letter, voted_count = vote.most_common(1)[0]
                    vote_ratio = voted_count / len(recent_top1)
                    if vote_ratio >= 0.57:
                        voted_prob = float(np.mean([p for l, p in recent_top1 if l == voted_letter]))
                        rest = [x for x in top3 if x[0] != voted_letter]
                        top3 = [(voted_letter, voted_prob)] + rest[:2]

                # If model is uncertain, fallback to coordinate templates.
                if (not idle_detected) and top3 and top3[0][1] < 0.75:
                    live_xyz = xyz_norm
                    fb_letter, fb_dist = _template_match_letter(live_xyz, landmark_refs, max_dist=args.template_max_dist)
                    if fb_letter is not None:
                        fb_conf = float(max(0.55, min(0.93, 1.0 - fb_dist)))
                        rest = [x for x in top3 if x[0] != fb_letter]
                        top3 = [(fb_letter, fb_conf)] + rest[:2]

                # If still uncertain, try dynamic motion templates (Sh/Zh/...).
                if (not idle_detected) and ((not top3) or (top3 and top3[0][1] < 0.75)):
                    dyn_letter, dyn_dist = _dynamic_match_letter(
                        feat_history, dynamic_templates, max_dist=args.dynamic_max_dist
                    )
                    if dyn_letter is not None:
                        dyn_conf = float(max(0.58, min(0.90, 1.0 - dyn_dist)))
                        rest = [x for x in top3 if x[0] != dyn_letter]
                        top3 = [(dyn_letter, dyn_conf)] + rest[:2]
                if (not idle_detected) and top3:
                    top3 = _boost_focus_letters(
                        top3,
                        focus_letters=focus_letters,
                        motion_energy=motion_energy,
                        base_boost=float(args.dynamic_focus_boost),
                        motion_boost=float(args.dynamic_focus_motion_boost),
                    )
                # For dynamic letters, allow a slightly lower auto-append gate.
                if args.auto_append and top3:
                    cand, conf = top3[0]
                    th = args.auto_min_conf_dynamic if cand in dynamic_letters else args.auto_min_conf
                    auto_letter = _auto_append_letter(
                        auto_state,
                        candidate=cand,
                        confidence=conf,
                        threshold=th,
                        repeat_cooldown_ms=int(args.auto_repeat_cooldown_ms),
                        now_ms=ts_ms,
                    )
                    if auto_letter is not None:
                        word_buffer.append(auto_letter)
            else:
                # Decay prediction memory quickly when no hand is detected.
                if ema_probs is not None:
                    ema_probs *= 0.85
                if len(recent_top1) > 0:
                    recent_top1.clear()
                if len(feat_history) > 0:
                    feat_history.clear()
                if len(xyz_history) > 0:
                    xyz_history.clear()
                auto_state["candidate"] = None
                auto_state["count"] = 0

            # --- Draw skeleton overlay (all detected hands) -----------------
            pts_px: List[Tuple[int, int]] = []
            for pts_px_h, h_color, h_side, h_conf in hands_draw:
                _draw_hand(frame, pts_px_h, h_color, label=h_side, conf=h_conf)
            if detected and candidates:
                pts_px = best_pts_px  # primary hand pts (for recording)

            # --- Border ------------------------------------------------------
            if not detected:
                cv2.rectangle(frame, (0, 0), (W - 1, H - 1), (0, 0, 255), 6)

            # --- Recording state machine ------------------------------------
            record_text = ""
            if recording_state["mode"] == "countdown":
                remaining = float(recording_state["countdown_until"]) - time.time()
                if remaining <= 0:
                    recording_state["mode"] = "record"
                    recording_state["frames"] = []
                else:
                    secs = int(math.ceil(remaining))
                    record_text = f"RECORDING IN {secs}s"
            elif recording_state["mode"] == "record":
                if detected:
                    recording_state["frames"].append(build_feature(xyz, is_left=is_left))
                captured = len(recording_state["frames"])
                record_text = f"REC {captured}/{RECORD_FRAME_COUNT}"
                if captured >= RECORD_FRAME_COUNT:
                    feats_arr = np.stack(recording_state["frames"], axis=0).astype(np.float32)
                    label = str(recording_state["record_label"])
                    stamp = time.strftime("%Y%m%d-%H%M%S")
                    _append_recording_to_h5(
                        args.recordings_h5,
                        feats_arr,
                        label=label,
                        source=f"live-{stamp}",
                    )
                    logger.info(
                        "recorded {} samples for letter {} -> {}",
                        feats_arr.shape[0],
                        _safe_letter(label),
                        args.recordings_h5,
                    )
                    recording_state = {"mode": "idle", "countdown_until": 0.0, "frames": [], "record_label": None}

            # --- HUD ---------------------------------------------------------
            selected_letter = ALBANIAN_LETTERS[selected_label_idx]
            _put_text(frame, f"label_select={_safe_letter(selected_letter)}", (10, 24), (255, 255, 255))
            if idle_detected:
                _put_text(frame, f"pred=IDLE  conf={idle_prob:.2f}", (10, 52), (180, 180, 255), scale=0.8)
            elif top3:
                top_letter, top_conf = top3[0]
                actionable = _top1_is_actionable(
                    top3,
                    float(args.pred_min_conf),
                    float(args.pred_margin),
                )
                color = (0, 255, 0) if actionable else (0, 165, 255)
                shown = _safe_letter(top_letter) if actionable else "UNCERTAIN"
                _put_text(frame, f"pred={shown}  conf={top_conf:.2f}", (10, 52), color, scale=0.8)
                y = 84
                for letter, prob in top3:
                    _put_text(frame, f"  {_safe_letter(letter):6s} {prob*100:5.1f}%", (10, y), (220, 220, 220), scale=0.6)
                    y += 22
            else:
                _put_text(frame, "pred=-", (10, 52), (0, 0, 255), scale=0.8)

            if record_text:
                _put_text(frame, record_text, (10, H - 56), (0, 0, 255), scale=1.0)

            _put_text(
                frame,
                f"word={''.join(_safe_letter(c) for c in word_buffer)}",
                (10, H - 18),
                (255, 255, 255),
                scale=0.8,
            )
            if args.auto_append:
                wb_key = tuple(word_buffer)
                if wb_key != last_word_key:
                    cached_suggestion = _suggest_word_from_index(word_buffer, words_index)
                    cached_exact = _match_word_from_index(word_buffer, words_index)
                    last_word_key = wb_key
                suggestion = cached_suggestion
                if suggestion:
                    _put_text(frame, f"suggest={suggestion}", (10, H - 84), (180, 220, 255), scale=0.6)
                # Real-time exact match: when buffer matches a known word, surface it.
                exact = cached_exact
                if exact and exact != last_detected_word:
                    last_detected_word = exact
                    logger.info("WORD-DETECTED: {}", exact)
                if exact:
                    _put_text(frame, f"WORD={exact}", (10, H - 110), (0, 255, 0), scale=0.7)

            cv2.imshow("AlbSL Live v2", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 0xFF:
                continue
            if key == ord("q"):
                break
            elif key == ord("l"):
                selected_label_idx = (selected_label_idx + 1) % len(ALBANIAN_LETTERS)
            elif key == ord("k"):
                selected_label_idx = (selected_label_idx - 1) % len(ALBANIAN_LETTERS)
            elif key == ord("r") and recording_state["mode"] == "idle":
                recording_state = {
                    "mode": "countdown",
                    "countdown_until": time.time() + COUNTDOWN_SECS,
                    "frames": [],
                    "record_label": selected_letter,
                }
            elif key == 32:  # SPACE
                if top3 and _top1_is_actionable(
                    top3,
                    float(args.pred_min_conf),
                    float(args.pred_margin),
                ):
                    word_buffer.append(top3[0][0])
            elif key == ord("y"):  # confirm prediction -> append coordinates to CSV
                if detected and top3 and top3[0][1] >= args.confirm_min_conf:
                    live_xyz = canonical_normalize_hand(xyz, is_left=is_left)
                    top2_margin = float(top3[0][1] - top3[1][1]) if len(top3) >= 2 else float(top3[0][1])
                    row_payload: Dict[str, object] = {
                        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                        "label": top3[0][0],
                        "confidence": round(float(top3[0][1]), 6),
                        "source": "albsl_app_v2_live_confirm",
                        "session_id": capture_session_id,
                        "frame_ts_ms": int(ts_ms),
                        "model_source": "primary" if loaded else ("iterative" if lm_loaded else "fallback"),
                        "top2_margin": round(top2_margin, 6),
                        "landmarks_63": json.dumps(live_xyz.reshape(-1).astype(np.float32).tolist(), ensure_ascii=False),
                    }
                    flat = live_xyz.reshape(-1)
                    for i in range(21):
                        base = i * 3
                        row_payload[f"lm{i}_x"] = round(float(flat[base + 0]), 6)
                        row_payload[f"lm{i}_y"] = round(float(flat[base + 1]), 6)
                        row_payload[f"lm{i}_z"] = round(float(flat[base + 2]), 6)
                    confirmed_rows_batch.append(row_payload)
                    if len(confirmed_rows_batch) >= confirm_flush_every:
                        _flush_confirmed_csv_queue(args.confirmed_csv, confirmed_rows_batch)
                    logger.info("confirmed {} -> {}", top3[0][0], args.confirmed_csv.resolve())
            elif key == 8:  # BACKSPACE
                if word_buffer:
                    word_buffer.pop()
            elif key == 13:  # ENTER
                if word_buffer:
                    raw_word = "".join(word_buffer)
                    matched = _match_word_from_letters(word_buffer, words_dict)
                    if matched is not None:
                        logger.info("WORD: {}  -> dictionary_match={}", raw_word, matched)
                    else:
                        logger.info("WORD: {}", raw_word)
                    word_buffer.clear()
            elif key == ord("c"):
                word_buffer.clear()
    finally:
        if confirmed_rows_batch:
            _flush_confirmed_csv_queue(args.confirmed_csv, confirmed_rows_batch)
        cap.release()
        cv2.destroyAllWindows()
        hand.close()


# --- CLI --------------------------------------------------------------------


def _inject_default_subcommand() -> None:
    """``python albsl_app_v2.py`` and ``...py --weights ...`` imply ``live``."""
    if len(sys.argv) <= 1:
        sys.argv.append("live")
        return
    first = sys.argv[1]
    if first in ("-h", "--help", "diagnose", "train", "live"):
        return
    sys.argv.insert(1, "live")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="AlbSL live recognition + recording v2")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("diagnose", help="Analyze training data sources")
    d.add_argument("--keypoints-dir", type=Path, default=Path("datasets/processed/core_data/data/keypoints"))
    d.add_argument("--alfabeti-h5", type=Path, default=Path("datasets/processed/core_data/data/alfabeti_keypoints.h5"))
    d.add_argument("--legacy-h5", type=Path, default=Path("keypoints.h5"))

    t = sub.add_parser("train", help="Train temporal LSTM landmark classifier")
    t.add_argument("--keypoints-dir", type=Path, default=Path("datasets/processed/core_data/data/keypoints"))
    t.add_argument("--alfabeti-h5", type=Path, default=Path("datasets/processed/core_data/data/alfabeti_keypoints.h5"))
    t.add_argument("--legacy-h5", type=Path, default=Path("keypoints.h5"))
    t.add_argument(
        "--confirmed-csv",
        type=Path,
        default=DEFAULT_CONFIRMED_CSV,
        help="Unified landmark CSV (live confirms + merged coordinates).",
    )
    t.add_argument(
        "--coordinates-csv",
        type=Path,
        default=None,
        help="Optional extra coordinates CSV; omit if rows are already in --confirmed-csv.",
    )
    t.add_argument("--out", type=Path, default=Path("outputs/albsl_mlp.pt"))
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--device", type=str, default="cuda", choices=["auto", "cuda", "xpu", "cpu"])
    t.add_argument("--sequence-len", type=int, default=DEFAULT_SEQUENCE_LEN)
    t.add_argument("--sequence-stride", type=int, default=DEFAULT_SEQUENCE_STRIDE)
    t.add_argument("--min-valid-frames", type=int, default=DEFAULT_MIN_VALID_FRAMES)
    t.add_argument("--idle-ratio", type=float, default=0.35)
    t.add_argument("--hidden-dim", type=int, default=192)
    t.add_argument("--layers", type=int, default=2)
    t.add_argument("--dropout", type=float, default=0.25)
    t.add_argument("--workers", type=int, default=0)
    t.add_argument("--no-augment", action="store_true")

    l = sub.add_parser("live", help="Run live recognition + recording app")
    l.add_argument("--weights", type=Path, default=Path("outputs/albsl_mlp.pt"))
    l.add_argument("--fused-weights", type=Path, default=Path("outputs/fused_phase3.pt"))
    l.add_argument("--albsl-model", type=Path, default=Path("models/trained/albsl_model_final/model_full.pt"))
    l.add_argument("--models-dir", type=Path, default=Path("models/mediapipe/mp_models"))
    l.add_argument("--camera", type=int, default=0)
    l.add_argument("--recordings-h5", type=Path, default=Path("keypoints.h5"))
    l.add_argument("--landmarks-json", type=Path, default=Path("datasets/processed/assets/albsl_landmarks.json"))
    l.add_argument("--dynamic-templates-json", type=Path, default=Path("datasets/processed/assets/albsl_dynamic_templates.json"))
    l.add_argument("--words-dict-json", type=Path, default=Path("datasets/processed/assets/albsl_words_dictionary.json"))
    l.add_argument("--template-max-dist", type=float, default=0.16)
    l.add_argument("--dynamic-max-dist", type=float, default=0.12)
    l.add_argument(
        "--confirmed-csv",
        type=Path,
        default=DEFAULT_CONFIRMED_CSV,
        help="Append live confirmations (Y key) to this CSV.",
    )
    l.add_argument("--confirm-min-conf", type=float, default=0.72)
    l.add_argument("--confirm-flush-every", type=int, default=8, help="Flush confirmed CSV rows every N confirmations.")
    l.add_argument("--strict-startup", action="store_true", help="Fail fast if required models/assets are unavailable.")
    l.add_argument(
        "--pred-min-conf",
        type=float,
        default=0.60,
        help="Min top-1 probability to show the letter (else UNCERTAIN unless margin rule applies).",
    )
    l.add_argument(
        "--pred-margin",
        type=float,
        default=0.10,
        help="If top-1 minus top-2 is at least this (e.g. 0.10), show top-1 even below --pred-min-conf.",
    )
    l.add_argument("--auto-append", action="store_true")
    l.add_argument("--auto-hold-frames", type=int, default=7)
    l.add_argument("--auto-min-conf", type=float, default=0.78)
    l.add_argument("--auto-min-conf-dynamic", type=float, default=0.72)
    l.add_argument("--auto-repeat-cooldown-ms", type=int, default=900)
    l.add_argument("--dynamic-focus-letters", type=str, default="Ë,Ç,Sh,Zh,Xh")
    l.add_argument("--dynamic-focus-boost", type=float, default=0.04)
    l.add_argument("--dynamic-focus-motion-boost", type=float, default=0.40)
    l.add_argument("--dynamic-focus-motion-threshold", type=float, default=0.010)
    l.add_argument("--min-hand-detection-confidence", type=float, default=0.70)
    l.add_argument("--min-hand-presence-confidence", type=float, default=0.70)
    l.add_argument("--min-hand-tracking-confidence", type=float, default=0.70)
    l.add_argument("--min-hand-score", type=float, default=0.65)
    l.add_argument("--min-hand-area", type=float, default=0.006)
    l.add_argument("--max-hand-area", type=float, default=0.55)
    l.add_argument("--landmark-smooth-alpha", type=float, default=0.70)
    l.add_argument("--detect-on-frames", type=int, default=2)
    l.add_argument("--detect-off-frames", type=int, default=4)
    l.add_argument(
        "--unified-coords-json",
        type=Path,
        default=Path("datasets/json_dataset/coordinates.json"),
        help="Optional unified coordinates JSON; words section overrides --words-dict-json.",
    )
    l.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "xpu", "cpu"])
    l.add_argument("--sequence-len", type=int, default=DEFAULT_SEQUENCE_LEN)
    l.add_argument("--no-sign-threshold", type=float, default=0.58)

    return ap.parse_args()


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    _inject_default_subcommand()
    args = parse_args()
    if args.cmd == "diagnose":
        cmd_diagnose(args)
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "live":
        cmd_live(args)


if __name__ == "__main__":
    main()
