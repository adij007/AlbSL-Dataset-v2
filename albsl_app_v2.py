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
  says UNCERTAIN when top-1 confidence < 0.75.
- Press R to start a 3-second countdown then record 30 frames of keypoints into
  ``keypoints.h5`` under the currently selected label.

Controls inside the video window
--------------------------------
  L / K        cycle selected label forward/backward
  R            start 3s countdown, then record 30 frames
  SPACE        append predicted top-1 to the word buffer (if confident)
  BACKSPACE    delete last letter
  ENTER        commit current word (printed to console, buffer cleared)
  C            clear word buffer
  Q            quit (saves live-trained model if training updates happened)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

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
from albsl_fusion.model import FusionBatch, build_model

BaseOptions = mp_tasks.BaseOptions

# --- Constants --------------------------------------------------------------

ALBANIAN_LETTERS: List[str] = [
    "A", "B", "C", "Ç", "D", "Dh", "E", "Ë", "F", "G", "Gj", "H", "I", "J", "K",
    "L", "Ll", "M", "N", "Nj", "O", "P", "Q", "R", "Rr", "S", "Sh", "T", "Th",
    "U", "V", "X", "Xh", "Y", "Z", "Zh",
]
LETTER_TO_IDX: Dict[str, int] = {l: i for i, l in enumerate(ALBANIAN_LETTERS)}

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
FUSION_IMAGE_SIZE = 224

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


def canonical_normalize_hand(xyz: np.ndarray, is_left: bool = False) -> np.ndarray:
    """Translate to wrist, scale by bbox diagonal, align MCP9 to +Y, mirror left to right."""
    if not np.any(xyz):
        return xyz.astype(np.float32, copy=True)
    out = xyz.astype(np.float32, copy=True)
    # Mirror left hand into right-hand space BEFORE normalization for chirality consistency.
    if is_left:
        out[:, 0] *= -1.0
    out -= out[0]  # wrist = origin
    diag = float(np.linalg.norm(out.max(axis=0) - out.min(axis=0)))
    if diag > 1e-8:
        out /= diag
    R = _rodrigues(out[9], np.array([0.0, 1.0, 0.0], dtype=np.float32))
    out = (R @ out.T).T
    # Secondary chirality fix via palm signed area (robust to handedness mislabels).
    signed = float(np.cross(out[5] - out[0], out[17] - out[0])[2])
    if signed < 0.0:
        out[:, 0] *= -1.0
    return out.astype(np.float32)


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


# --- Data loading -----------------------------------------------------------


def _pick_hand_from_h5(right: np.ndarray, left: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Select the hand with non-zero data; returns (xyz, is_left)."""
    if np.any(right):
        return right, False
    if np.any(left):
        return left, True
    return right, False


def load_labeled_samples(
    keypoints_dir: Path,
    alfabeti_h5: Path,
    legacy_h5: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, Counter]:
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
                T = xyz_r.shape[0]
                for t in range(T):
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
                T = xyz.shape[0]
                for t in range(T):
                    if conf[t].max() < 0.5:
                        continue
                    # Old-format clips already stored normalized xyz; re-extract anyway
                    # for consistency via build_feature (which re-normalizes).
                    features.append(build_feature(xyz[t], is_left=False))
                    labels.append(letter)

    # (2) Alfabeti per-image H5
    if alfabeti_h5.exists():
        with h5py.File(alfabeti_h5, "r") as f:
            xyz = f["xyz"][...]
            det = f["detected"][...]
            lbls = [x.decode() if isinstance(x, bytes) else x for x in f["labels"][...]]
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
            lbls = [x.decode() if isinstance(x, bytes) else x for x in f["labels"][...]]
            right = f["right_hand"][...]
            left = f["left_hand"][...]
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


# --- Subcommand: diagnose ---------------------------------------------------


def cmd_diagnose(args: argparse.Namespace) -> None:
    print("=" * 72)
    print("AlbSL dataset diagnostics")
    print("=" * 72)

    if args.legacy_h5 and args.legacy_h5.exists():
        with h5py.File(args.legacy_h5, "r") as f:
            lbls = [x.decode() if isinstance(x, bytes) else x for x in f["labels"][...]]
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
            lbls = [x.decode() if isinstance(x, bytes) else x for x in f["labels"][...]]
            det = f["detected"][...]
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
            bar = "█" * min(40, v // max(1, total_frames // 400))
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
    X, Y, counts = load_labeled_samples(args.keypoints_dir, args.alfabeti_h5, legacy_h5=args.legacy_h5)
    if len(X) == 0:
        logger.error("No labeled samples found. Run extraction first.")
        return
    present = sorted(counts.keys())
    logger.info("training samples: {} across {} letters", len(X), len(present))

    y_idx = np.array([LETTER_TO_IDX[l] for l in Y], dtype=np.int64)

    # Compute class weights (inverse-frequency) to counter imbalance.
    n_classes = len(ALBANIAN_LETTERS)
    class_counts = np.ones(n_classes, dtype=np.float32) * 1.0
    for i, letter in enumerate(ALBANIAN_LETTERS):
        class_counts[i] = max(1.0, float(counts.get(letter, 0)))
    class_weights = torch.tensor((class_counts.sum() / (n_classes * class_counts)), dtype=torch.float32)

    # 80/20 train/val split.
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    cut = int(0.8 * len(X))
    tr_idx, va_idx = perm[:cut], perm[cut:]

    Xt = torch.from_numpy(X[tr_idx])
    Yt = torch.from_numpy(y_idx[tr_idx])
    Xv = torch.from_numpy(X[va_idx])
    Yv = torch.from_numpy(y_idx[va_idx])

    device = torch.device("xpu") if hasattr(torch, "xpu") and torch.xpu.is_available() else torch.device("cpu")
    model = LetterMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    bs = int(args.batch_size)
    best_acc = 0.0
    best_state: Optional[dict] = None

    for epoch in range(int(args.epochs)):
        model.train()
        idx = rng.permutation(len(Xt))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(Xt), bs):
            b = idx[start : start + bs]
            x = Xt[b].to(device)
            y = Yt[b].to(device)
            logits = model(x)
            loss = criterion(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += float(loss.detach().cpu())
            n_batches += 1

        model.eval()
        with torch.no_grad():
            logits_v = model(Xv.to(device))
            pred_v = logits_v.argmax(dim=-1).cpu()
            acc = float((pred_v == Yv).float().mean())

        logger.info(
            "epoch {:3d}  loss={:.4f}  val_acc={:.3f}",
            epoch + 1, epoch_loss / max(n_batches, 1), acc,
        )
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": best_state or model.state_dict(),
        "classes": ALBANIAN_LETTERS,
        "in_dim": KEY_DIM,
        "val_acc": best_acc,
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
            feats_ds = f["features"]
            lbl_ds = f["labels"]
            src_ds = f["sources"]
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


def _load_landmark_refs(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[str, np.ndarray] = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        try:
            arr = np.array(v, dtype=np.float32)
        except Exception:
            continue
        if arr.shape == (21, 3) and k in LETTER_TO_IDX:
            out[str(k)] = arr
    return out


def _resolve_json_path(path_like: Path) -> Path:
    """Accept both 'foo' and 'foo.json' path styles."""
    if path_like.exists():
        return path_like
    if path_like.suffix.lower() != ".json":
        alt = Path(str(path_like) + ".json")
        if alt.exists():
            return alt
        # Also try the same basename in current working directory.
        cwd_alt = Path(path_like.name + ".json")
        if cwd_alt.exists():
            return cwd_alt
    # Common fallback names for this app.
    if path_like.name.lower() in ("albsl_landmarks", "albsl_landmarks.json"):
        fallback = Path("albsl_landmarks.json")
        if fallback.exists():
            return fallback
    return path_like


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


def _load_dynamic_templates(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[str, np.ndarray] = {}
    if not isinstance(raw, dict):
        return out
    for letter, payload in raw.items():
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
            out[letter] = arr
    return out


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
    templates: Dict[str, np.ndarray],
    max_dist: float = 0.12,
) -> Tuple[Optional[str], float]:
    if not templates:
        return None, float("inf")
    if len(feat_history) < 8:
        return None, float("inf")
    best_letter: Optional[str] = None
    best_dist = float("inf")
    hist = np.stack(list(feat_history), axis=0).astype(np.float32)
    for letter, tmpl in templates.items():
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
        d = float(np.mean(np.abs(cand[:, :d_dim] - tmpl[:, :d_dim])))
        if d < best_dist:
            best_dist = d
            best_letter = letter
    if best_letter is not None and best_dist <= max_dist:
        return best_letter, best_dist
    return None, best_dist


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


def _match_word_from_letters(letter_seq: List[str], words_dict: List[Dict[str, object]]) -> Optional[str]:
    if not letter_seq:
        return None
    for item in words_dict:
        letters = item.get("letters")
        word = item.get("word")
        if isinstance(letters, list) and isinstance(word, str):
            if [str(x) for x in letters] == letter_seq:
                return word
    return None


def _load_landmark_model_checkpoint(path: Path, device: torch.device) -> Tuple[Optional[torch.nn.Module], Optional[Dict[str, Any]]]:
    """
    Load model exported by train_albsl.py -> albsl_model_final/model_full.pt.
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


def cmd_live(args: argparse.Namespace) -> None:
    device = torch.device("xpu") if hasattr(torch, "xpu") and torch.xpu.is_available() else torch.device("cpu")
    model = LetterMLP().to(device)
    loaded = False
    if args.weights.exists():
        payload = torch.load(str(args.weights), map_location=device, weights_only=True)
        state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        try:
            model.load_state_dict(state)
            loaded = True
            logger.info("loaded weights from {}", args.weights)
        except Exception as exc:
            logger.warning("failed to load weights: {} — using random init", exc)
    if not loaded:
        logger.warning("no trained weights loaded — run `train` first for meaningful predictions")
    model.eval()
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

    hand_model = _ensure_hand_model(args.models_dir)
    hand = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(hand_model)),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_hands=2,          # ← detect both hands simultaneously
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
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

    recording_state: Dict[str, object] = {"mode": "idle", "countdown_until": 0.0, "frames": [], "record_label": None}
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

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # selfie view
            H, W = frame.shape[:2]

            ts_ms = max(last_ts_ms + 1, int((time.time() - t_start) * 1000))
            last_ts_ms = ts_ms

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = hand.detect_for_video(mp_image, ts_ms)

            # ── Detect up to 2 hands ─────────────────────────────────────────
            # Primary hand selection is temporally stable:
            # score = detection confidence + side stickiness - center-jump penalty.
            xyz = np.zeros((21, 3), dtype=np.float32)
            is_left = False
            detected = False
            hand_score = 0.0

            # Per-hand draw info: list of (pts_px, color, side_label, conf)
            hands_draw: List[Tuple[List[Tuple[int, int]], Tuple[int,int,int], str, float]] = []

            candidates: List[Tuple[float, np.ndarray, bool, float, List[Tuple[int, int]], str]] = []
            hand_lms = result.hand_landmarks if result.hand_landmarks is not None else []
            for h_idx in range(len(hand_lms)):
                pts_raw = hand_lms[h_idx]
                hand_xyz = np.array([[p.x, p.y, p.z] for p in pts_raw], dtype=np.float32)
                hand_side = "right"
                hand_conf = 0.0
                if result.handedness and h_idx < len(result.handedness) and result.handedness[h_idx]:
                    # MediaPipe label is from signer's perspective; frame is mirrored for selfie.
                    raw_side = result.handedness[h_idx][0].display_name.lower()
                    hand_conf = float(result.handedness[h_idx][0].score)
                    hand_side = "left" if raw_side == "right" else "right"  # flip for mirror

                h_is_left = (hand_side == "left")
                # Right-hand skeleton = green; left-hand skeleton = cyan
                h_color = (0, 200, 255) if h_is_left else (0, 255, 0)
                pts_px_h = [(int(p[0] * W), int(p[1] * H)) for p in hand_xyz]
                hands_draw.append((pts_px_h, h_color, hand_side.upper(), hand_conf))
                center = hand_xyz[:, :2].mean(axis=0)
                center_penalty = 0.0
                if prev_center is not None:
                    center_penalty = float(np.linalg.norm(center - prev_center))
                sticky_bonus = 0.15 if (side_sticky is not None and side_sticky == hand_side) else 0.0
                # Prefer right hand very slightly, but do not force hard switching.
                side_bias = 0.05 if hand_side == "right" else 0.0
                score = float(hand_conf) + sticky_bonus + side_bias - 0.35 * center_penalty
                candidates.append((score, hand_xyz, h_is_left, hand_conf, pts_px_h, hand_side))

            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                _, xyz, is_left, hand_score, best_pts_px, best_side = candidates[0]
                detected = True
                side_sticky = best_side
                prev_center = xyz[:, :2].mean(axis=0)
            else:
                side_sticky = None
                prev_center = None

            # --- Prediction --------------------------------------------------
            top3: List[Tuple[str, float]] = []
            if detected and loaded:
                feat = build_feature(xyz, is_left=is_left)
                feat_history.append(feat.copy())
                with torch.no_grad():
                    x = torch.from_numpy(feat).unsqueeze(0).to(device)
                    logits = model(x)[0]
                    probs_mlp = F.softmax(logits, dim=-1).float().cpu().numpy()
                    probs = probs_mlp
                    # Blend MLP and fusion probabilities when available.
                    probs_lm: Optional[np.ndarray] = None
                    if lm_loaded and lm_model is not None:
                        # train_albsl model expects 63-dim (21*3), while v2 feature is 123-dim.
                        feat63 = feat[:63].astype(np.float32, copy=False)
                        logits_lm = lm_model(
                            torch.from_numpy(feat63).view(1, -1).to(device=device, dtype=torch.float32)
                        )[0]
                        probs_lm = F.softmax(logits_lm, dim=-1).float().cpu().numpy()
                    if fusion_loaded and fusion_model is not None:
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
                        probs = 0.55 * probs_mlp + 0.45 * probs_f
                        s = probs.sum()
                        if s > 1e-8:
                            probs = probs / s
                    if probs_lm is not None and len(lm_idx_to_letter) > 0:
                        # Map iterative model class order to v2 alphabet order.
                        mapped = np.zeros_like(probs, dtype=np.float32)
                        for j, p_val in enumerate(probs_lm):
                            letter = lm_idx_to_letter.get(int(j))
                            if letter is None:
                                continue
                            idx = LETTER_TO_IDX.get(letter)
                            if idx is None:
                                continue
                            mapped[idx] += float(p_val)
                        ms = mapped.sum()
                        if ms > 1e-8:
                            mapped /= ms
                            # Blend all available models; give iterative model meaningful influence.
                            probs = 0.50 * probs + 0.50 * mapped
                            ps = probs.sum()
                            if ps > 1e-8:
                                probs /= ps
                ema_probs = probs if ema_probs is None else ema_alpha * ema_probs + (1 - ema_alpha) * probs
                top_k = np.argsort(-ema_probs)[:3]
                top3 = [(ALBANIAN_LETTERS[int(i)], float(ema_probs[int(i)])) for i in top_k]
                # Consensus smoothing on recent confident predictions.
                if len(top3) >= 2:
                    margin = top3[0][1] - top3[1][1]
                else:
                    margin = 0.0
                if top3 and (top3[0][1] >= 0.60 or margin >= 0.15):
                    recent_top1.append((top3[0][0], top3[0][1]))
                if len(recent_top1) >= 4:
                    vote = Counter([x[0] for x in recent_top1])
                    voted_letter, voted_count = vote.most_common(1)[0]
                    vote_ratio = voted_count / len(recent_top1)
                    if vote_ratio >= 0.57:
                        voted_prob = float(np.mean([p for l, p in recent_top1 if l == voted_letter]))
                        rest = [x for x in top3 if x[0] != voted_letter]
                        top3 = [(voted_letter, voted_prob)] + rest[:2]

                # If model is uncertain, fallback to coordinate templates.
                if top3 and top3[0][1] < 0.75:
                    live_xyz = canonical_normalize_hand(xyz, is_left=is_left)
                    fb_letter, fb_dist = _template_match_letter(live_xyz, landmark_refs, max_dist=args.template_max_dist)
                    if fb_letter is not None:
                        fb_conf = float(max(0.55, min(0.93, 1.0 - fb_dist)))
                        rest = [x for x in top3 if x[0] != fb_letter]
                        top3 = [(fb_letter, fb_conf)] + rest[:2]

                # If still uncertain, try dynamic motion templates (Sh/Zh/...)
                if (not top3) or (top3 and top3[0][1] < 0.75):
                    dyn_letter, dyn_dist = _dynamic_match_letter(
                        feat_history, dynamic_templates, max_dist=args.dynamic_max_dist
                    )
                    if dyn_letter is not None:
                        dyn_conf = float(max(0.58, min(0.90, 1.0 - dyn_dist)))
                        rest = [x for x in top3 if x[0] != dyn_letter]
                        top3 = [(dyn_letter, dyn_conf)] + rest[:2]
            else:
                # Decay prediction memory quickly when no hand is detected.
                if ema_probs is not None:
                    ema_probs *= 0.85
                if len(recent_top1) > 0:
                    recent_top1.clear()
                if len(feat_history) > 0:
                    feat_history.clear()

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
                captured = len(recording_state["frames"])  # type: ignore[arg-type]
                record_text = f"REC {captured}/{RECORD_FRAME_COUNT}"
                if captured >= RECORD_FRAME_COUNT:
                    feats_arr = np.stack(recording_state["frames"], axis=0).astype(np.float32)  # type: ignore[arg-type]
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
            if top3:
                top_letter, top_conf = top3[0]
                color = (0, 255, 0) if top_conf >= 0.75 else (0, 165, 255)
                shown = _safe_letter(top_letter) if top_conf >= 0.75 else "UNCERTAIN"
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
                if top3 and top3[0][1] >= 0.75:
                    word_buffer.append(top3[0][0])
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
    d.add_argument("--keypoints-dir", type=Path, default=Path("data/keypoints"))
    d.add_argument("--alfabeti-h5", type=Path, default=Path("data/alfabeti_keypoints.h5"))
    d.add_argument("--legacy-h5", type=Path, default=Path("keypoints.h5"))

    t = sub.add_parser("train", help="Train MLP classifier")
    t.add_argument("--keypoints-dir", type=Path, default=Path("data/keypoints"))
    t.add_argument("--alfabeti-h5", type=Path, default=Path("data/alfabeti_keypoints.h5"))
    t.add_argument("--legacy-h5", type=Path, default=Path("keypoints.h5"))
    t.add_argument("--out", type=Path, default=Path("outputs/albsl_mlp.pt"))
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-3)

    l = sub.add_parser("live", help="Run live recognition + recording app")
    l.add_argument("--weights", type=Path, default=Path("outputs/albsl_mlp.pt"))
    l.add_argument("--fused-weights", type=Path, default=Path("outputs/fused_phase3.pt"))
    l.add_argument("--albsl-model", type=Path, default=Path("albsl_model_final/model_full.pt"))
    l.add_argument("--models-dir", type=Path, default=Path("mp_models"))
    l.add_argument("--camera", type=int, default=0)
    l.add_argument("--recordings-h5", type=Path, default=Path("keypoints.h5"))
    l.add_argument("--landmarks-json", type=Path, default=Path("albsl_landmarks.json"))
    l.add_argument("--dynamic-templates-json", type=Path, default=Path("albsl_dynamic_templates.json"))
    l.add_argument("--words-dict-json", type=Path, default=Path("albsl_words_dictionary.json"))
    l.add_argument("--template-max-dist", type=float, default=0.16)
    l.add_argument("--dynamic-max-dist", type=float, default=0.12)

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
