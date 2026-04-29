"""Alfabeti image-folder keypoint extractor.

Scans a flat folder of JPEG/PNG images (one hand pose per image), runs
MediaPipe HandLandmarker in IMAGE mode, applies the canonical normalization
and dihedral features from ``extract_keypoints_v2``, and writes an HDF5
dataset plus a per-label stats JSON.

Supported filename conventions (case-insensitive):
- ``handN_<letter>_<view>_seg_<idx>_cropped.jpeg``
  (e.g., ``hand1_a_bot_seg_3_cropped.jpeg`` -> label ``A``)
- ``<letter>_<anything>.jpg``
  (e.g., ``A_01.jpg`` -> label ``A``)
- GUID-only filenames are included as ``UNKNOWN``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import mediapipe as mp
import numpy as np
from loguru import logger
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from extract_keypoints_v2 import (
    ALBANIAN_LETTERS,
    HAND_BONES,
    canonical_normalize_hand,
    dihedral_features,
    ensure_models,
    parse_hand,
)

BaseOptions = mp_tasks.BaseOptions
console = Console()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
HAND_N_PATTERN = re.compile(r"^hand\d+_([a-zA-ZËÇëç]+)_", re.IGNORECASE)
LEADING_LETTER_PATTERN = re.compile(r"^([a-zA-ZËÇëç]+)[_\-.]", re.IGNORECASE)
LETTER_ALIASES: Dict[str, str] = {
    "c_cedilla": "Ç",
    "e_umlaut": "Ë",
    "dh": "Dh", "ll": "Ll", "nj": "Nj", "rr": "Rr", "sh": "Sh",
    "th": "Th", "xh": "Xh", "zh": "Zh", "gj": "Gj",
}
GUID_PATTERN = re.compile(r"^\{[0-9a-fA-F-]{36}\}$")


@dataclass
class Sample:
    path: Path
    label: str


def parse_label(path: Path) -> Optional[str]:
    name = path.stem
    m = HAND_N_PATTERN.match(name)
    if m:
        token = m.group(1).lower()
    else:
        m2 = LEADING_LETTER_PATTERN.match(name + "_")
        if not m2:
            if GUID_PATTERN.match(name):
                return "UNKNOWN"
            return None
        token = m2.group(1).lower()
    if token in LETTER_ALIASES:
        return LETTER_ALIASES[token]
    canonical_map = {letter.lower(): letter for letter in ALBANIAN_LETTERS}
    return canonical_map.get(token)


def collect_samples(root: Path) -> Tuple[List[Sample], int]:
    samples: List[Sample] = []
    skipped = 0
    for entry in sorted(root.iterdir()):
        if not entry.is_file() or entry.suffix.lower() not in IMAGE_EXTS:
            continue
        label = parse_label(entry)
        if label is None:
            logger.debug("Skipping unparseable filename: {}", entry.name)
            skipped += 1
            continue
        samples.append(Sample(entry, label))
    return samples, skipped


def make_image_hand_detector(model_path: Path) -> HandLandmarker:
    return HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_hands=1,
        )
    )


def extract_image(detector: HandLandmarker, image_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    data = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if data is None:
        return (
            np.zeros((21, 3), np.float32),
            np.zeros((21,), np.float32),
            np.zeros((20, 3), np.float32),
            False,
        )
    rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(image)
    xyz, conf = parse_hand(result)
    detected = bool(np.any(xyz))
    if detected:
        xyz = canonical_normalize_hand(xyz)
        xyz = np.clip(xyz, -1.0, 1.0)
        angles = dihedral_features(xyz).astype(np.float32)
    else:
        angles = np.zeros((20, 3), np.float32)
    return xyz.astype(np.float32), conf.astype(np.float32), angles, detected


def write_hdf5(path: Path, samples: List[Sample], payload: List[Dict[str, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with h5py.File(path, "w") as f:
        N = len(samples)
        f.create_dataset(
            "xyz", data=np.stack([p["xyz"] for p in payload], axis=0),
            compression="gzip", compression_opts=4,
        )
        f.create_dataset(
            "angles", data=np.stack([p["angles"] for p in payload], axis=0),
            compression="gzip", compression_opts=4,
        )
        f.create_dataset(
            "conf", data=np.stack([p["conf"] for p in payload], axis=0),
            compression="gzip", compression_opts=4,
        )
        f.create_dataset(
            "detected",
            data=np.array([p["detected"] for p in payload], dtype=bool),
        )
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("labels", data=np.array([s.label for s in samples], dtype=object), dtype=dt)
        f.create_dataset("sources", data=np.array([str(s.path) for s in samples], dtype=object), dtype=dt)
        f.attrs["total_samples"] = N
        f.attrs["detected_samples"] = int(sum(p["detected"] for p in payload))


def build_stats(samples: List[Sample], payload: List[Dict[str, np.ndarray]]) -> Dict[str, object]:
    per_letter: Dict[str, Dict[str, float]] = {}
    for s, p in zip(samples, payload):
        entry = per_letter.setdefault(
            s.label,
            {"samples": 0.0, "detected": 0.0, "mean_confidence_sum": 0.0},
        )
        entry["samples"] += 1
        if p["detected"]:
            entry["detected"] += 1
            entry["mean_confidence_sum"] += float(p["conf"].mean())
    finalized: Dict[str, Dict[str, float]] = {}
    for letter, entry in sorted(per_letter.items()):
        detected = entry["detected"]
        finalized[letter] = {
            "samples": int(entry["samples"]),
            "detected": int(detected),
            "detection_rate_pct": (detected / entry["samples"] * 100.0) if entry["samples"] else 0.0,
            "mean_confidence": (entry["mean_confidence_sum"] / detected) if detected else 0.0,
        }
    total = len(samples)
    total_det = int(sum(p["detected"] for p in payload))
    return {
        "per_letter": finalized,
        "global": {
            "total_samples": total,
            "detected_samples": total_det,
            "detection_rate_pct": (total_det / total * 100.0) if total else 0.0,
            "unique_letters": len(finalized),
        },
    }


def run(input_dir: Path, output_h5: Path, stats_json: Path, models_dir: Path) -> None:
    samples, skipped = collect_samples(input_dir)
    if not samples:
        logger.warning("No recognizable images found in {}", input_dir)
        return
    logger.info(
        "Found {} parseable images across {} labels ({} skipped).",
        len(samples),
        len({s.label for s in samples}),
        skipped,
    )

    model_paths = ensure_models(models_dir)
    detector = make_image_hand_detector(model_paths["hand"])

    payload: List[Dict[str, np.ndarray]] = []
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("Extracting Alfabeti"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("images", total=len(samples))
            for sample in samples:
                xyz, conf, angles, detected = extract_image(detector, sample.path)
                payload.append({"xyz": xyz, "conf": conf, "angles": angles, "detected": detected})
                progress.advance(task)
    finally:
        detector.close()

    write_hdf5(output_h5, samples, payload)
    stats = build_stats(samples, payload)
    stats["global"]["skipped_unparseable"] = skipped
    stats_json.parent.mkdir(parents=True, exist_ok=True)
    stats_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "Done. Detected {}/{} ({:.1f}%). HDF5={} Stats={}",
        stats["global"]["detected_samples"],
        stats["global"]["total_samples"],
        stats["global"]["detection_rate_pct"],
        output_h5,
        stats_json,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="AlbSL Alfabeti image extractor")
    ap.add_argument("--input", type=Path, default=Path("datasets/raw/images/Alfabeti"))
    ap.add_argument("--output", type=Path, default=Path("datasets/processed/core_data/data/alfabeti_keypoints.h5"))
    ap.add_argument("--stats", type=Path, default=Path("datasets/processed/core_data/data/alfabeti_stats.json"))
    ap.add_argument("--models-dir", type=Path, default=Path("models/mediapipe/mp_models"))
    return ap.parse_args()


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    args = parse_args()
    run(args.input, args.output, args.stats, args.models_dir)


if __name__ == "__main__":
    main()
