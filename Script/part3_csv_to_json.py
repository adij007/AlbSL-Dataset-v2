"""
AlbSL Pipeline — Part 3: CSV → Per-Letter JSON Knowledge Base
=============================================================
Converts the keypoint CSV produced by Part 2 into a rich, queryable JSON
structure that the recognition model, a real-time inference engine, or any
downstream consumer can load without touching the original images or CSV.

JSON schema per letter
----------------------
{
  "letter": "A",
  "total_samples": 47,
  "statistics": {
    "mean_confidence": 0.963,
    "hands": {"Right": 40, "Left": 7},
    "per_landmark": {
      "WRIST": {"mean_x": ..., "std_x": ..., ...},
      ...
    }
  },
  "representative": {
    "source_image": "A_042.jpg",
    "confidence": 0.991,
    "landmarks": [
      {"id": 0, "name": "WRIST",       "x": 0.0, "y": 0.0, "z": 0.0},
      {"id": 1, "name": "THUMB_CMC",   "x": 0.12, "y": -0.08, "z": ...},
      ...
    ]
  },
  "samples": [
    {
      "id": 0,
      "source_image": "A_001.jpg",
      "hand_label": "Right",
      "confidence": 0.98,
      "landmarks": [
        {"id": 0, "name": "WRIST", "x_raw": 0.51, "y_raw": 0.73, "z_raw": 0.0,
                                    "x_norm": 0.0,  "y_norm": 0.0,  "z_norm": 0.0},
        ...
      ],
      "bone_vectors": [
        {"bone": "bone_0_1", "dx": 0.04, "dy": -0.07, "dz": 0.0},
        ...
      ]
    },
    ...
  ]
}

Output files
------------
datasets/processed/landmarks/albsl_landmarks\
    A.json
    B.json
    ...
    _index.json          ← lightweight index over all letters
    _full_dataset.json   ← single merged file (optional, can be large)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
INPUT_CSV    = Path("datasets/processed/core_data/data/csv/alfabeti_keypoints.csv")
OUTPUT_DIR   = Path("datasets/processed/landmarks/albsl_landmarks")
WRITE_FULL   = True   # Set False to skip the large merged file

NUM_LANDMARKS = 21
BONE_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
BONE_NAMES = [f"bone_{p}_{c}" for p, c in BONE_PAIRS]

LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP",  "INDEX_PIP",  "INDEX_DIP",  "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP",   "RING_PIP",   "RING_DIP",   "RING_TIP",
    "PINKY_MCP",  "PINKY_PIP",  "PINKY_DIP",  "PINKY_TIP",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR.parent / "json_build_log.txt"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def f(v) -> float:
    """Round to 6 decimal places for compact JSON."""
    return round(float(v), 6)


def row_to_sample(row: pd.Series, sample_id: int) -> dict:
    """Convert a single CSV row into the sample sub-document."""
    landmarks = []
    for i, name in enumerate(LANDMARK_NAMES):
        landmarks.append({
            "id":     i,
            "name":   name,
            "x_raw":  f(row[f"lm{i}_x_raw"]),
            "y_raw":  f(row[f"lm{i}_y_raw"]),
            "z_raw":  f(row[f"lm{i}_z_raw"]),
            "x_norm": f(row[f"lm{i}_x_norm"]),
            "y_norm": f(row[f"lm{i}_y_norm"]),
            "z_norm": f(row[f"lm{i}_z_norm"]),
        })

    bone_vectors = []
    for bone_name in BONE_NAMES:
        bone_vectors.append({
            "bone": bone_name,
            "dx": f(row[f"{bone_name}_dx"]),
            "dy": f(row[f"{bone_name}_dy"]),
            "dz": f(row[f"{bone_name}_dz"]),
        })

    return {
        "id":           sample_id,
        "source_image": str(row["image_file"]),
        "hand_label":   str(row["hand_label"]),
        "confidence":   f(row["detection_confidence"]),
        "img_width":    int(row["img_width"]),
        "img_height":   int(row["img_height"]),
        "landmarks":    landmarks,
        "bone_vectors": bone_vectors,
    }


def compute_statistics(letter_df: pd.DataFrame) -> dict:
    """
    Compute per-letter aggregate statistics for quick model sanity checks
    and as a compact 'fingerprint' of each letter's hand shape.
    """
    stats: dict = {}

    # Overall detection quality
    stats["mean_confidence"] = f(letter_df["detection_confidence"].mean())
    stats["std_confidence"]  = f(letter_df["detection_confidence"].std())

    # Hand chirality distribution
    hand_counts = letter_df["hand_label"].value_counts().to_dict()
    stats["hands"] = {k: int(v) for k, v in hand_counts.items()}

    # Per-landmark statistics on the NORMALIZED coordinates
    # (these values describe the typical hand shape for this letter)
    per_lm: dict = {}
    for i, name in enumerate(LANDMARK_NAMES):
        xc = letter_df[f"lm{i}_x_norm"]
        yc = letter_df[f"lm{i}_y_norm"]
        zc = letter_df[f"lm{i}_z_norm"]
        per_lm[name] = {
            "mean_x": f(xc.mean()), "std_x": f(xc.std()),
            "mean_y": f(yc.mean()), "std_y": f(yc.std()),
            "mean_z": f(zc.mean()), "std_z": f(zc.std()),
        }
    stats["per_landmark"] = per_lm

    # Median normalized pose (compact reference vector for fast distance matching)
    median_pose = []
    for i, name in enumerate(LANDMARK_NAMES):
        median_pose.append({
            "id":   i,
            "name": name,
            "x":    f(letter_df[f"lm{i}_x_norm"].median()),
            "y":    f(letter_df[f"lm{i}_y_norm"].median()),
            "z":    f(letter_df[f"lm{i}_z_norm"].median()),
        })
    stats["median_pose"] = median_pose

    return stats


def pick_representative(letter_df: pd.DataFrame) -> tuple[pd.Series, int]:
    """
    Select the sample with the highest detection confidence as the
    'canonical' example of this letter.
    """
    best_idx = letter_df["detection_confidence"].idxmax()
    return letter_df.loc[best_idx], int(best_idx)


def build_representative_doc(row: pd.Series) -> dict:
    """Compact landmarks-only view of the representative sample."""
    landmarks = []
    for i, name in enumerate(LANDMARK_NAMES):
        landmarks.append({
            "id":   i,
            "name": name,
            "x":    f(row[f"lm{i}_x_norm"]),
            "y":    f(row[f"lm{i}_y_norm"]),
            "z":    f(row[f"lm{i}_z_norm"]),
        })
    return {
        "source_image": str(row["image_file"]),
        "hand_label":   str(row["hand_label"]),
        "confidence":   f(row["detection_confidence"]),
        "landmarks":    landmarks,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main builder
# ──────────────────────────────────────────────────────────────────────────────
def build_json_knowledge_base() -> None:
    """
    Read INPUT_CSV, split by letter, write one JSON file per letter plus
    a lightweight _index.json and optional _full_dataset.json.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Reading CSV: %s", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV)
    log.info("  Loaded %d rows, %d columns", len(df), len(df.columns))

    letters = sorted(df["letter"].unique())
    log.info("  Letters found: %s", letters)

    index_records: list[dict] = []
    full_dataset: list[dict]  = []

    for letter in tqdm(letters, desc="Building JSON"):
        letter_df = df[df["letter"] == letter].reset_index(drop=True)
        n = len(letter_df)

        # Build statistics
        stats = compute_statistics(letter_df)

        # Best single example
        rep_row, rep_local_idx = pick_representative(letter_df)
        representative = build_representative_doc(rep_row)

        # All samples
        samples = [
            row_to_sample(letter_df.iloc[i], i)
            for i in range(n)
        ]

        letter_doc = {
            "letter":          letter,
            "total_samples":   n,
            "description": (
                (
                    f"AlbSL fingerspelling letter '{letter}'. "
                    if len(letter) == 1 else
                    f"AlbSL dataset group '{letter}' (images with no letter label). "
                ) +
                f"Contains {n} annotated hand pose samples. "
                f"Use 'representative.landmarks' for fast nearest-neighbour "
                f"matching or 'statistics.median_pose' for a robust reference. "
                f"All coordinate values in 'x_norm/y_norm/z_norm' fields are "
                f"wrist-centered and scale-normalised to [-1,+1]."
            ),
            "coordinate_system": {
                "origin":    "WRIST joint (landmark 0)",
                "scale":     "Divided by max absolute coordinate → range [-1, +1]",
                "x_axis":    "Horizontal (positive = right in image)",
                "y_axis":    "Vertical (positive = down in image)",
                "z_axis":    "Depth estimate from MediaPipe (positive = toward camera)",
                "raw_range": "[0.0, 1.0] image fraction (MediaPipe default)",
            },
            "statistics":    stats,
            "representative": representative,
            "samples":        samples,
        }

        # Write individual letter file
        out_path = OUTPUT_DIR / f"{letter}.json"
        with open(out_path, "w", encoding="utf-8") as f_out:
            json.dump(letter_doc, f_out, ensure_ascii=False, indent=2)

        log.info("  Wrote %s (%d samples, %.1f KB)",
                 out_path.name, n, out_path.stat().st_size / 1024)

        # Index entry (lightweight, no per-sample data)
        index_records.append({
            "letter":          letter,
            "total_samples":   n,
            "mean_confidence": stats["mean_confidence"],
            "hands":           stats["hands"],
            "file":            f"{letter}.json",
            "representative_image": representative["source_image"],
        })

        if WRITE_FULL:
            full_dataset.append(letter_doc)

    # ── _index.json ────────────────────────────────────────────────────────
    index_doc = {
        "description": (
            "AlbSL fingerspelling alphabet — landmark JSON knowledge base index. "
            "Load a specific letter by reading the file named in the 'file' field. "
            "Use 'total_samples' and 'mean_confidence' to assess data quality per letter."
        ),
        "total_letters": len(letters),
        "total_samples": int(df.shape[0]),
        "letters": index_records,
    }
    index_path = OUTPUT_DIR / "_index.json"
    with open(index_path, "w", encoding="utf-8") as f_out:
        json.dump(index_doc, f_out, ensure_ascii=False, indent=2)
    log.info("\nIndex written: %s", index_path)

    # ── _full_dataset.json (optional) ──────────────────────────────────────
    if WRITE_FULL:
        full_path = OUTPUT_DIR / "_full_dataset.json"
        with open(full_path, "w", encoding="utf-8") as f_out:
            json.dump({
                "description": "Complete AlbSL landmark knowledge base (all letters merged).",
                "total_letters": len(letters),
                "total_samples": int(df.shape[0]),
                "alphabet": full_dataset,
            }, f_out, ensure_ascii=False, indent=2)
        log.info("Full dataset written: %s (%.1f MB)",
                 full_path, full_path.stat().st_size / 1024 / 1024)

    log.info("\nDone. JSON files saved to: %s", OUTPUT_DIR)


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    build_json_knowledge_base()
