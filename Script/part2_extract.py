"""
AlbSL Pipeline — Part 2: Keypoint Coordinate Extraction → CSV
==============================================================
Reads the flat datasets/raw/images/Alfabeti/ folder, parses the letter from each filename,
extracts normalised hand keypoints, and writes to datasets/processed/core_data/data/csv/alfabeti_keypoints.csv.

Filename format: hand{N}_{letter}_{angle}_{seg}_{frame}_cropped.{ext}
Images that do not match the pattern (e.g. GUID-named PNGs) are
labelled as UNKNOWN.

The final CSV is combined with the existing albsl_keypoints.csv.
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import urllib.request
import logging
import sys
import re
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR        = Path("datasets/raw/images/Alfabeti")
OUTPUT_CSV         = Path("datasets/processed/core_data/data/csv/alfabeti_keypoints.csv")
EXISTING_CSV       = Path("datasets/processed/core_data/exports/albsl_keypoints.csv")
MODEL_PATH         = Path("models/base/hand_landmarker.task")
SUPPORTED_EXT      = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_DETECTION_CONF = 0.4
MAX_HANDS          = 2

FILENAME_RE = re.compile(r'^hand\d+_([a-zA-Z])_', re.IGNORECASE)

NUM_LANDMARKS = 21
BONE_PAIRS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]
BONE_NAMES = [f"bone_{p}_{c}" for p, c in BONE_PAIRS]
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC","THUMB_MCP","THUMB_IP","THUMB_TIP",
    "INDEX_MCP","INDEX_PIP","INDEX_DIP","INDEX_TIP",
    "MIDDLE_MCP","MIDDLE_PIP","MIDDLE_DIP","MIDDLE_TIP",
    "RING_MCP","RING_PIP","RING_DIP","RING_TIP",
    "PINKY_MCP","PINKY_PIP","PINKY_DIP","PINKY_TIP",
]
RAW_COLS  = [f"lm{i}_{ax}_raw"  for i in range(NUM_LANDMARKS) for ax in ("x","y","z")]
NORM_COLS = [f"lm{i}_{ax}_norm" for i in range(NUM_LANDMARKS) for ax in ("x","y","z")]
BONE_COLS = [f"{bn}_{ax}" for bn in BONE_NAMES for ax in ("dx","dy","dz")]
META_COLS = ["letter","image_file","hand_label","detection_confidence",
             "img_width","img_height"]
ALL_COLS  = META_COLS + RAW_COLS + NORM_COLS + BONE_COLS

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# ── Logging ───────────────────────────────────────────────────────────────────
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(OUTPUT_CSV.parent / "extraction_log.txt")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def extract_letter(filename: str) -> str | None:
    m = FILENAME_RE.match(filename)
    return m.group(1).upper() if m else None


def get_detector():
    # New Task API
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python.vision import (
            HandLandmarker, HandLandmarkerOptions, RunningMode
        )
        if not MODEL_PATH.exists():
            log.info("Downloading hand_landmarker.task (~25 MB) ...")
            urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))

        opts = HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=RunningMode.IMAGE,
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=MIN_DETECTION_CONF,
            min_hand_presence_confidence=MIN_DETECTION_CONF,
        )
        _lm = HandLandmarker.create_from_options(opts)
        log.info("Using MediaPipe NEW Task API")

        class NewAPI:
            def detect(self, bgr):
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = _lm.detect(img)
                out = []
                for lms, handedness in zip(res.hand_landmarks, res.handedness):
                    out.append((lms, handedness[0].display_name, handedness[0].score))
                return out
            def close(self): _lm.close()

        return NewAPI()
    except Exception as e:
        log.warning("New Task API failed (%s), trying legacy ...", e)

    # Legacy solutions API
    try:
        _hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=MAX_HANDS,
            min_detection_confidence=MIN_DETECTION_CONF,
        )
        log.info("Using MediaPipe LEGACY solutions API")

        class LegacyAPI:
            def detect(self, bgr):
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                res = _hands.process(rgb)
                if not res.multi_hand_landmarks:
                    return []
                out = []
                for lms, handedness in zip(res.multi_hand_landmarks,
                                           res.multi_handedness):
                    label = handedness.classification[0].label
                    conf  = handedness.classification[0].score
                    out.append((lms.landmark, label, conf))
                return out
            def close(self): _hands.close()

        return LegacyAPI()
    except Exception as e:
        log.error("Both APIs failed: %s", e)
        sys.exit(1)


def landmarks_to_features(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    raw      = pts.flatten()
    relative = pts - pts[0]
    max_abs  = np.abs(relative).max()
    normalized = (relative / max(max_abs, 1e-6)).flatten()
    bone_vecs  = np.array(
        [relative[c] - relative[p] for (p,c) in BONE_PAIRS],
        dtype=np.float32,
    ).flatten()
    return raw, normalized, bone_vecs


def extract_keypoints():
    detector = get_detector()
    rows  = []
    stats = {"processed": 0, "no_hand": 0, "error": 0}

    all_images = sorted([
        p for p in DATASET_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    ])

    if not all_images:
        log.error("No images found in %s", DATASET_DIR)
        detector.close()
        return pd.DataFrame(columns=ALL_COLS)

    log.info("Found %d images — extracting keypoints ...", len(all_images))

    try:
        for img_path in tqdm(all_images, desc="Extracting"):
            letter = extract_letter(img_path.name)
            if letter is None:
                letter = "UNKNOWN"

            try:
                frame = cv2.imread(str(img_path))
                if frame is None:
                    raise ValueError("cv2.imread returned None")

                h, w  = frame.shape[:2]
                hands = detector.detect(frame)

                if not hands:
                    stats["no_hand"] += 1
                    continue

                for (lms, label, conf) in hands:
                    raw, normalized, bone_vecs = landmarks_to_features(lms)
                    row = {
                        "letter":               letter,
                        "image_file":           img_path.name,
                        "hand_label":           label,
                        "detection_confidence": round(float(conf), 4),
                        "img_width":            w,
                        "img_height":           h,
                    }
                    for col, val in zip(RAW_COLS,  raw):
                        row[col] = round(float(val), 6)
                    for col, val in zip(NORM_COLS, normalized):
                        row[col] = round(float(val), 6)
                    for col, val in zip(BONE_COLS, bone_vecs):
                        row[col] = round(float(val), 6)
                    rows.append(row)

                stats["processed"] += 1

            except Exception as exc:
                log.error("  Error on %s: %s", img_path.name, exc)
                stats["error"] += 1

    finally:
        detector.close()

    df_new = pd.DataFrame(rows, columns=ALL_COLS)

    # ── Combine with existing albsl_keypoints.csv ─────────────────────────
    if EXISTING_CSV.exists():
        df_old = pd.read_csv(str(EXISTING_CSV))
        log.info("Loaded %d existing rows from %s", len(df_old), EXISTING_CSV)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["image_file", "hand_label"], keep="last")
        log.info("After dedup: %d rows", len(df))
    else:
        df = df_new
        log.info("No existing CSV found at %s — using new data only", EXISTING_CSV)

    df.to_csv(str(OUTPUT_CSV), index=False)

    log.info("\nExtraction complete.")
    log.info("  Total images    : %d", len(all_images))
    log.info("  New rows        : %d", len(df_new))
    log.info("  Combined rows   : %d", len(df))
    log.info("  Processed OK    : %d", stats["processed"])
    log.info("  No hand found   : %d", stats["no_hand"])
    log.info("  Errors          : %d", stats["error"])
    log.info("  CSV saved to    : %s", OUTPUT_CSV)

    log.info("\nPer-letter row counts:")
    for letter, grp in df.groupby("letter"):
        log.info("  %s : %d rows", letter, len(grp))

    return df


if __name__ == "__main__":
    extract_keypoints()