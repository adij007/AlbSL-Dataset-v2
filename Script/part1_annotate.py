"""
AlbSL Pipeline — Part 1: Hand Keypoint Annotation
==================================================
Recursively scans the entire datasets/raw/images/Alfabeti/ folder tree at any depth.
Parses the letter from each filename using the pattern:
    hand{N}_{letter}_{...}_cropped.{ext}  →  letter = parts[1]

Images whose filenames do not match the pattern (e.g. GUID-named
PNGs) are annotated and saved under _UNKNOWN/.

All annotated copies are saved to:
    datasets/raw/images/Alfabeti_Annotated/{LETTER or _UNKNOWN}/{original_stem}_annotated{ext}
"""

import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import json
import logging
import sys
import re
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR        = Path("datasets/raw/images/Alfabeti")
OUTPUT_ROOT        = Path("datasets/raw/images/Alfabeti_Annotated")
MODEL_PATH         = Path("models/base/hand_landmarker.task")
SUPPORTED_EXT      = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_DETECTION_CONF = 0.4
MAX_HANDS          = 2

# Matches: hand1_a_bot_seg_1_cropped.jpeg  →  group(1) = 'a'
FILENAME_RE = re.compile(r'^hand\d+_([a-zA-Z])_', re.IGNORECASE)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ── Logging ───────────────────────────────────────────────────────────────────
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(OUTPUT_ROOT.parent / "annotation_log.txt")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def extract_letter(filename: str):
    m = FILENAME_RE.match(filename)
    return m.group(1).upper() if m else None


def get_detector():
    # ── Try new Task API ──────────────────────────────────────────────────
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python.vision import (
            HandLandmarker, HandLandmarkerOptions, RunningMode
        )
        if not MODEL_PATH.exists():
            log.info("Downloading hand_landmarker.task (~25 MB) ...")
            urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
            log.info("Model saved: %s", MODEL_PATH)

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
                return _lm.detect(img).hand_landmarks
            def close(self): _lm.close()

        return NewAPI()
    except Exception as e:
        log.warning("New Task API failed (%s), trying legacy ...", e)

    # ── Fall back to legacy solutions API ─────────────────────────────────
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
                return res.multi_hand_landmarks or []
            def close(self): _hands.close()

        return LegacyAPI()
    except Exception as e:
        log.error("Both MediaPipe APIs failed: %s", e)
        sys.exit(1)


def draw_hand(frame: np.ndarray, hand_landmarks) -> None:
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for (a, b) in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2, cv2.LINE_AA)
    for idx, (cx, cy) in enumerate(pts):
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, str(idx), (cx + 5, cy - 4),
                    FONT, 0.35, (255, 255, 255), 1, cv2.LINE_AA)


def annotate_dataset():
    detector = get_detector()

    # ── Collect ALL images recursively under DATASET_DIR ─────────────────
    all_images = sorted([
        p for p in DATASET_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    ])

    if not all_images:
        log.error("No images found anywhere under %s", DATASET_DIR)
        detector.close()
        return

    log.info("Found %d images (recursive scan of %s)", len(all_images), DATASET_DIR)

    # ── Group by letter parsed from filename ──────────────────────────────
    by_letter: dict[str, list[Path]] = {}
    unmatched_count = 0

    for p in all_images:
        letter = extract_letter(p.name)
        if letter is None:
            by_letter.setdefault("_UNKNOWN", []).append(p)
            unmatched_count += 1
        else:
            by_letter.setdefault(letter, []).append(p)

    log.info("Letters found: %s  (%d images matched, %d unmatched → _UNKNOWN)",
             sorted(k for k in by_letter.keys() if k != "_UNKNOWN"),
             sum(len(v) for k, v in by_letter.items() if k != "_UNKNOWN"),
             unmatched_count)

    summary: dict = {}
    failed:  list = []

    try:
        for letter in sorted(by_letter.keys()):
            images  = by_letter[letter]
            out_dir = OUTPUT_ROOT / letter
            out_dir.mkdir(parents=True, exist_ok=True)
            found = missed = 0

            for img_path in tqdm(images, desc=f"  Letter {letter}", leave=False):
                frame = cv2.imread(str(img_path))
                if frame is None:
                    log.warning("Cannot read: %s", img_path)
                    missed += 1
                    failed.append(str(img_path))
                    continue

                hands = detector.detect(frame)

                if hands:
                    for h_lms in hands:
                        draw_hand(frame, h_lms)
                    found += 1
                else:
                    cv2.putText(frame, "NO HAND DETECTED", (10, 30),
                                FONT, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                    missed += 1
                    failed.append(str(img_path))

                out_path = out_dir / f"{img_path.stem}_annotated{img_path.suffix}"
                cv2.imwrite(str(out_path), frame)

            summary[letter] = {
                "found": found, "missed": missed, "total": found + missed
            }
            pct = 100 * found / max(found + missed, 1)
            log.info("  %s : %d/%d detected (%.0f%%)", letter, found, found + missed, pct)

    finally:
        detector.close()

    # ── Save summary JSON ─────────────────────────────────────────────────
    s_path = OUTPUT_ROOT.parent / "annotation_summary.json"
    with open(str(s_path), "w", encoding="utf-8") as fh:
        json.dump({"per_letter": summary, "failed_images": failed}, fh, indent=2)

    tf = sum(v["found"]  for v in summary.values())
    tm = sum(v["missed"] for v in summary.values())
    log.info("")
    log.info("════ ANNOTATION COMPLETE ════")
    log.info("  Total detected  : %d / %d  (%.1f%%)", tf, tf + tm,
             100 * tf / max(tf + tm, 1))
    log.info("  Annotated images → %s", OUTPUT_ROOT)
    log.info("  Summary JSON     → %s", s_path)
    if tm:
        log.warning("  %d images had no hand detected. "
                    "Lower MIN_DETECTION_CONF (currently %.2f) to catch more.",
                    tm, MIN_DETECTION_CONF)


if __name__ == "__main__":
    annotate_dataset()