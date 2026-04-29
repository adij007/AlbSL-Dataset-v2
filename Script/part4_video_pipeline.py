"""
AlbSL Pipeline — Video Annotation & Keypoint Extraction
=========================================================
Processes every video clip in D:\\AlbSl-Dataset-v2\\split_clips\\
Recursively scans all sub-folders. Letter is parsed from the video filename
using the same pattern as the image pipeline:
    hand{N}_{letter}_{angle}_{clip}.mp4  ->  letter = parts[1]

TWO outputs per video
---------------------
1. ANNOTATED VIDEO
   D:\\AlbSl-Dataset-v2\\split_clips_annotated\\{LETTER}\\{stem}_annotated.mp4
   Every frame has:
     - Green skeleton lines  (21 hand joints × 2 hands)
     - Red landmark dots + joint index numbers
     - Blue pose skeleton    (33 body joints)
     - Teal face mesh        (468 face landmarks, subsampled for clarity)
     - HUD overlay: letter label, frame number, FPS, detection confidence

2. KEYPOINT CSV
   D:\\AlbSl-Dataset-v2\\video_keypoints.csv
   One row per frame per video. Columns:
     Meta  : letter, video_file, relative_path, frame_idx, timestamp_ms,
              img_width, img_height
     Hands : 21 landmarks x 2 hands x {x_raw, y_raw, z_raw,
                                        x_norm, y_norm, z_norm}
             + 20 bone vectors x 2 hands x {dx, dy, dz}
             + hand_detected_left, hand_detected_right (bool)
     Pose  : 33 landmarks x {x, y, z, visibility}
     Face  : 10 key face points x {x, y, z}   (reduced from 468)
     Motion: per-joint velocity (delta from previous frame) for both hands

Why three streams?
------------------
This directly feeds the STMC + MS-TCN++ architecture from the AlbSL skill:
  Stream 1 (Manual)     : hand + bone + velocity columns
  Stream 2 (Non-Manual) : pose + face columns
  Stream 3 (Temporal)   : motion/velocity + optical-flow magnitude

Dependencies
------------
  pip install mediapipe opencv-python pandas numpy tqdm
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import urllib.request
import json
import logging
import sys
import re
import time
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_DIR          = Path("datasets/processed/clips/split_clips")
ANNOTATED_ROOT     = Path("datasets/processed/clips/split_clips_annotated")
OUTPUT_CSV         = Path("datasets/processed/core_data/video_keypoints.csv")
MODEL_PATH         = Path("models/base/hand_landmarker.task")
SUPPORTED_VID_EXT  = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

MIN_HAND_CONF      = 0.2    # low threshold — same as image pipeline
MIN_POSE_CONF      = 0.5
MAX_HANDS          = 2

# Output video codec — change to "XVID" if mp4v causes issues
FOURCC             = cv2.VideoWriter_fourcc(*"mp4v")

# Filename pattern  hand{N}_{letter}_{...}.mp4
# Supports Albanian digraph letters (Dh, Gj, Ll, Nj, Rr, Sh, Th, Xh, Zh) and Ç/Ë.
FILENAME_RE = re.compile(r'^hand\d+_([a-zA-ZÇËçë]{1,2})_', re.IGNORECASE)
ALLOWED_LETTERS = {
    "A", "B", "C", "Ç", "D", "DH", "E", "Ë", "F", "G", "GJ", "H", "I", "J", "K",
    "L", "LL", "M", "N", "NJ", "O", "P", "Q", "R", "RR", "S", "SH", "T", "TH",
    "U", "V", "X", "XH", "Y", "Z", "ZH",
}

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# MediaPipe hand skeleton connections
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

# Bone pairs for feature extraction (same as image pipeline)
BONE_PAIRS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]
BONE_NAMES = [f"bone_{p}_{c}" for p, c in BONE_PAIRS]

# Key face landmark indices (from 468-point mesh — anatomically stable points)
FACE_KEY_INDICES = [
    1,    # nose tip
    33,   # left eye outer corner
    263,  # right eye outer corner
    61,   # left mouth corner
    291,  # right mouth corner
    199,  # chin
    10,   # forehead centre
    152,  # jaw bottom
    234,  # left cheek
    454,  # right cheek
]

NUM_HAND_LM   = 21
NUM_POSE_LM   = 33
NUM_FACE_KEY  = len(FACE_KEY_INDICES)
NUM_BONES     = len(BONE_PAIRS)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ── Column definitions ────────────────────────────────────────────────────────
META_COLS = [
    "letter", "video_file", "relative_path",
    "frame_idx", "timestamp_ms", "img_width", "img_height",
    "hand_detected_left", "hand_detected_right",
]

def _hand_cols(side: str) -> list[str]:
    raw  = [f"{side}_lm{i}_{ax}_raw"  for i in range(NUM_HAND_LM) for ax in ("x","y","z")]
    norm = [f"{side}_lm{i}_{ax}_norm" for i in range(NUM_HAND_LM) for ax in ("x","y","z")]
    bone = [f"{side}_{bn}_{ax}"       for bn in BONE_NAMES for ax in ("dx","dy","dz")]
    vel  = [f"{side}_vel{i}_{ax}"     for i in range(NUM_HAND_LM) for ax in ("vx","vy","vz")]
    return raw + norm + bone + vel

POSE_COLS = [
    f"pose_lm{i}_{ax}"
    for i in range(NUM_POSE_LM)
    for ax in ("x","y","z","vis")
]
FACE_COLS = [
    f"face_{idx}_{ax}"
    for idx in FACE_KEY_INDICES
    for ax in ("x","y","z")
]
OPTICAL_FLOW_COLS = ["opt_flow_mean_mag", "opt_flow_max_mag"]

ALL_COLS = (
    META_COLS
    + _hand_cols("left")
    + _hand_cols("right")
    + POSE_COLS
    + FACE_COLS
    + OPTICAL_FLOW_COLS
)

# ── Logging ───────────────────────────────────────────────────────────────────
ANNOTATED_ROOT.mkdir(parents=True, exist_ok=True)

# Force UTF-8 on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            str(VIDEO_DIR.parent / "video_extraction_log.txt"),
            encoding="utf-8",
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ── Helper: download model ────────────────────────────────────────────────────
def ensure_model():
    if not MODEL_PATH.exists():
        log.info("Downloading hand_landmarker.task (~25 MB) ...")
        urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
        log.info("Model saved: %s", MODEL_PATH)


# ── Helper: parse letter ──────────────────────────────────────────────────────
def extract_letter(filename: str):
    m = FILENAME_RE.match(filename)
    if m:
        token = m.group(1)
        up = token.upper()
        if up in ALLOWED_LETTERS:
            return token[0].upper() + token[1:].lower()
    
    # Support A.mp4, B.mp4 filenames directly
    stem = Path(filename).stem
    if len(stem) <= 2:
        up = stem.upper()
        if up in ALLOWED_LETTERS:
            return stem[0].upper() + stem[1:].lower()
    
    return None


# ── Helper: feature extraction ───────────────────────────────────────────────
def hand_to_features(landmarks, prev_pts=None):
    """
    Returns (raw, norm, bones, velocity, pts_array)
    prev_pts: previous frame's (21,3) array or None
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks],
                   dtype=np.float32)                          # (21,3)
    raw      = pts.flatten()
    relative = pts - pts[0]
    max_abs  = np.abs(relative).max()
    norm     = (relative / max(max_abs, 1e-6)).flatten()
    bones    = np.array(
        [relative[c] - relative[p] for (p,c) in BONE_PAIRS],
        dtype=np.float32,
    ).flatten()
    if prev_pts is not None:
        vel = (pts - prev_pts).flatten()
    else:
        vel = np.zeros(NUM_HAND_LM * 3, dtype=np.float32)
    return raw, norm, bones, vel, pts


def empty_hand_features():
    n = NUM_HAND_LM * 3
    b = NUM_BONES * 3
    return (
        np.zeros(n, np.float32),   # raw
        np.zeros(n, np.float32),   # norm
        np.zeros(b, np.float32),   # bones
        np.zeros(n, np.float32),   # vel
        None,                       # pts (no detection)
    )


# ── Helper: optical flow magnitude ───────────────────────────────────────────
def optical_flow_stats(prev_gray, curr_gray):
    if prev_gray is None:
        return 0.0, 0.0
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None, 0.5, 3, 15, 3, 5, 1.2, 0,
    )
    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return float(mag.mean()), float(mag.max())


# ── Helper: drawing ───────────────────────────────────────────────────────────
def draw_hand_on_frame(frame, hand_lms, color_line=(0,255,0), color_dot=(0,0,255)):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
    for (a, b) in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color_line, 2, cv2.LINE_AA)
    for idx, (cx, cy) in enumerate(pts):
        cv2.circle(frame, (cx, cy), 5, color_dot, -1, cv2.LINE_AA)
        cv2.putText(frame, str(idx), (cx+4, cy-3),
                    FONT, 0.28, (255,255,255), 1, cv2.LINE_AA)


def draw_pose_on_frame(frame, pose_lms):
    """Draw a simplified body skeleton (shoulders, elbows, wrists, hips)."""
    if pose_lms is None:
        return
    h, w = frame.shape[:2]
    # Key connections: shoulders(11,12), arms(11,13,15), (12,14,16), hips(23,24)
    POSE_CONN = [
        (11,12),(11,13),(13,15),(12,14),(14,16),
        (11,23),(12,24),(23,24),
    ]
    lms = pose_lms.landmark
    for (a, b) in POSE_CONN:
        xa, ya = int(lms[a].x * w), int(lms[a].y * h)
        xb, yb = int(lms[b].x * w), int(lms[b].y * h)
        if lms[a].visibility > 0.4 and lms[b].visibility > 0.4:
            cv2.line(frame, (xa, ya), (xb, yb), (255, 200, 0), 2, cv2.LINE_AA)


def draw_face_dots(frame, face_lms):
    """Draw the 10 key face landmarks as small teal dots."""
    if face_lms is None:
        return
    h, w = frame.shape[:2]
    for idx in FACE_KEY_INDICES:
        lm = face_lms.landmark[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 3, (200, 255, 150), -1, cv2.LINE_AA)


def draw_hud(frame, letter, frame_idx, fps, left_conf, right_conf):
    """Draw semi-transparent HUD overlay at the top of the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.putText(frame, f"Letter: {letter}", (8, 18),
                FONT, 0.58, (0, 255, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Frame: {frame_idx}", (8, 36),
                FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w//2 - 40, 18),
                FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    l_color = (0, 255, 0) if left_conf  > 0 else (80, 80, 80)
    r_color = (0, 200, 255) if right_conf > 0 else (80, 80, 80)
    cv2.putText(frame, f"L:{left_conf:.2f}" if left_conf  > 0 else "L: --",
                (w - 130, 18), FONT, 0.45, l_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"R:{right_conf:.2f}" if right_conf > 0 else "R: --",
                (w - 130, 36), FONT, 0.45, r_color, 1, cv2.LINE_AA)


# ── Core: process one video ───────────────────────────────────────────────────
def process_video(
    video_path: Path,
    letter: str,
    hand_detector,
    holistic_detector,
    annotated_out_dir: Path,
) -> list[dict]:
    """
    Annotates one video and extracts per-frame keypoint rows.
    Returns list of row dicts (one per frame where at least one hand was found,
    plus rows with zeros for frames with no detection so the sequence is complete).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error("  Cannot open: %s", video_path.name)
        return []

    fps_in  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video writer
    annotated_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = annotated_out_dir / f"{video_path.stem}_annotated{video_path.suffix}"
    writer   = cv2.VideoWriter(str(out_path), FOURCC, fps_in, (width, height))

    try:
        rel_path = str(video_path.relative_to(VIDEO_DIR))
    except ValueError:
        rel_path = video_path.name

    rows: list[dict] = []
    prev_gray  = None
    prev_left  = None   # previous frame left  hand pts (21,3)
    prev_right = None   # previous frame right hand pts (21,3)
    frame_idx  = 0
    t0 = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        curr_gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # ── Hand detection (new Task API) ──────────────────────────────────
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hand_result = hand_detector.detect(mp_img)

        left_raw = left_norm = left_bones = left_vel = None
        right_raw= right_norm= right_bones= right_vel= None
        left_pts = right_pts = None
        left_conf = right_conf = 0.0
        hand_det_left = hand_det_right = False

        for lms, handedness in zip(
            hand_result.hand_landmarks,
            hand_result.handedness,
        ):
            label = handedness[0].display_name  # "Left" / "Right"
            conf  = handedness[0].score
            if label == "Left":
                left_raw, left_norm, left_bones, left_vel, left_pts = \
                    hand_to_features(lms, prev_left)
                left_conf = conf
                hand_det_left = True
                draw_hand_on_frame(frame, lms,
                                   color_line=(0,255,0), color_dot=(0,0,255))
            else:
                right_raw, right_norm, right_bones, right_vel, right_pts = \
                    hand_to_features(lms, prev_right)
                right_conf = conf
                hand_det_right = True
                draw_hand_on_frame(frame, lms,
                                   color_line=(255,165,0), color_dot=(0,128,255))

        # Fill zeros for undetected hands
        if left_raw  is None: left_raw,  left_norm,  left_bones,  left_vel,  _ = empty_hand_features()
        if right_raw is None: right_raw, right_norm, right_bones, right_vel, _ = empty_hand_features()

        # ── Pose + Face (legacy Holistic) ─────────────────────────────────
        pose_lms = face_lms = None
        pose_vals = np.zeros(NUM_POSE_LM * 4, dtype=np.float32)
        face_vals = np.zeros(NUM_FACE_KEY * 3, dtype=np.float32)

        holistic_result = holistic_detector.process(rgb)
        if holistic_result.pose_landmarks:
            pose_lms = holistic_result.pose_landmarks
            pose_arr = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility]
                 for lm in pose_lms.landmark],
                dtype=np.float32,
            )
            pose_vals = pose_arr.flatten()
            draw_pose_on_frame(frame, pose_lms)

        if holistic_result.face_landmarks:
            face_lms = holistic_result.face_landmarks
            face_arr = np.array(
                [[face_lms.landmark[i].x,
                  face_lms.landmark[i].y,
                  face_lms.landmark[i].z]
                 for i in FACE_KEY_INDICES],
                dtype=np.float32,
            )
            face_vals = face_arr.flatten()
            draw_face_dots(frame, face_lms)

        # ── Optical flow ───────────────────────────────────────────────────
        flow_mean, flow_max = optical_flow_stats(prev_gray, curr_gray)

        # ── HUD ────────────────────────────────────────────────────────────
        elapsed = time.perf_counter() - t0
        live_fps = (frame_idx + 1) / max(elapsed, 1e-6)
        draw_hud(frame, letter, frame_idx, live_fps, left_conf, right_conf)

        writer.write(frame)

        # ── Build CSV row ──────────────────────────────────────────────────
        row: dict = {
            "letter":              letter,
            "video_file":          video_path.name,
            "relative_path":       rel_path,
            "frame_idx":           frame_idx,
            "timestamp_ms":        timestamp_ms,
            "img_width":           width,
            "img_height":          height,
            "hand_detected_left":  hand_det_left,
            "hand_detected_right": hand_det_right,
        }

        def _add(prefix, raw, norm, bones, vel):
            n = NUM_HAND_LM
            for i in range(n):
                row[f"{prefix}_lm{i}_x_raw"]  = round(float(raw[i*3+0]), 6)
                row[f"{prefix}_lm{i}_y_raw"]  = round(float(raw[i*3+1]), 6)
                row[f"{prefix}_lm{i}_z_raw"]  = round(float(raw[i*3+2]), 6)
                row[f"{prefix}_lm{i}_x_norm"] = round(float(norm[i*3+0]), 6)
                row[f"{prefix}_lm{i}_y_norm"] = round(float(norm[i*3+1]), 6)
                row[f"{prefix}_lm{i}_z_norm"] = round(float(norm[i*3+2]), 6)
                row[f"{prefix}_vel{i}_vx"]    = round(float(vel[i*3+0]), 6)
                row[f"{prefix}_vel{i}_vy"]    = round(float(vel[i*3+1]), 6)
                row[f"{prefix}_vel{i}_vz"]    = round(float(vel[i*3+2]), 6)
            for bi, bn in enumerate(BONE_NAMES):
                row[f"{prefix}_{bn}_dx"] = round(float(bones[bi*3+0]), 6)
                row[f"{prefix}_{bn}_dy"] = round(float(bones[bi*3+1]), 6)
                row[f"{prefix}_{bn}_dz"] = round(float(bones[bi*3+2]), 6)

        _add("left",  left_raw,  left_norm,  left_bones,  left_vel)
        _add("right", right_raw, right_norm, right_bones, right_vel)

        for i in range(NUM_POSE_LM):
            row[f"pose_lm{i}_x"]   = round(float(pose_vals[i*4+0]), 6)
            row[f"pose_lm{i}_y"]   = round(float(pose_vals[i*4+1]), 6)
            row[f"pose_lm{i}_z"]   = round(float(pose_vals[i*4+2]), 6)
            row[f"pose_lm{i}_vis"] = round(float(pose_vals[i*4+3]), 6)

        for fi, idx in enumerate(FACE_KEY_INDICES):
            row[f"face_{idx}_x"] = round(float(face_vals[fi*3+0]), 6)
            row[f"face_{idx}_y"] = round(float(face_vals[fi*3+1]), 6)
            row[f"face_{idx}_z"] = round(float(face_vals[fi*3+2]), 6)

        row["opt_flow_mean_mag"] = round(flow_mean, 6)
        row["opt_flow_max_mag"]  = round(flow_max,  6)

        rows.append(row)

        # ── Update previous-frame state ────────────────────────────────────
        prev_gray  = curr_gray
        prev_left  = left_pts  if left_pts  is not None else prev_left
        prev_right = right_pts if right_pts is not None else prev_right
        frame_idx += 1

    cap.release()
    writer.release()
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────
def process_all_videos():
    ensure_model()

    # ── Collect all videos recursively ────────────────────────────────────
    all_videos = sorted([
        p for p in VIDEO_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_VID_EXT
    ])

    if not all_videos:
        log.error("No video files found under %s", VIDEO_DIR)
        log.error("Supported extensions: %s", SUPPORTED_VID_EXT)
        return

    matched = [(p, extract_letter(p.name)) for p in all_videos]
    matched_ok  = [(p, l) for (p, l) in matched if l is not None]
    matched_bad = [p.name for (p, l) in matched if l is None]

    log.info("Found %d videos (%d matched, %d skipped)",
             len(all_videos), len(matched_ok), len(matched_bad))
    if matched_bad:
        log.warning("Skipped (unrecognised filename): %s", matched_bad[:5])

    letters = sorted({l for (_, l) in matched_ok})
    log.info("Letters to process: %s", letters)

    # ── Build detectors ───────────────────────────────────────────────────
    # Hand detector: new Task API
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python.vision import (
        HandLandmarker, HandLandmarkerOptions, RunningMode
    )
    hand_opts = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=MIN_HAND_CONF,
        min_hand_presence_confidence=MIN_HAND_CONF,
    )
    hand_detector = HandLandmarker.create_from_options(hand_opts)

    # Holistic: legacy solutions API (pose + face in one call)
    holistic_detector = mp.solutions.holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=MIN_POSE_CONF,
        min_tracking_confidence=MIN_POSE_CONF,
    )

    # ── Process videos ────────────────────────────────────────────────────
    all_rows: list[dict] = []
    summary:  dict       = {}

    try:
        for video_path, letter in tqdm(matched_ok, desc="Videos", unit="vid"):
            out_dir = ANNOTATED_ROOT / letter
            log.info("Processing [%s] %s", letter, video_path.name)

            rows = process_video(
                video_path, letter,
                hand_detector, holistic_detector,
                out_dir,
            )
            all_rows.extend(rows)

            frames_with_hand = sum(
                1 for r in rows
                if r["hand_detected_left"] or r["hand_detected_right"]
            )
            summary.setdefault(letter, {"videos": 0, "frames": 0, "hand_frames": 0})
            summary[letter]["videos"]      += 1
            summary[letter]["frames"]      += len(rows)
            summary[letter]["hand_frames"] += frames_with_hand

            pct = 100 * frames_with_hand / max(len(rows), 1)
            log.info("  -> %d/%d frames with hand (%.0f%%)",
                     frames_with_hand, len(rows), pct)

    finally:
        hand_detector.close()
        holistic_detector.close()

    # ── Write CSV ─────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    # Reorder to ALL_COLS where they exist, then remaining cols
    existing = [c for c in ALL_COLS if c in df.columns]
    extra    = [c for c in df.columns if c not in ALL_COLS]
    df = df[existing + extra]
    df.to_csv(str(OUTPUT_CSV), index=False)

    # ── Summary JSON ──────────────────────────────────────────────────────
    s_path = VIDEO_DIR.parent / "video_extraction_summary.json"
    with open(str(s_path), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # ── Final report ──────────────────────────────────────────────────────
    log.info("")
    log.info("=== VIDEO PIPELINE COMPLETE ===")
    log.info("  Total rows (frames): %d", len(df))
    log.info("  CSV saved to       : %s", OUTPUT_CSV)
    log.info("  Annotated videos   : %s", ANNOTATED_ROOT)
    log.info("")
    log.info("  Per-letter summary:")
    for letter in sorted(summary.keys()):
        s = summary[letter]
        pct = 100 * s["hand_frames"] / max(s["frames"], 1)
        log.info("    %s : %d videos, %d frames, %.0f%% with hand",
                 letter, s["videos"], s["frames"], pct)


if __name__ == "__main__":
    process_all_videos()
