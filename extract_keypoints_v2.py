"""Intel/OpenVINO-first keypoint extraction pipeline for AlbSL.

All heavy Intel backends (IPEX, OpenVINO, decord) are imported defensively so
the extractor still runs end-to-end on a CPU-only Windows box.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from loguru import logger
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    PoseLandmarker,
    PoseLandmarkerOptions,
)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

try:
    import torch
except Exception:  # pragma: no cover - Windows DLL failures, etc.
    torch = None  # type: ignore[assignment]

try:
    import intel_extension_for_pytorch as _ipex  # noqa: F401
    HAS_IPEX = True
except Exception:
    HAS_IPEX = False

try:
    import openvino as ov
    HAS_OPENVINO = True
except Exception:
    ov = None  # type: ignore[assignment]
    HAS_OPENVINO = False

try:
    import decord  # type: ignore
    HAS_DECORD = True
except Exception:
    decord = None  # type: ignore[assignment]
    HAS_DECORD = False

BaseOptions = mp_tasks.BaseOptions
console = Console()

ALBANIAN_LETTERS: List[str] = [
    "A", "B", "C", "Ç", "D", "Dh", "E", "Ë", "F", "G", "Gj", "H", "I", "J", "K",
    "L", "Ll", "M", "N", "Nj", "O", "P", "Q", "R", "Rr", "S", "Sh", "T", "Th",
    "U", "V", "X", "Xh", "Y", "Z", "Zh",
]
MODEL_URLS: Dict[str, str] = {
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
}
HAND_BONES: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


@dataclass
class RuntimeContext:
    device_str: str
    openvino_core: Optional[Any]


def init_runtime(prefer_gpu: bool = True) -> RuntimeContext:
    device_str = "CPU"
    if torch is not None and hasattr(torch, "xpu") and torch.xpu.is_available():
        device_str = "GPU.0" if prefer_gpu else "CPU"
    elif torch is None:
        logger.warning("torch unavailable; running extractor without torch device.")
    else:
        logger.warning("torch.xpu is unavailable; falling back to CPU execution.")
    core = ov.Core() if HAS_OPENVINO else None
    if core is None:
        logger.warning("OpenVINO runtime not available; MediaPipe CPU path will be used.")
    return RuntimeContext(device_str, core)


def ensure_models(models_dir: Path) -> Dict[str, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    for key, url in MODEL_URLS.items():
        dst = models_dir / f"{key}_landmarker.task"
        if not dst.exists():
            logger.info("Downloading {} model to {}", key, dst)
            urllib.request.urlretrieve(url, dst)
        out[key] = dst
    return out


def make_hand_detector(path: Path, num_hands: int = 2) -> HandLandmarker:
    return HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_hands=int(num_hands),
        )
    )


def make_pose_detector(path: Path) -> PoseLandmarker:
    return PoseLandmarker.create_from_options(
        PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_poses=1,
        )
    )


def parse_hand(result: Any) -> Tuple[np.ndarray, np.ndarray]:
    xyz = np.zeros((21, 3), dtype=np.float32)
    conf = np.zeros((21,), dtype=np.float32)
    if not result.hand_landmarks:
        return xyz, conf
    points = result.hand_landmarks[0]
    xyz = np.array([[p.x, p.y, p.z] for p in points], dtype=np.float32)
    hand_conf = 0.0
    if result.handedness and result.handedness[0]:
        hand_conf = float(result.handedness[0][0].score)
    conf[:] = hand_conf
    return xyz, conf


def parse_hands_dual(result: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (xyz_right, xyz_left, conf_right, conf_left) for up to two hands."""
    xyz_r = np.zeros((21, 3), dtype=np.float32)
    xyz_l = np.zeros((21, 3), dtype=np.float32)
    c_r = np.zeros((21,), dtype=np.float32)
    c_l = np.zeros((21,), dtype=np.float32)
    if not result.hand_landmarks:
        return xyz_r, xyz_l, c_r, c_l
    for i, pts in enumerate(result.hand_landmarks):
        if i >= len(result.handedness):
            break
        side = result.handedness[i][0].display_name.lower()  # 'Left'/'Right'
        score = float(result.handedness[i][0].score)
        arr = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float32)
        if side == "right":
            xyz_r = arr
            c_r[:] = score
        else:
            xyz_l = arr
            c_l[:] = score
    return xyz_r, xyz_l, c_r, c_l


class _YoloCropper:
    def __init__(self, weights: Path, conf_threshold: float = 0.2) -> None:
        self.model = None
        self.conf_threshold = conf_threshold
        try:
            from ultralytics import YOLO  # type: ignore

            self.model = YOLO(str(weights))
        except Exception as exc:
            logger.warning("YOLO cropper disabled (load failed): {}", exc)

    def crop(self, frame_rgb: np.ndarray) -> np.ndarray:
        if self.model is None:
            return frame_rgb
        try:
            results = self.model.predict(frame_rgb, verbose=False, classes=[0], conf=self.conf_threshold)
            if not results:
                return frame_rgb
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                return frame_rgb
            xyxy = boxes.xyxy[int(boxes.conf.argmax())].tolist()
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            H, W = frame_rgb.shape[:2]
            pad_x, pad_y = int(0.1 * (x2 - x1)), int(0.1 * (y2 - y1))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(W, x2 + pad_x)
            y2 = min(H, y2 + pad_y)
            if x2 - x1 < 20 or y2 - y1 < 20:
                return frame_rgb
            return frame_rgb[y1:y2, x1:x2]
        except Exception:
            return frame_rgb


def parse_pose(result: Any) -> Tuple[np.ndarray, np.ndarray]:
    xyz = np.zeros((33, 3), dtype=np.float32)
    conf = np.zeros((33,), dtype=np.float32)
    if not result.pose_landmarks:
        return xyz, conf
    points = result.pose_landmarks[0]
    xyz = np.array([[p.x, p.y, p.z] for p in points], dtype=np.float32)
    conf = np.array(
        [getattr(p, "visibility", 0.0) for p in points], dtype=np.float32,
    )
    return xyz, conf


def rodrigues_rotation(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
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


def canonical_normalize_hand(xyz: np.ndarray) -> np.ndarray:
    if not np.any(xyz):
        return xyz.copy()
    out = xyz.copy()
    out -= out[0]
    diag = np.linalg.norm(np.max(out, axis=0) - np.min(out, axis=0))
    if diag > 1e-8:
        out /= diag
    R = rodrigues_rotation(out[9], np.array([0.0, 1.0, 0.0], dtype=np.float32))
    out = (R @ out.T).T
    p0, p5, p17 = out[0], out[5], out[17]
    signed = np.cross(p5 - p0, p17 - p0)[2]
    if signed < 0.0:
        out[:, 0] *= -1.0
    return out.astype(np.float32)


def procrustes_rotation(prev_xyz: np.ndarray, curr_xyz: np.ndarray) -> np.ndarray:
    if not np.any(prev_xyz) or not np.any(curr_xyz):
        return curr_xyz.copy()
    H = curr_xyz.T @ prev_xyz
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    return (curr_xyz @ R).astype(np.float32)


def bone_directions(xyz: np.ndarray) -> np.ndarray:
    dirs: List[np.ndarray] = []
    for parent, child in HAND_BONES:
        vec = xyz[child] - xyz[parent]
        norm = np.linalg.norm(vec) + 1e-8
        dirs.append(vec / norm)
    return np.stack(dirs, axis=0).astype(np.float32)


def dihedral_features(xyz: np.ndarray) -> np.ndarray:
    dirs = bone_directions(xyz)
    out = np.zeros((20, 3), dtype=np.float32)
    for i in range(1, len(dirs) - 1):
        b_prev, b_curr, b_next = dirs[i - 1], dirs[i], dirs[i + 1]
        n1 = np.cross(b_prev, b_curr)
        n2 = np.cross(b_curr, b_next)
        n1 /= np.linalg.norm(n1) + 1e-8
        n2 /= np.linalg.norm(n2) + 1e-8
        theta = math.atan2(float(np.dot(np.cross(n1, n2), b_curr)), float(np.dot(n1, n2)))
        out[i] = np.array(
            [theta, float(np.dot(b_prev, b_curr)), float(np.dot(b_curr, b_next))],
            dtype=np.float32,
        )
    return out


def unproject_pose_landmarks(pose_xyz: np.ndarray, width: int, height: int) -> np.ndarray:
    f = float(max(width, height))
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    K_inv = np.linalg.inv(K)
    out = np.zeros_like(pose_xyz)
    for i, (x, y, z) in enumerate(pose_xyz):
        u, v = x * width, y * height
        ray = K_inv @ np.array([u, v, 1.0], dtype=np.float32)
        out[i] = (z * ray).astype(np.float32)
    return out


def confidence_interpolate(sequence: np.ndarray, conf: np.ndarray) -> np.ndarray:
    out = sequence.copy()
    T, L, D = out.shape
    idx = np.arange(T)
    for l in range(L):
        valid = conf[:, l] >= 0.5
        if valid.sum() < 4:
            continue
        for d in range(D):
            spline = CubicSpline(idx[valid], out[valid, l, d], extrapolate=True)
            out[~valid, l, d] = spline(idx[~valid])
    return out.astype(np.float32)


def smooth_sequence(sequence: np.ndarray) -> np.ndarray:
    out = sequence.copy()
    T = out.shape[0]
    if T >= 7:
        # Savitzky-Golay along time axis, vectorized per-dimension.
        out = savgol_filter(out, window_length=7, polyorder=3, mode="nearest", axis=0)
    return out.astype(np.float32)


def iter_video_frames(video_path: Path) -> Tuple[Any, float]:
    """Return a lazy iterator over frames and fps. Tries decord GPU, falls back to CPU/OpenCV."""
    if HAS_DECORD:
        try:
            ctx = decord.cpu(0)
            vr = decord.VideoReader(str(video_path), ctx=ctx)
            fps = float(vr.get_avg_fps()) or 30.0

            def gen() -> Any:
                for i in range(len(vr)):
                    yield vr[i].asnumpy()

            return gen(), fps
        except Exception as exc:
            logger.warning("decord failed ({}); falling back to OpenCV.", exc)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    def gen() -> Any:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()

    return gen(), float(fps)


def extract_clip(
    video_path: Path,
    output_npz: Path,
    runtime: RuntimeContext,
    models_dir: Path,
    num_hands: int = 2,
    min_confidence: float = 0.0,
    yolo_weights: Optional[Path] = None,
) -> Dict[str, Any]:
    model_paths = ensure_models(models_dir)
    hand = make_hand_detector(model_paths["hand"], num_hands=num_hands)
    pose = make_pose_detector(model_paths["pose"])
    cropper: Optional[_YoloCropper] = _YoloCropper(yolo_weights) if yolo_weights else None

    frames_iter, fps = iter_video_frames(video_path)

    xyz_r_list: List[np.ndarray] = []
    xyz_l_list: List[np.ndarray] = []
    conf_r_list: List[np.ndarray] = []
    conf_l_list: List[np.ndarray] = []
    prev_r = np.zeros((21, 3), dtype=np.float32)
    prev_l = np.zeros((21, 3), dtype=np.float32)

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Extracting {video_path.name}", total=None)
        for t, frame in enumerate(frames_iter):
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            if cropper is not None:
                frame = cropper.crop(frame)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            ts_ms = int((1000.0 * t) / max(fps, 1.0))
            hres = hand.detect_for_video(image, ts_ms)
            pose.detect_for_video(image, ts_ms)  # kept to exercise pose pipeline

            xyz_r, xyz_l, c_r, c_l = parse_hands_dual(hres)
            xyz_r = canonical_normalize_hand(xyz_r)
            xyz_l = canonical_normalize_hand(xyz_l)
            xyz_r = procrustes_rotation(prev_r, xyz_r)
            xyz_l = procrustes_rotation(prev_l, xyz_l)
            prev_r, prev_l = xyz_r, xyz_l

            xyz_r_list.append(xyz_r)
            xyz_l_list.append(xyz_l)
            conf_r_list.append(c_r)
            conf_l_list.append(c_l)
            progress.advance(task)

    hand.close()
    pose.close()

    def _finalize(xyz_list: List[np.ndarray], conf_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not xyz_list:
            return (
                np.zeros((0, 21, 3), dtype=np.float32),
                np.zeros((0, 21), dtype=np.float32),
                np.zeros((0, 20, 3), dtype=np.float32),
            )
        xyz = np.stack(xyz_list, axis=0)
        conf = np.stack(conf_list, axis=0)
        xyz = confidence_interpolate(xyz, conf)
        xyz = smooth_sequence(xyz)
        xyz = np.clip(xyz, -1.0, 1.0)
        angles = np.stack([dihedral_features(xyz[i]) for i in range(xyz.shape[0])], axis=0).astype(np.float32)
        return xyz, conf, angles

    xyz_r_arr, conf_r_arr, angles_r = _finalize(xyz_r_list, conf_r_list)
    xyz_l_arr, conf_l_arr, angles_l = _finalize(xyz_l_list, conf_l_list)

    T = xyz_r_arr.shape[0]
    if min_confidence > 0.0 and T > 0:
        frame_max_conf = np.maximum(conf_r_arr.max(axis=1), conf_l_arr.max(axis=1))
        keep = frame_max_conf >= min_confidence
        if keep.any():
            xyz_r_arr, xyz_l_arr = xyz_r_arr[keep], xyz_l_arr[keep]
            conf_r_arr, conf_l_arr = conf_r_arr[keep], conf_l_arr[keep]
            angles_r, angles_l = angles_r[keep], angles_l[keep]
            T = int(keep.sum())

    occluded = int(((conf_r_arr < 0.5) & (conf_l_arr < 0.5)).sum()) if T else 0
    mean_conf = float(np.maximum(conf_r_arr, conf_l_arr).mean()) if T else 0.0
    meta: Dict[str, Any] = {
        "fps": int(round(fps)),
        "total_frames": T,
        "occluded_frames": occluded,
        "mean_confidence": mean_conf,
        "device": runtime.device_str,
        "num_hands": num_hands,
        "min_confidence": min_confidence,
        "yolo_crop": bool(yolo_weights),
    }
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    # Preserve backward-compatible `xyz`/`conf`/`angles` fields using the right hand as primary.
    primary_xyz = xyz_r_arr if np.any(xyz_r_arr) else xyz_l_arr
    primary_conf = conf_r_arr if np.any(xyz_r_arr) else conf_l_arr
    primary_angles = angles_r if np.any(xyz_r_arr) else angles_l
    np.savez_compressed(
        output_npz,
        xyz=primary_xyz,
        conf=primary_conf,
        angles=primary_angles,
        xyz_right=xyz_r_arr,
        xyz_left=xyz_l_arr,
        conf_right=conf_r_arr,
        conf_left=conf_l_arr,
        angles_right=angles_r,
        angles_left=angles_l,
        meta=np.array(meta, dtype=object),
    )
    logger.info("Wrote {} ({} frames, mean_conf={:.3f})", output_npz, T, mean_conf)
    return meta


def run_on_segments(
    input_dir: Path,
    output_dir: Path,
    models_dir: Path,
    runtime: RuntimeContext,
    num_hands: int = 2,
    min_confidence: float = 0.0,
    yolo_weights: Optional[Path] = None,
) -> None:
    records: Dict[str, Dict[str, Any]] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for clip in sorted(input_dir.glob("*.mp4")):
        letter = clip.stem.split("_", 1)[-1]
        if letter not in ALBANIAN_LETTERS:
            logger.warning("Skipping clip with unexpected name: {}", clip.name)
            continue
        started = time.perf_counter()
        out_npz = output_dir / f"{clip.stem}.npz"
        try:
            meta = extract_clip(
                clip,
                out_npz,
                runtime,
                models_dir,
                num_hands=num_hands,
                min_confidence=min_confidence,
                yolo_weights=yolo_weights,
            )
        except Exception as exc:
            logger.exception("Failed to extract {}: {}", clip.name, exc)
            continue
        elapsed = max(1e-6, time.perf_counter() - started)
        frames = int(meta["total_frames"])
        records[letter] = {
            "frame_count": frames,
            "mean_confidence": float(meta["mean_confidence"]),
            "occlusion_rate_pct": (
                float(meta["occluded_frames"]) / max(frames * 21, 1) * 100.0
            ),
            "extraction_time_s": elapsed,
            "throughput_fps": frames / elapsed if elapsed > 0 else 0.0,
        }
    stats = {
        "per_letter": records,
        "global": {
            "total_frames": sum(r["frame_count"] for r in records.values()),
            "total_features_extracted": sum(r["frame_count"] for r in records.values()) * 123,
            "avg_throughput_fps": (
                float(np.mean([r["throughput_fps"] for r in records.values()]))
                if records else 0.0
            ),
            "device": runtime.device_str,
        },
    }
    (output_dir / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intel Arc keypoint extractor")
    parser.add_argument("--input", type=Path, required=True, help="Input clip or folder")
    parser.add_argument("--output", type=Path, required=True, help="Output clip npz or folder")
    parser.add_argument("--models-dir", type=Path, default=Path("mp_models"))
    parser.add_argument("--segments", action="store_true", help="Process 36 segmented clips")
    parser.add_argument("--prefer-cpu", action="store_true")
    parser.add_argument("--num-hands", type=int, default=2, help="Detect up to N hands per frame")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Drop frames whose best-hand confidence is below this threshold",
    )
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=None,
        help="Optional YOLO weights (e.g., yolov8n.pt) for a person-crop preprocessor",
    )
    return parser.parse_args()


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    args = parse_args()
    runtime = init_runtime(prefer_gpu=not args.prefer_cpu)
    yolo_weights = args.yolo_weights if args.yolo_weights and args.yolo_weights.exists() else None
    if args.segments:
        run_on_segments(
            args.input,
            args.output,
            args.models_dir,
            runtime,
            num_hands=args.num_hands,
            min_confidence=args.min_confidence,
            yolo_weights=yolo_weights,
        )
        return
    meta = extract_clip(
        args.input,
        args.output,
        runtime,
        args.models_dir,
        num_hands=args.num_hands,
        min_confidence=args.min_confidence,
        yolo_weights=yolo_weights,
    )
    logger.info("Saved extraction to {} with meta {}", args.output, meta)


if __name__ == "__main__":
    main()
