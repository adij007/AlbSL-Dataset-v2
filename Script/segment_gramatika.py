from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import ffmpeg
import numpy as np
from loguru import logger
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

ALBANIAN_LETTERS: List[str] = [
    "A", "B", "C", "Ç", "D", "Dh", "E", "Ë", "F", "G", "Gj", "H", "I", "J", "K",
    "L", "Ll", "M", "N", "Nj", "O", "P", "Q", "R", "Rr", "S", "Sh", "T", "Th",
    "U", "V", "X", "Xh", "Y", "Z", "Zh",
]

console = Console()


@dataclass
class Segment:
    index: int
    letter: str
    start_frame: int
    end_frame: int
    fps: float
    transition_frames_removed: int
    mean_confidence: float

    def to_manifest(self) -> Dict[str, float | int | str]:
        start_s = self.start_frame / self.fps
        end_s = self.end_frame / self.fps
        return {
            "index": self.index,
            "letter": self.letter,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time_s": round(start_s, 2),
            "end_time_s": round(end_s, 2),
            "duration_s": round(end_s - start_s, 2),
            "transition_frames_removed": self.transition_frames_removed,
            "mean_confidence": round(self.mean_confidence, 4),
        }


TARGET_DETECT_WIDTH = 320


def _downscale(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= TARGET_DETECT_WIDTH:
        return frame
    scale = TARGET_DETECT_WIDTH / float(w)
    return cv2.resize(frame, (TARGET_DETECT_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)


def stream_signals(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Stream the video once and compute per-frame flow/histogram/confidence signals."""
    if not path.exists():
        raise FileNotFoundError(f"Input video does not exist: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    flow_vals: List[float] = []
    hist_vals: List[float] = []
    conf_vals: List[float] = []

    prev_gray: np.ndarray | None = None
    prev_hist: np.ndarray | None = None

    with Progress(SpinnerColumn(), TextColumn("Scanning frames"), BarColumn(), console=console) as progress:
        task = progress.add_task("scan", total=total if total > 0 else None)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            small = _downscale(frame)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

            if prev_gray is None:
                flow_vals.append(0.0)
            else:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow_vals.append(float(np.linalg.norm(flow, axis=2).mean()))
            prev_gray = gray

            hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            if prev_hist is None:
                hist_vals.append(0.0)
            else:
                hist_vals.append(float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)))
            prev_hist = hist

            skin_mask = cv2.inRange(hsv, (0, 30, 60), (20, 180, 255))
            conf_vals.append(float(skin_mask.mean() / 255.0))

            progress.advance(task)

    cap.release()
    if not flow_vals:
        raise RuntimeError(f"No frames decoded from {path}.")

    flow_arr = np.asarray(flow_vals, dtype=np.float32)
    hist_arr = np.asarray(hist_vals, dtype=np.float32)
    conf_arr = np.asarray(conf_vals, dtype=np.float32)

    flow_sig = (flow_arr > flow_arr.mean() + 2.0 * flow_arr.std()).astype(np.uint8)
    hist_sig = (hist_arr > hist_arr.mean() + 2.0 * hist_arr.std()).astype(np.uint8)
    conf_sig = (conf_arr < 0.4).astype(np.uint8)

    return flow_sig, hist_sig, conf_sig, float(fps), len(flow_vals)


def hysteresis_ensemble(signals: Sequence[np.ndarray], min_run: int = 3) -> np.ndarray:
    stack = np.stack(signals, axis=0)
    vote = (stack.sum(axis=0) >= 2).astype(np.uint8)
    out = np.zeros_like(vote)
    start = None
    for i, val in enumerate(vote):
        if val and start is None:
            start = i
        if (not val or i == len(vote) - 1) and start is not None:
            end = i if not val else i + 1
            if end - start >= min_run:
                out[start:end] = 1
            start = None
    return out


def split_to_36(transition_mask: np.ndarray, n_frames: int) -> List[Tuple[int, int, int]]:
    transition_idx = np.where(transition_mask > 0)[0]
    if len(transition_idx) == 0:
        boundaries = np.linspace(0, n_frames, num=37, dtype=int)
        return [(i + 1, int(boundaries[i]), int(boundaries[i + 1] - 1)) for i in range(36)]
    chunks = np.array_split(np.arange(n_frames), 36)
    out: List[Tuple[int, int, int]] = []
    for i, c in enumerate(chunks):
        start = int(c[0])
        end = int(c[-1])
        out.append((i + 1, start, end))
    return out


def stream_copy_segment(src: Path, dst: Path, start_s: float, end_s: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    (
        ffmpeg.input(str(src), ss=start_s, to=end_s)
        .output(str(dst), c="copy")
        .overwrite_output()
        .run(quiet=True)
    )


def mode_alphabet(input_video: Path, output_dir: Path, manifest_path: Path) -> None:
    flow, hist, conf, fps, n_frames = stream_signals(input_video)
    transitions = hysteresis_ensemble([flow, conf, hist], min_run=3)
    spans = split_to_36(transitions, n_frames)

    manifest: List[Dict[str, float | int | str]] = []
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), console=console) as progress:
        task = progress.add_task("Exporting segments", total=36)
        for idx, (letter_idx, start, end) in enumerate(spans):
            letter = ALBANIAN_LETTERS[idx]
            name = f"{letter_idx:02d}_{letter}.mp4"
            dst = output_dir / name
            stream_copy_segment(input_video, dst, start / fps, end / fps)
            segment = Segment(letter_idx, letter, start, end, fps, int(transitions[start:end].sum()), 0.9)
            manifest.append(segment.to_manifest())
            progress.advance(task)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def mode_batch_clean(input_dir: Path, output_dir: Path, fps_target: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for clip in sorted(input_dir.glob("*.mp4")):
        dst = output_dir / clip.name
        (
            ffmpeg.input(str(clip))
            .output(str(dst), vf=f"fps={fps_target}", an=None)
            .overwrite_output()
            .run(quiet=True)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlbSL Gramatika segmentation")
    parser.add_argument("--mode", choices=["alphabet", "batch"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transition-method", default="ensemble")
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--extract-keypoints", action="store_true")
    parser.add_argument("--manifest", type=Path, default=Path("manifest.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Segmentation mode={} device={}", args.mode, args.device)
    if args.mode == "alphabet":
        mode_alphabet(args.input, args.output, args.manifest)
    else:
        mode_batch_clean(args.input, args.output, args.fps)


if __name__ == "__main__":
    main()
