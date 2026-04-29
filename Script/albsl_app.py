"""Real-time AlbSL recognition + online training app.

Controls (press a key inside the video window):
  - SPACE  append current top-1 letter to the word buffer
  - BACKSP delete last letter from word buffer
  - ENTER  commit current word (logged to console, then cleared)
  - 0..9   shortcut: select training letter from the label quickbar
  - T      toggle training mode (live optimizer steps on labeled samples)
  - L      cycle training label to the next alphabet letter
  - K      cycle training label to the previous alphabet letter
  - S      store the current frame as a labeled sample for the selected letter
  - C      clear word buffer
  - Q      quit

Modes:
  - INFER (default): shows top-1 prediction and confidence, builds words.
  - TRAIN: everytime you press S, the current frame+keypoints are stored as a
    labeled sample for the currently selected letter. A background thread
    periodically runs optimizer steps on the replay buffer.

Notes:
  - Uses CPU/XPU automatically. BF16 on XPU, FP32 on CPU.
  - Uses MediaPipe HandLandmarker (VIDEO mode) for 21 hand keypoints.
  - Keypoint features per frame = 63 normalized xyz + 60 dihedral = 123.
  - If ``outputs/fused_phase3.pt`` exists it is loaded as init weights.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Deque, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from albsl_fusion.model import FusionBatch, build_model
from albsl_fusion.utils.hardware import build_runtime
from extract_keypoints_v2 import (
    ALBANIAN_LETTERS,
    canonical_normalize_hand,
    dihedral_features,
    ensure_models,
)

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

BaseOptions = mp_tasks.BaseOptions

IMAGE_SIZE = 224
KEY_DIM = 123
REPLAY_MAX = 2048
BATCH_SIZE = 8
TRAIN_STEPS_PER_TICK = 2


def _safe_show(name: str, letter: str) -> str:
    """Avoid console garbling by mapping Albanian letters to ASCII-friendly tags."""
    mapping = {"Ç": "Cc", "Ë": "Ee"}
    return mapping.get(letter, letter)


def build_feature(xyz: np.ndarray) -> np.ndarray:
    if not np.any(xyz):
        return np.zeros(KEY_DIM, dtype=np.float32)
    normalized = canonical_normalize_hand(xyz)
    normalized = np.clip(normalized, -1.0, 1.0)
    angles = dihedral_features(normalized)
    flat = np.concatenate([normalized.reshape(-1), angles.reshape(-1)], axis=0)
    return flat.astype(np.float32)


def hand_bbox(frame_shape: Tuple[int, int], xyz_image: np.ndarray, pad: float = 0.2) -> Tuple[int, int, int, int]:
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


def preprocess_crop(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = frame_bgr
    crop = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE))
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


class ReplayBuffer:
    def __init__(self, capacity: int = REPLAY_MAX) -> None:
        self.capacity = capacity
        self.images: Deque[np.ndarray] = deque(maxlen=capacity)
        self.keypoints: Deque[np.ndarray] = deque(maxlen=capacity)
        self.bboxes: Deque[np.ndarray] = deque(maxlen=capacity)
        self.labels: Deque[int] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def add(self, img: np.ndarray, key: np.ndarray, bbox: np.ndarray, label: int) -> None:
        with self._lock:
            self.images.append(img.copy())
            self.keypoints.append(key.copy())
            self.bboxes.append(bbox.copy())
            self.labels.append(int(label))

    def __len__(self) -> int:
        return len(self.labels)

    def sample(self, n: int) -> Optional[dict[str, torch.Tensor]]:
        with self._lock:
            if len(self.labels) == 0:
                return None
            n = min(n, len(self.labels))
            idxs = np.random.choice(len(self.labels), size=n, replace=False)
            images = np.stack([self.images[i] for i in idxs], axis=0)
            keypoints = np.stack([self.keypoints[i] for i in idxs], axis=0)
            bboxes = np.stack([self.bboxes[i] for i in idxs], axis=0)
            labels = np.array([self.labels[i] for i in idxs], dtype=np.int64)
        return {
            "image": torch.from_numpy(images).permute(0, 3, 1, 2).contiguous(),
            "keypoints": torch.from_numpy(keypoints).unsqueeze(1),  # [B,1,123]
            "bbox": torch.from_numpy(bboxes),
            "letter_index": torch.from_numpy(labels),
        }


class OnlineTrainer:
    def __init__(self, model: torch.nn.Module, device: torch.device, dtype: torch.dtype, lr: float = 1e-4) -> None:
        self.model = model
        self.device = device
        self.dtype = dtype
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.loss_weights = {"ce": 1.0, "focal": 0.0, "align": 0.1}
        self.latest_stats: dict[str, float] = {"loss": 0.0, "ce": 0.0, "align": 0.0}
        self._buf = ReplayBuffer()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    @property
    def buffer(self) -> ReplayBuffer:
        return self._buf

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        while self._running:
            time.sleep(0.1)
            if len(self._buf) < 4:
                continue
            batch = self._buf.sample(BATCH_SIZE)
            if batch is None:
                continue
            image = batch["image"].to(device=self.device, dtype=self.dtype)
            keypoints = batch["keypoints"].to(device=self.device, dtype=self.dtype)
            bbox = batch["bbox"].to(device=self.device, dtype=self.dtype)
            labels = batch["letter_index"].to(device=self.device)
            fb = FusionBatch(image=image, keypoints=keypoints, bbox=bbox, letter_index=labels)
            self.model.train()
            for _ in range(TRAIN_STEPS_PER_TICK):
                loss, stats = self.model.training_step(fb, loss_weights=self.loss_weights)
                if torch.isnan(loss).any():
                    continue
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.latest_stats = stats


def draw_overlay(frame: np.ndarray, text_lines: List[str], bbox: Tuple[int, int, int, int], color: Tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    y = 28
    for line in text_lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        y += 28


def run_app(camera_index: int, weights: Optional[Path]) -> None:
    runtime = build_runtime("xpu")
    logger.info("Runtime: {}", runtime)

    cfg = SimpleNamespace(
        model=SimpleNamespace(hidden_dim=1152, fusion=SimpleNamespace(num_heads=4)),
        data=SimpleNamespace(num_letters=36),
    )
    model = build_model(cfg).to(runtime.device)
    if runtime.using_xpu:
        model = model.to(dtype=torch.bfloat16)
    if weights and weights.exists():
        state = torch.load(str(weights), map_location=runtime.device)
        try:
            model.load_state_dict(state)
            logger.info("Loaded weights from {}", weights)
        except Exception as exc:
            logger.warning("Could not load weights cleanly: {}", exc)

    trainer = OnlineTrainer(model, runtime.device, runtime.dtype)
    trainer.start()

    model_paths = ensure_models(Path("models/mediapipe/mp_models"))
    hand = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_paths["hand"])),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_hands=1,
        )
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Cannot open camera index {}", camera_index)
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    word_buffer: List[str] = []
    selected_letter_idx = 0
    training_mode = False
    last_ts_ms = 0
    t_start = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            ts_ms = max(last_ts_ms + 1, int((time.time() - t_start) * 1000))
            last_ts_ms = ts_ms
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = hand.detect_for_video(mp_image, ts_ms)

            xyz = np.zeros((21, 3), dtype=np.float32)
            detected = False
            if result.hand_landmarks:
                pts = result.hand_landmarks[0]
                xyz = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float32)
                detected = True

            bbox = hand_bbox(frame.shape, xyz)
            feature_vec = build_feature(xyz)
            crop = preprocess_crop(frame, bbox)

            pred_letter = "?"
            conf = 0.0
            if detected:
                with torch.no_grad():
                    model.eval()
                    fb = FusionBatch(
                        image=torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).to(runtime.device, dtype=runtime.dtype),
                        keypoints=torch.from_numpy(feature_vec).view(1, 1, -1).to(runtime.device, dtype=runtime.dtype),
                        bbox=torch.tensor([[bbox[0], bbox[1], bbox[2], bbox[3]]], dtype=runtime.dtype, device=runtime.device),
                        letter_index=torch.zeros(1, dtype=torch.long, device=runtime.device),
                    )
                    logits = model(fb)["logits"].float()
                    probs = F.softmax(logits, dim=-1)[0]
                    top_idx = int(probs.argmax().item())
                    pred_letter = ALBANIAN_LETTERS[top_idx]
                    conf = float(probs[top_idx].item())

            selected_letter = ALBANIAN_LETTERS[selected_letter_idx]
            status = "TRAIN" if training_mode else "INFER"
            word_text = "".join(word_buffer)
            color = (0, 200, 0) if detected else (0, 0, 200)
            lines = [
                f"mode={status}  device={runtime.device}",
                f"pred={_safe_show('pred', pred_letter)}  conf={conf:.2f}",
                f"label_select={_safe_show('label', selected_letter)}  buffer={len(trainer.buffer)}  loss={trainer.latest_stats.get('loss', 0.0):.4f}",
                f"word={word_text}",
            ]
            draw_overlay(frame, lines, bbox, color)
            cv2.imshow("AlbSL Live", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == 32:  # SPACE
                if detected and pred_letter != "?":
                    word_buffer.append(pred_letter)
            elif key == 8:  # BACKSPACE
                if word_buffer:
                    word_buffer.pop()
            elif key == 13:  # ENTER
                if word_buffer:
                    logger.info("WORD: {}", "".join(word_buffer))
                    word_buffer.clear()
            elif key == ord("c"):
                word_buffer.clear()
            elif key == ord("t"):
                training_mode = not training_mode
                logger.info("training_mode={}", training_mode)
            elif key == ord("l"):
                selected_letter_idx = (selected_letter_idx + 1) % len(ALBANIAN_LETTERS)
            elif key == ord("k"):
                selected_letter_idx = (selected_letter_idx - 1) % len(ALBANIAN_LETTERS)
            elif key == ord("s"):
                if detected and training_mode:
                    trainer.buffer.add(
                        crop,
                        feature_vec,
                        np.array(bbox, dtype=np.float32),
                        selected_letter_idx,
                    )
                    logger.info(
                        "added sample: letter={} buffer={}",
                        _safe_show("label", selected_letter),
                        len(trainer.buffer),
                    )
    finally:
        trainer.stop()
        cap.release()
        cv2.destroyAllWindows()
        hand.close()
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), out_dir / "fused_live.pt")
        logger.info("Saved live-trained weights to outputs/fused_live.pt")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="AlbSL live recognition + online training app")
    ap.add_argument("--camera", type=int, default=0, help="Webcam index")
    ap.add_argument("--weights", type=Path, default=Path("outputs/fused_phase3.pt"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_app(args.camera, args.weights)


if __name__ == "__main__":
    main()
