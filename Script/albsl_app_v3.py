"""
albsl_app v3: live 63-d landmark sign classifier + optional template fallback.
Loads the same landmarks JSON (dict of letter -> 21x3) as v2 for fallback.

Requires: train_albsl.py -> models/trained/albsl_model_final/model_full.pt,
models/mediapipe/mp_models from extract_keypoints_v2.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mediapipe.tasks import python as mp_task
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

BERT_HIDDEN, BERT_LAYERS, BERT_HEADS = 256, 2, 4
MODEL_URLS: Dict[str, str] = {
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}


def ensure_models_local(models_dir: Path) -> Dict[str, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    for key, url in MODEL_URLS.items():
        dst = models_dir / f"{key}_landmarker.task"
        if not dst.exists():
            urllib.request.urlretrieve(url, dst)
        out[key] = dst
    return out


def _rodrigues(
    a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    v = np.cross(a, b)
    c, s = float(np.dot(a, b)), float(np.linalg.norm(v))
    if s < 1e-8:
        return np.eye(3, dtype=np.float32)
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float32
    )
    return (np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s * s))).astype(np.float32)


def canonicalize(xyz: np.ndarray) -> np.ndarray:
    if not np.any(xyz):
        return np.zeros((21, 3), np.float32)
    o = xyz.astype(np.float32).copy()
    o -= o[0]
    d = float(np.linalg.norm(o.max(0) - o.min(0)))
    if d > 1e-8:
        o /= d
    r = _rodrigues(o[9], np.array([0.0, 1.0, 0.0], np.float32))
    o = (r @ o.T).T
    c = np.cross(o[5] - o[0], o[17] - o[0])
    if c[2] < 0:
        o[:, 0] *= -1.0
    return np.clip(o, -1, 1)


def pack63(x: np.ndarray) -> np.ndarray:
    return x.reshape(63).astype(np.float32)


def load_refs(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    o: Dict[str, np.ndarray] = {}
    for k, v in raw.items():
        a = np.array(v, np.float32)
        if a.shape == (21, 3):
            o[k] = a
    return o


def template_fallback(
    live: np.ndarray, ref: Dict[str, np.ndarray], t: float
) -> Optional[str]:
    best, bd = None, 1e9
    for L, r in ref.items():
        d = float(np.mean(np.linalg.norm(live - r, axis=1)))
        if d < bd:
            best, bd = L, d
    if best and bd <= t:
        return best
    return None


class SignLandmarkModel(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        from transformers import BertConfig, BertModel

        c = BertConfig(
            vocab_size=2, hidden_size=BERT_HIDDEN, num_hidden_layers=BERT_LAYERS,
            num_attention_heads=BERT_HEADS, intermediate_size=BERT_HIDDEN * 4,
            max_position_embeddings=32, type_vocab_size=1, hidden_dropout_prob=0.1, use_cache=False,
        )
        self.bert = BertModel(c)
        self.joint_embed = nn.Linear(3, BERT_HIDDEN)
        self.pos_embed = nn.Parameter(torch.zeros(1, 22, BERT_HIDDEN))
        self.cls = nn.Parameter(torch.zeros(1, 1, BERT_HIDDEN))
        self.d = nn.Dropout(0.1)
        self.head = nn.Linear(BERT_HIDDEN, n)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        n = 21
        b = u.size(0)
        x = u.view(b, n, 3)
        h = self.joint_embed(x)
        t = self.cls.expand(b, -1, -1)
        s = torch.cat([t, h], 1) + self.pos_embed
        z = self.bert(inputs_embeds=s).last_hidden_state[:, 0]
        return self.head(self.d(z))


def _load(ck: Path) -> Tuple[SignLandmarkModel, Dict[str, Any]]:
    d: Any = torch.load(ck, map_location="cpu", weights_only=False)
    lm = d.get("lmap", d)
    st: Dict[str, Any] = d.get("state_dict", d)  # type: ignore[assignment]
    n = int(max(lm["label_to_id"].values()) + 1) if "label_to_id" in lm else 36
    m = SignLandmarkModel(n)
    m.load_state_dict(st, strict=False)
    m.eval()
    return m, lm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks-json", type=Path, default=Path("datasets/processed/assets/albsl_landmarks.json"))
    ap.add_argument("--model", type=Path, default=Path("models/trained/albsl_model_final/model_full.pt"))
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--min-conf", type=float, default=0.2)
    ap.add_argument("--template-thr", type=float, default=0.2)
    a = ap.parse_args()
    m = None
    lm: Dict = {}
    if a.model.is_file():
        m, lm = _load(a.model)
    i2: Dict[int, str] = {
        v: k for k, v in (lm.get("label_to_id") or {}).items()
    }
    refs = load_refs(a.landmarks_json)
    p = ensure_models_local(Path("models/mediapipe/mp_models"))
    h = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=mp_task.BaseOptions(model_asset_path=str(p["hand"])),
            running_mode=VisionTaskRunningMode.VIDEO, num_hands=1
        )
    )
    cap = cv2.VideoCapture(int(a.camera))
    t0 = time.time()
    last = 0
    while cap.isOpened():
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.flip(fr, 1)
        ts = int(1000 * (time.time() - t0)) + 1
        if ts <= last:
            ts = last + 1
        last = ts
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=g)
        r = h.detect_for_video(img, ts)
        s, c = "?", 0.0
        if r.hand_landmarks:
            pts = r.hand_landmarks[0]
            arr = np.array([[e.x, e.y, e.z] for e in pts], np.float32)
            vec = pack63(canonicalize(arr))
            if m is not None and i2:
                with torch.no_grad():
                    lg = m(torch.from_numpy(vec).view(1, 63).float())
                    pz = F.softmax(lg, 1).squeeze(0)
                    j = int(pz.argmax().item())
                    s = str(i2.get(j, "?"))
                    c = float(pz[j].item())
            if c < a.min_conf and refs:
                fb = template_fallback(canonicalize(arr), refs, a.template_thr)
                if fb:
                    s, c = fb, 0.55
        else:
            s, c = "—", 0.0
        cv2.putText(
            fr, f"pred={s} conf={c:.2f}", (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 0), 2
        )
        cv2.imshow("AlbSL v3", fr)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break
    cap.release()
    h.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
