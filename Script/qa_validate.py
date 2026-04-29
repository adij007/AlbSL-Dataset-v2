from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from extract_keypoints_v2 import ALBANIAN_LETTERS, HAND_BONES


def load_npz(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    xyz = data["xyz"]
    angles = data["angles"]
    return xyz, angles


def render_contact_sheet(npz_dir: Path, out_path: Path, per_letter: int = 5) -> None:
    fig, axes = plt.subplots(len(ALBANIAN_LETTERS), per_letter, figsize=(per_letter * 2.5, len(ALBANIAN_LETTERS) * 1.8))
    if len(ALBANIAN_LETTERS) == 1:
        axes = np.array([axes])
    for i, letter in enumerate(ALBANIAN_LETTERS):
        candidates = sorted(npz_dir.glob(f"*_{letter}.npz"))
        if not candidates:
            continue
        xyz, angles = load_npz(candidates[0])
        T = xyz.shape[0]
        picks = sorted(random.sample(range(T), k=min(per_letter, T)))
        for j in range(per_letter):
            ax = axes[i, j]
            ax.axis("off")
            if j >= len(picks):
                continue
            t = picks[j]
            pts = xyz[t]
            amp = float(np.abs(angles[t]).mean()) if t < angles.shape[0] else 0.0
            color = plt.cm.viridis(min(1.0, amp))
            ax.scatter(pts[:, 0], -pts[:, 1], s=8, color=color)
            for parent, child in HAND_BONES:
                ax.plot([pts[parent, 0], pts[child, 0]], [-pts[parent, 1], -pts[child, 1]], color=color, linewidth=1)
            if j == 0:
                ax.set_title(letter, fontsize=8, loc="left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def summarize_stats(npz_dir: Path, out_json: Path) -> None:
    per_letter: Dict[str, Dict[str, float]] = {}
    for letter in ALBANIAN_LETTERS:
        clips = sorted(npz_dir.glob(f"*_{letter}.npz"))
        if not clips:
            continue
        data = np.load(clips[0], allow_pickle=True)
        xyz = data["xyz"]
        conf = data["conf"]
        per_letter[letter] = {
            "frames": float(xyz.shape[0]),
            "mean_confidence": float(conf.mean()),
            "min_confidence": float(conf.min()),
            "occlusion_rate_pct": float((conf < 0.5).sum()) / max(conf.size, 1) * 100.0,
        }
    out_json.write_text(json.dumps({"per_letter": per_letter}, indent=2, ensure_ascii=False), encoding="utf-8")
