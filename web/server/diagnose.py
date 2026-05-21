"""Structured dataset diagnostics (same sources as ``albsl_app_v2.cmd_diagnose``)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "Script") not in sys.path:
    sys.path.insert(0, str(_REPO / "Script"))

from collections import Counter
from typing import Any, Dict, List, Optional

import h5py

import albsl_app_v2 as app


def run_diagnose(
    keypoints_dir: Path,
    alfabeti_h5: Path,
    legacy_h5: Optional[Path] = None,
) -> Dict[str, Any]:
    keypoints_dir = app._resolve_model_path(keypoints_dir)
    alfabeti_h5 = app._resolve_model_path(alfabeti_h5)
    if legacy_h5 is not None:
        legacy_h5 = app._resolve_model_path(legacy_h5)

    out: Dict[str, Any] = {"legacy": None, "alfabeti": None, "clips": None, "combined": None}

    if legacy_h5 and legacy_h5.exists():
        with h5py.File(legacy_h5, "r") as f:
            lbls = [x.decode() if isinstance(x, bytes) else x for x in f["labels"][...]]
            c = Counter(lbls)
            out["legacy"] = {
                "path": str(legacy_h5),
                "samples": len(lbls),
                "unique_labels": len(c),
                "top10": [{"label": k, "count": v} for k, v in c.most_common(10)],
                "single_label_warning": len(c) == 1,
            }
    else:
        out["legacy"] = {"skipped": True, "path": str(legacy_h5) if legacy_h5 else None}

    if alfabeti_h5.exists():
        with h5py.File(alfabeti_h5, "r") as f:
            lbls = [x.decode() if isinstance(x, bytes) else x for x in f["labels"][...]]
            det = f["detected"][...]
            c = Counter([lbls[i] for i in range(len(lbls)) if det[i]])
            out["alfabeti"] = {
                "path": str(alfabeti_h5),
                "detected": int(det.sum()),
                "total": len(lbls),
                "labels": {k: v for k, v in sorted(c.items())},
            }

    if keypoints_dir.exists():
        counts: Counter[str] = Counter()
        total_frames = 0
        for npz in sorted(keypoints_dir.glob("*.npz")):
            letter = npz.stem.split("_", 1)[-1]
            if letter not in app.LETTER_TO_IDX:
                continue
            data = np.load(npz, allow_pickle=True)
            n = int(data["xyz"].shape[0]) if "xyz" in data.files else 0
            if "conf" in data.files:
                n = int((data["conf"].max(axis=1) >= 0.5).sum())
            counts[letter] += n
            total_frames += n
        per_letter: List[Dict[str, Any]] = []
        for k in app.ALBANIAN_LETTERS:
            v = counts.get(k, 0)
            per_letter.append({"letter": k, "frames": v})
        out["clips"] = {
            "path": str(keypoints_dir),
            "total_frames": total_frames,
            "letters_with_data": len(counts),
            "per_letter": per_letter,
        }

    X, Y, c = app.load_labeled_samples(keypoints_dir, alfabeti_h5, legacy_h5=legacy_h5)
    missing = [l for l in app.ALBANIAN_LETTERS if l not in c]
    combined: Dict[str, Any] = {
        "samples": int(X.shape[0]),
        "unique_letters": len(c),
        "missing_letters": missing,
    }
    if X.size:
        combined["feature_stats"] = {
            "min": float(X.min()),
            "max": float(X.max()),
            "mean": float(X.mean()),
            "std": float(X.std()),
        }
    out["combined"] = combined
    return out
