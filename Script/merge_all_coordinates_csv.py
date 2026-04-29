from __future__ import annotations

import argparse
import json
import hashlib
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

ALBANIAN_LETTERS: List[str] = [
    "A", "B", "C", "Ç", "D", "Dh", "E", "Ë", "F", "G", "Gj", "H", "I", "J", "K",
    "L", "Ll", "M", "N", "Nj", "O", "P", "Q", "R", "Rr", "S", "Sh", "T", "Th",
    "U", "V", "X", "Xh", "Y", "Z", "Zh",
]
LM63_COLS = [f"lm{i}_{ax}" for i in range(21) for ax in ("x", "y", "z")]
VIDEO_63 = [
    f"{name}_{ax}"
    for name in [
        "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
        "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
        "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
        "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
    ]
    for ax in ("x", "y", "z")
]


def _normalize_letter(token: str) -> Optional[str]:
    t = str(token).strip()
    for letter in ALBANIAN_LETTERS:
        if t.upper() == letter.upper():
            return letter
    return None


def _hash_coords(vals: np.ndarray) -> str:
    q = np.round(vals.astype(np.float32), 5)
    return hashlib.blake2b(q.tobytes(), digest_size=16).hexdigest()


def _pack_row(label: str, vals: np.ndarray, source_type: str, source_file: Path) -> Dict[str, object]:
    row: Dict[str, object] = {"label": label, "source_type": source_type, "source_file": str(source_file.as_posix())}
    for i in range(21):
        base = i * 3
        row[f"lm{i}_x"] = float(vals[base + 0])
        row[f"lm{i}_y"] = float(vals[base + 1])
        row[f"lm{i}_z"] = float(vals[base + 2])
    return row


def _extract_from_standard_lm(df: pd.DataFrame, path: Path) -> Iterable[Dict[str, object]]:
    if not all(c in df.columns for c in LM63_COLS):
        return []
    label_col = "label" if "label" in df.columns else ("letter" if "letter" in df.columns else None)
    if label_col is None:
        return []
    rows: List[Dict[str, object]] = []
    for _, r in df.iterrows():
        lab = _normalize_letter(str(r[label_col]))
        if lab is None:
            continue
        vals = np.array([float(r[c]) for c in LM63_COLS], dtype=np.float32)
        if vals.shape[0] != 63 or not np.isfinite(vals).all():
            continue
        rows.append(_pack_row(lab, vals, "standard_lm", path))
    return rows


def _extract_from_landmarks63_json(df: pd.DataFrame, path: Path) -> Iterable[Dict[str, object]]:
    if "landmarks_63" not in df.columns:
        return []
    label_col = "label" if "label" in df.columns else ("letter" if "letter" in df.columns else None)
    if label_col is None:
        return []
    rows: List[Dict[str, object]] = []
    for _, r in df.iterrows():
        lab = _normalize_letter(str(r[label_col]))
        if lab is None:
            continue
        try:
            raw = str(r["landmarks_63"]).strip()
            vals = np.array(json.loads(raw) if raw.startswith("[") else [], dtype=np.float32)
        except Exception:
            continue
        if vals.shape[0] != 63 or not np.isfinite(vals).all():
            continue
        rows.append(_pack_row(lab, vals, "landmarks_63", path))
    return rows


def _extract_from_alfabeti(df: pd.DataFrame, path: Path) -> Iterable[Dict[str, object]]:
    need = [f"lm{i}_{ax}_norm" for i in range(21) for ax in ("x", "y", "z")]
    if not all(c in df.columns for c in need):
        return []
    label_col = "letter" if "letter" in df.columns else ("label" if "label" in df.columns else None)
    if label_col is None:
        return []
    rows: List[Dict[str, object]] = []
    for _, r in df.iterrows():
        lab = _normalize_letter(str(r[label_col]))
        if lab is None:
            continue
        vals = np.array([float(r[c]) for c in need], dtype=np.float32)
        if vals.shape[0] != 63 or not np.isfinite(vals).all():
            continue
        rows.append(_pack_row(lab, vals, "alfabeti_norm", path))
    return rows


def _extract_from_video_letter_file(df: pd.DataFrame, path: Path) -> Iterable[Dict[str, object]]:
    if not all(c in df.columns for c in VIDEO_63):
        return []
    m = re.match(r"^\d+_(.+)\.csv$", path.name, re.IGNORECASE)
    if not m:
        return []
    lab = _normalize_letter(m.group(1))
    if lab is None:
        return []
    rows: List[Dict[str, object]] = []
    for _, r in df.iterrows():
        vals = np.array([float(r[c]) for c in VIDEO_63], dtype=np.float32)
        if vals.shape[0] != 63 or not np.isfinite(vals).all():
            continue
        rows.append(_pack_row(lab, vals, "video_63", path))
    return rows


def _extract_from_part4(df: pd.DataFrame, path: Path) -> Iterable[Dict[str, object]]:
    right_cols = [f"right_lm{i}_{ax}_norm" for i in range(21) for ax in ("x", "y", "z")]
    left_cols = [f"left_lm{i}_{ax}_norm" for i in range(21) for ax in ("x", "y", "z")]
    if not (all(c in df.columns for c in right_cols) or all(c in df.columns for c in left_cols)):
        return []
    label_col = "letter" if "letter" in df.columns else ("label" if "label" in df.columns else None)
    if label_col is None:
        return []
    rows: List[Dict[str, object]] = []
    has_right_det = "hand_detected_right" in df.columns
    has_left_det = "hand_detected_left" in df.columns
    for _, r in df.iterrows():
        lab = _normalize_letter(str(r[label_col]))
        if lab is None:
            continue
        use_right = bool(r["hand_detected_right"]) if has_right_det else True
        use_left = bool(r["hand_detected_left"]) if has_left_det else False
        vals = None
        if use_right and all(c in df.columns for c in right_cols):
            vals = np.array([float(r[c]) for c in right_cols], dtype=np.float32)
        elif use_left and all(c in df.columns for c in left_cols):
            vals = np.array([float(r[c]) for c in left_cols], dtype=np.float32)
        elif all(c in df.columns for c in right_cols):
            vals = np.array([float(r[c]) for c in right_cols], dtype=np.float32)
        elif all(c in df.columns for c in left_cols):
            vals = np.array([float(r[c]) for c in left_cols], dtype=np.float32)
        if vals is None or vals.shape[0] != 63 or not np.isfinite(vals).all():
            continue
        rows.append(_pack_row(lab, vals, "part4_norm", path))
    return rows


def _extract_rows(df: pd.DataFrame, path: Path) -> List[Dict[str, object]]:
    for extractor in (
        _extract_from_standard_lm,
        _extract_from_landmarks63_json,
        _extract_from_alfabeti,
        _extract_from_video_letter_file,
        _extract_from_part4,
    ):
        out = list(extractor(df, path))
        if out:
            return out
    return []


def merge_coordinates(input_root: Path, output_csv: Path) -> None:
    rows: List[Dict[str, object]] = []
    for p in sorted(input_root.rglob("*.csv")):
        if p.resolve() == output_csv.resolve():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        rows.extend(_extract_rows(df, p))

    if not rows:
        raise SystemExit("No coordinate rows found in CSV files.")

    seen = set()
    deduped: List[Dict[str, object]] = []
    for r in rows:
        vals = np.array([r[c] for c in LM63_COLS], dtype=np.float32)
        key = (str(r["label"]), _hash_coords(vals))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    out_df = pd.DataFrame(deduped)
    col_order = ["label"] + LM63_COLS + ["source_type", "source_file"]
    out_df = out_df[col_order]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Wrote {len(out_df)} rows -> {output_csv}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge all CSV coordinate rows into one coordinates.csv")
    ap.add_argument("--input-root", type=Path, default=Path("datasets/processed"))
    ap.add_argument("--output-csv", type=Path, default=Path("datasets/processed/core_data/data/csv/coordinates.csv"))
    args = ap.parse_args()
    merge_coordinates(args.input_root, args.output_csv)


if __name__ == "__main__":
    main()
