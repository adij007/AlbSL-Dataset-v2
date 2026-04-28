"""
TASK 1 — Data consolidation for AlbSL.

Scans a data root (default: ./data), parses supported formats, harmonizes
schemas, deduplicates, stratify-splits 80/10/10, writes Parquet + reports.

Run:
  python consolidate_data.py
  python consolidate_data.py --data-root D:/AlbSL-Dataset-v2/data --out-dir albsl_dataset_v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# --- Albanian alphabet (same order as rest of project) -----------------------
ALBANIAN_LETTERS: List[str] = [
    "A", "B", "C", "Ç", "D", "Dh", "E", "Ë", "F", "G", "Gj", "H", "I", "J", "K",
    "L", "Ll", "M", "N", "Nj", "O", "P", "Q", "R", "Rr", "S", "Sh", "T", "Th",
    "U", "V", "X", "Xh", "Y", "Z", "Zh",
]
VALID_LABELS: Set[str] = set(ALBANIAN_LETTERS)
LETTER_RE = re.compile(
    r"^([A-Za-z\u00C7\u00E7Ëë]{1,2}|D[hH]|G[jJ]|L[lL]|N[jJ]|R[rR]|S[hH]|T[hH]|X[hH]|Z[hH]|[Rr]{2}|[Çç])$"
)

# Video CSV landmark columns (first 63 = 21x3)
_LM = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]
VIDEO_63: List[str] = [f"{n}_{a}" for n in _LM for a in ("x", "y", "z")]

DEDUP_EPS = 1e-4
RANDOM_SEED = 42


# --- snake_case ----------------------------------------------------------------

def to_snake_case(name: str) -> str:
    s = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", str(name).replace("-", "_"))
    s = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s)
    s = s.lower().replace(" ", "_")
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "field"


# --- row schema ----------------------------------------------------------------
@dataclass
class Row:
    label: str
    landmarks: np.ndarray  # (21, 3) float32
    subject_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: Optional[str] = None
    source_file: str = ""
    source_type: str = ""


# --- label normalization -------------------------------------------------------

def _normalize_letter(s: str) -> Optional[str]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    t = str(s).strip()
    for letter in ALBANIAN_LETTERS:
        if t.upper() == letter.upper():
            return letter
    up = t.upper()
    m = {
        "C+": "Ç", "C\\u00C7": "Ç", "E+": "Ë",
    }
    if t in m:
        return m[t]
    if up == "CC" or t == "Ç":
        return "Ç"
    if t == "Ë" or up == "E" and "Ë" in t:
        if len(t) > 1:
            pass
    for letter in ALBANIAN_LETTERS:
        if t == letter or up == letter.upper():
            return letter
    if up in ("UNKNOWN", "UNKN", "?"):
        return None
    if t in VALID_LABELS:
        return t
    return None


# --- file loaders ---------------------------------------------------------------


def _rows_from_video_csv(path: Path) -> Iterator[Row]:
    m = re.match(r"^(\d+)_(.+)\.csv$", path.name, re.IGNORECASE)
    if not m:
        return
    letter = _normalize_letter(m.group(2))
    if letter is None:
        return
    try:
        df = pd.read_csv(path)
    except Exception:
        return
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    get: List[str] = []
    for orig in VIDEO_63:
        k = orig.lower()
        if k in lower:
            get.append(lower[k])
        else:
            return
    for i in range(len(df)):
        try:
            vals = [float(df.iloc[i][c]) for c in get]
        except Exception:
            continue
        arr = np.array(vals, dtype=np.float32).reshape(21, 3)
        if not np.isfinite(arr).all():
            continue
        yield Row(
            label=letter,
            landmarks=arr,
            session_id=path.stem,
            source_file=str(path.as_posix()),
            source_type="video_csv",
        )


def _rows_from_alfabeti_csv(path: Path) -> Iterator[Row]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return
    df.columns = [to_snake_case(c) for c in df.columns]
    for _, r in df.iterrows():
        letter = _normalize_letter(r.get("letter", r.get("label", "")))
        if letter is None:
            continue
        zmx = [f"lm{i}_x_norm" for i in range(21)]
        zmy = [f"lm{i}_y_norm" for i in range(21)]
        zmz = [f"lm{i}_z_norm" for i in range(21)]
        if not all(c in r.index for c in zmx + zmy + zmz):
            continue
        x = [float(r[c]) for c in zmx]
        yv = [float(r[c]) for c in zmy]
        z = [float(r[c]) for c in zmz]
        arr = np.stack([x, yv, z], axis=1).astype(np.float32)
        sid = r.get("subject_id") or r.get("image_file", "")
        yield Row(
            label=letter,
            landmarks=arr,
            subject_id=str(sid) if pd.notna(sid) else None,
            source_file=str(path.as_posix()),
            source_type="alfabeti_csv",
        )


def _rows_from_npz(path: Path) -> Iterator[Row]:
    m = re.match(r"^(\d+)_(.+)\.npz$", path.name, re.IGNORECASE)
    if not m:
        return
    letter = _normalize_letter(m.group(2))
    if letter is None:
        return
    try:
        d = np.load(path, allow_pickle=True)
    except Exception:
        return
    xyz = None
    for k in ("xyz", "xyz_right", "data"):
        if k in d.files:
            try:
                xyz = np.asarray(d[k], dtype=np.float32)
            except Exception:
                pass
            if xyz is not None and xyz.ndim == 3 and xyz.shape[1:] == (21, 3):
                break
            xyz = None
    if xyz is None or xyz.size == 0:
        return
    T = xyz.shape[0]
    for t in range(T):
        arr = np.asarray(xyz[t], dtype=np.float32)
        if arr.shape != (21, 3) or not np.isfinite(arr).all():
            continue
        yield Row(
            label=letter,
            landmarks=arr,
            session_id=path.stem,
            source_file=str(path.as_posix()),
            source_type="npz",
        )


def _rows_from_json_file(path: Path) -> Iterator[Row]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return
    if isinstance(data, dict) and "samples" in data:
        data = data["samples"]
    if not isinstance(data, list):
        return
    for item in data:
        if not isinstance(item, dict):
            continue
        keys = {to_snake_case(k): v for k, v in item.items()}
        label = _normalize_letter(keys.get("label", keys.get("letter", None)))
        lm = keys.get("landmarks", keys.get("xyz", None))
        if label is None or lm is None:
            continue
        try:
            arr = np.array(lm, dtype=np.float32)
            if arr.size == 63:
                arr = arr.reshape(21, 3)
        except Exception:
            continue
        if arr.shape != (21, 3):
            continue
        if not np.isfinite(arr).all():
            continue
        yield Row(
            label=label,
            landmarks=arr,
            subject_id=str(keys["subject_id"]) if keys.get("subject_id") is not None else None,
            session_id=str(keys["session_id"]) if keys.get("session_id") is not None else None,
            timestamp=str(keys["timestamp"]) if keys.get("timestamp") is not None else None,
            source_file=str(path.as_posix()),
            source_type="json",
        )


def _rows_from_h5(path: Path) -> Iterator[Row]:
    try:
        import h5py
    except ImportError:
        return
    try:
        f = h5py.File(path, "r")
    except Exception:
        return
    try:
        for key in ("labels", "letter", "y"):
            if key in f:
                labels = f[key][:]
                break
        else:
            return
        for fk in ("features", "keypoints", "data", "xyz", "right_hand", "right_hand_c"):
            if fk in f:
                feats = f[fk][:]
                break
        else:
            return
        n = min(len(labels), len(feats))
        for i in range(n):
            try:
                lab = _normalize_letter(
                    labels[i].decode() if isinstance(labels[i], (bytes, np.bytes_)) else str(labels[i])
                )
            except Exception:
                continue
            if lab is None:
                continue
            row = np.asarray(feats[i], dtype=np.float32).ravel()
            if row.size >= 63:
                row = row[:63]
            if row.size != 63:
                continue
            arr = row.reshape(21, 3)
            if not np.isfinite(arr).all():
                continue
            yield Row(
                label=lab,
                landmarks=arr,
                source_file=str(path.as_posix()),
                source_type="h5",
            )
    finally:
        f.close()


def _scan_data_root(root: Path) -> List[Row]:
    rows: List[Row] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".mp4", ".mp3", ".txt", ".log", ".md", ".pyc"):
            if "meta" in p.name.lower() and p.suffix.lower() == ".txt":
                pass
            else:
                if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".mp4", ".mp3", ".pyc", ".log"):
                    continue
        if p.suffix.lower() == ".csv":
            if p.parent.name == "videos" and re.match(r"^\d+_.+\.csv$", p.name):
                rows.extend(_rows_from_video_csv(p))
            elif p.name == "alfabeti_keypoints.csv" or "alfabeti" in p.name:
                rows.extend(_rows_from_alfabeti_csv(p))
        elif p.suffix.lower() == ".npz":
            rows.extend(_rows_from_npz(p))
        elif p.suffix.lower() == ".json" and p.name not in ("manifest.json", "dataset_stats.json"):
            if "schema" in p.name.lower() or "manifest" in p.name.lower():
                continue
            rows.extend(_rows_from_json_file(p))
        elif p.suffix.lower() in (".h5", ".hdf5"):
            rows.extend(_rows_from_h5(p))
    return rows


# --- hash / dedup ---------------------------------------------------------------

def _landmark_hash(landmarks: np.ndarray) -> str:
    q = np.round(landmarks / DEDUP_EPS).astype(np.int64)
    h = hashlib.blake2b(q.tobytes(), digest_size=16).hexdigest()
    return h


def _deduplicate(rows: List[Row]) -> Tuple[List[Row], int]:
    seen: Set[Tuple[str, str]] = set()
    out: List[Row] = []
    dropped = 0
    for r in rows:
        key = (r.label, _landmark_hash(r.landmarks))
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        out.append(r)
    return out, dropped


# --- stratified split (min per class) ------------------------------------------
def _stratified_split(
    df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float, min_per_split: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    from sklearn.model_selection import train_test_split

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    rng = np.random.RandomState(RANDOM_SEED)
    labels = df["label"].values
    all_idx = np.arange(len(df))
    by_label: Dict[str, List[int]] = {}
    for i, lab in enumerate(labels):
        by_label.setdefault(lab, []).append(i)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    warnings: List[str] = []
    for lab, idxs in by_label.items():
        idxs = list(idxs)
        if len(idxs) < 3:
            train_idx.extend(idxs)
            warnings.append(f"Class {lab}: only {len(idxs)} sample(s) — all in train")
            continue
        # first: train vs temp
        tr, te = train_test_split(
            idxs,
            test_size=(1.0 - train_ratio),
            random_state=rng,
            shuffle=True,
        )
        if len(tr) < min_per_split or len(te) < 2:
            train_idx.extend(idxs)
            warnings.append(f"Class {lab}: forced all to train (small n)")
            continue
        # split te into val + test by ratio
        v_share = val_ratio / (val_ratio + test_ratio)
        val_ids, te_ids = train_test_split(
            te,
            test_size=(1.0 - v_share) if (val_ratio + test_ratio) > 0 else 0.5,
            random_state=rng,
            shuffle=True,
        ) if te.size else ([], [])
        if len(val_ids) < min_per_split or len(te_ids) < min_per_split:
            train_idx.extend(idxs)
            continue
        train_idx.extend(tr)
        val_idx.extend(list(val_ids))
        test_idx.extend(list(te_ids))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True) if val_idx else pd.DataFrame(columns=df.columns)
    test_df = df.iloc[test_idx].reset_index(drop=True) if test_idx else pd.DataFrame(columns=df.columns)
    return train_df, val_df, test_df, {"warnings": warnings, "dropped": []}


def _simpler_stratify(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    from sklearn.model_selection import train_test_split

    rng = np.random.RandomState(RANDOM_SEED)
    y = df["label"].values
    if len(df) < 3:
        return df, pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns), {
            "warnings": ["Entire dataset < 3 rows: all in train."],
        }
    idx = np.arange(len(df))
    train_i, te = train_test_split(idx, test_size=0.2, random_state=rng, stratify=y)
    y_tr = df.iloc[train_i]["label"].values
    val_i, test_i = train_test_split(
        np.arange(len(train_i)),
        test_size=0.5,
        random_state=rng,
        stratify=y_tr,
    )
    v_abs = train_i[val_i]
    tr_abs = np.setdiff1d(train_i, v_abs)
    t_abs = te
    train_df = df.iloc[tr_abs].reset_index(drop=True)
    val_df = df.iloc[v_abs].reset_index(drop=True)
    test_df = df.iloc[t_abs].reset_index(drop=True)
    for split_name, sdf in [("val", val_df), ("test", test_df), ("train", train_df)]:
        if len(sdf) and sdf["label"].nunique() < y.nunique():
            return _fallback_split(df)
    return train_df, val_df, test_df, {"warnings": []}


def _fallback_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    rng = np.random.RandomState(RANDOM_SEED)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n = len(idx)
    nt = int(0.8 * n)
    nv = int(0.1 * n)
    tr = idx[:nt]
    va = idx[nt : nt + nv]
    te = idx[nt + nv :]
    w = "Stratification failed; used shuffled 80/10/10 index split (may be imbalanced per class)"
    return (
        df.iloc[tr].reset_index(drop=True),
        df.iloc[va].reset_index(drop=True) if len(va) else pd.DataFrame(columns=df.columns),
        df.iloc[te].reset_index(drop=True) if len(te) else pd.DataFrame(columns=df.columns),
        {"warnings": [w]},
    )


def _safe_stratify(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    w: List[str] = []
    if df["label"].value_counts().min() < 2 or len(df) < 6:
        train, val, test, meta = _fallback_split(df)
        w.extend(meta.get("warnings", []))
    else:
        try:
            train, val, test, meta = _simpler_stratify(df)
            w.extend(meta.get("warnings", []))
        except Exception:
            train, val, test, meta = _fallback_split(df)
            w.extend(meta.get("warnings", []))
    return train, val, test, w


# --- main consolidation ---------------------------------------------------------

def _rows_to_dataframe(rows: List[Row]) -> pd.DataFrame:
    data = {
        "label": [r.label for r in rows],
        "landmarks": [r.landmarks.astype(np.float32) for r in rows],
        "landmarks_flat": [r.landmarks.astype(np.float32).ravel() for r in rows],
        "subject_id": [r.subject_id for r in rows],
        "session_id": [r.session_id for r in rows],
        "timestamp": [r.timestamp for r in rows],
        "source_file": [r.source_file for r in rows],
        "source_type": [r.source_type for r in rows],
    }
    return pd.DataFrame(data)


def _write_markdown(
    out_dir: Path,
    sources: Dict[str, int],
    n_before: int,
    n_after: int,
    n_drop: int,
    w: List[str],
) -> None:
    lines = [
        "# Data consolidation report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Source counts (by source_type)",
        "",
    ]
    for k, v in sorted(sources.items()):
        lines.append(f"- **{k}**: {v}")
    lines.extend(
        [
            "",
            f"- **Rows before deduplication**: {n_before}",
            f"- **Rows after deduplication**: {n_after} (dropped {n_drop} duplicates)",
            "",
        ]
    )
    if w:
        lines.append("## Warnings / split notes\n")
        for x in w:
            lines.append(f"- {x}")
    path = out_dir / "consolidation_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")


def consolidate(data_root: Path, out_dir: Path) -> None:
    if not data_root.is_dir():
        print(f"error: data root not found: {data_root}", file=sys.stderr)
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Scanning {data_root} ...", file=sys.stderr)
    rows = _scan_data_root(data_root)
    n_before = len(rows)
    rows, n_drop = _deduplicate(rows)
    n_after = len(rows)
    if n_after == 0:
        print("error: no valid samples; check data formats.", file=sys.stderr)
        sys.exit(2)
    by_src: Dict[str, int] = {}
    for r in rows:
        by_src[r.source_type] = by_src.get(r.source_type, 0) + 1
    df = _rows_to_dataframe(rows)
    label_map: Dict[str, int] = {
        str(lb): i for i, lb in enumerate(sorted({str(x) for x in df["label"]}))
    }
    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(
            {"label_to_id": label_map, "id_to_label": {v: k for k, v in label_map.items()}},
            f,
            ensure_ascii=False,
            indent=2,
        )
    df["label_id"] = df["label"].map(label_map)
    tr, va, te, wlist = _safe_stratify(df)
    stats: Dict[str, Any] = {
        "train": {str(k): int(v) for k, v in tr["label"].value_counts().items()} if len(tr) else {},
        "val": {str(k): int(v) for k, v in va["label"].value_counts().items()} if len(va) else {},
        "test": {str(k): int(v) for k, v in te["label"].value_counts().items()} if len(te) else {},
    }
    with open(out_dir / "split_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    def pack_frame(part: pd.DataFrame) -> pd.DataFrame:
        t = part.copy()
        t["landmarks_63"] = t["landmarks"].apply(
            lambda a: np.asarray(a, dtype=np.float32).reshape(-1).tolist()
        )
        t = t.drop(columns=["landmarks", "landmarks_flat"], errors="ignore")
        return t

    for name, part in (("train", tr), ("val", va), ("test", te)):
        pack_frame(part).to_parquet(out_dir / f"{name}.parquet", index=False)
    _write_markdown(out_dir, by_src, n_before, n_after, n_drop, wlist)
    print(
        f"Wrote {out_dir}: train={len(tr)} val={len(va)} test={len(te)} (dedup dropped {n_drop})",
        file=sys.stderr,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data"), help="Root to scan (recursive)")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("albsl_dataset_v2"),
        help="Output directory for parquet and reports",
    )
    return ap.parse_args()


def main() -> None:
    a = parse_args()
    consolidate(a.data_root, a.out_dir)


if __name__ == "__main__":
    main()
