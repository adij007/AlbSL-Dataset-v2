"""Shared parsing for confirmed_labels.csv and coordinates.csv landmark rows."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np

LM_COLS = [f"lm{i}_{ax}" for i in range(21) for ax in ("x", "y", "z")]

ALBANIAN_LETTERS: List[str] = [
    "A", "B", "C", "Ç", "D", "Dh", "E", "Ë", "F", "G", "Gj", "H", "I", "J", "K",
    "L", "Ll", "M", "N", "Nj", "O", "P", "Q", "R", "Rr", "S", "Sh", "T", "Th",
    "U", "V", "X", "Xh", "Y", "Z", "Zh",
]

CONFIRMED_CSV_FIELDNAMES: List[str] = [
    "timestamp",
    "label",
    "confidence",
    "source",
    "session_id",
    "frame_ts_ms",
    "model_source",
    "top2_margin",
    "landmarks_63",
] + LM_COLS

OLD_SCHEMA_COLS = 68
NEW_SCHEMA_COLS = 72


@dataclass
class LandmarkCsvRecord:
    label: str
    landmarks: np.ndarray  # (21, 3) float32
    timestamp: Optional[str] = None
    confidence: Optional[float] = None
    session_id: Optional[str] = None
    source: Optional[str] = None
    source_file: str = ""


def _optional_float(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _landmarks_from_parts(lm_json: str, flat_vals: list[str]) -> Optional[np.ndarray]:
    arr: Optional[np.ndarray] = None
    if lm_json:
        try:
            vals = json.loads(str(lm_json))
            arr = np.array(vals, dtype=np.float32).reshape(21, 3)
        except Exception:
            arr = None
    if arr is None and len(flat_vals) >= 63:
        try:
            arr = np.array([float(v) for v in flat_vals[:63]], dtype=np.float32).reshape(21, 3)
        except Exception:
            arr = None
    if arr is None or arr.shape != (21, 3) or not np.isfinite(arr).all():
        return None
    return arr


def iter_confirmed_csv_records(path: Path) -> Iterator[LandmarkCsvRecord]:
    """Yield landmark rows from confirmed_labels.csv (68- or 72-column schemas)."""
    path = Path(path)
    if not path.is_file():
        return
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 5:
                continue
            label = str(row[1]).strip()
            if not label or label.lower() == "label":
                continue
            if len(row) == OLD_SCHEMA_COLS:
                lm_json_idx = 4
                lm_flat_start = 5
                session_id = None
            elif len(row) == NEW_SCHEMA_COLS:
                lm_json_idx = 8
                lm_flat_start = 9
                session_id = str(row[4]).strip() or None
            else:
                continue
            arr = _landmarks_from_parts(row[lm_json_idx], row[lm_flat_start : lm_flat_start + 63])
            if arr is None:
                continue
            yield LandmarkCsvRecord(
                label=label,
                landmarks=arr,
                timestamp=str(row[0]).strip() or None,
                confidence=_optional_float(row[2]),
                session_id=session_id,
                source=str(row[3]).strip() or None if len(row) > 3 else None,
                source_file=str(path.as_posix()),
            )


def iter_coordinates_csv_records(path: Path) -> Iterator[LandmarkCsvRecord]:
    """Yield landmark rows from coordinates.csv (label + lm0_x..lm20_z)."""
    path = Path(path)
    if not path.is_file():
        return
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return
        fields = set(reader.fieldnames)
        if not all(c in fields for c in LM_COLS):
            return
        label_col = "label" if "label" in fields else ("letter" if "letter" in fields else None)
        if label_col is None:
            return
        for row in reader:
            label = str(row.get(label_col, "")).strip()
            if not label or label.lower() == "label":
                continue
            try:
                arr = np.array([float(row[c]) for c in LM_COLS], dtype=np.float32).reshape(21, 3)
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite(arr).all():
                continue
            yield LandmarkCsvRecord(
                label=label,
                landmarks=arr,
                confidence=_optional_float(row.get("confidence")),
                source_file=str(path.as_posix()),
            )


def normalize_letter(token: str) -> Optional[str]:
    t = str(token).strip()
    for letter in ALBANIAN_LETTERS:
        if t.upper() == letter.upper():
            return letter
    return None


def _landmark_dedupe_key(label: str, landmarks: np.ndarray) -> str:
    lab = normalize_letter(label) or str(label).strip()
    q = np.round(landmarks.astype(np.float32), 5)
    digest = hashlib.blake2b(q.tobytes(), digest_size=16).hexdigest()
    return f"{lab}|{digest}"


def record_to_confirmed_row(
    rec: LandmarkCsvRecord,
    *,
    source: str,
    timestamp: Optional[str] = None,
    confidence: Optional[float] = None,
) -> Dict[str, object]:
    """Build a confirmed_labels.csv row dict (72-column / DictWriter schema)."""
    lab = normalize_letter(rec.label) or str(rec.label).strip()
    xyz = rec.landmarks.astype(np.float32)
    flat = xyz.reshape(-1)
    ts = timestamp or rec.timestamp
    if not ts:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    conf = confidence if confidence is not None else rec.confidence
    row: Dict[str, object] = {
        "timestamp": ts,
        "label": lab,
        "confidence": "" if conf is None else round(float(conf), 6),
        "source": source,
        "session_id": rec.session_id or "",
        "frame_ts_ms": "",
        "model_source": "",
        "top2_margin": "",
        "landmarks_63": json.dumps(flat.tolist(), ensure_ascii=False),
    }
    for i in range(21):
        base = i * 3
        row[f"lm{i}_x"] = round(float(flat[base + 0]), 6)
        row[f"lm{i}_y"] = round(float(flat[base + 1]), 6)
        row[f"lm{i}_z"] = round(float(flat[base + 2]), 6)
    return row


def iter_coordinates_json_records(path: Path) -> Iterator[LandmarkCsvRecord]:
    """Yield rows from datasets/json_dataset/coordinates.json (static + dynamic samples)."""
    path = Path(path)
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    letters = data.get("letters") or {}
    static = letters.get("static") or {}
    for label, pts in static.items():
        lab = normalize_letter(str(label))
        if lab is None:
            continue
        try:
            arr = np.array(pts, dtype=np.float32).reshape(21, 3)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(arr).all():
            continue
        yield LandmarkCsvRecord(
            label=lab,
            landmarks=arr,
            source="coordinates_json:static",
            source_file=str(path.as_posix()),
        )
    dynamic = letters.get("dynamic") or {}
    for label, block in dynamic.items():
        lab = normalize_letter(str(label))
        if lab is None or not isinstance(block, dict):
            continue
        for sample in block.get("samples") or []:
            if not isinstance(sample, dict):
                continue
            lm_list = sample.get("landmarks") or []
            if len(lm_list) < 21:
                continue
            try:
                ordered = sorted(lm_list, key=lambda p: int(p.get("id", 0)))
                arr = np.array(
                    [[float(p["x_norm"]), float(p["y_norm"]), float(p["z_norm"])] for p in ordered[:21]],
                    dtype=np.float32,
                )
            except (KeyError, TypeError, ValueError):
                continue
            if arr.shape != (21, 3) or not np.isfinite(arr).all():
                continue
            yield LandmarkCsvRecord(
                label=lab,
                landmarks=arr,
                confidence=_optional_float(sample.get("confidence")),
                source="coordinates_json:dynamic",
                source_file=str(path.as_posix()),
            )


def merge_coordinates_into_confirmed(
    confirmed_path: Path,
    coordinate_paths: Sequence[Path],
    *,
    json_paths: Optional[Sequence[Path]] = None,
    dedupe: bool = True,
    backup: bool = True,
) -> Dict[str, int]:
    """
    Merge coordinates-style CSVs into confirmed_labels.csv.
    Existing confirmed rows are kept; coordinate rows are appended unless deduped.
    """
    confirmed_path = Path(confirmed_path)
    confirmed_path.parent.mkdir(parents=True, exist_ok=True)

    rows_by_key: Dict[str, Dict[str, object]] = {}
    stats = {
        "existing_confirmed": 0,
        "from_coordinates": 0,
        "from_json": 0,
        "skipped_duplicate": 0,
        "skipped_bad_label": 0,
        "total_written": 0,
    }

    if confirmed_path.is_file():
        for rec in iter_confirmed_csv_records(confirmed_path):
            lab = normalize_letter(rec.label)
            if lab is None:
                stats["skipped_bad_label"] += 1
                continue
            rec = LandmarkCsvRecord(
                label=lab,
                landmarks=rec.landmarks,
                timestamp=rec.timestamp,
                confidence=rec.confidence,
                session_id=rec.session_id,
                source=rec.source,
                source_file=rec.source_file,
            )
            src = rec.source or "confirmed_labels"
            row = record_to_confirmed_row(rec, source=src, timestamp=rec.timestamp)
            key = _landmark_dedupe_key(lab, rec.landmarks)
            rows_by_key[key] = row
            stats["existing_confirmed"] += 1

    for coord_path in coordinate_paths:
        coord_path = Path(coord_path)
        if not coord_path.is_file():
            continue
        tag = coord_path.stem
        for rec in iter_coordinates_csv_records(coord_path):
            lab = normalize_letter(rec.label)
            if lab is None:
                stats["skipped_bad_label"] += 1
                continue
            rec = LandmarkCsvRecord(
                label=lab,
                landmarks=rec.landmarks,
                confidence=rec.confidence,
                source=rec.source,
                source_file=str(coord_path.as_posix()),
            )
            key = _landmark_dedupe_key(lab, rec.landmarks)
            if dedupe and key in rows_by_key:
                stats["skipped_duplicate"] += 1
                continue
            rows_by_key[key] = record_to_confirmed_row(
                rec,
                source=f"coordinates_merge:{tag}",
                confidence=1.0 if rec.confidence is None else rec.confidence,
            )
            stats["from_coordinates"] += 1

    for json_path in json_paths or []:
        json_path = Path(json_path)
        if not json_path.is_file():
            continue
        tag = json_path.stem
        for rec in iter_coordinates_json_records(json_path):
            lab = normalize_letter(rec.label)
            if lab is None:
                stats["skipped_bad_label"] += 1
                continue
            rec = LandmarkCsvRecord(
                label=lab,
                landmarks=rec.landmarks,
                confidence=rec.confidence,
                source=rec.source,
                source_file=str(json_path.as_posix()),
            )
            key = _landmark_dedupe_key(lab, rec.landmarks)
            if dedupe and key in rows_by_key:
                stats["skipped_duplicate"] += 1
                continue
            rows_by_key[key] = record_to_confirmed_row(
                rec,
                source=f"coordinates_merge:{tag}:{rec.source or 'json'}",
                confidence=1.0 if rec.confidence is None else rec.confidence,
            )
            stats["from_json"] += 1

    if backup and confirmed_path.is_file() and confirmed_path.stat().st_size > 0:
        bak = confirmed_path.with_suffix(confirmed_path.suffix + ".bak")
        bak.write_bytes(confirmed_path.read_bytes())

    ordered = list(rows_by_key.values())
    with confirmed_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CONFIRMED_CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)

    stats["total_written"] = len(ordered)
    return stats
