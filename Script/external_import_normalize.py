from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ALBANIAN_LETTERS: List[str] = [
    "A", "B", "C", "Ç", "D", "Dh", "E", "Ë", "F", "G", "Gj", "H", "I", "J", "K",
    "L", "Ll", "M", "N", "Nj", "O", "P", "Q", "R", "Rr", "S", "Sh", "T", "Th",
    "U", "V", "X", "Xh", "Y", "Z", "Zh",
]

# Conservative transfer map:
# - High confidence for direct manual alphabet overlap.
# - Lower confidence for labels likely to drift across languages/dialects.
ASL_TO_ALBSL_MAP: Dict[str, Dict[str, Any]] = {
    "A": {"target": "A", "confidence": 0.95},
    "B": {"target": "B", "confidence": 0.95},
    "C": {"target": "C", "confidence": 0.9},
    "D": {"target": "D", "confidence": 0.9},
    "E": {"target": "E", "confidence": 0.9},
    "F": {"target": "F", "confidence": 0.92},
    "G": {"target": "G", "confidence": 0.85},
    "H": {"target": "H", "confidence": 0.82},
    "I": {"target": "I", "confidence": 0.95},
    "K": {"target": "K", "confidence": 0.88},
    "L": {"target": "L", "confidence": 0.95},
    "M": {"target": "M", "confidence": 0.88},
    "N": {"target": "N", "confidence": 0.88},
    "O": {"target": "O", "confidence": 0.92},
    "P": {"target": "P", "confidence": 0.84},
    "Q": {"target": "Q", "confidence": 0.8},
    "R": {"target": "R", "confidence": 0.85},
    "S": {"target": "S", "confidence": 0.94},
    "T": {"target": "T", "confidence": 0.9},
    "U": {"target": "U", "confidence": 0.86},
    "V": {"target": "V", "confidence": 0.9},
    "X": {"target": "X", "confidence": 0.82},
    "Y": {"target": "Y", "confidence": 0.9},
    "Z": {"target": "Z", "confidence": 0.75},
    # Dynamic letters in ASL differ semantically from Albanian digraph signs.
    "J": {"target": "J", "confidence": 0.6},
}


def _canonicalize(xyz: np.ndarray) -> np.ndarray:
    if xyz.shape != (21, 3):
        return np.zeros((21, 3), dtype=np.float32)
    out = xyz.astype(np.float32, copy=True)
    out -= out[0]
    diag = float(np.linalg.norm(out.max(axis=0) - out.min(axis=0)))
    if diag > 1e-8:
        out /= diag
    p9 = out[9] / (np.linalg.norm(out[9]) + 1e-8)
    target = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v = np.cross(p9, target)
    c = float(np.dot(p9, target))
    s = float(np.linalg.norm(v))
    if s > 1e-8:
        vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float32)
        rot = np.eye(3, dtype=np.float32) + vx + (vx @ vx) * ((1.0 - c) / (s * s))
        out = (rot @ out.T).T
    signed = float(np.cross(out[5] - out[0], out[17] - out[0])[2])
    if signed < 0.0:
        out[:, 0] *= -1.0
    return np.clip(out, -1.0, 1.0)


def _download(url: str, dst: Path) -> bool:
    """Download URL to dst. Returns True on success, False on HTTP/network errors."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return True
    try:
        urllib.request.urlretrieve(url, str(dst))
        return True
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
        print(f"warning: download failed ({exc}) — url={url}", file=sys.stderr)
        return False


def _load_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            for k in ("rows", "samples", "data"):
                if isinstance(data.get(k), list):
                    return pd.DataFrame(data[k])
        return pd.DataFrame()
    raise ValueError(f"Unsupported file type: {path}")


def _extract_xyz_row(row: pd.Series) -> Optional[np.ndarray]:
    lm_cols = [f"lm{i}_{ax}" for i in range(21) for ax in ("x", "y", "z")]
    if all(c in row.index for c in lm_cols):
        vals = np.array([float(row[c]) for c in lm_cols], dtype=np.float32).reshape(21, 3)
        return vals
    for key in ("landmarks_63", "xyz63", "coords_63"):
        if key in row.index:
            try:
                arr = np.array(json.loads(str(row[key])), dtype=np.float32).reshape(21, 3)
                return arr
            except Exception:
                pass
    return None


def _map_label(label: str) -> Tuple[Optional[str], float]:
    src = label.strip().upper()
    if src in ASL_TO_ALBSL_MAP:
        ent = ASL_TO_ALBSL_MAP[src]
        return str(ent["target"]), float(ent["confidence"])
    # If dataset is already AlbSL-labeled.
    for l in ALBANIAN_LETTERS:
        if src == l.upper():
            return l, 1.0
    return None, 0.0


def _normalize_df(df: pd.DataFrame, source_name: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if df.empty:
        return out
    label_col = None
    for candidate in ("label", "letter", "class", "target"):
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        return out
    for _, r in df.iterrows():
        mapped, conf = _map_label(str(r[label_col]))
        if mapped is None:
            continue
        xyz = _extract_xyz_row(r)
        if xyz is None or xyz.shape != (21, 3) or not np.isfinite(xyz).all():
            continue
        norm = _canonicalize(xyz)
        out.append(
            {
                "label": mapped,
                "mapping_confidence": round(conf, 4),
                "source_dataset": source_name,
                "source_label": str(r[label_col]),
                "landmarks_63": json.dumps(norm.reshape(-1).tolist(), ensure_ascii=False),
                **{f"lm{i}_{ax}": round(float(norm.reshape(-1)[i * 3 + j]), 6)
                   for i in range(21) for j, ax in enumerate(("x", "y", "z"))},
            }
        )
    return out


def _expand_archive_if_needed(path: Path, target_dir: Path) -> List[Path]:
    if path.suffix.lower() != ".zip":
        return [path]
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(target_dir)
    files: List[Path] = []
    for ext in ("*.csv", "*.json", "*.parquet"):
        files.extend(sorted(target_dir.rglob(ext)))
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description="Import and normalize external sign coordinate datasets.")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/processed/external/sources_manifest.json"),
        help="JSON list of dataset sources with optional URL and local path.",
    )
    ap.add_argument(
        "--output-csv",
        type=Path,
        default=Path("datasets/processed/external/external_normalized.csv"),
    )
    ap.add_argument(
        "--downloads-dir",
        type=Path,
        default=Path("datasets/processed/external/downloads"),
    )
    ap.add_argument(
        "--strict-downloads",
        action="store_true",
        help="Exit with error if any URL download fails (default: skip failed URLs).",
    )
    args = ap.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"Missing manifest: {args.manifest}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise SystemExit("Manifest must be a JSON list.")

    all_rows: List[Dict[str, Any]] = []
    for item in manifest:
        if not isinstance(item, dict):
            continue
        source_name = str(item.get("name", "external_unknown"))
        local_path = item.get("path")
        url = item.get("url")
        if local_path:
            src_path = Path(str(local_path))
            if not src_path.is_file():
                print(
                    f"warning: skip '{source_name}' — path missing: {src_path}",
                    file=sys.stderr,
                )
                continue
        elif url:
            filename = str(item.get("filename", Path(str(url)).name or f"{source_name}.csv"))
            dst = args.downloads_dir / filename
            ok = _download(str(url), dst)
            if not ok:
                if args.strict_downloads:
                    raise SystemExit(f"Download failed for {source_name}: {url}")
                continue
            src_path = dst
        else:
            continue
        expand_dir = args.downloads_dir / f"{source_name}_unzipped"
        tables = _expand_archive_if_needed(src_path, expand_dir)
        for table in tables:
            try:
                df = _load_table(table)
            except Exception:
                continue
            all_rows.extend(_normalize_df(df, source_name=source_name))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(args.output_csv, index=False)

    mapping_dir = args.output_csv.parent / "mappings"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    (mapping_dir / "asl_to_albsl_map.json").write_text(
        json.dumps(ASL_TO_ALBSL_MAP, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(out_df)} normalized rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
