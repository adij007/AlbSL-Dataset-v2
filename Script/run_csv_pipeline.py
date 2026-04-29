"""Run end-to-end CSV pipeline: external import → consolidate → embedding experiment.

Optional: snapshot CSV inputs under datasets/processed/backups/ (no git changes).

Usage (repo root):
  python Script/run_csv_pipeline.py
  python Script/run_csv_pipeline.py --no-backup
  python Script/run_csv_pipeline.py --train   # also runs train_albsl.py (long)
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), file=sys.stderr)
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def _backup_csvs() -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    snap = ROOT / "datasets" / "processed" / "backups" / f"csv_snapshot_{ts}"
    snap.mkdir(parents=True, exist_ok=True)

    copies: list[tuple[Path, Path]] = []
    data_csv = ROOT / "datasets" / "processed" / "core_data" / "data" / "csv"
    if data_csv.is_dir():
        copies.append((data_csv, snap / "data_csv"))
    vk = ROOT / "datasets" / "processed" / "core_data" / "video_keypoints.csv"
    if vk.is_file():
        copies.append((vk, snap / vk.name))
    ext = ROOT / "datasets" / "processed" / "external" / "external_normalized.csv"
    if ext.is_file():
        copies.append((ext, snap / ext.name))
    conf = ROOT / "datasets" / "processed" / "core_data" / "data" / "csv" / "confirmed_labels.csv"
    if conf.is_file():
        copies.append((conf, snap / "confirmed_labels.csv"))

    for src, dst in copies:
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    print(f"Backup snapshot: {snap}", file=sys.stderr)
    return snap


def main() -> None:
    ap = argparse.ArgumentParser(description="Run CSV merge + benchmark pipeline.")
    ap.add_argument("--no-backup", action="store_true", help="Skip CSV snapshot.")
    ap.add_argument(
        "--train",
        action="store_true",
        help="Also run train_albsl.py on consolidated parquet (can take a long time).",
    )
    args = ap.parse_args()

    py = sys.executable
    if not args.no_backup:
        _backup_csvs()

    _run([py, str(ROOT / "Script" / "external_import_normalize.py")])
    _run(
        [
            py,
            str(ROOT / "Script" / "consolidate_data.py"),
            "--data-root",
            "datasets/processed/core_data/data",
            "--out-dir",
            "datasets/processed/consolidated/albsl_dataset_v2",
        ]
    )
    _run(
        [
            py,
            str(ROOT / "Script" / "embedding_experiment.py"),
            "--data-dir",
            "datasets/processed/consolidated/albsl_dataset_v2",
        ]
    )
    if args.train:
        _run([py, str(ROOT / "Script" / "train_albsl.py"), "--data-dir", "datasets/processed/consolidated/albsl_dataset_v2"])

    print("Pipeline finished.", file=sys.stderr)


if __name__ == "__main__":
    main()
