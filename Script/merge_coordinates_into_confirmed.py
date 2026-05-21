"""Merge coordinates CSV exports into confirmed_labels.csv."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from confirmed_csv_io import merge_coordinates_into_confirmed
from path_utils import resolve_repo_path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description="Merge coordinates CSVs into confirmed_labels.csv")
    p.add_argument(
        "--confirmed-csv",
        type=Path,
        default=Path("datasets/csv_dataset/confirmed_labels.csv"),
    )
    p.add_argument(
        "--coordinates",
        type=Path,
        nargs="*",
        default=[
            Path("datasets/csv_dataset/coordinates.csv"),
            Path("datasets/csv_dataset/coordinates_legacy_subset.csv"),
        ],
        help="Coordinates-style CSV files to import",
    )
    p.add_argument(
        "--json-coordinates",
        type=Path,
        nargs="*",
        default=[Path("datasets/json_dataset/coordinates.json")],
        help="Unified coordinates JSON (static + dynamic samples) if CSVs are missing",
    )
    p.add_argument("--no-dedupe", action="store_true")
    p.add_argument("--no-backup", action="store_true")
    args = p.parse_args()

    confirmed = resolve_repo_path(args.confirmed_csv)
    coords = [resolve_repo_path(c) for c in args.coordinates]
    json_paths = [resolve_repo_path(j) for j in args.json_coordinates]
    stats = merge_coordinates_into_confirmed(
        confirmed,
        coords,
        json_paths=json_paths,
        dedupe=not args.no_dedupe,
        backup=not args.no_backup,
    )
    print(json.dumps(stats, indent=2))
    print(f"Wrote {stats['total_written']} rows -> {confirmed}", file=sys.stderr)


if __name__ == "__main__":
    main()
