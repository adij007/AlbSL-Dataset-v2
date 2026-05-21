# Datasets layout

Canonical training and live-inference data lives under **`processed/`**. Everything else is either raw source media, optional export bundles, or archives.

```
datasets/
├── README.md                 ← this file
├── raw/                      Original images/videos (~1.5 GB). Rebuild pipeline input.
├── processed/                ★ Use this for training and apps (see processed/README.md)
│   ├── core_data/data/       NPZ clips, H5, CSV landmarks
│   ├── consolidated/         Parquet splits (BERT / train_albsl.py)
│   ├── assets/               JSON for live apps (landmarks, words, templates)
│   ├── external/             Import manifests and mappings
│   ├── landmarks/            Per-letter JSON KB (overlaps assets; legacy)
│   └── clips/                Clip split reports
├── csv_dataset/              Optional CSV exports / staging (see csv_dataset/README.md)
├── json_dataset/             Unified coordinates bundle for live (see json_dataset/README.md)
└── archive/                  Old pipeline snapshots (safe to delete or move off-disk)
```

## Quick reference — which path for what

| Task | Path |
|------|------|
| v2 train (`albsl_app_v2.py train`) | `processed/core_data/data/keypoints/*.npz`, H5, plus CSV below |
| Confirmed live captures | `csv_dataset/confirmed_labels.csv` (or `processed/.../csv/` if copied) |
| Merged static coordinates | `csv_dataset/coordinates.csv` (~53 MB) |
| Legacy coordinates subset | `csv_dataset/coordinates_legacy_subset.csv` (~44k rows; pre-merge) |
| BERT train (`train_albsl.py`) | `processed/consolidated/albsl_dataset_v2/*.parquet` |
| v2 live templates / words | `processed/assets/*.json` or `json_dataset/coordinates.json` |
| Re-run consolidation | `python Script/consolidate_data.py --data-root processed/core_data/data` |

## `csv_dataset/` CSV trio

- **`coordinates.csv`** — merged file for training (largest).
- **`coordinates_legacy_subset.csv`** — older subset (~44,251 rows); safe to archive if you only use the merged file.
- **`confirmed_labels.csv`** — live confirmation rows.

## Size (approximate)

| Folder | Size |
|--------|------|
| `raw/` | ~1.5 GB |
| `processed/` | ~0.75 GB |
| `csv_dataset/` | ~85 MB (merged + legacy + confirmed CSVs) |
| `json_dataset/` | ~3 MB |
| `archive/` | negligible |
