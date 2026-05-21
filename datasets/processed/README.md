# Processed data (canonical)

All scripts should read and write training-ready artifacts here.

## `core_data/data/`

| Path | Used by |
|------|---------|
| `keypoints/*.npz` | `albsl_app_v2.py` train/live (temporal windows) |
| `alfabeti_keypoints.h5` | v2 train/live (static pseudo-sequences) |
| `csv/confirmed_labels.csv` | Consolidation, v2 train (CSV pseudo-sequences), live confirm (`y` key) |
| `csv/coordinates.csv` | Consolidation, v2 train (merged static landmarks) |
| `manifest.json`, `segments/` | Pipeline metadata |

## `consolidated/albsl_dataset_v2/`

Output of `Script/consolidate_data.py`:

- `train.parquet`, `val.parquet`, `test.parquet`
- `label_map.json`, `split_stats.json`, `consolidation_report.md`

Consumed by `Script/train_albsl.py` (BERT).

## `assets/`

Small JSON bundles for **live inference** (template matching, words, dynamics):

- `albsl_landmarks.json`
- `albsl_dynamic_templates.json`
- `albsl_words_dictionary.json`

## `landmarks/albsl_landmarks/`

Per-letter JSON files; largely superseded by `assets/albsl_landmarks.json`. Kept for reference / older tools.

## `external/`

External dataset import metadata (`sources_manifest.json`, `mappings/`).
