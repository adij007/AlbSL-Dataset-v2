# JSON dataset bundle

## `coordinates.json`

Unified bundle used by **live inference** when you pass `--unified-coords-json` (default in `albsl_app_v2.py`).

It can supply:

- Word dictionary (`words` section)
- Static letter templates
- Dynamic letter templates

Production-sized copies also live under [`../processed/assets/`](../processed/assets/) (`albsl_landmarks.json`, etc.). Apps load assets first; the unified JSON fills gaps.

Built by:

```powershell
python Script/merge_all_coordinates_json.py
```

Default output path: `datasets/json_dataset/coordinates.json`.

This file is **not** fed directly into BERT training (that uses Parquet under `processed/consolidated/`).
