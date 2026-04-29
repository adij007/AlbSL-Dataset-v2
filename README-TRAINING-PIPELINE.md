# AlbSL: raw data to Parquet, QLoRA-style training, live v3

This document wires **your existing repo** (MediaPipe extraction in `datasets/processed/core_data/data/`, CSV/NPZ layouts) to the new pipeline: **consolidation → `Script/train_albsl.py` → `models/trained/albsl_model_final/` → `Script/albsl_app_v3.py`**.

## 1) Environment

From the repository root (e.g. `D:\AlbSL-Dataset-v2`):

```powershell
.\.venv\Scripts\activate
pip install -r Dependencies/requirements.txt
```

- **PyTorch with CUDA (optional)**: use the [official](https://pytorch.org) wheel that matches your GPU, then re-run `pip install` for the rest.  
- **4-bit (bitsandbytes)**: on Windows you may need `--no-4bit` when training if a compatible wheel is unavailable.

## 2) Consolidate `data/**` to Parquet

The script **recursively** walks `./datasets/processed/core_data/data` (or `--data-root`), ingests `data/csv/videos/*.csv`, `alfabeti_keypoints.csv`, `data/keypoints/*.npz`, and optional JSON / HDF5 where parsable, normalizes to **label + 63 floats** (21×3) per row, **deduplicates** with ε = 1e−4, and writes **stratified** train/val/test (with fallback to shuffled split if a class is too small).

```powershell
python Script/consolidate_data.py --data-root datasets/processed/core_data/data --out-dir datasets/processed/consolidated/albsl_dataset_v2
```

Output:

- `datasets/processed/consolidated/albsl_dataset_v2/train.parquet`, `val.parquet`, `test.parquet`
- `label_map.json` — `label_to_id` / `id_to_label`
- `split_stats.json` — per-split label counts
- `consolidation_report.md` — source mix and warnings

**Relation to the rest of the project**: video CSVs use the same 63 hand columns as `albsl_fusion.data.VIDEO_63` (and `datasets/processed/core_data/data/csv/videos/NN_Letter.csv`). NPZ files follow `extract_keypoints_v2` (`xyz` / `xyz_right` per frame). This keeps consolidation aligned with the fusion / CSV ecosystem, not a separate ad-hoc format.

## 3) Train (`Script/train_albsl.py`)

- **Model**: small **BERT** over **22 tokens** (1 CLS + 21 joints), each joint is **3D → 256D**, **LoRA** on attention/FFN (`r=16`, `alpha=32`, `dropout=0.05`), optional **4-bit** on the BERT body when CUDA + `bitsandbytes` work.  
- **Input**: **flat 63** (`landmarks_63` column) — the same 21×3 semantic as the fusion pipeline’s 63 geometry slot (we do *not* append dihedrals in this spec; you can extend the dataset to 123D later if you align `train_albsl.py` + v3).  
- **Iterative loop**: for each **round** (up to `MAX_ROUNDS`), up to `MAX_EPOCHS_PER_ROUND` epochs; per-round **validation per-letter** accuracy; letters below `THRESHOLD` get **duplicated** rows in training and **2×** cross-entropy weight for those classes.  
- **Logs**: `Log/training_log.jsonl` (per epoch), `Log/convergence_log.jsonl` (per round, checkpoint path).

```powershell
# GPU + 4-bit (if available)
python Script/train_albsl.py --data-dir datasets/processed/consolidated/albsl_dataset_v2 --4bit

# CPU or 4bit broken on your machine
python Script/train_albsl.py --data-dir datasets/processed/consolidated/albsl_dataset_v2 --no-4bit
```

**Outputs**:

- `models/trained/checkpoints/state.pt` (last) and `models/trained/checkpoints/round_*.pt`
- `models/trained/albsl_model_final/model_full.pt` — `state_dict` + `lmap` (unless `--no-export`)
- `models/trained/albsl_model_final/label_map.json`, optional `albsl_landmark.onnx`

## 4) `models/trained/albsl_model_final` and `datasets/processed/assets/albsl_landmarks.json` for v3

- **`Script/albsl_app_v3.py`**: loads `models/trained/albsl_model_final/model_full.pt` for the **63-D** model. It **still** reads `datasets/processed/assets/albsl_landmarks.json` (same `letter → 21×3` layout as the older letter collector) and uses it **only** when softmax confidence is below `--min-conf` (template + distance in canonical 21×3).  
- **Recorder/viewer** compatibility: the JSON path and schema stay stable so external tools that only need that file and the same hand landmark order do not have to change.

**Generate a baseline JSON** (if missing), same way as the earlier flow: e.g. median per letter from `data/keypoints/*.npz`, or the helper that produced `albsl_landmarks.json` in the main project.

## 5) Live app

```powershell
python Script/albsl_app_v3.py --landmarks-json datasets/processed/assets/albsl_landmarks.json --model models/trained/albsl_model_final/model_full.pt --camera 0
```

Press **Q** in the window to quit.

## 6) Knobs (edit constants in `Script/train_albsl.py` if needed)

| Name | Default | Role |
|------|---------|------|
| `MAX_ROUNDS` | 20 | Outer curriculum rounds |
| `MAX_EPOCHS_PER_ROUND` | 10 | Inner epochs per round |
| `THRESHOLD` | 0.90 | Per-letter val accuracy to stop upsampling a class |
| `CHECKPOINT_DIR` | `models/trained/checkpoints/` | Checkpoints |
| `GRAD_ACCUM` | 4 | Gradient accumulation |
| AUG + `HARD_CLASS_LOSS_MULT` | see file | Augmentation and harder-class loss |

## 7) Limitations and linking to the fusion model

- **Fused** training (`fused_phase3_best.pt` + `albsl_fusion/`) is **image + 123D windows**; this new track is **63D single-frame (or one row = one pose)**. To unify with fusion later: extend consolidation to 123D features (copy `albsl_fusion` feature builder), widen `landmarks_63` → `landmarks_123` in Parquet, and change `joint_embed` to accept 123-D as **one token** or 41×3 (with dihedral packing) in a v4 script.  
- **QLoRA** in the spec is approximated: **4-bit** BERT with **LoRA** on target linear layers. Very small BERTs may be slower to quantize; `--no-4bit` is always valid.

## 8) Checklist

1. [ ] `python Script/consolidate_data.py`  
2. [ ] Inspect `datasets/processed/consolidated/albsl_dataset_v2/consolidation_report.md`  
3. [ ] `python Script/train_albsl.py --data-dir datasets/processed/consolidated/albsl_dataset_v2`  
4. [ ] `models/trained/albsl_model_final/model_full.pt` exists  
5. [ ] `datasets/processed/assets/albsl_landmarks.json` present (fallback)  
6. [ ] `python Script/albsl_app_v3.py`

---

For **Intel-only** extraction and legacy apps, keep using `Script/extract_keypoints_v2.py`, `Script/albsl_app_v2.py` (MLP + diagnose/train/live) as in the main README you maintain for that track.
