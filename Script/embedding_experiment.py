from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def _load_split(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(path)
    x = np.stack([np.asarray(v, dtype=np.float32) for v in df["landmarks_63"]], axis=0)
    y = df["label_id"].to_numpy(dtype=np.int64)
    return x, y


def _fit_encoder(train_x: np.ndarray, embed_dim: int, seed: int) -> Tuple[StandardScaler, MLPRegressor]:
    scaler = StandardScaler()
    z_train = scaler.fit_transform(train_x)
    # Autoencoder-like bottleneck via shallow MLP regressor.
    # hidden layer is the embedding space used for downstream classifier input.
    ae = MLPRegressor(
        hidden_layer_sizes=(embed_dim,),
        activation="tanh",
        solver="adam",
        max_iter=120,
        random_state=seed,
    )
    ae.fit(z_train, z_train)
    return scaler, ae


def _embed(scaler: StandardScaler, ae: MLPRegressor, x: np.ndarray) -> np.ndarray:
    z = scaler.transform(x)
    w = ae.coefs_[0]
    b = ae.intercepts_[0]
    h = np.tanh(z @ w + b)
    return h.astype(np.float32)


def _train_eval_classifier(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    seed: int,
) -> Dict[str, object]:
    clf = LogisticRegression(
        max_iter=1000,
        random_state=seed,
        n_jobs=1,
    )
    clf.fit(train_x, train_y)
    pred = clf.predict(test_x)
    acc = float(accuracy_score(test_y, pred))
    cm = confusion_matrix(test_y, pred).tolist()
    return {"accuracy": acc, "confusion_matrix": cm}


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare raw 63D vs embedding-based AlbSL training.")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/processed/consolidated/albsl_dataset_v2"),
    )
    ap.add_argument("--embed-dim", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("datasets/processed/consolidated/albsl_dataset_v2/embedding_experiment"),
    )
    args = ap.parse_args()

    train_x, train_y = _load_split(args.data_dir / "train.parquet")
    test_x, test_y = _load_split(args.data_dir / "test.parquet")
    val_path = args.data_dir / "val.parquet"
    if val_path.exists():
        val_x, val_y = _load_split(val_path)
        train_x = np.concatenate([train_x, val_x], axis=0)
        train_y = np.concatenate([train_y, val_y], axis=0)

    baseline = _train_eval_classifier(train_x, train_y, test_x, test_y, seed=args.seed)
    scaler, ae = _fit_encoder(train_x, embed_dim=args.embed_dim, seed=args.seed)
    train_e = _embed(scaler, ae, train_x)
    test_e = _embed(scaler, ae, test_x)
    embedded = _train_eval_classifier(train_e, train_y, test_e, test_y, seed=args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "baseline_raw63": baseline,
        "embedding_dim": args.embed_dim,
        "embedding_classifier": embedded,
        "delta_accuracy": float(embedded["accuracy"]) - float(baseline["accuracy"]),
    }
    (args.out_dir / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    md = [
        "# Embedding Experiment Report",
        "",
        f"- Baseline raw63 accuracy: **{baseline['accuracy']:.4f}**",
        f"- Embedding ({args.embed_dim}d) accuracy: **{embedded['accuracy']:.4f}**",
        f"- Delta: **{result['delta_accuracy']:+.4f}**",
        "",
        "Confusion matrices are exported in `report.json`.",
    ]
    (args.out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote experiment report to {args.out_dir}")


if __name__ == "__main__":
    main()
