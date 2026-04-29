"""RAG-style augmentation for AlbSL keypoint data.

Builds a searchable vector index of normalized keypoint features from the
extracted video clips and alfabeti images, then uses retrieval to synthesize
augmented samples via k-NN mixup and prototype blending.

Why this is RAG: instead of free-form data synthesis, each augmented sample
is conditioned on real, retrieved exemplars for the same letter from the
dataset — giving a strong, label-consistent generative signal.

Two augmentation strategies are implemented:

1. retrieve_and_mix:  sample = α*query + (1-α)*retrieved_neighbor
   Interpolates in keypoint-feature space within the same class.

2. prototype_blend:   sample = β*query + (1-β) * avg(top-k same-class)
   Blends the query with the class prototype (retrieved centroid).

Usage:
  python Script/rag_augment.py build --out datasets/processed/core_data/data/rag/index.npz
  python Script/rag_augment.py augment --index datasets/processed/core_data/data/rag/index.npz --out datasets/processed/core_data/data/rag/augmented.h5 --copies 5
  python Script/rag_augment.py preview --index datasets/processed/core_data/data/rag/index.npz --letter B
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from loguru import logger
from sklearn.neighbors import NearestNeighbors

from extract_keypoints_v2 import (
    ALBANIAN_LETTERS,
    canonical_normalize_hand,
    dihedral_features,
)

KEY_DIM = 123  # 21*3 xyz + 20*3 angles


def build_feature_vector(xyz_21x3: np.ndarray) -> np.ndarray:
    """Build the canonical 123-d keypoint feature used by the fused model."""
    if not np.any(xyz_21x3):
        return np.zeros(KEY_DIM, dtype=np.float32)
    normalized = canonical_normalize_hand(xyz_21x3)
    normalized = np.clip(normalized, -1.0, 1.0)
    angles = dihedral_features(normalized)
    return np.concatenate([normalized.reshape(-1), angles.reshape(-1)], axis=0).astype(np.float32)


def _load_from_alfabeti(h5_path: Path, features: List[np.ndarray], labels: List[str], sources: List[str]) -> None:
    if not h5_path.exists():
        return
    with h5py.File(h5_path, "r") as f:
        xyz = f["xyz"][...]
        det = f["detected"][...]
        lbls = [x.decode() if isinstance(x, bytes) else x for x in f["labels"][...]]
        srcs = [x.decode() if isinstance(x, bytes) else x for x in f["sources"][...]]
        for i in range(xyz.shape[0]):
            if not det[i]:
                continue
            features.append(build_feature_vector(xyz[i]))
            labels.append(str(lbls[i]))
            sources.append(str(srcs[i]))


def _load_from_videos(npz_dir: Path, features: List[np.ndarray], labels: List[str], sources: List[str]) -> None:
    if not npz_dir.exists():
        return
    for npz_path in sorted(npz_dir.glob("*.npz")):
        letter = npz_path.stem.split("_", 1)[-1]
        if letter not in ALBANIAN_LETTERS:
            continue
        data = np.load(npz_path, allow_pickle=True)
        xyz = data["xyz_right"] if "xyz_right" in data.files else data["xyz"]
        conf = data["conf_right"] if "conf_right" in data.files else data["conf"]
        T = xyz.shape[0]
        for t in range(T):
            if conf[t].max() < 0.3:
                continue
            features.append(build_feature_vector(xyz[t]))
            labels.append(letter)
            sources.append(f"{npz_path.name}#frame={t}")


def build_index(
    keypoints_dir: Path,
    alfabeti_h5: Path,
    out_path: Path,
) -> Path:
    features: List[np.ndarray] = []
    labels: List[str] = []
    sources: List[str] = []
    _load_from_alfabeti(alfabeti_h5, features, labels, sources)
    _load_from_videos(keypoints_dir, features, labels, sources)
    if not features:
        raise RuntimeError("No features collected. Did you run extraction first?")
    X = np.stack(features, axis=0).astype(np.float32)
    Y = np.array(labels, dtype=object)
    S = np.array(sources, dtype=object)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, features=X, labels=Y, sources=S)
    logger.info(
        "RAG index built: {} samples across {} labels -> {}",
        X.shape[0],
        len(set(labels)),
        out_path,
    )
    return out_path


@dataclass
class RAGIndex:
    features: np.ndarray
    labels: np.ndarray
    sources: np.ndarray
    _nn_by_label: Dict[str, NearestNeighbors]
    _idx_by_label: Dict[str, np.ndarray]
    _global_nn: NearestNeighbors


def load_index(index_path: Path) -> RAGIndex:
    data = np.load(index_path, allow_pickle=True)
    feats = data["features"].astype(np.float32)
    labels = data["labels"]
    sources = data["sources"]
    nn_by_label: Dict[str, NearestNeighbors] = {}
    idx_by_label: Dict[str, np.ndarray] = {}
    for letter in sorted(set(labels.tolist())):
        mask = labels == letter
        idxs = np.where(mask)[0]
        if len(idxs) < 1:
            continue
        k = min(16, len(idxs))
        nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(feats[idxs])
        nn_by_label[str(letter)] = nn
        idx_by_label[str(letter)] = idxs
    global_nn = NearestNeighbors(n_neighbors=min(16, len(feats)), metric="cosine").fit(feats)
    return RAGIndex(
        features=feats,
        labels=labels,
        sources=sources,
        _nn_by_label=nn_by_label,
        _idx_by_label=idx_by_label,
        _global_nn=global_nn,
    )


def retrieve(
    index: RAGIndex,
    query: np.ndarray,
    k: int = 5,
    same_label: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (neighbor_features, neighbor_labels, neighbor_sources)."""
    q = query.reshape(1, -1).astype(np.float32)
    if same_label and same_label in index._nn_by_label:
        nn = index._nn_by_label[same_label]
        parent = index._idx_by_label[same_label]
        distances, local_idx = nn.kneighbors(q, n_neighbors=min(k, nn.n_neighbors))
        global_idx = parent[local_idx[0]]
    else:
        distances, global_idx = index._global_nn.kneighbors(q, n_neighbors=min(k, index._global_nn.n_neighbors))
        global_idx = global_idx[0]
    return (
        index.features[global_idx],
        index.labels[global_idx],
        index.sources[global_idx],
    )


def retrieve_and_mix(
    index: RAGIndex,
    query: np.ndarray,
    label: str,
    k: int = 3,
    alpha_range: Tuple[float, float] = (0.5, 0.9),
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    neighbors, _, _ = retrieve(index, query, k=k, same_label=label)
    if neighbors.size == 0:
        return query
    picked = neighbors[rng.integers(len(neighbors))]
    alpha = float(rng.uniform(*alpha_range))
    mixed = alpha * query + (1.0 - alpha) * picked
    return mixed.astype(np.float32)


def prototype_blend(
    index: RAGIndex,
    query: np.ndarray,
    label: str,
    k: int = 8,
    beta_range: Tuple[float, float] = (0.4, 0.8),
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    neighbors, _, _ = retrieve(index, query, k=k, same_label=label)
    if neighbors.size == 0:
        return query
    prototype = neighbors.mean(axis=0)
    beta = float(rng.uniform(*beta_range))
    blended = beta * query + (1.0 - beta) * prototype
    return blended.astype(np.float32)


def augment_dataset(
    index: RAGIndex,
    out_path: Path,
    copies_per_source: int = 4,
    mix_ratio: float = 0.5,
    seed: int = 7,
) -> Path:
    rng = np.random.default_rng(seed)
    orig_features = index.features
    orig_labels = index.labels
    orig_sources = index.sources
    n_copies = max(1, int(copies_per_source))
    aug_feats: List[np.ndarray] = []
    aug_labels: List[str] = []
    aug_sources: List[str] = []
    aug_kinds: List[str] = []

    for i in range(orig_features.shape[0]):
        label = str(orig_labels[i])
        query = orig_features[i]
        source = str(orig_sources[i])
        for j in range(n_copies):
            if rng.random() < mix_ratio:
                feat = retrieve_and_mix(index, query, label, rng=rng)
                kind = "retrieve_and_mix"
            else:
                feat = prototype_blend(index, query, label, rng=rng)
                kind = "prototype_blend"
            aug_feats.append(feat)
            aug_labels.append(label)
            aug_sources.append(f"{source}#rag_{kind}_{j}")
            aug_kinds.append(kind)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset(
            "features", data=np.stack(aug_feats, axis=0),
            compression="gzip", compression_opts=4,
        )
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("labels", data=np.array(aug_labels, dtype=object), dtype=dt)
        f.create_dataset("sources", data=np.array(aug_sources, dtype=object), dtype=dt)
        f.create_dataset("kinds", data=np.array(aug_kinds, dtype=object), dtype=dt)
        f.attrs["total_augmented"] = len(aug_feats)
        f.attrs["source_samples"] = int(orig_features.shape[0])
    logger.info("Augmented dataset: {} samples -> {}", len(aug_feats), out_path)
    return out_path


def preview(index_path: Path, letter: str, top: int = 5) -> None:
    index = load_index(index_path)
    mask = index.labels == letter
    if not mask.any():
        print(f"No samples for letter '{letter}'.")
        return
    sample_i = int(np.where(mask)[0][0])
    query = index.features[sample_i]
    nbrs_feats, nbrs_lbls, nbrs_srcs = retrieve(index, query, k=top, same_label=letter)
    report = {
        "letter": letter,
        "query_source": str(index.sources[sample_i]),
        "neighbors": [
            {"label": str(l), "source": str(s)}
            for l, s in zip(nbrs_lbls, nbrs_srcs)
        ],
        "retrieve_and_mix_sample_norm": float(np.linalg.norm(retrieve_and_mix(index, query, letter))),
        "prototype_blend_sample_norm": float(np.linalg.norm(prototype_blend(index, query, letter))),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="RAG augmentation for AlbSL keypoints")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--keypoints-dir", type=Path, default=Path("datasets/processed/core_data/data/keypoints"))
    b.add_argument("--alfabeti-h5", type=Path, default=Path("datasets/processed/core_data/data/alfabeti_keypoints.h5"))
    b.add_argument("--out", type=Path, default=Path("datasets/processed/core_data/data/rag/index.npz"))

    a = sub.add_parser("augment")
    a.add_argument("--index", type=Path, default=Path("datasets/processed/core_data/data/rag/index.npz"))
    a.add_argument("--out", type=Path, default=Path("datasets/processed/core_data/data/rag/augmented.h5"))
    a.add_argument("--copies", type=int, default=4)
    a.add_argument("--mix-ratio", type=float, default=0.5)
    a.add_argument("--seed", type=int, default=7)

    p = sub.add_parser("preview")
    p.add_argument("--index", type=Path, default=Path("datasets/processed/core_data/data/rag/index.npz"))
    p.add_argument("--letter", type=str, required=True)
    p.add_argument("--top", type=int, default=5)

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "build":
        build_index(args.keypoints_dir, args.alfabeti_h5, args.out)
    elif args.cmd == "augment":
        index = load_index(args.index)
        augment_dataset(index, args.out, copies_per_source=args.copies, mix_ratio=args.mix_ratio, seed=args.seed)
    elif args.cmd == "preview":
        preview(args.index, args.letter, top=args.top)


if __name__ == "__main__":
    main()
