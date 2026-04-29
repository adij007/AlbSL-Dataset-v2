from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ALBANIAN_LETTERS = [
    "A", "B", "C", "Ç", "D", "Dh", "E", "Ë", "F", "G", "Gj", "H", "I", "J", "K",
    "L", "Ll", "M", "N", "Nj", "O", "P", "Q", "R", "Rr", "S", "Sh", "T", "Th",
    "U", "V", "X", "Xh", "Y", "Z", "Zh",
]


def test_npz_shapes_and_values() -> None:
    search_dirs = [
        Path("datasets/processed/core_data/data"),
        Path("datasets/processed/clips/split_clips"),
        Path("outputs"),
    ]
    fixtures = []
    for root in search_dirs:
        if root.exists():
            fixtures.extend(sorted(root.glob("**/*.npz")))
    if not fixtures:
        return
    data = np.load(fixtures[0], allow_pickle=True)
    xyz = data["xyz"]
    assert xyz.ndim == 3
    assert xyz.shape[1:] in {(543, 3), (21, 3)}
    assert np.isfinite(xyz).all()
    assert np.max(xyz) <= 1.0 + 1e-5
    assert np.min(xyz) >= -1.0 - 1e-5


def test_manifest_has_36_letters() -> None:
    manifest_path = Path("datasets/processed/core_data/data/manifest.json")
    if not manifest_path.exists():
        return
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(payload) == 36
    got = [entry["letter"] for entry in payload]
    assert got == ALBANIAN_LETTERS
