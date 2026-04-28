from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ALBANIAN_LETTERS: List[str] = [
    "A", "B", "C", "Ç", "D", "Dh", "E", "Ë", "F", "G", "Gj", "H", "I", "J", "K",
    "L", "Ll", "M", "N", "Nj", "O", "P", "Q", "R", "Rr", "S", "Sh", "T", "Th",
    "U", "V", "X", "Xh", "Y", "Z", "Zh",
]

LM = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]
VIDEO_63 = [f"{n}_{a}" for n in LM for a in ("x", "y", "z")]

DEFAULT_WORDS = [
    "arra", "shtepi", "zog", "drite", "yll", "nene", "baba", "mire", "faleminderit",
    "shqip", "zhurme", "qytet", "bukë", "çaj", "ëmbël", "rruge", "shok", "zemer",
]


def canonicalize(xyz: np.ndarray) -> np.ndarray:
    if not np.any(xyz):
        return np.zeros((21, 3), np.float32)
    out = xyz.astype(np.float32).copy()
    out -= out[0]
    diag = float(np.linalg.norm(out.max(axis=0) - out.min(axis=0)))
    if diag > 1e-8:
        out /= diag
    p9 = out[9] / (np.linalg.norm(out[9]) + 1e-8)
    t = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v = np.cross(p9, t)
    c = float(np.dot(p9, t))
    s = float(np.linalg.norm(v))
    if s > 1e-8:
        vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float32)
        r = np.eye(3, dtype=np.float32) + vx + (vx @ vx) * ((1.0 - c) / (s * s))
        out = (r @ out.T).T
    signed = float(np.cross(out[5] - out[0], out[17] - out[0])[2])
    if signed < 0.0:
        out[:, 0] *= -1.0
    return np.clip(out, -1.0, 1.0)


def resample_seq(seq: np.ndarray, n: int = 20) -> np.ndarray:
    if len(seq) == 0:
        return np.zeros((n, 63), np.float32)
    if len(seq) == n:
        return seq.astype(np.float32)
    src = np.linspace(0.0, 1.0, len(seq))
    dst = np.linspace(0.0, 1.0, n)
    out = np.zeros((n, seq.shape[1]), np.float32)
    for j in range(seq.shape[1]):
        out[:, j] = np.interp(dst, src, seq[:, j])
    return out


def parse_letter_from_name(name: str) -> str | None:
    m = re.match(r"^\d+_(.+)\.csv$", name, re.IGNORECASE)
    if not m:
        return None
    letter = m.group(1)
    for L in ALBANIAN_LETTERS:
        if L.lower() == letter.lower():
            return L
    return None


def main() -> None:
    root = Path("data/csv/videos")
    if not root.exists():
        raise SystemExit(f"Missing folder: {root}")

    per_letter_frames: Dict[str, List[np.ndarray]] = {k: [] for k in ALBANIAN_LETTERS}
    per_letter_seq: Dict[str, List[np.ndarray]] = {k: [] for k in ALBANIAN_LETTERS}

    for csv_path in sorted(root.glob("*.csv")):
        letter = parse_letter_from_name(csv_path.name)
        if letter is None:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if not all(c in df.columns for c in VIDEO_63):
            continue
        arr = df[VIDEO_63].to_numpy(np.float32).reshape(-1, 21, 3)
        seq = np.stack([canonicalize(x).reshape(-1) for x in arr], axis=0)
        if len(seq) == 0:
            continue
        per_letter_frames[letter].append(np.median(seq, axis=0).reshape(21, 3))
        per_letter_seq[letter].append(resample_seq(seq, n=20))

    # 1) per-letter static coordinates
    landmarks: Dict[str, List[List[float]]] = {}
    for letter in ALBANIAN_LETTERS:
        if per_letter_frames[letter]:
            med = np.median(np.stack(per_letter_frames[letter], axis=0), axis=0).astype(np.float32)
            landmarks[letter] = med.tolist()
    Path("albsl_landmarks.json").write_text(
        json.dumps(landmarks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 2) dynamic templates (for motion-sensitive letters)
    dyn_letters = ["Sh", "Zh", "Ç", "Ë", "Rr", "Dh", "Gj", "Ll", "Nj", "Th", "Xh"]
    dynamic: Dict[str, Dict[str, object]] = {}
    for letter in dyn_letters:
        seqs = per_letter_seq.get(letter, [])
        if not seqs:
            continue
        stack = np.stack(seqs, axis=0)  # [N,20,63]
        mean = np.mean(stack, axis=0).astype(np.float32)
        std = np.std(stack, axis=0).astype(np.float32)
        dynamic[letter] = {
            "sequence_len": 20,
            "feature_dim": 63,
            "template": mean.tolist(),
            "std": std.tolist(),
            "num_sequences": int(stack.shape[0]),
        }
    Path("albsl_dynamic_templates.json").write_text(
        json.dumps(dynamic, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 3) word dictionary with coordinates per letter token
    words_payload: Dict[str, object] = {
        "token_alphabet": ALBANIAN_LETTERS,
        "words": [],
    }
    for w in DEFAULT_WORDS:
        tokens: List[str] = []
        i = 0
        # greedy digraph tokenize against Albanian alphabet
        while i < len(w):
            two = w[i : i + 2]
            if i + 1 < len(w) and any(L.lower() == two.lower() for L in ALBANIAN_LETTERS):
                tok = next(L for L in ALBANIAN_LETTERS if L.lower() == two.lower())
                tokens.append(tok)
                i += 2
            else:
                one = w[i]
                mapped = next((L for L in ALBANIAN_LETTERS if L.lower() == one.lower()), None)
                if mapped is not None:
                    tokens.append(mapped)
                i += 1
        coords = [landmarks[t] for t in tokens if t in landmarks]
        words_payload["words"].append(
            {
                "word": w,
                "letters": tokens,
                "letter_count": len(tokens),
                "coordinates": coords,
            }
        )

    Path("albsl_words_dictionary.json").write_text(
        json.dumps(words_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Wrote: albsl_landmarks.json, albsl_dynamic_templates.json, albsl_words_dictionary.json")


if __name__ == "__main__":
    main()

