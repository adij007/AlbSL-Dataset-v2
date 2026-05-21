"""Consolidate every coordinate-related JSON in datasets/processed/ into one file.

Merges:
  - datasets/processed/assets/albsl_landmarks.json          (static per-letter coords)
  - datasets/processed/assets/albsl_dynamic_templates.json  (dynamic letters)
  - datasets/processed/assets/albsl_words_dictionary.json   (words + letter coords)
  - datasets/processed/landmarks/albsl_landmarks/*.json     (per-letter knowledge base)
  - datasets/processed/external/mappings/asl_to_albsl_map.json
  - datasets/processed/external/sources_manifest.json

Output:
  datasets/json_dataset/coordinates.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "datasets" / "json_dataset" / "coordinates.json",
    )
    args = ap.parse_args()

    assets_dir = ROOT / "datasets" / "processed" / "assets"
    landmarks_kb_dir = ROOT / "datasets" / "processed" / "landmarks" / "albsl_landmarks"
    external_dir = ROOT / "datasets" / "processed" / "external"

    static_landmarks = _read_json(assets_dir / "albsl_landmarks.json") or {}
    dynamic_templates = _read_json(assets_dir / "albsl_dynamic_templates.json") or {}
    words_payload = _read_json(assets_dir / "albsl_words_dictionary.json") or {}

    landmark_kb: Dict[str, Any] = {}
    if landmarks_kb_dir.is_dir():
        for jf in sorted(landmarks_kb_dir.glob("*.json")):
            if jf.name.startswith("_"):
                continue
            data = _read_json(jf)
            if isinstance(data, dict) and "letter" in data:
                landmark_kb[str(data["letter"])] = data

    asl_map = _read_json(external_dir / "mappings" / "asl_to_albsl_map.json") or {}
    manifest = _read_json(external_dir / "sources_manifest.json") or []

    unified: Dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "letters": {
            "static": static_landmarks,
            "dynamic": dynamic_templates,
            "knowledge_base": landmark_kb,
        },
        "words": words_payload,
        "transfer_maps": {
            "asl_to_albsl": asl_map,
        },
        "external_sources_manifest": manifest,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(unified, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    static_n = len(static_landmarks) if isinstance(static_landmarks, dict) else 0
    dyn_n = len(dynamic_templates) if isinstance(dynamic_templates, dict) else 0
    kb_n = len(landmark_kb)
    words_n = 0
    if isinstance(words_payload, dict):
        words_n = len(words_payload.get("words", [])) if isinstance(words_payload.get("words"), list) else 0
    print(
        f"Wrote unified coordinates JSON -> {args.output_json}\n"
        f"  static_letters={static_n} dynamic_letters={dyn_n} kb_letters={kb_n} words={words_n}"
    )


if __name__ == "__main__":
    main()
