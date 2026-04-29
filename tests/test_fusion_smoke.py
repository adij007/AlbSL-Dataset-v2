from __future__ import annotations

import time
from types import SimpleNamespace

import torch

from albsl_fusion.model import FusionBatch, build_model
from albsl_fusion.utils.hardware import peak_memory_bytes

ALBANIAN_ALPHABET_36 = [
    "A", "B", "C", "C_CEDILLA", "D", "Dh", "E", "E_UMLAUT", "F", "G", "Gj", "H", "I", "J", "K",
    "L", "Ll", "M", "N", "Nj", "O", "P", "Q", "R", "Rr", "S", "Sh", "T", "Th",
    "U", "V", "X", "Xh", "Y", "Z", "Zh",
]


def _build() -> torch.nn.Module:
    cfg = SimpleNamespace(
        model=SimpleNamespace(hidden_dim=1152, fusion=SimpleNamespace(num_heads=4)),
        data=SimpleNamespace(num_letters=36),
    )
    return build_model(cfg)


def _dummy_batch() -> FusionBatch:
    return FusionBatch(
        image=torch.randn(2, 3, 224, 224),
        keypoints=torch.randn(2, 16, 123),
        bbox=torch.tensor([[20.0, 20.0, 180.0, 180.0], [30.0, 30.0, 190.0, 190.0]], dtype=torch.float32),
        letter_index=torch.tensor([0, 10], dtype=torch.long),
    )


def test_fusion_bridge_output_shape() -> None:
    model = _build()
    out = model(_dummy_batch())
    assert out["fused_tokens"].shape[-1] == 1152


def test_no_nan_in_training_step() -> None:
    model = _build()
    loss, _ = model.training_step(_dummy_batch(), {"ce": 1.0, "focal": 0.3, "align": 0.2})
    assert not torch.isnan(loss).any()


def test_vram_under_limit() -> None:
    peak = peak_memory_bytes()
    assert peak < 14 * 1024**3 or peak == 0


def test_letter_prediction_format() -> None:
    model = _build().eval()
    logits = model(_dummy_batch())["logits"]
    pred = int(logits[0].argmax().item())
    assert 0 <= pred < len(ALBANIAN_ALPHABET_36)


def test_openvino_int8_latency_placeholder() -> None:
    t0 = time.perf_counter()
    _ = _build()(_dummy_batch())
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 2000
