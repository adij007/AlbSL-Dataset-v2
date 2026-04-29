from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def detect_device(preferred: str = "xpu") -> torch.device:
    if preferred == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def bf16_if_supported(device: torch.device) -> torch.dtype:
    if device.type == "xpu":
        return torch.bfloat16
    return torch.float32


def peak_memory_bytes() -> int:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        stats = torch.xpu.memory_stats()
        return int(stats.get("allocated_bytes.all.peak", 0))
    return 0


@dataclass
class RuntimeInfo:
    device: torch.device
    dtype: torch.dtype
    using_xpu: bool


def build_runtime(preferred: str = "xpu") -> RuntimeInfo:
    device = detect_device(preferred)
    dtype = bf16_if_supported(device)
    return RuntimeInfo(device=device, dtype=dtype, using_xpu=device.type == "xpu")
