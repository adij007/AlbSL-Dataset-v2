from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


@dataclass
class FusionBatch:
    image: torch.Tensor          # [B,3,H,W]
    keypoints: torch.Tensor      # [B,T,123]
    bbox: torch.Tensor           # [B,4] xyxy in image coords
    letter_index: torch.Tensor   # [B]


class DummyYoloBackbone(nn.Module):
    """Lightweight stand-in for YOLO features (CUDA-free, XPU-safe)."""

    def __init__(self, out_channels: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RoiPatchBridge(nn.Module):
    def __init__(self, in_channels: int = 256, hidden_dim: int = 1152, roi_size: int = 7) -> None:
        super().__init__()
        self.roi_size = roi_size
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * roi_size * roi_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.sign_cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(self, feats: torch.Tensor, boxes_xyxy: torch.Tensor) -> torch.Tensor:
        # boxes format for roi_align: [K,5] with batch index.
        B = feats.shape[0]
        rois = []
        for i in range(B):
            x1, y1, x2, y2 = boxes_xyxy[i]
            rois.append(torch.tensor([i, x1, y1, x2, y2], device=feats.device, dtype=feats.dtype))
        rois_t = torch.stack(rois, dim=0)
        pooled = roi_align(feats, rois_t, output_size=self.roi_size, spatial_scale=0.25, aligned=True)
        roi_tokens = self.proj(pooled).unsqueeze(1)  # [B,1,D]
        sign_cls = self.sign_cls.expand(B, -1, -1)
        return torch.cat([sign_cls, roi_tokens], dim=1)


class KeypointEmbedder(nn.Module):
    def __init__(self, in_dim: int = 123, hidden_dim: int = 1152) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class CrossModalFusion(nn.Module):
    def __init__(self, hidden_dim: int = 1152, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, visual_tokens: torch.Tensor, keypoint_tokens: torch.Tensor) -> torch.Tensor:
        fused, _ = self.attn(query=visual_tokens, key=keypoint_tokens, value=keypoint_tokens)
        return self.ln(visual_tokens + fused)


class LightweightDecoder(nn.Module):
    def __init__(self, hidden_dim: int = 1152, num_classes: int = 36) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use SIGN_CLS token
        cls = x[:, 0, :]
        return self.head(cls)


class AlbslFusedModel(nn.Module):
    def __init__(self, hidden_dim: int = 1152, num_heads: int = 4, num_classes: int = 36) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.yolo = DummyYoloBackbone(out_channels=256)
        self.bridge = RoiPatchBridge(in_channels=256, hidden_dim=hidden_dim)
        self.key_embed = KeypointEmbedder(in_dim=123, hidden_dim=hidden_dim)
        self.fusion = CrossModalFusion(hidden_dim=hidden_dim, num_heads=num_heads)
        self.decoder = LightweightDecoder(hidden_dim=hidden_dim, num_classes=num_classes)

    def forward(self, batch: FusionBatch) -> dict[str, torch.Tensor]:
        feats = self.yolo(batch.image)
        visual_tokens = self.bridge(feats, batch.bbox)
        key_tokens = self.key_embed(batch.keypoints)
        fused = self.fusion(visual_tokens, key_tokens)
        logits = self.decoder(fused)
        return {"logits": logits, "fused_tokens": fused, "key_tokens": key_tokens}

    def training_step(self, batch: FusionBatch, loss_weights: dict[str, float]) -> tuple[torch.Tensor, dict[str, float]]:
        out = self.forward(batch)
        ce = F.cross_entropy(out["logits"], batch.letter_index)
        # Auxiliary placeholders to preserve required multi-loss structure.
        focal = torch.zeros_like(ce)
        align = 1.0 - F.cosine_similarity(out["fused_tokens"][:, 1, :], out["key_tokens"].mean(dim=1), dim=-1).mean()
        total = loss_weights.get("ce", 1.0) * ce + loss_weights.get("focal", 0.0) * focal + loss_weights.get("align", 0.0) * align
        stats = {
            "loss": float(total.detach().cpu()),
            "ce": float(ce.detach().cpu()),
            "focal": float(focal.detach().cpu()),
            "align": float(align.detach().cpu()),
        }
        return total, stats


def build_model(cfg: Any) -> AlbslFusedModel:
    return AlbslFusedModel(
        hidden_dim=int(cfg.model.hidden_dim),
        num_heads=int(cfg.model.fusion.num_heads),
        num_classes=int(cfg.data.num_letters),
    )
