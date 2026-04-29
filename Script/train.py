from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from albsl_fusion.data import build_loader
from albsl_fusion.model import FusionBatch, build_model
from albsl_fusion.utils.hardware import build_runtime, peak_memory_bytes


def _to_batch(raw: dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> FusionBatch:
    image      = raw["image"].to(device=device, dtype=dtype)
    keypoints  = raw["keypoints"].to(device=device, dtype=dtype)
    bbox       = raw["bbox"].to(device=device, dtype=dtype)
    labels     = raw["letter_index"].to(device=device)
    return FusionBatch(image=image, keypoints=keypoints, bbox=bbox, letter_index=labels)


def run_training(cfg: DictConfig) -> None:
    runtime = build_runtime(str(cfg.hardware.device))
    print("Runtime:", runtime)
    model = build_model(cfg).to(runtime.device)

    # BF16 everywhere on XPU by default.
    if runtime.using_xpu:
        model = model.to(dtype=torch.bfloat16)

    # ── Data loaders — real dataset ───────────────────────────────────────
    batch_size  = int(cfg.data.batch_size)
    num_workers = int(cfg.data.num_workers)
    seed        = int(cfg.seed)

    train_loader = build_loader(
        batch_size=batch_size,
        split="train",
        num_workers=num_workers,
        seed=seed,
    )
    val_loader = build_loader(
        batch_size=batch_size,
        split="val",
        num_workers=num_workers,
        seed=seed,
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.optimizer.lr),
        weight_decay=float(cfg.training.optimizer.weight_decay),
    )

    # Cosine LR scheduler (one cycle over all epochs)
    epochs = int(cfg.training.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * len(train_loader), eta_min=1e-6
    )

    weights = {
        "ce":    float(cfg.training.loss_weights.ce),
        "focal": float(cfg.training.loss_weights.focal),
        "align": float(cfg.training.loss_weights.align),
    }

    best_val_loss = float("inf")
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    for epoch in range(epochs):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        for step, raw in enumerate(train_loader):
            batch = _to_batch(raw, runtime.device, runtime.dtype)
            loss, stats = model.training_step(batch, loss_weights=weights)
            if torch.isnan(loss).any():
                raise RuntimeError("NaN loss encountered")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(getattr(cfg.training, "grad_clip", 1.0)),
            )
            opt.step()
            scheduler.step()
            if step % 20 == 0:
                peak = peak_memory_bytes()
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"epoch={epoch} step={step} "
                    f"loss={stats['loss']:.4f} ce={stats['ce']:.4f} "
                    f"align={stats['align']:.4f} lr={lr_now:.2e} "
                    f"peak_mem={peak}"
                )

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for raw in val_loader:
                batch = _to_batch(raw, runtime.device, runtime.dtype)
                _, stats = model.training_step(batch, loss_weights=weights)
                val_losses.append(stats["loss"])
        val_loss = sum(val_losses) / max(len(val_losses), 1)
        print(f"[val] epoch={epoch} val_loss={val_loss:.4f}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = out_dir / f"fused_phase{cfg.training.phase}_best.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"  ✓ Best checkpoint saved: {ckpt}  (val_loss={val_loss:.4f})")

    # Always save the final checkpoint
    ckpt = out_dir / f"fused_phase{cfg.training.phase}.pt"
    torch.save(model.state_dict(), ckpt)
    print("Saved final checkpoint:", ckpt)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    run_training(cfg)


if __name__ == "__main__":
    main()
