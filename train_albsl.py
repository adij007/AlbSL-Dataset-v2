"""
Tasks 2 & 3: small BERT on 21x3 joint tokens, LoRA on attention/FFN, optional 4-bit on GPU,
iterative re-training with upsampling and class-weighted loss for low-accuracy letters.

GPU backend priority (auto-detected at startup)
────────────────────────────────────────────────
  1. NVIDIA CUDA      – full AMP + 4-bit bitsandbytes
  2. Intel Arc / GPU  – torch-directml  (pip install torch-directml)
                        or Intel IPEX XPU  (pip install intel-extension-for-pytorch)
                        AMP and 4-bit are disabled on these backends.
  3. CPU              – fallback

Other GPU optimisations
───────────────────────
• AMP mixed-precision  (autocast + GradScaler)    →  CUDA only, ~2-3× speedup
• Batch-level augmentation on GPU (vectorised)    →  all GPU backends
• torch.compile()  (PyTorch ≥ 2.0, CUDA only)    →  graph fusion
• cudnn.benchmark = True  (CUDA only)             →  auto-tunes kernels
• persistent_workers + prefetch_factor            →  keeps DataLoader pipeline warm
• non_blocking=True transfers                     →  overlaps H→D copy with compute
• Gradient clipping  (max_norm=1.0)               →  stabilises mixed-precision

Input : albsl_dataset_v2/{train,val}.parquet  +  label_map.json
Logs  : training_log.jsonl, convergence_log.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── knobs ──────────────────────────────────────────────────────────────────────
MAX_ROUNDS: int           = 6
MAX_EPOCHS_PER_ROUND: int = 4
THRESHOLD: float          = 0.80
CHECKPOINT_DIR: str       = "checkpoints/"

BATCH_SIZE: int           = 128
GRAD_ACCUM: int           = 1
LEARNING_RATE: float      = 3e-4
WEIGHT_DECAY: float       = 0.01
WARMUP_FRAC: float        = 0.1
GRAD_CLIP_NORM: float     = 1.0          # NEW – gradient clipping

AUG_GAUSS_STD: float      = 0.005
AUG_ROT_DEG: float        = 15.0
AUG_SCALE_MIN: float      = 0.95
AUG_SCALE_MAX: float      = 1.05
HARD_CLASS_LOSS_MULT: float = 2.0

RANDOM_SEED: int          = 42
DATASET_DIR               = Path("albsl_dataset_v2")
TRAINING_LOG              = Path("training_log.jsonl")
CONVERGENCE_LOG           = Path("convergence_log.jsonl")
EXPORT_DIR                = Path("albsl_model_final")

BERT_HIDDEN: int  = 128
BERT_LAYERS: int  = 1
BERT_HEADS: int   = 4
LORA_R: int       = 16
LORA_ALPHA: int   = 32
LORA_DROPOUT: float = 0.05

# Number of DataLoader worker processes.
# 0 = main process only (safe on Windows / MPS); auto-detect otherwise.
NUM_WORKERS: int = min(8, max(0, (os.cpu_count() or 4) - 1))


# ── reproducibility ────────────────────────────────────────────────────────────
def set_seed() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


# ── multi-backend GPU detection ────────────────────────────────────────────────
# Priority: CUDA (NVIDIA) → XPU (Intel IPEX) → DirectML (Intel/AMD Windows) → CPU
#
# Install instructions per backend
#   CUDA    : ships with standard PyTorch
#   XPU     : pip install intel-extension-for-pytorch
#   DirectML: pip install torch-directml

def _try_directml():
    """Return a torch_directml device or None."""
    try:
        import torch_directml  # type: ignore[import]
        dev = torch_directml.device(torch_directml.default_device())
        # Smoke-test: allocate a tiny tensor to confirm the device works
        torch.zeros(1).to(dev)
        return dev
    except Exception:
        return None


def _try_xpu():
    """Return torch.device('xpu') if Intel IPEX is available, else None."""
    # Avoid noisy IPEX import errors when installed wheel doesn't match torch minor.
    ver = (getattr(torch, "__version__", "") or "").split("+", 1)[0]
    parts = ver.split(".")
    try:
        major_minor = (int(parts[0]), int(parts[1]))
    except Exception:
        major_minor = None
    if major_minor is not None and major_minor != (2, 8):
        return None
    try:
        import intel_extension_for_pytorch as ipex  # type: ignore[import]  # noqa: F401
        if hasattr(torch, "xpu") and torch.xpu.is_available():  # type: ignore[attr-defined]
            return torch.device("xpu")
    except Exception:
        pass
    return None


def get_device() -> Tuple[torch.device, str]:
    """
    Returns (device, backend_name) for the best available hardware.
    backend_name is one of: 'cuda', 'xpu', 'directml', 'cpu'
    """
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    xpu = _try_xpu()
    if xpu is not None:
        return xpu, "xpu"
    dml = _try_directml()
    if dml is not None:
        return dml, "directml"  # type: ignore[return-value]
    return torch.device("cpu"), "cpu"


def configure_backend(backend: str) -> None:
    """Apply backend-specific performance flags."""
    if backend == "cuda":
        torch.backends.cudnn.benchmark     = True
        torch.backends.cudnn.deterministic = False


# ── dataset ───────────────────────────────────────────────────────────────────
class ParquetRows(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        # pre-stack into a contiguous float32 array so workers copy nothing
        self.X = np.stack([np.asarray(x, dtype=np.float32) for x in df["landmarks_63"]])
        self.y = np.asarray(df["label_id"], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.X[i].copy()), torch.tensor(self.y[i], dtype=torch.long)


def make_loader(
    df: pd.DataFrame,
    shuffle: bool,
    batch_size: int = BATCH_SIZE,
    backend: str = "cpu",
) -> DataLoader:
    """
    Build a DataLoader tuned for the active backend.

    pin_memory is CUDA-only — DirectML and XPU don't support it and will
    error or silently fall back to pageable memory.

    On Windows, multiprocessing with spawn (the default) can deadlock inside
    DataLoader workers when using non-CUDA backends; num_workers=0 is safest
    for DirectML.  CUDA and XPU work fine with multiple workers.
    """
    pin  = backend == "cuda"
    # DirectML on Windows: worker processes can't share the DML context
    nw   = 0 if backend == "directml" else NUM_WORKERS
    kwargs: Dict[str, Any] = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=pin,
    )
    if nw > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"]    = 2
    return DataLoader(ParquetRows(df), **kwargs)


# ── GPU-side batch augmentation (vectorised) ───────────────────────────────────
def augment_batch(x: torch.Tensor, hard_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Apply random augmentation to a whole batch on-device in one pass.

    Original code looped in Python over every sample (slow).  Here every op
    is a batched tensor kernel so the GPU stays busy.

    x          : (B, 63)  float32
    hard_mask  : (B,)     bool    – only augment hard-class samples (optional)
    returns    : (B, 63)
    """
    B = x.size(0)
    device = x.device
    v = x.view(B, 21, 3).clone()

    # ── Gaussian noise ──────────────────────────────────────────────────────
    v = v + torch.randn_like(v) * AUG_GAUSS_STD

    # ── Random Z-axis rotation (one angle per sample) ───────────────────────
    angles = (torch.rand(B, device=device) * 2 - 1) * math.radians(AUG_ROT_DEG)  # (B,)
    c = angles.cos()   # (B,)
    s = angles.sin()   # (B,)
    zeros = torch.zeros(B, device=device)
    ones  = torch.ones(B, device=device)
    # Build (B, 3, 3) rotation matrices
    R = torch.stack([
        torch.stack([ c, -s, zeros], dim=1),   # row 0
        torch.stack([ s,  c, zeros], dim=1),   # row 1
        torch.stack([zeros, zeros, ones], dim=1),  # row 2
    ], dim=1)   # (B, 3, 3)
    v = torch.bmm(v, R)   # (B, 21, 3)  — bmm = batched matmul

    # ── Random scale (one factor per sample) ────────────────────────────────
    sc = (torch.rand(B, device=device) * (AUG_SCALE_MAX - AUG_SCALE_MIN)
          + AUG_SCALE_MIN).view(B, 1, 1)
    v = v * sc

    out = v.reshape(B, 63)

    if hard_mask is not None:
        # Only replace samples whose hard_mask==True
        return torch.where(hard_mask.view(B, 1), out, x)
    return out


# ── LR schedule ───────────────────────────────────────────────────────────────
def _cosine_schedule(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup: int
) -> Any:
    from torch.optim.lr_scheduler import LambdaLR

    def fn(step: int) -> float:
        if step < warmup:
            return float(step) / max(1, warmup)
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    return LambdaLR(optimizer, fn)


# ── class weights for hard letters ────────────────────────────────────────────
def build_class_weights(
    train_df: pd.DataFrame, hard: Set[str], lmap: Dict[str, int]
) -> torch.Tensor:
    n_cls = int(train_df["label_id"].max() + 1)
    w = np.ones(n_cls, dtype=np.float32)
    for _, row in train_df.iterrows():
        if str(row["label"]) in hard:
            w[int(row["label_id"])] = HARD_CLASS_LOSS_MULT
    return torch.from_numpy(w)


# ── per-class validation accuracy ─────────────────────────────────────────────
def per_class_by_label(
    model: nn.Module,
    loader: DataLoader,
    lmap: Dict[str, Any],
    device: torch.device,
) -> Dict[str, float]:
    from collections import defaultdict

    id2: Dict[int, str] = {v: k for k, v in lmap["label_to_id"].items()}
    tot: Dict[str, int] = defaultdict(int)
    ok:  Dict[str, int] = defaultdict(int)

    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = (xb.to(device, non_blocking=True),
                      yb.to(device, non_blocking=True))
            pr = model(xb).argmax(-1)
            for j in range(yb.size(0)):
                y  = int(yb[j].item())
                le = id2.get(y, str(y))
                tot[le] += 1
                if int(pr[j].item()) == y:
                    ok[le] += 1
    return {L: ok.get(L, 0) / max(1, tot.get(L, 0)) for L in tot}


# ── model ──────────────────────────────────────────────────────────────────────
class SignLandmarkModel(nn.Module):
    def __init__(self, num_labels: int, use_4bit: bool) -> None:
        super().__init__()
        from transformers import BertConfig, BertModel

        bcfg = BertConfig(
            vocab_size=2,
            hidden_size=BERT_HIDDEN,
            num_hidden_layers=BERT_LAYERS,
            num_attention_heads=BERT_HEADS,
            intermediate_size=BERT_HIDDEN * 4,
            max_position_embeddings=32,
            type_vocab_size=1,
            hidden_dropout_prob=0.1,
            use_cache=False,
        )
        q = None
        if use_4bit and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig
                q = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            except Exception:
                q = None

        if q is not None:
            self.bert = BertModel(bcfg, quantization_config=q)  # type: ignore[call-arg]
        else:
            self.bert = BertModel(bcfg)

        self.joint_embed = nn.Linear(3, BERT_HIDDEN)
        self.pos_embed   = nn.Parameter(torch.zeros(1, 22, BERT_HIDDEN))
        self.cls         = nn.Parameter(torch.zeros(1, 1, BERT_HIDDEN))
        self.dropout     = nn.Dropout(0.1)
        self.classifier  = nn.Linear(BERT_HIDDEN, num_labels)
        self.use_4bit    = bool(q)
        self._lora       = False

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        x  = landmarks.view(-1, 21, 3)
        bz = x.size(0)
        h  = self.joint_embed(x)
        cls = self.cls.expand(bz, -1, -1)
        h   = torch.cat([cls, h], dim=1) + self.pos_embed
        dtype = self.bert.embeddings.word_embeddings.weight.dtype
        h = h.to(dtype)
        o = self.bert(inputs_embeds=h).last_hidden_state[:, 0]
        return self.classifier(self.dropout(o))

    def add_lora(self) -> "SignLandmarkModel":
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if self.use_4bit and torch.cuda.is_available():
            try:
                self.bert = prepare_model_for_kbit_training(self.bert)  # type: ignore[assignment]
            except Exception:
                pass
        lcfg = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=["query", "key", "value", "intermediate.dense", "output.dense"],
        )
        self.bert  = get_peft_model(self.bert, lcfg)  # type: ignore[assignment]
        self._lora = True
        return self


# ── export ─────────────────────────────────────────────────────────────────────
def export_final(
    n_class: int,
    use_4bit: bool,
    best_state: Path,
    lmap: Dict[str, Any],
) -> None:
    m = SignLandmarkModel(n_class, use_4bit)
    m.add_lora()
    m.load_state_dict(
        torch.load(str(best_state), map_location="cpu", weights_only=True), strict=False
    )
    m.eval()
    out = EXPORT_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "label_map.json").write_text(
        json.dumps(lmap, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    try:
        m.bert = m.bert.merge_and_unload()  # type: ignore[assignment, attr-defined]
    except Exception:
        pass
    pack = {"state_dict": m.state_dict(), "lmap": lmap}
    torch.save(pack, out / "model_full.pt")

    m.to("cpu").float()
    eg = torch.randn(1, 63)
    try:
        torch.onnx.export(
            m, eg, str(out / "albsl_landmark.onnx"),
            input_names=["landmarks_63"],
            output_names=["logits"],
            opset_version=17,
        )
    except Exception as exc:
        (out / "onnx_export_error.txt").write_text(str(exc), encoding="utf-8")


# ── main training loop ─────────────────────────────────────────────────────────
def train_loop(
    train: pd.DataFrame,
    val: pd.DataFrame,
    lmap: Dict[str, Any],
    device: torch.device,
    use_4b: bool,
    backend: str = "cpu",
) -> Path:
    n_class = int(train["label_id"].max() + 1)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    last_cp = Path(CHECKPOINT_DIR) / "state.pt"
    for p in (TRAINING_LOG, CONVERGENCE_LOG):
        if p.exists():
            p.unlink()

    # ── AMP: only supported on CUDA ──────────────────────────────────────────
    # DirectML and XPU do not implement the float16 accumulation kernels that
    # GradScaler relies on.  On those backends we train in full fp32.
    use_amp = backend == "cuda"
    scaler  = torch.amp.GradScaler(device.type if backend == "cuda" else "cpu",
                                   enabled=use_amp)
    def autocast_ctx():
        if use_amp:
            return torch.amp.autocast(device_type="cuda", enabled=True)
        import contextlib
        return contextlib.nullcontext()

    for rd in range(MAX_ROUNDS):
        # ── CUDA timing event ────────────────────────────────────────────────
        t_start = _cuda_event_record(device)
        print(f"\n=== round {rd + 1} / {MAX_ROUNDS} ===", file=sys.stderr)

        # ── identify hard classes from previous checkpoint ───────────────────
        hard: Set[str] = set()
        if rd and last_cp.exists():
            t_prev = SignLandmarkModel(n_class, use_4b)
            t_prev.add_lora()
            t_prev.load_state_dict(
                torch.load(str(last_cp), map_location=device, weights_only=True), strict=False
            )
            t_prev.to(device)
            vl_prev = make_loader(val, shuffle=False, backend=backend)
            pc0 = per_class_by_label(t_prev, vl_prev, lmap, device)
            del t_prev
            for L, a in pc0.items():
                if a < THRESHOLD:
                    hard.add(L)
            if not hard:
                print("All classes above threshold — stopping early.", file=sys.stderr)
                break

        # ── upsample hard letters ────────────────────────────────────────────
        tr_df = train.copy()
        for h in hard:
            sub = train[train["label"] == h]
            if len(sub):
                tr_df = pd.concat([tr_df, sub, sub], ignore_index=True)

        wmat: Optional[torch.Tensor] = None
        if hard:
            wmat = build_class_weights(train, hard, lmap["label_to_id"]).to(device)

        # ── build / reload model ─────────────────────────────────────────────
        m = SignLandmarkModel(n_class, use_4b).to(device)
        if last_cp.exists() and rd:
            m.load_state_dict(
                torch.load(str(last_cp), map_location=device, weights_only=True), strict=False
            )
        m = m.add_lora()
        # !! PEFT's get_peft_model wraps the model but does NOT preserve the
        #    original device — adapter weights land on CPU.  Explicit .to()
        #    after add_lora() is the fix.
        m = m.to(device)

        # ── guard: crash early if the model is still on CPU ─────────────────
        _first = next(m.parameters())
        if _first.device.type != device.type:
            raise RuntimeError(
                f"Model parameter device mismatch after add_lora(): "
                f"got {_first.device}, expected {device}.  "
                f"Check that CUDA is accessible and bitsandbytes is installed."
            )

        # ── optional torch.compile (PyTorch ≥ 2.0, CUDA only) ───────────────
        if use_amp and hasattr(torch, "compile"):
            try:
                m = torch.compile(m)                    # type: ignore[assignment]
                print("  torch.compile() active", file=sys.stderr)
            except Exception as exc:
                print(f"  torch.compile() skipped: {exc}", file=sys.stderr)

        # ── optimiser + scheduler ────────────────────────────────────────────
        trainable = [p for p in m.parameters() if p.requires_grad]
        opt   = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        tr_ld = make_loader(tr_df, shuffle=True,  backend=backend)
        v_ld  = make_loader(val,   shuffle=False, backend=backend)
        n_it  = max(1, len(tr_ld) * MAX_EPOCHS_PER_ROUND * GRAD_ACCUM)
        sched = _cosine_schedule(opt, n_it, int(n_it * WARMUP_FRAC))

        # ── epoch loop ───────────────────────────────────────────────────────
        step = 0
        gacc = 0
        for ep in range(MAX_EPOCHS_PER_ROUND):
            m.train()
            tloss = 0.0
            ni    = 0
            opt.zero_grad()

            for xb, yb in tr_ld:
                # non_blocking=True overlaps the H→D DMA transfer with CPU work
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                # ── GPU batch augmentation (25 % of the time, hard classes only)
                if hard and random.random() < 0.25:
                    hard_ids  = {lmap["label_to_id"][h] for h in hard
                                 if h in lmap["label_to_id"]}
                    hard_mask = torch.tensor(
                        [int(y.item()) in hard_ids for y in yb],
                        dtype=torch.bool, device=device
                    )
                    xb = augment_batch(xb, hard_mask)

                # ── forward + loss under AMP ─────────────────────────────────
                with autocast_ctx():
                    logits = m(xb)
                    if wmat is not None:
                        lvec = F.cross_entropy(logits, yb, reduction="none")
                        loss = (lvec * wmat[yb]).mean()
                    else:
                        loss = F.cross_entropy(logits, yb)

                # ── scaled backward ──────────────────────────────────────────
                scaler.scale(loss / GRAD_ACCUM).backward()
                tloss += float(loss.detach())
                ni    += 1
                gacc  += 1

                if gacc % GRAD_ACCUM == 0:
                    # unscale before clipping so the clip threshold is in fp32
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(trainable, GRAD_CLIP_NORM)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()
                    sched.step()
                    step += 1

            # flush leftover gradients
            if gacc % GRAD_ACCUM:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(trainable, GRAD_CLIP_NORM)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            # ── per-epoch validation ─────────────────────────────────────────
            pc_ep = per_class_by_label(m, v_ld, lmap, device)
            below = [k for k, v in pc_ep.items() if v < THRESHOLD]
            print(
                f"  ep {ep+1:02d}  loss {tloss/max(ni,1):.4f}  "
                f"below={below}",
                file=sys.stderr,
            )
            _append_jsonl(
                TRAINING_LOG,
                {
                    "round": rd, "epoch": ep,
                    "loss": tloss / max(ni, 1),
                    "per_class_val": pc_ep,
                },
            )

        # ── round-end checkpoint ─────────────────────────────────────────────
        pc_f   = per_class_by_label(m, v_ld, lmap, device)
        rpath  = Path(CHECKPOINT_DIR) / f"round_{rd}.pt"
        # unwrap possible torch.compile wrapper before saving
        state  = (m._orig_mod if hasattr(m, "_orig_mod") else m).state_dict()
        torch.save(state, rpath)
        torch.save(state, last_cp)

        # GPU wall-time for the round
        elapsed_ms = _cuda_event_elapsed(t_start, device)

        _append_jsonl(
            CONVERGENCE_LOG,
            {
                "round": rd,
                "pass_letters": [k for k, v in pc_f.items() if v >= THRESHOLD],
                "fail_letters": [k for k, v in pc_f.items() if v < THRESHOLD],
                "checkpoint":   str(rpath),
                "per_class_val_acc": pc_f,
                "round_time_ms": elapsed_ms,
            },
        )
        print(
            f"  ✓ round {rd+1} done  "
            f"({elapsed_ms/1000:.1f}s)  "
            f"fail={[k for k, v in pc_f.items() if v < THRESHOLD]}",
            file=sys.stderr,
        )

        if not any(a < THRESHOLD for a in pc_f.values()):
            break

    return last_cp


# ── helpers ────────────────────────────────────────────────────────────────────
def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _cuda_event_record(device: torch.device) -> Any:
    if device.type == "cuda":
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        return ev
    import time
    return time.perf_counter()


def _cuda_event_elapsed(start: Any, device: torch.device) -> float:
    """Return elapsed milliseconds since *start* (works on all backends)."""
    if device.type == "cuda":
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)   # type: ignore[attr-defined]
    import time
    return (time.perf_counter() - start) * 1000.0


# ── entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    global NUM_WORKERS
    ap = argparse.ArgumentParser(
        description="Train ALBSL sign-language BERT  (CUDA / Intel Arc / CPU)"
    )
    ap.add_argument("--data-dir", type=Path, default=DATASET_DIR)
    # 4-bit quant only works with bitsandbytes on CUDA; silently ignored elsewhere
    ap.add_argument("--4bit",    dest="use_4b", action="store_true")
    ap.set_defaults(use_4b=True)
    ap.add_argument("--no-4bit", dest="use_4b", action="store_false")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument(
        "--workers", type=int, default=NUM_WORKERS,
        help="DataLoader worker processes (0 = main-process only, default: auto)"
    )
    ap.add_argument(
        "--backend", type=str, default=None,
        choices=["cuda", "xpu", "directml", "cpu"],
        help="Force a specific backend instead of auto-detecting"
    )
    a = ap.parse_args()
    NUM_WORKERS = a.workers

    set_seed()

    # ── device selection ─────────────────────────────────────────────────────
    if a.backend:
        # Manual override
        if a.backend == "directml":
            dml_dev = _try_directml()
            if dml_dev is None:
                print("DirectML requested but torch-directml not found.\n"
                      "  Run:  pip install torch-directml", file=sys.stderr)
                sys.exit(1)
            device, backend = dml_dev, "directml"
        elif a.backend == "xpu":
            xpu_dev = _try_xpu()
            if xpu_dev is None:
                print("XPU requested but intel-extension-for-pytorch not found.\n"
                      "  Run:  pip install intel-extension-for-pytorch", file=sys.stderr)
                sys.exit(1)
            device, backend = xpu_dev, "xpu"
        else:
            device, backend = torch.device(a.backend), a.backend
    else:
        device, backend = get_device()

    configure_backend(backend)

    # 4-bit quantisation requires bitsandbytes which is CUDA-only
    use_4b = bool(a.use_4b) and (backend == "cuda")
    if bool(a.use_4b) and backend != "cuda":
        print(f"4-bit quant not supported on '{backend}' backend — using fp32.", file=sys.stderr)

    # ── startup banner ───────────────────────────────────────────────────────
    print(f"Backend: {backend.upper()}", file=sys.stderr)
    print(f"Device : {device}", file=sys.stderr)
    if backend == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU    : {props.name}  "
              f"{props.total_memory // 1024**2} MiB  "
              f"cc={props.major}.{props.minor}", file=sys.stderr)
        print("AMP    : enabled (fp16 GradScaler + autocast)", file=sys.stderr)
    elif backend == "directml":
        print("AMP    : disabled (not supported by DirectML)", file=sys.stderr)
        print("Tip    : install torch-directml if you haven't already:\n"
              "         pip install torch-directml", file=sys.stderr)
    elif backend == "xpu":
        print("AMP    : disabled (use IPEX's native optimiser instead)", file=sys.stderr)
    else:
        print("AMP    : disabled (CPU run — consider --backend directml)", file=sys.stderr)

    if not a.data_dir.is_dir():
        print("Missing dataset — run: python consolidate_data.py", file=sys.stderr)
        sys.exit(1)

    train = pd.read_parquet(a.data_dir / "train.parquet")
    val   = pd.read_parquet(a.data_dir / "val.parquet")
    with open(a.data_dir / "label_map.json", "r", encoding="utf-8") as f:
        lmap = json.load(f)

    cp = train_loop(train, val, lmap, device, use_4b, backend=backend)

    if not a.no_export and cp.exists():
        export_final(int(train["label_id"].max() + 1), use_4b, cp, lmap)

    print(f"\nDone.  Checkpoint → {cp}", file=sys.stderr)
    if not a.no_export:
        print(f"       Export     → {EXPORT_DIR}", file=sys.stderr)


if __name__ == "__main__":
    main()