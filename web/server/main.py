"""FastAPI entry: REST + WebSocket + train log streaming."""

from __future__ import annotations

import asyncio
import base64
import json
import sys
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "Script") not in sys.path:
    sys.path.insert(0, str(_REPO / "Script"))

import albsl_app_v2 as app  # noqa: E402
from albsl_live_engine import ClientHand, LiveWebConfig, LiveWebSession  # noqa: E402

from .diagnose import run_diagnose  # noqa: E402

PYTHON = sys.executable
SCRIPT = _REPO / "Script" / "albsl_app_v2.py"


def _exists(p: str | Path) -> bool:
    try:
        return Path(p).expanduser().resolve().exists() or Path(p).exists()
    except Exception:
        return False


class DiagnoseRequest(BaseModel):
    keypoints_dir: str = "datasets/processed/core_data/data/keypoints"
    alfabeti_h5: str = "datasets/processed/core_data/data/alfabeti_keypoints.h5"
    legacy_h5: Optional[str] = "keypoints.h5"


class TrainRequest(BaseModel):
    keypoints_dir: str = "datasets/processed/core_data/data/keypoints"
    alfabeti_h5: str = "datasets/processed/core_data/data/alfabeti_keypoints.h5"
    legacy_h5: Optional[str] = "keypoints.h5"
    out: str = "outputs/albsl_mlp.pt"
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    device: str = "cuda"
    sequence_len: int = 8
    sequence_stride: int = 2
    min_valid_frames: int = 4
    idle_ratio: float = 0.35
    hidden_dim: int = 192
    layers: int = 2
    dropout: float = 0.25
    workers: int = 0
    no_augment: bool = False


class RecordRequest(BaseModel):
    recordings_h5: str = "keypoints.h5"
    label: str
    source: str = "webui"
    features: List[List[float]] = Field(..., description="List of 123-d feature vectors")


app = FastAPI(title="AlbSL WebUI API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> Dict[str, Any]:
    root = _REPO
    checks = {
        "repo": str(root),
        "weights": _exists(root / "outputs" / "albsl_mlp.pt"),
        "fused_weights": _exists(root / "outputs" / "fused_phase3.pt"),
        "albsl_model": _exists(root / "models" / "trained" / "albsl_model_final" / "model_full.pt"),
        "landmarks_json": _exists(root / "datasets" / "processed" / "assets" / "albsl_landmarks.json"),
        "dynamic_templates_json": _exists(root / "datasets" / "processed" / "assets" / "albsl_dynamic_templates.json"),
        "words_dict_json": _exists(root / "datasets" / "processed" / "assets" / "albsl_words_dictionary.json"),
        "unified_coords_json": _exists(root / "datasets" / "json_dataset" / "coordinates.json"),
    }
    return {"ok": True, "paths": checks}


@app.post("/api/diagnose")
def diagnose(req: DiagnoseRequest) -> Dict[str, Any]:
    legacy = Path(req.legacy_h5) if req.legacy_h5 else None
    return run_diagnose(Path(req.keypoints_dir), Path(req.alfabeti_h5), legacy_h5=legacy)


@app.post("/api/train/stream")
async def train_stream(req: TrainRequest) -> StreamingResponse:
    cmd: List[str] = [
        PYTHON,
        str(SCRIPT),
        "train",
        "--keypoints-dir",
        req.keypoints_dir,
        "--alfabeti-h5",
        req.alfabeti_h5,
        "--out",
        req.out,
        "--epochs",
        str(req.epochs),
        "--batch-size",
        str(req.batch_size),
        "--lr",
        str(req.lr),
        "--device",
        req.device,
        "--sequence-len",
        str(req.sequence_len),
        "--sequence-stride",
        str(req.sequence_stride),
        "--min-valid-frames",
        str(req.min_valid_frames),
        "--idle-ratio",
        str(req.idle_ratio),
        "--hidden-dim",
        str(req.hidden_dim),
        "--layers",
        str(req.layers),
        "--dropout",
        str(req.dropout),
        "--workers",
        str(req.workers),
    ]
    if req.legacy_h5:
        cmd.extend(["--legacy-h5", req.legacy_h5])
    if req.no_augment:
        cmd.append("--no-augment")

    async def gen() -> AsyncIterator[bytes]:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(_REPO),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            yield f"data: {line.decode(errors='replace').rstrip()}\n\n".encode()
        code = await proc.wait()
        yield f"data: __exit_code__:{code}\n\n".encode()

    return StreamingResponse(gen(), media_type="text/event-stream")


class RecordLandmarksRequest(BaseModel):
    recordings_h5: str = "keypoints.h5"
    label: str
    source: str = "webui"
    frames: List[Dict[str, Any]] = Field(..., description="Each item: xyz (63 floats), is_left bool")


@app.post("/api/record")
def record_append(req: RecordRequest) -> Dict[str, Any]:
    arr = np.array(req.features, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != app.KEY_DIM:
        return {"ok": False, "error": f"features must be [N, {app.KEY_DIM}], got {arr.shape}"}
    h5_path = (Path(req.recordings_h5) if Path(req.recordings_h5).is_absolute() else _REPO / req.recordings_h5).resolve()
    app._append_recording_to_h5(h5_path, arr, label=req.label, source=req.source)
    return {"ok": True, "path": str(h5_path), "rows": int(arr.shape[0])}


@app.post("/api/record/landmarks")
def record_from_landmarks(req: RecordLandmarksRequest) -> Dict[str, Any]:
    feats: List[np.ndarray] = []
    for fr in req.frames:
        xyz = np.array(fr.get("xyz"), dtype=np.float32).reshape(21, 3)
        is_left = bool(fr.get("is_left", False))
        feats.append(app.build_feature(xyz, is_left=is_left))
    arr = np.stack(feats, axis=0).astype(np.float32)
    h5_path = Path(req.recordings_h5)
    h5_path = h5_path if h5_path.is_absolute() else _REPO / h5_path
    h5_path = h5_path.resolve()
    app._append_recording_to_h5(h5_path, arr, label=req.label, source=req.source)
    return {"ok": True, "path": str(h5_path), "rows": int(arr.shape[0])}


def _decode_fusion_jpeg(b64: str) -> Optional[np.ndarray]:
    if not b64:
        return None
    try:
        raw = base64.b64decode(b64)
        buf = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return None
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return rgb
    except Exception:
        return None


@app.websocket("/api/ws/live")
async def ws_live(ws: WebSocket) -> None:
    await ws.accept()
    session: Optional[LiveWebSession] = None
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            mtype = msg.get("type")
            if mtype == "init":
                cfg_dict = msg.get("config") or {}

                def _path(key: str, default: str) -> Path:
                    return Path(str(cfg_dict.get(key, default)))

                uni = cfg_dict.get("unified_coords_json")
                cfg = LiveWebConfig(
                    weights=_path("weights", "outputs/albsl_mlp.pt"),
                    fused_weights=_path("fused_weights", "outputs/fused_phase3.pt"),
                    albsl_model=_path("albsl_model", "models/trained/albsl_model_final/model_full.pt"),
                    landmarks_json=_path("landmarks_json", "datasets/processed/assets/albsl_landmarks.json"),
                    dynamic_templates_json=_path(
                        "dynamic_templates_json", "datasets/processed/assets/albsl_dynamic_templates.json"
                    ),
                    words_dict_json=_path("words_dict_json", "datasets/processed/assets/albsl_words_dictionary.json"),
                    unified_coords_json=Path(str(uni)) if uni else Path("datasets/json_dataset/coordinates.json"),
                    pred_min_conf=float(cfg_dict.get("pred_min_conf", 0.60)),
                    pred_margin=float(cfg_dict.get("pred_margin", 0.10)),
                    device=str(cfg_dict.get("device", "auto")),
                    sequence_len=int(cfg_dict.get("sequence_len", app.DEFAULT_SEQUENCE_LEN)),
                    no_sign_threshold=float(cfg_dict.get("no_sign_threshold", 0.58)),
                    auto_append=bool(cfg_dict.get("auto_append", False)),
                    auto_hold_frames=int(cfg_dict.get("auto_hold_frames", 7)),
                    auto_min_conf=float(cfg_dict.get("auto_min_conf", 0.78)),
                    auto_min_conf_dynamic=float(cfg_dict.get("auto_min_conf_dynamic", 0.72)),
                    auto_repeat_cooldown_ms=int(cfg_dict.get("auto_repeat_cooldown_ms", 900)),
                )
                session = LiveWebSession(cfg)
                status = session.load()
                await ws.send_json({"type": "ready", "status": status})
            elif mtype == "frame" and session is not None:
                ts_ms = int(msg.get("ts_ms", 0))
                fh = int(msg.get("frame_h", 480))
                fw = int(msg.get("frame_w", 640))
                hands_in = msg.get("hands") or []
                hands: List[ClientHand] = []
                for h in hands_in:
                    xyz = np.array(h.get("xyz"), dtype=np.float32).reshape(21, 3)
                    hands.append(
                        ClientHand(
                            xyz=xyz,
                            is_left=bool(h.get("is_left", False)),
                            score=float(h.get("score", 0)),
                            side=str(h.get("side", "right")).lower(),
                        )
                    )
                fusion_rgb = _decode_fusion_jpeg(msg.get("fusion_jpeg") or "")
                res = session.step(
                    hands,
                    (fh, fw),
                    ts_ms,
                    frame_bgr=None,
                    fusion_crop_rgb=fusion_rgb,
                )
                await ws.send_json(
                    {
                        "type": "result",
                        "top3": [[a, b] for a, b in res.top3],
                        "idle_detected": res.idle_detected,
                        "idle_prob": res.idle_prob,
                        "detected": res.detected,
                        "actionable": res.actionable,
                        "shown_letter": res.shown_letter,
                        "auto_append_letter": res.auto_append_letter,
                        "primary_loaded": res.primary_loaded,
                        "fusion_used": res.fusion_used,
                    }
                )
            else:
                await ws.send_json({"type": "error", "message": "send init first or unknown type"})
    except WebSocketDisconnect:
        return
    except Exception as exc:
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass


def create_app() -> FastAPI:
    return app


_spa_dir = _REPO / "web" / "dist"
if _spa_dir.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(_spa_dir), html=True), name="spa")
