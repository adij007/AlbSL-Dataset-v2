"""
Reusable live inference session for AlbSL v2 (browser WebSocket + optional fusion crop).

Imports model and feature helpers from ``albsl_app_v2``; keeps temporal state in this class.
CLI ``cmd_live`` behavior is unchanged — this module is for programmatic / WebUI use.
"""

from __future__ import annotations

import sys
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "Script") not in sys.path:
    sys.path.insert(0, str(_REPO / "Script"))

import albsl_app_v2 as app  # noqa: E402
from albsl_fusion.model import FusionBatch, build_model  # noqa: E402


@dataclass
class LiveWebConfig:
    """Mirrors ``live`` argparse defaults from ``albsl_app_v2.parse_args``."""

    weights: Path = Path("outputs/albsl_mlp.pt")
    fused_weights: Path = Path("outputs/fused_phase3.pt")
    albsl_model: Path = Path("models/trained/albsl_model_final/model_full.pt")
    models_dir: Path = Path("models/mediapipe/mp_models")
    landmarks_json: Path = Path("datasets/processed/assets/albsl_landmarks.json")
    dynamic_templates_json: Path = Path("datasets/processed/assets/albsl_dynamic_templates.json")
    words_dict_json: Path = Path("datasets/processed/assets/albsl_words_dictionary.json")
    unified_coords_json: Optional[Path] = Path("datasets/json_dataset/coordinates.json")
    template_max_dist: float = 0.16
    dynamic_max_dist: float = 0.12
    pred_min_conf: float = 0.60
    pred_margin: float = 0.10
    auto_append: bool = False
    auto_hold_frames: int = 7
    auto_min_conf: float = 0.78
    auto_min_conf_dynamic: float = 0.72
    auto_repeat_cooldown_ms: int = 900
    dynamic_focus_letters: str = "Ë,Ç,Sh,Zh,Xh"
    dynamic_focus_boost: float = 0.04
    dynamic_focus_motion_boost: float = 0.40
    dynamic_focus_motion_threshold: float = 0.010
    min_hand_score: float = 0.65
    min_hand_area: float = 0.006
    max_hand_area: float = 0.55
    landmark_smooth_alpha: float = 0.70
    detect_on_frames: int = 2
    detect_off_frames: int = 4
    device: str = "auto"
    sequence_len: int = app.DEFAULT_SEQUENCE_LEN
    no_sign_threshold: float = 0.58


@dataclass
class LiveStepResult:
    top3: List[Tuple[str, float]]
    idle_detected: bool
    idle_prob: float
    detected: bool
    actionable: bool
    shown_letter: str  # letter or "UNCERTAIN" or "-"
    auto_append_letter: Optional[str]
    primary_loaded: bool
    fusion_used: bool


@dataclass
class ClientHand:
    xyz: np.ndarray  # (21, 3) float32 image-normalized
    is_left: bool
    score: float
    side: str  # "left" | "right"


class LiveWebSession:
    """Stateful inference aligned with ``cmd_live`` prediction path (no OpenCV GUI)."""

    def __init__(self, cfg: LiveWebConfig) -> None:
        self.cfg = cfg
        self.device = app._select_device(cfg.device)
        self.primary_model: Optional[torch.nn.Module] = None
        self.primary_meta: Dict[str, Any] = {}
        self.lm_model: Optional[torch.nn.Module] = None
        self.lm_payload: Optional[Dict[str, Any]] = None
        self.lm_idx_to_letter: Dict[int, str] = {}
        self.fusion_model: Optional[torch.nn.Module] = None
        self.landmark_refs: Dict[str, np.ndarray] = {}
        self.dynamic_templates: Dict[str, Dict[str, Any]] = {}
        self.words_dict: List[Dict[str, Any]] = []

        self.loaded = False
        self.temporal_loaded = False
        self.sequence_len = int(cfg.sequence_len)
        self.primary_classes: List[str] = list(app.ALBANIAN_LETTERS)
        self.primary_class_to_idx: Dict[str, int] = {}

        self.ema_probs: Optional[np.ndarray] = None
        self.ema_alpha = 0.7
        self.side_sticky: Optional[str] = None
        self.prev_center: Optional[np.ndarray] = None
        self.recent_top1: Deque[Tuple[str, float]] = deque(maxlen=7)
        self.feat_history: Deque[np.ndarray] = deque(maxlen=24)
        self.xyz_history: Deque[np.ndarray] = deque(maxlen=max(24, self.sequence_len * 3))
        self.auto_state: Dict[str, Any] = {
            "candidate": None,
            "count": 0,
            "hold_frames": max(2, int(cfg.auto_hold_frames)),
            "last_emit_letter": "",
            "last_emit_ms": -10**9,
        }
        self.smoothed_by_side: Dict[str, np.ndarray] = {}
        self.detect_state: Dict[str, int] = {"on_count": 0, "off_count": 0, "active": 0}
        self.dynamic_letters: set[str] = set()
        self.focus_letters: set[str] = set()
        self.fusion_loaded = False

    def load(self) -> Dict[str, Any]:
        cfg = self.cfg
        w = app._resolve_model_path(cfg.weights)
        fw = app._resolve_model_path(cfg.fused_weights)
        am = app._resolve_model_path(cfg.albsl_model)

        self.primary_meta = {
            "model_type": "frame_mlp",
            "classes": app.ALBANIAN_LETTERS,
            "sequence_len": max(5, min(10, int(cfg.sequence_len))),
        }
        loaded = False
        try:
            pm, meta = app._load_primary_model_checkpoint(w, self.device, int(cfg.sequence_len))
            if pm is not None:
                self.primary_model = pm
                self.primary_meta = meta
                loaded = True
        except Exception:
            self.primary_model = None
        self.loaded = loaded and self.primary_model is not None

        self.primary_classes = [str(x) for x in self.primary_meta.get("classes", app.ALBANIAN_LETTERS)]
        self.primary_class_to_idx = {label: i for i, label in enumerate(self.primary_classes)}
        self.temporal_loaded = self.loaded and self.primary_meta.get("model_type") == "temporal_lstm"
        self.sequence_len = int(self.primary_meta.get("sequence_len", cfg.sequence_len))

        self.lm_model, self.lm_payload = app._load_landmark_model_checkpoint(am, self.device)
        lm_loaded = self.lm_model is not None and self.lm_payload is not None
        self.lm_idx_to_letter = {}
        if lm_loaded and isinstance(self.lm_payload, dict):
            self.lm_idx_to_letter = {
                int(v): str(k) for k, v in self.lm_payload.get("label_to_id", {}).items()
            }

        self.fusion_model = None
        self.fusion_loaded = False
        if fw.exists():
            try:
                cfg_f = type(
                    "Cfg",
                    (),
                    {
                        "model": type(
                            "M",
                            (),
                            {"hidden_dim": 1152, "fusion": type("F", (), {"num_heads": 4})()},
                        )(),
                        "data": type("D", (), {"num_letters": len(app.ALBANIAN_LETTERS)})(),
                    },
                )()
                self.fusion_model = build_model(cfg_f).to(self.device)
                state_f = torch.load(str(fw), map_location=self.device, weights_only=True)
                self.fusion_model.load_state_dict(state_f, strict=False)
                self.fusion_model.eval()
                self.fusion_loaded = True
            except Exception:
                self.fusion_model = None
                self.fusion_loaded = False

        lj = app._resolve_json_path(cfg.landmarks_json)
        dj = app._resolve_json_path(cfg.dynamic_templates_json)
        wj = app._resolve_json_path(cfg.words_dict_json)
        self.landmark_refs = app._load_landmark_refs(lj)
        self.dynamic_templates = app._load_dynamic_templates(dj)
        self.words_dict = app._load_words_dictionary(wj)

        unified_payload: Optional[dict] = None
        up = cfg.unified_coords_json
        if up:
            unified_path = app._resolve_json_path(up)
            if unified_path.exists():
                try:
                    import json

                    unified_payload = json.loads(unified_path.read_text(encoding="utf-8"))
                    words_section = unified_payload.get("words", {})
                    if isinstance(words_section, dict):
                        words_list = words_section.get("words", [])
                        if isinstance(words_list, list) and words_list:
                            self.words_dict = [w for w in words_list if isinstance(w, dict)]
                except Exception:
                    pass
        if not self.landmark_refs and isinstance(unified_payload, dict):
            merged = app._landmark_refs_from_json_root(unified_payload)
            if merged:
                self.landmark_refs = merged
        if not self.dynamic_templates and isinstance(unified_payload, dict):
            merged_d = app._dynamic_templates_from_json_root(unified_payload)
            if merged_d:
                self.dynamic_templates = merged_d

        self.dynamic_letters = app._dynamic_letter_set(self.dynamic_templates)
        requested = app._parse_letter_set(cfg.dynamic_focus_letters)
        self.focus_letters = set(self.dynamic_letters) | requested | {"Ë", "Ç", "Sh", "Zh", "Xh"}

        return {
            "primary_model": self.loaded,
            "iterative_model": lm_loaded,
            "fused_model": self.fusion_loaded,
            "landmark_refs": len(self.landmark_refs) > 0,
            "dynamic_templates": len(self.dynamic_templates) > 0,
            "words_dict": len(self.words_dict) > 0,
        }

    def _pick_hand(
        self, hands: List[ClientHand], frame_hw: Tuple[int, int]
    ) -> Tuple[bool, np.ndarray, np.ndarray, bool, float]:
        """Return detected, xyz, xyz_norm, is_left, score — mirrors cmd_live selection."""
        H, W = frame_hw
        candidates: List[Tuple[float, ClientHand]] = []
        frame_list = [
            app.HandLandmarkFrame(
                xyz=h.xyz,
                normalized_xyz=h.xyz,
                is_left=h.is_left,
                score=h.score,
                side=h.side,
            )
            for h in hands
        ]
        filtered_frames = app.filter_hand_landmark_frames(
            frame_list,
            min_score=float(self.cfg.min_hand_score),
            min_area=float(self.cfg.min_hand_area),
            max_area=float(self.cfg.max_hand_area),
        )
        filtered: List[ClientHand] = []
        for ff in filtered_frames:
            for h in hands:
                if h.side == ff.side and np.allclose(h.xyz, ff.xyz, rtol=0.0, atol=1e-4):
                    filtered.append(h)
                    break

        for hand_frame in filtered:
            center = hand_frame.xyz[:, :2].mean(axis=0)
            center_penalty = 0.0
            if self.prev_center is not None:
                center_penalty = float(np.linalg.norm(center - self.prev_center))
            sticky_bonus = 0.15 if (self.side_sticky is not None and self.side_sticky == hand_frame.side) else 0.0
            side_bias = 0.05 if hand_frame.side == "right" else 0.0
            score = float(hand_frame.score) + sticky_bonus + side_bias - 0.35 * center_penalty
            candidates.append((score, hand_frame))

        if not candidates:
            self.detect_state["off_count"] = int(self.detect_state.get("off_count", 0)) + 1
            self.detect_state["on_count"] = 0
            if self.detect_state["off_count"] >= int(self.cfg.detect_off_frames):
                self.detect_state["active"] = 0
            detected = bool(self.detect_state.get("active", 0))
            self.side_sticky = None
            self.prev_center = None
            self.smoothed_by_side.clear()
            return detected, np.zeros((21, 3), np.float32), np.zeros((21, 3), np.float32), False, 0.0

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_frame = candidates[0][1]
        prev_side_xyz = self.smoothed_by_side.get(best_frame.side)
        xyz = app._smooth_hand_landmarks(
            best_frame.xyz,
            prev_side_xyz,
            alpha=float(self.cfg.landmark_smooth_alpha),
        )
        self.smoothed_by_side[best_frame.side] = xyz.copy()
        xyz_norm = app.normalize_hand_landmarks(xyz, is_left=best_frame.is_left)
        is_left = best_frame.is_left
        hand_score = best_frame.score

        self.detect_state["on_count"] = int(self.detect_state.get("on_count", 0)) + 1
        self.detect_state["off_count"] = 0
        det_on = max(1, int(self.cfg.detect_on_frames))
        if int(self.detect_state.get("active", 0)) == 1 or self.detect_state["on_count"] >= det_on:
            self.detect_state["active"] = 1
            detected = True
        else:
            detected = False
        self.side_sticky = best_frame.side
        self.prev_center = xyz[:, :2].mean(axis=0)
        return detected, xyz.astype(np.float32), xyz_norm.astype(np.float32), is_left, float(hand_score)

    def step(
        self,
        hands: List[ClientHand],
        frame_hw: Tuple[int, int],
        ts_ms: int,
        frame_bgr: Optional[np.ndarray] = None,
        fusion_crop_rgb: Optional[np.ndarray] = None,
    ) -> LiveStepResult:
        """One inference step. ``frame_bgr`` is optional BGR full frame for fusion crop."""
        cfg = self.cfg
        args = cfg
        top3: List[Tuple[str, float]] = []
        idle_detected = False
        idle_prob = 0.0
        fusion_used = False
        auto_letter: Optional[str] = None
        actionable = False
        shown = "-"

        lm_loaded = self.lm_model is not None
        detected, xyz, xyz_norm, is_left, _hs = self._pick_hand(hands, frame_hw)

        if detected and ((self.loaded and self.primary_model is not None) or lm_loaded):
            feat = app.build_feature(xyz, is_left=is_left)
            self.xyz_history.append(xyz_norm.copy())
            self.feat_history.append(feat.copy())
            with torch.no_grad():
                probs_primary: Optional[np.ndarray] = None
                probs: Optional[np.ndarray] = None
                if self.loaded and self.primary_model is not None:
                    if self.temporal_loaded:
                        seq_np = app._sequence_from_history(self.xyz_history, self.sequence_len)
                        x = torch.from_numpy(seq_np).unsqueeze(0).to(device=self.device, dtype=torch.float32)
                    else:
                        x = torch.from_numpy(feat).unsqueeze(0).to(device=self.device, dtype=torch.float32)
                    logits = self.primary_model(x)[0]
                    probs_primary = F.softmax(logits, dim=-1).float().cpu().numpy().astype(np.float32)
                    probs = probs_primary.copy()

                probs_lm: Optional[np.ndarray] = None
                if lm_loaded and self.lm_model is not None:
                    feat63 = feat[:63].astype(np.float32, copy=False)
                    logits_lm = self.lm_model(
                        torch.from_numpy(feat63).view(1, -1).to(device=self.device, dtype=torch.float32)
                    )[0]
                    probs_lm = F.softmax(logits_lm, dim=-1).float().cpu().numpy()

                if self.fusion_loaded and self.fusion_model is not None:
                    crop_chw: Optional[torch.Tensor] = None
                    bbox: Tuple[int, int, int, int]
                    if fusion_crop_rgb is not None and fusion_crop_rgb.size > 0:
                        cr = fusion_crop_rgb.astype(np.float32)
                        if cr.max() > 1.5:
                            cr = cr / 255.0
                        if cr.ndim == 3 and cr.shape[-1] == 3:
                            if cr.shape[0] != app.FUSION_IMAGE_SIZE or cr.shape[1] != app.FUSION_IMAGE_SIZE:
                                cr = cv2.resize(cr, (app.FUSION_IMAGE_SIZE, app.FUSION_IMAGE_SIZE))
                            cr = np.transpose(cr, (2, 0, 1))
                        crop_chw = torch.from_numpy(cr).unsqueeze(0).to(device=self.device, dtype=torch.float32)
                        bbox = app._hand_bbox((frame_hw[0], frame_hw[1], 3), xyz)
                        fusion_used = True
                    elif frame_bgr is not None and frame_bgr.size > 0:
                        bbox = app._hand_bbox(frame_bgr.shape, xyz)
                        crop = app._preprocess_crop(frame_bgr, bbox)
                        crop_chw = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).to(
                            device=self.device, dtype=torch.float32
                        )
                        fusion_used = True
                    else:
                        crop_chw = None
                        bbox = (0, 0, 1, 1)

                    if crop_chw is not None:
                        fb = FusionBatch(
                            image=crop_chw,
                            keypoints=torch.from_numpy(feat).view(1, 1, -1).to(
                                device=self.device, dtype=torch.float32
                            ),
                            bbox=torch.tensor(
                                [[bbox[0], bbox[1], bbox[2], bbox[3]]], dtype=torch.float32, device=self.device
                            ),
                            letter_index=torch.zeros(1, dtype=torch.long, device=self.device),
                        )
                        logits_f = self.fusion_model(fb)["logits"][0]
                        probs_f = F.softmax(logits_f, dim=-1).float().cpu().numpy()
                        if self.temporal_loaded:
                            mapped_f = np.zeros(len(self.primary_classes), dtype=np.float32)
                            for idx_f, letter in enumerate(app.ALBANIAN_LETTERS):
                                dst_idx = self.primary_class_to_idx.get(letter)
                                if dst_idx is not None and idx_f < len(probs_f):
                                    mapped_f[dst_idx] += float(probs_f[idx_f])
                            if probs is None:
                                probs = mapped_f
                            else:
                                probs = 0.75 * probs + 0.25 * mapped_f
                        else:
                            if probs_primary is None:
                                mapped_f = np.zeros(len(self.primary_classes), dtype=np.float32)
                                for idx_f, letter in enumerate(app.ALBANIAN_LETTERS):
                                    dst_idx = self.primary_class_to_idx.get(letter)
                                    if dst_idx is not None and idx_f < len(probs_f):
                                        mapped_f[dst_idx] += float(probs_f[idx_f])
                                probs = mapped_f
                            else:
                                probs = 0.55 * probs_primary + 0.45 * probs_f
                        s = probs.sum()
                        if s > 1e-8:
                            probs = probs / s

                if probs_lm is not None and len(self.lm_idx_to_letter) > 0:
                    mapped = np.zeros(len(self.primary_classes), dtype=np.float32)
                    for j, p_val in enumerate(probs_lm):
                        letter = self.lm_idx_to_letter.get(int(j))
                        if letter is None:
                            continue
                        idx = self.primary_class_to_idx.get(letter)
                        if idx is None:
                            continue
                        mapped[idx] += float(p_val)
                    ms = mapped.sum()
                    if ms > 1e-8:
                        mapped /= ms
                        blend = 0.35 if self.temporal_loaded else 0.50
                        if probs is None:
                            probs = mapped
                        else:
                            probs = (1.0 - blend) * probs + blend * mapped
                        ps = probs.sum()
                        if ps > 1e-8:
                            probs /= ps
                if probs is None:
                    pass
                else:
                    if self.ema_probs is None or len(self.ema_probs) != len(probs):
                        self.ema_probs = probs
                    else:
                        self.ema_probs = self.ema_alpha * self.ema_probs + (1.0 - self.ema_alpha) * probs
                    no_sign_idx = self.primary_class_to_idx.get(app.NO_SIGN_LABEL)
                    if no_sign_idx is not None:
                        idle_prob = float(self.ema_probs[no_sign_idx])
                    sorted_idx = np.argsort(-self.ema_probs)
                    for idx_sorted in sorted_idx:
                        label = self.primary_classes[int(idx_sorted)]
                        if label == app.NO_SIGN_LABEL:
                            continue
                        top3.append((label, float(self.ema_probs[int(idx_sorted)])))
                        if len(top3) == 3:
                            break
                    idle_detected = (
                        no_sign_idx is not None
                        and idle_prob >= float(args.no_sign_threshold)
                        and idle_prob >= (top3[0][1] if top3 else 0.0)
                    )
                    motion_energy = app._motion_energy_from_xyz_history(
                        self.xyz_history, frames=max(5, int(args.sequence_len))
                    )
                    if idle_detected:
                        if motion_energy >= float(args.dynamic_focus_motion_threshold):
                            dyn_letter, dyn_dist = app._dynamic_match_letter(
                                self.feat_history, self.dynamic_templates, max_dist=args.dynamic_max_dist
                            )
                            if dyn_letter is not None and dyn_letter in self.focus_letters:
                                dyn_conf = float(max(0.58, min(0.92, 1.0 - dyn_dist)))
                                top3 = [(dyn_letter, dyn_conf)]
                                idle_detected = False
                        if idle_detected:
                            top3 = []
                            self.recent_top1.clear()
                    if not idle_detected and len(top3) >= 2:
                        margin = top3[0][1] - top3[1][1]
                    else:
                        margin = 0.0
                    if (not idle_detected) and top3 and (top3[0][1] >= 0.60 or margin >= 0.15):
                        self.recent_top1.append((top3[0][0], top3[0][1]))
                    if (not idle_detected) and len(self.recent_top1) >= 4:
                        vote = Counter([x[0] for x in self.recent_top1])
                        voted_letter, voted_count = vote.most_common(1)[0]
                        vote_ratio = voted_count / len(self.recent_top1)
                        if vote_ratio >= 0.57:
                            voted_prob = float(np.mean([p for l, p in self.recent_top1 if l == voted_letter]))
                            rest = [x for x in top3 if x[0] != voted_letter]
                            top3 = [(voted_letter, voted_prob)] + rest[:2]

                    if (not idle_detected) and top3 and top3[0][1] < 0.75:
                        live_xyz = xyz_norm
                        fb_letter, fb_dist = app._template_match_letter(
                            live_xyz, self.landmark_refs, max_dist=args.template_max_dist
                        )
                        if fb_letter is not None:
                            fb_conf = float(max(0.55, min(0.93, 1.0 - fb_dist)))
                            rest = [x for x in top3 if x[0] != fb_letter]
                            top3 = [(fb_letter, fb_conf)] + rest[:2]

                    if (not idle_detected) and ((not top3) or (top3 and top3[0][1] < 0.75)):
                        dyn_letter, dyn_dist = app._dynamic_match_letter(
                            self.feat_history, self.dynamic_templates, max_dist=args.dynamic_max_dist
                        )
                        if dyn_letter is not None:
                            dyn_conf = float(max(0.58, min(0.90, 1.0 - dyn_dist)))
                            rest = [x for x in top3 if x[0] != dyn_letter]
                            top3 = [(dyn_letter, dyn_conf)] + rest[:2]
                    if (not idle_detected) and top3:
                        top3 = app._boost_focus_letters(
                            top3,
                            focus_letters=self.focus_letters,
                            motion_energy=motion_energy,
                            base_boost=float(args.dynamic_focus_boost),
                            motion_boost=float(args.dynamic_focus_motion_boost),
                        )
                    if args.auto_append and top3:
                        cand, conf = top3[0]
                        th = (
                            args.auto_min_conf_dynamic if cand in self.dynamic_letters else args.auto_min_conf
                        )
                        auto_letter = app._auto_append_letter(
                            self.auto_state,
                            candidate=cand,
                            confidence=conf,
                            threshold=th,
                            repeat_cooldown_ms=int(args.auto_repeat_cooldown_ms),
                            now_ms=int(ts_ms),
                        )
        else:
            if self.ema_probs is not None:
                self.ema_probs *= 0.85
            self.recent_top1.clear()
            self.feat_history.clear()
            self.xyz_history.clear()
            self.auto_state["candidate"] = None
            self.auto_state["count"] = 0

        if idle_detected:
            shown = "IDLE"
        elif top3:
            actionable = app._top1_is_actionable(top3, float(cfg.pred_min_conf), float(cfg.pred_margin))
            top_letter, top_conf = top3[0]
            shown = top_letter if actionable else "UNCERTAIN"
        else:
            shown = "-"

        return LiveStepResult(
            top3=top3,
            idle_detected=idle_detected,
            idle_prob=idle_prob,
            detected=detected,
            actionable=actionable,
            shown_letter=shown,
            auto_append_letter=auto_letter,
            primary_loaded=bool(self.loaded),
            fusion_used=fusion_used,
        )
