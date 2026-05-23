
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm
try:
    from deepface import DeepFace
except Exception:                                                             
    DeepFace = None                            
try:
    import mediapipe as mp
except Exception:                                                       
    mp = None                            

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.pose_detect.pose_detection_video import (              
    annotate_frame_with_pose,
    process_frame_bgr,
)
from models.score_pain import analyze_pain_in_frame              
from models.action_events.infer_temporal_events import (              
    TemporalEventsInferencer,
)
from models.yolo_instruments import (              
    annotate_frame_with_detections,
    detect_instruments_in_frame,
    get_default_detector,
    is_red_bbox_class_id,
)
from src.hf_assets import resolve_temporal_events_root
from src.bleeding_spectral_filter import SpectralConfig, evaluate_bleeding_roi              

CSV_COLUMNS: list[str] = [
    "frame_idx",
    "timestamp_sec",
    "pain_proxy",
    "pain_smoothed",
    "behavior_score",
    "flag_pose_visible",
    "flag_left_arm_bent",
    "flag_right_arm_bent",
    "flag_curved_posture",
    "flag_asymmetric_shoulders",
    "n_faces",
    "n_instruments",
    "instruments_seen",
    "instrument_top_conf",
    "visual_aggression_score",
    "visual_aggression_persisted",
    "score_sangue_identificado",
    "flag_sangue_identificado",
    "score_equimose_identificada",
    "flag_equimose_identificada",
    "emotion_angry_prob",
    "emotion_disgust_prob",
    "emotion_fear_prob",
    "emotion_happy_prob",
    "emotion_neutral_prob",
    "emotion_sad_prob",
    "emotion_face_detected",
    "pose_curvature_deg",
    "vision_critical_alert",
    "temporal_action_label",
    "temporal_action_conf",
    "temporal_bleeding_label",
    "temporal_bleeding_conf",
    "temporal_bleeding_positive",
    "temporal_smoke_label",
    "temporal_smoke_conf",
    "temporal_smoke_positive",
]

_patient_visual_thresholds: dict[str, float | int] = {}


def set_patient_visual_thresholds(**values: float | int) -> None:
    """Permite ajustar limiares visuais sem depender de kwargs novos em run_pipeline."""
    global _patient_visual_thresholds
    _patient_visual_thresholds = dict(values)


def _resolve_patient_visual_thresholds(
    *,
    visual_blood_persist_threshold: float,
    visual_blood_score_threshold: float,
    visual_bruise_score_threshold: float,
    visual_bruise_min_pixels: int,
) -> tuple[float, float, float, int]:
    if not _patient_visual_thresholds:
        return (
            visual_blood_persist_threshold,
            visual_blood_score_threshold,
            visual_bruise_score_threshold,
            visual_bruise_min_pixels,
        )
    return (
        float(
            _patient_visual_thresholds.get(
                "visual_blood_persist_threshold", visual_blood_persist_threshold
            )
        ),
        float(
            _patient_visual_thresholds.get(
                "visual_blood_score_threshold", visual_blood_score_threshold
            )
        ),
        float(
            _patient_visual_thresholds.get(
                "visual_bruise_score_threshold", visual_bruise_score_threshold
            )
        ),
        int(
            _patient_visual_thresholds.get(
                "visual_bruise_min_pixels", visual_bruise_min_pixels
            )
        ),
    )

INSTRUMENT_EVENT_COLUMNS: list[str] = [
    "frame_idx",
    "timestamp_sec",
    "timestamp",
    "class_id",
    "class_name",
    "class_group",
    "confidence",
    "spectral_confirmed",
    "rg_ratio",
    "rb_ratio",
    "red_mean",
    "pixel_count",
    "risk_alert",
    "x1",
    "y1",
    "x2",
    "y2",
]


@dataclass
class VisionRuntime:
    pose_model: Any | None
    deepface_available: bool


_VISION_RUNTIME: VisionRuntime | None = None


def _build_default_emotions_payload() -> dict[str, float]:
    return {
        "angry": 0.0,
        "disgust": 0.0,
        "fear": 0.0,
        "happy": 0.0,
        "neutral": 0.0,
        "sad": 0.0,
    }


def get_vision_runtime() -> VisionRuntime:
    global _VISION_RUNTIME
    if _VISION_RUNTIME is not None:
        return _VISION_RUNTIME

    pose_model = None
    if mp is not None:
        try:
            pose_model = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception:
            pose_model = None

    deepface_available = False
    if DeepFace is not None:
        try:
                                                                               
            DeepFace.build_model("Emotion")
            deepface_available = True
        except Exception:
            deepface_available = False

    _VISION_RUNTIME = VisionRuntime(
        pose_model=pose_model,
        deepface_available=deepface_available,
    )
    return _VISION_RUNTIME


def preprocess_frame_node(
    frame_bgr: np.ndarray,
    *,
    target_size: tuple[int, int] = (224, 224),
) -> dict[str, Any]:
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return {
            "ok": False,
            "error": "empty_frame",
            "frame_bgr": None,
            "frame_rgb": None,
            "frame_resized_bgr": None,
            "tensor": None,
        }
    resized = cv2.resize(frame_bgr, target_size)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = resized.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    return {
        "ok": True,
        "error": "",
        "frame_bgr": frame_bgr,
        "frame_rgb": rgb,
        "frame_resized_bgr": resized,
        "tensor": tensor,
    }


def emotions_deepface_node(
    preprocessed: dict[str, Any],
    runtime: VisionRuntime,
) -> dict[str, Any]:
    payload = {
        "available": False,
        "face_detected": False,
        "emotions": _build_default_emotions_payload(),
        "fear_probability": 0.0,
        "error": "",
    }
    if not preprocessed.get("ok"):
        payload["error"] = "invalid_preprocessed_frame"
        return payload
    if not runtime.deepface_available:
        payload["error"] = "deepface_unavailable"
        return payload

    frame_rgb = preprocessed.get("frame_rgb")
    if frame_rgb is None:
        payload["error"] = "missing_frame_rgb"
        return payload

    try:
        result = DeepFace.analyze(
            img_path=frame_rgb,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True,
        )
        if isinstance(result, list):
            result = result[0] if result else {}
        emotions_raw = result.get("emotion", {}) if isinstance(result, dict) else {}
        mapped = {}
        for key in ("angry", "disgust", "fear", "happy", "neutral", "sad"):
            mapped[key] = float(emotions_raw.get(key, 0.0) or 0.0) / 100.0
        payload["available"] = True
        payload["emotions"] = mapped
        payload["fear_probability"] = float(mapped.get("fear", 0.0))
                                                                                           
        region = result.get("region", {}) if isinstance(result, dict) else {}
        payload["face_detected"] = bool(region and int(region.get("w", 0) or 0) > 0)
    except Exception as exc:
        payload["error"] = f"deepface_error:{exc}"
    return payload


def _calculate_joint_angle(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float32)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float32)
    nba = float(np.linalg.norm(ba))
    nbc = float(np.linalg.norm(bc))
    if nba <= 1e-6 or nbc <= 1e-6:
        return 180.0
    cosang = float(np.dot(ba, bc) / (nba * nbc))
    cosang = max(-1.0, min(1.0, cosang))
    return float(np.degrees(np.arccos(cosang)))


def _landmark_xy(landmarks: Any, index: int) -> tuple[float, float, float]:
    lm = landmarks[index]
    return float(lm.x), float(lm.y), float(lm.visibility)


def pose_mediapipe_node(
    preprocessed: dict[str, Any],
    runtime: VisionRuntime,
    *,
    visibility_threshold: float = 0.45,
) -> dict[str, Any]:
    payload = {
        "available": False,
        "pose_visible": False,
        "left_hip_angle": 180.0,
        "right_hip_angle": 180.0,
        "curvature": 0.0,
        "error": "",
    }
    if not preprocessed.get("ok"):
        payload["error"] = "invalid_preprocessed_frame"
        return payload
    if runtime.pose_model is None:
        payload["error"] = "mediapipe_unavailable"
        return payload

    frame_rgb = preprocessed.get("frame_rgb")
    if frame_rgb is None:
        payload["error"] = "missing_frame_rgb"
        return payload

    try:
        pose_result = runtime.pose_model.process(frame_rgb)
        landmarks = getattr(pose_result, "pose_landmarks", None)
        if landmarks is None or not getattr(landmarks, "landmark", None):
            payload["error"] = "pose_not_visible"
            return payload

        lms = landmarks.landmark
        lsx, lsy, lsv = _landmark_xy(lms, 11)
        lhx, lhy, lhv = _landmark_xy(lms, 23)
        lkx, lky, lkv = _landmark_xy(lms, 25)
        rsx, rsy, rsv = _landmark_xy(lms, 12)
        rhx, rhy, rhv = _landmark_xy(lms, 24)
        rkx, rky, rkv = _landmark_xy(lms, 26)

        left_angle = 180.0
        right_angle = 180.0
        left_visible = min(lsv, lhv, lkv) >= visibility_threshold
        right_visible = min(rsv, rhv, rkv) >= visibility_threshold
        if left_visible:
            left_angle = _calculate_joint_angle((lsx, lsy), (lhx, lhy), (lkx, lky))
        if right_visible:
            right_angle = _calculate_joint_angle((rsx, rsy), (rhx, rhy), (rkx, rky))

        curvature_left = max(0.0, 180.0 - left_angle)
        curvature_right = max(0.0, 180.0 - right_angle)
        curvature = max(curvature_left, curvature_right)
        payload["available"] = True
        payload["pose_visible"] = bool(left_visible or right_visible)
        payload["left_hip_angle"] = float(left_angle)
        payload["right_hip_angle"] = float(right_angle)
        payload["curvature"] = float(curvature)
        if not payload["pose_visible"]:
            payload["error"] = "pose_low_visibility"
    except Exception as exc:
        payload["error"] = f"pose_error:{exc}"

    return payload


def decision_alert_node(
    emotions_payload: dict[str, Any],
    pose_payload: dict[str, Any],
    *,
    fear_threshold: float = 0.70,
    curvature_threshold: float = 35.0,
) -> dict[str, Any]:
    fear = float(emotions_payload.get("fear_probability", 0.0) or 0.0)
    curvature = float(pose_payload.get("curvature", 0.0) or 0.0)
    critical = bool(fear > fear_threshold and curvature > curvature_threshold)
    return {
        "fear_probability": fear,
        "curvature": curvature,
        "fear_threshold": fear_threshold,
        "curvature_threshold": curvature_threshold,
        "critical_alert": critical,
    }


def _fallback_emotions_from_pain_payload(
    pain_payload: dict[str, Any],
) -> dict[str, Any]:
    default = {
        "available": False,
        "face_detected": False,
        "emotions": _build_default_emotions_payload(),
        "fear_probability": 0.0,
        "error": "fallback_unavailable",
    }
    faces = pain_payload.get("faces", []) if isinstance(pain_payload, dict) else []
    if not isinstance(faces, list) or not faces:
        return default
    agg = _build_default_emotions_payload()
    detected = False
    for face in faces:
        emotions = face.get("emotions", {}) if isinstance(face, dict) else {}
        if not isinstance(emotions, dict):
            continue
        detected = True
        for key in agg:
            agg[key] = max(agg[key], float(emotions.get(key, 0.0) or 0.0) / 100.0)
    if not detected:
        return default
    return {
        "available": True,
        "face_detected": True,
        "emotions": agg,
        "fear_probability": float(agg.get("fear", 0.0)),
        "error": "",
    }


def _fallback_pose_payload_from_pose_result(
    pose_result: dict[str, Any],
    curvature_threshold: float,
) -> dict[str, Any]:
    flags = pose_result.get("flags", {}) if isinstance(pose_result, dict) else {}
    behavior = float(pose_result.get("behavior_score", 0.0) or 0.0) if isinstance(pose_result, dict) else 0.0
    pose_visible = bool(flags.get("pose_visible", False))
    curved = bool(flags.get("curved_posture", False))
    asymmetric = bool(flags.get("asymmetric_shoulders", False))
    proxy_curvature = 0.0
    if curved:
        proxy_curvature = max(float(curvature_threshold) + 5.0, 40.0)
    elif asymmetric:
        proxy_curvature = max(float(curvature_threshold) * 0.75, 20.0)
    elif pose_visible:
        proxy_curvature = max(0.0, min(35.0, behavior * 0.4))
    return {
        "available": True,
        "pose_visible": pose_visible,
        "left_hip_angle": max(180.0 - proxy_curvature, 90.0),
        "right_hip_angle": max(180.0 - proxy_curvature, 90.0),
        "curvature": proxy_curvature,
        "error": "fallback_from_pose_flags",
    }


def _bool_to_int(value: Any) -> int:
    return 1 if bool(value) else 0


def _pain_color(score: float) -> tuple[int, int, int]:
    if score >= 40:
        return (0, 0, 255)
    if score >= 25:
        return (0, 165, 255)
    return (0, 255, 0)


def _pain_label(score: float) -> str:
    if score >= 40:
        return "DOR SEVERA"
    if score >= 25:
        return "DESCONFORTO MODERADO"
    return "ESTAVEL"


def _clip_bbox_to_frame(
    bbox: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), frame_width - 1))
    y1 = max(0, min(int(y1), frame_height - 1))
    x2 = max(0, min(int(x2), frame_width))
    y2 = max(0, min(int(y2), frame_height))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return (x1, y1, x2, y2)


def _build_patient_roi_mask(
    frame_shape: tuple[int, int, int],
    landmarks: list[dict[str, Any]] | None,
) -> np.ndarray | None:
    if not landmarks:
        return None
    frame_h, frame_w = frame_shape[:2]
    points: list[list[int]] = []
    for lm in landmarks:
        if not isinstance(lm, dict):
            continue
        visibility = float(lm.get("visibility", 0.0) or 0.0)
        if visibility < 0.35:
            continue
        x = int(round(float(lm.get("x", 0.0) or 0.0) * frame_w))
        y = int(round(float(lm.get("y", 0.0) or 0.0) * frame_h))
        if 0 <= x < frame_w and 0 <= y < frame_h:
            points.append([x, y])
    if len(points) < 3:
        return None

    pts = np.array(points, dtype=np.int32)
    hull = cv2.convexHull(pts)
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    # Mantém ROI mais "justa" para evitar vazamento em fundo/objetos vermelhos.
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    mask = cv2.erode(mask, erode_kernel, iterations=1)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.dilate(mask, dilate_kernel, iterations=1)
    return mask


def _estimate_face_asymmetry(face_roi_bgr: np.ndarray) -> float:
    if face_roi_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    half = w // 2
    if half < 4:
        return 0.0
    left = gray[:, :half]
    right = cv2.flip(gray[:, w - half :], 1)
    if left.shape != right.shape:
        return 0.0
    diff = np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))) / 255.0
    return float(max(0.0, min(1.0, diff / 0.25)))


def _estimate_face_blood_score(
    face_roi_bgr: np.ndarray,
    *,
    face_roi_mask: np.ndarray | None = None,
    blood_score_threshold: float = 0.85,
) -> tuple[float, bool]:
    if face_roi_bgr.size == 0:
        return 0.0, False

    roi = cv2.GaussianBlur(face_roi_bgr, (5, 5), 0)
    h, w = roi.shape[:2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                                                                              
    valid = np.ones((h, w), dtype=np.uint8)
    y0 = int(h * 0.6)
    x1 = int(w * 0.3)
    x2 = int(w * 0.7)
    valid[y0:, x1:x2] = 0
    if face_roi_mask is not None and face_roi_mask.shape[:2] == valid.shape[:2]:
        valid = np.where((valid > 0) & (face_roi_mask > 0), 1, 0).astype(np.uint8)

    valid_pixels = int(np.count_nonzero(valid))
    if valid_pixels <= 0:
        return 0.0, False

    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(
        ycrcb,
        np.array([0, 138, 85]),
        np.array([255, 170, 125]),
    )
    skin_valid = (skin_mask > 0) & (valid > 0)
    skin_pixels = int(np.count_nonzero(skin_valid))
    if skin_pixels <= 0:
        return 0.0, False

    b, g, r = cv2.split(roi.astype(np.float32))
    rg_ratio = r / (g + 1.0)
    rb_ratio = r / (b + 1.0)
    red_dominance = (
        (rg_ratio >= 1.25)
        & (rb_ratio >= 1.25)
        & ((r - g) >= 28.0)
        & ((r - b) >= 22.0)
    )
    red_hsv_mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 120, 45]), np.array([10, 255, 255])),
        cv2.inRange(hsv, np.array([170, 120, 45]), np.array([180, 255, 255])),
    )
    candidate_mask = np.where(
        (red_hsv_mask > 0) & skin_valid & red_dominance,
        255,
        0,
    ).astype(np.uint8)
    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )
    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate_mask, connectivity=8
    )
    if num_labels <= 1:
        return 0.0, False
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = int(np.argmax(component_areas)) + 1
    largest_area = int(component_areas[largest_idx - 1])
    if largest_area <= 0:
        return 0.0, False
    largest_component = labels == largest_idx
    largest_ratio = float(largest_area / max(skin_pixels, 1))
    if largest_ratio > 0.22:
        return 0.0, False

    red_excess = np.maximum(0.0, r - np.maximum(g, b))
    red_excess_mean = float(
        np.mean(red_excess[largest_component]) if np.any(largest_component) else 0.0
    )

    score_area = max(0.0, min(1.0, (largest_ratio - 0.01) / 0.10))
    score_excess = max(0.0, min(1.0, (red_excess_mean - 35.0) / 80.0))
    score = min(1.0, score_area * 0.55 + score_excess * 0.45)
    min_skin_pixels = max(80, int(valid_pixels * 0.10))
    blood_detected = bool(
        skin_pixels >= min_skin_pixels
        and largest_ratio >= 0.015
        and red_excess_mean >= 45.0
        and score >= float(blood_score_threshold)
    )
    return float(score), blood_detected


def _estimate_face_bruise_score(
    face_roi_bgr: np.ndarray,
    *,
    face_roi_mask: np.ndarray | None = None,
    bruise_score_threshold: float = 0.06,
    bruise_min_pixels: int = 12,
) -> tuple[float, bool]:
    if face_roi_bgr.size == 0:
        return 0.0, False

    roi = cv2.GaussianBlur(face_roi_bgr, (3, 3), 0)
    h, w = roi.shape[:2]
    if h <= 0 or w <= 0:
        return 0.0, False

    valid = np.ones((h, w), dtype=np.uint8)
    if face_roi_mask is not None and face_roi_mask.shape[:2] == valid.shape[:2]:
        valid = np.where(face_roi_mask > 0, 1, 0).astype(np.uint8)
    valid_pixels = int(np.count_nonzero(valid))
    if valid_pixels <= 0:
        return 0.0, False

    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(
        ycrcb,
        np.array([0, 130, 80]),
        np.array([255, 190, 145]),
    )
    skin_valid = (skin_mask > 0) & (valid > 0)
    skin_pixels = int(np.count_nonzero(skin_valid))
    base_mask = skin_valid if skin_pixels >= max(12, int(valid_pixels * 0.05)) else (valid > 0)
    base_pixels = int(np.count_nonzero(base_mask))
    if base_pixels <= 0:
        return 0.0, False

    valid_uint8 = np.where(base_mask, 255, 0).astype(np.uint8)
    bruise_mask = _build_ultra_sensitive_bruise_mask(roi, valid_uint8)
    bruise_pixels = int(np.count_nonzero(bruise_mask))
    ratio = float(bruise_pixels / max(base_pixels, 1))
    score = max(0.0, min(1.0, (ratio - 0.002) / 0.20))

    # Reforço de score na faixa abaixo dos olhos (mancha grande e escura).
    y_band = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    periorbital_band = (y_band >= 0.20) & (y_band <= 0.56)
    peri_pixels = int(np.count_nonzero((bruise_mask > 0) & periorbital_band))
    if peri_pixels > 0:
        peri_ratio = float(peri_pixels / max(base_pixels, 1))
        score = max(score, min(1.0, (peri_ratio - 0.003) / 0.16))

    bruised = bool(
        bruise_pixels >= max(int(bruise_min_pixels), 1)
        and ratio >= 0.008
        and score >= float(bruise_score_threshold)
    )
    return float(score), bruised


# Espectro de hematoma: âncora #4a1f2d + vermelho/roxo + mancha periorbital escura.
_BRUISE_REFERENCES_BGR = np.array(
    [
        [45, 31, 74],  # #4a1f2d (âncora)
        [48, 29, 72],
        [40, 26, 66],
        [61, 45, 107],  # vermelho-vinho
        [53, 37, 90],
        [79, 40, 95],
        [47, 37, 77],  # roxo-vinho
        [63, 31, 69],
        [35, 32, 53],
        # Mancha abaixo do olho (marrom-arroxeado mais escuro/acinzentado)
        [52, 38, 68],
        [46, 34, 62],
        [40, 30, 56],
        [36, 28, 50],
        [58, 42, 74],
        [44, 36, 60],
    ],
    dtype=np.uint8,
)
_BRUISE_LAB_MAX_DIST_ANCHOR = 18.0
_BRUISE_LAB_MAX_DIST_VARIANT = 24.0
_BRUISE_LAB_MAX_DIST_PERIORBITAL = 24.0


def _build_bruise_spatial_prior_map(h: int, w: int) -> np.ndarray:
    """Prior espacial: pico periorbital clínico, bochecha secundária, testa penalizada."""
    y_grid = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    x_grid = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]

    # Faixa periorbital (bolsa do olho / hematoma frequente).
    peri_band = (y_grid >= 0.18) & (y_grid <= 0.60)
    peri_peak = np.exp(-((y_grid - 0.38) ** 2) / (2.0 * 0.11**2)).astype(np.float32)
    prior = np.where(peri_band, 0.75 + 0.95 * peri_peak, 0.45).astype(np.float32)

    # Bochecha lateral (hematoma tardio).
    cheek_band = (y_grid >= 0.40) & (y_grid <= 0.82)
    lateral = (x_grid <= 0.42) | (x_grid >= 0.58)
    prior = np.where(
        cheek_band & lateral,
        np.maximum(prior, 1.05),
        prior,
    )

    # Testa/cabelo: penalização forte (evita falso positivo, sem bloquear periorbital).
    prior = np.where(y_grid < 0.16, prior * 0.12, prior)
    prior = np.where(y_grid > 0.88, prior * 0.55, prior)
    return np.clip(prior, 0.08, 2.0).astype(np.float32)


def _build_cheek_prior_map(h: int, w: int) -> np.ndarray:
    return _build_bruise_spatial_prior_map(h, w)


def _classify_bruise_zone(cy_norm: float) -> str:
    if cy_norm < 0.16:
        return "forehead"
    if cy_norm <= 0.60:
        return "periorbital"
    if cy_norm <= 0.78:
        return "mid_face"
    return "chin"


@dataclass
class _BruiseComponentCandidate:
    label_id: int
    area: int
    rank: float
    zone: str
    prior_mean: float
    fill_ratio: float


def _refine_bruise_mask(
    bruise_mask: np.ndarray,
    valid: np.ndarray,
    *,
    max_cover_ratio: float = 0.14,
    max_component_ratio: float = 0.08,
) -> np.ndarray:
    """
    Refina máscara de hematoma com seleção em duas passagens:
    1) reserva quota para o melhor componente periorbital;
    2) preenche o restante sem deixar ruído pequeno consumir o orçamento global.
    """
    h, w = bruise_mask.shape[:2]
    if h <= 0 or w <= 0:
        return bruise_mask

    valid_pixels = max(int(np.count_nonzero(valid > 0)), 1)
    prior = _build_bruise_spatial_prior_map(h, w)
    binary = (bruise_mask > 0).astype(np.uint8)
    if int(np.count_nonzero(binary)) == 0:
        return bruise_mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if num_labels <= 1:
        return np.zeros((h, w), dtype=np.uint8)

    min_area = 3
    max_comp_default = max(10, int(valid_pixels * max_component_ratio))
    max_comp_periorbital = max(32, int(valid_pixels * 0.30))
    peri_budget = max(20, int(valid_pixels * 0.26))
    global_budget = max(peri_budget, int(valid_pixels * max(max_cover_ratio, 0.22)))
    other_budget = max(10, int(valid_pixels * 0.10))

    zone_boost = {
        "periorbital": 2.20,
        "mid_face": 1.05,
        "cheek": 1.15,
        "chin": 0.65,
        "forehead": 0.20,
    }

    candidates: list[_BruiseComponentCandidate] = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        left = int(stats[label_id, cv2.CC_STAT_LEFT])
        top = int(stats[label_id, cv2.CC_STAT_TOP])
        comp_w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        cy_norm = (top + comp_h * 0.5) / max(h - 1, 1)
        zone = _classify_bruise_zone(cy_norm)
        if zone == "forehead":
            continue

        comp_cap = max_comp_periorbital if zone == "periorbital" else max_comp_default
        if area > comp_cap:
            continue

        comp_mask = labels == label_id
        prior_mean = float(np.mean(prior[comp_mask]))
        bbox_area = max(comp_w * comp_h, 1)
        fill_ratio = float(area / bbox_area)

        # Periorbital: priorização suave (sombra não descarta por limiar rígido).
        if zone == "periorbital":
            if prior_mean < 0.12 and fill_ratio < 0.12:
                continue
        elif prior_mean < 0.28:
            continue

        area_factor = float(np.sqrt(min(area, 2500) / 70.0))
        rank = (
            prior_mean
            * zone_boost.get(zone, 1.0)
            * area_factor
            * (0.55 + 0.45 * min(fill_ratio * 2.5, 1.0))
        )
        candidates.append(
            _BruiseComponentCandidate(
                label_id=label_id,
                area=area,
                rank=float(rank),
                zone=zone,
                prior_mean=prior_mean,
                fill_ratio=fill_ratio,
            )
        )

    if not candidates:
        return np.zeros((h, w), dtype=np.uint8)

    def _paint(label_id: int, target: np.ndarray) -> None:
        target[labels == label_id] = 255

    refined = np.zeros((h, w), dtype=np.uint8)
    peri_candidates = sorted(
        [c for c in candidates if c.zone == "periorbital"],
        key=lambda item: item.rank,
        reverse=True,
    )
    other_candidates = sorted(
        [c for c in candidates if c.zone != "periorbital"],
        key=lambda item: item.rank,
        reverse=True,
    )

    peri_used = 0
    for candidate in peri_candidates[:3]:
        if candidate.rank < 0.10:
            continue
        if peri_used + candidate.area > peri_budget:
            continue
        _paint(candidate.label_id, refined)
        peri_used += candidate.area

    total_used = peri_used
    other_used = 0
    for candidate in other_candidates:
        if candidate.rank < 0.14:
            continue
        if total_used + candidate.area > global_budget:
            continue
        if other_used + candidate.area > other_budget and peri_used > 0:
            continue
        _paint(candidate.label_id, refined)
        total_used += candidate.area
        other_used += candidate.area

    if total_used == 0:
        fallback = max(candidates, key=lambda item: item.rank)
        _paint(fallback.label_id, refined)

    return refined


def _build_ultra_sensitive_bruise_mask(
    face_roi_bgr: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Máscara de hematoma focada no espectro #4a1f2d (vinho/roxo escuro)."""
    h, w = face_roi_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((h, w), dtype=np.uint8)

    valid_bool = valid > 0
    lab_img = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2HSV)
    b_ch, g_ch, r_ch = cv2.split(face_roi_bgr.astype(np.float32))

    n_periorbital_refs = 6
    color_match = np.zeros((h, w), dtype=bool)
    color_match_periorbital = np.zeros((h, w), dtype=bool)
    for idx, ref_bgr in enumerate(_BRUISE_REFERENCES_BGR):
        ref_lab = cv2.cvtColor(
            ref_bgr.reshape(1, 1, 3),
            cv2.COLOR_BGR2LAB,
        ).astype(np.float32)[0, 0]
        dist = np.linalg.norm(lab_img - ref_lab, axis=2)
        if idx == 0:
            dist_limit = _BRUISE_LAB_MAX_DIST_ANCHOR
        elif idx >= len(_BRUISE_REFERENCES_BGR) - n_periorbital_refs:
            dist_limit = _BRUISE_LAB_MAX_DIST_PERIORBITAL
            color_match_periorbital |= dist <= dist_limit
        else:
            dist_limit = _BRUISE_LAB_MAX_DIST_VARIANT
        color_match |= dist <= dist_limit

    # Vermelho-vinho + roxo (#4a1f2d e afins).
    hsv_red_wine = cv2.inRange(
        hsv,
        np.array([140, 48, 34]),
        np.array([157, 255, 118]),
    )
    hsv_purple = cv2.inRange(
        hsv,
        np.array([158, 48, 34]),
        np.array([179, 255, 118]),
    )
    hsv_band = cv2.bitwise_or(hsv_red_wine, hsv_purple)
    # Mancha periorbital: exige saturação mínima (evita cinza neutro).
    hsv_periorbital = cv2.inRange(
        hsv,
        np.array([132, 24, 26]),
        np.array([180, 220, 108]),
    )

    # Envelope BGR: vinho (#4a1f2d), desvio vermelho ou roxo.
    spectrum_bgr = (
        (r_ch >= 42.0)
        & (r_ch <= 112.0)
        & (g_ch >= 14.0)
        & (g_ch <= 54.0)
        & (b_ch >= 28.0)
        & (b_ch <= 95.0)
        & (r_ch > g_ch * 1.20)
        & (b_ch > g_ch * 1.00)
        & (
            ((r_ch >= b_ch * 0.78) & (r_ch <= b_ch * 1.42))
            | (r_ch >= b_ch * 1.06)
            | (b_ch >= r_ch * 1.05)
        )
    )
    max_c = np.maximum(np.maximum(r_ch, g_ch), b_ch)
    min_c = np.minimum(np.minimum(r_ch, g_ch), b_ch)
    chroma = (max_c - min_c) / (max_c + 1.0)
    l_channel = lab_img[:, :, 0]

    # Marrom-arroxeado / cinza-plum abaixo do olho (#4a1f2d escurecido).
    periorbital_bgr = (
        (r_ch >= 22.0)
        & (r_ch <= 88.0)
        & (g_ch >= 10.0)
        & (g_ch <= 52.0)
        & (b_ch >= 18.0)
        & (b_ch <= 86.0)
        & (r_ch > g_ch * 1.01)
        & (b_ch > g_ch * 0.72)
        & (r_ch >= b_ch * 0.58)
        & (r_ch <= b_ch * 1.45)
    )
    grey_plum = (
        (max_c >= 30.0)
        & (max_c <= 72.0)
        & (chroma >= 0.055)
        & (chroma <= 0.17)
        & (r_ch > g_ch * 1.06)
        & (b_ch >= g_ch * 0.88)
        & (b_ch >= r_ch * 0.72)
        & ((r_ch + b_ch) >= g_ch * 2.25)
    )
    y_norm = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    periorbital_zone = (y_norm >= 0.18) & (y_norm <= 0.58)

    purple_red_tint = (r_ch > g_ch * 1.12) & (b_ch > g_ch * 0.90) & (r_ch >= 38.0)
    periorbital_tint = (r_ch > g_ch * 1.04) & (b_ch > g_ch * 0.78) & (r_ch >= 26.0)
    is_pure_black = (max_c < 26.0) & (chroma < 0.05)
    is_neutral_gray = (
        (chroma < 0.085)
        & (max_c < 85.0)
        & (np.abs(r_ch - g_ch) < 14.0)
        & (np.abs(r_ch - b_ch) < 20.0)
        & (np.abs(g_ch - b_ch) < 14.0)
    )
    not_black = (~is_pure_black) & (~is_neutral_gray) & (
        (
            (max_c >= 40.0)
            & ((chroma >= 0.11) | (purple_red_tint & (max_c >= 34.0)))
            & ~((l_channel < 32.0) & (chroma < 0.08))
        )
        | (
            periorbital_tint
            & (max_c >= 28.0)
            & (chroma >= 0.055)
            & ~((l_channel < 22.0) & (chroma < 0.05))
        )
    )

    blood_red = (r_ch > g_ch * 1.30) & (r_ch > b_ch * 1.28) & (chroma > 0.14)

    standard_match = color_match & spectrum_bgr & (hsv_band > 0)
    periorbital_match = periorbital_zone & (
        (periorbital_bgr & (hsv_periorbital > 0) & (chroma >= 0.06))
        | (grey_plum & (color_match_periorbital > 0) & (chroma >= 0.06))
        | (color_match_periorbital & periorbital_bgr & (chroma >= 0.05))
    )
    combined_bool = (
        (standard_match | periorbital_match)
        & not_black
        & valid_bool
        & (~blood_red)
    )

    combined = np.where(combined_bool, 255, 0).astype(np.uint8)
    combined = cv2.morphologyEx(
        combined, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8), iterations=1
    )
    return _refine_bruise_mask(combined, valid)


def _pick_best_bruise_contour(
    contours: list,
    bruise_mask: np.ndarray,
    *,
    valid_pixels: int,
    cheek_prior: np.ndarray | None = None,
) -> np.ndarray | None:
    if not contours:
        return None

    h, w = bruise_mask.shape[:2]
    if cheek_prior is None:
        cheek_prior = _build_cheek_prior_map(h, w)

    face_area = float(max(valid_pixels, 1))
    min_area = max(4.0, face_area * 0.0008)
    best_contour = None
    best_score = -1.0

    for contour in contours:
        area = float(cv2.contourArea(contour))
        rx, ry, rw, rh = cv2.boundingRect(contour)
        if rw < 2 or rh < 2:
            continue
        cy_norm = (ry + rh * 0.5) / max(h - 1, 1)
        if cy_norm < 0.18:
            continue
        is_periorbital = 0.20 <= cy_norm <= 0.58
        max_area = max(face_area * (0.22 if is_periorbital else 0.14), 20.0)
        if area < min_area or area > max_area:
            continue

        patch = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(patch, [contour], -1, 255, thickness=-1)
        overlap = int(np.count_nonzero(cv2.bitwise_and(patch, bruise_mask)))
        if overlap <= 0:
            continue

        density = float(overlap / max(area, 1.0))
        perimeter = float(cv2.arcLength(contour, True))
        compactness = (
            float((4.0 * np.pi * area) / max(perimeter * perimeter, 1.0))
            if perimeter > 0
            else 0.0
        )
        prior_mean = float(np.mean(cheek_prior[patch > 0])) if overlap > 0 else 0.0
        if prior_mean < (0.22 if is_periorbital else 0.38):
            continue
        size_penalty = 1.0 - min(1.0, area / max_area)
        score = (
            density
            * (0.50 + 0.50 * compactness)
            * (0.25 + 0.75 * size_penalty)
            * (0.40 + 0.60 * prior_mean)
            * (1.45 if is_periorbital else 1.0)
        )
        if score > best_score:
            best_score = score
            best_contour = contour

    return best_contour


def _locate_face_blood_bbox(
    frame_bgr: np.ndarray,
    face_bbox: tuple[int, int, int, int],
    *,
    face_roi_mask: np.ndarray | None = None,
) -> tuple[int, int, int, int] | None:
    frame_h, frame_w = frame_bgr.shape[:2]
    clipped = _clip_bbox_to_frame(face_bbox, frame_w, frame_h)
    if clipped is None:
        return None

    x1, y1, x2, y2 = clipped
    face_roi = frame_bgr[y1:y2, x1:x2]
    if face_roi.size == 0:
        return None

    roi = cv2.GaussianBlur(face_roi, (5, 5), 0)
    h, w = roi.shape[:2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    valid = np.ones((h, w), dtype=np.uint8)
    y0 = int(h * 0.6)
    vx1 = int(w * 0.3)
    vx2 = int(w * 0.7)
    valid[y0:, vx1:vx2] = 0
    if face_roi_mask is not None and face_roi_mask.shape[:2] == valid.shape[:2]:
        valid = np.where((valid > 0) & (face_roi_mask > 0), 1, 0).astype(np.uint8)

    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(
        ycrcb,
        np.array([0, 138, 85]),
        np.array([255, 170, 125]),
    )
    b, g, r = cv2.split(roi.astype(np.float32))
    rg_ratio = r / (g + 1.0)
    rb_ratio = r / (b + 1.0)
    red_dominance = (
        (rg_ratio >= 1.25)
        & (rb_ratio >= 1.25)
        & ((r - g) >= 28.0)
        & ((r - b) >= 22.0)
    )
    red_hsv_mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 120, 45]), np.array([10, 255, 255])),
        cv2.inRange(hsv, np.array([170, 120, 45]), np.array([180, 255, 255])),
    )
    blood_mask = np.where(
        (red_hsv_mask > 0) & (valid > 0) & (skin_mask > 0) & red_dominance,
        255,
        0,
    ).astype(np.uint8)
    blood_mask = cv2.morphologyEx(
        blood_mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )
    contours, _ = cv2.findContours(
        blood_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    valid_pixels = max(int(np.count_nonzero(valid)), 1)
    min_area = max(12.0, valid_pixels * 0.0015)
    if area < min_area:
        return None

    rx, ry, rw, rh = cv2.boundingRect(contour)
    if rw < 3 or rh < 3:
        return None
    bx1 = x1 + rx
    by1 = y1 + ry
    bx2 = x1 + rx + rw
    by2 = y1 + ry + rh

    # Ajuste apenas visual: reduz levemente a caixa para não "engolir" a face.
    shrink_ratio = 0.12
    dx = int((bx2 - bx1) * shrink_ratio)
    dy = int((by2 - by1) * shrink_ratio)
    sbx1 = min(max(bx1 + dx, x1), bx2 - 1)
    sby1 = min(max(by1 + dy, y1), by2 - 1)
    sbx2 = max(min(bx2 - dx, x2), sbx1 + 1)
    sby2 = max(min(by2 - dy, y2), sby1 + 1)
    return (sbx1, sby1, sbx2, sby2)


def _locate_face_bruise_bbox(
    frame_bgr: np.ndarray,
    face_bbox: tuple[int, int, int, int],
    *,
    face_roi_mask: np.ndarray | None = None,
    min_score_hint: float = 0.0,
) -> tuple[int, int, int, int] | None:
    frame_h, frame_w = frame_bgr.shape[:2]
    clipped = _clip_bbox_to_frame(face_bbox, frame_w, frame_h)
    if clipped is None:
        return None

    x1, y1, x2, y2 = clipped
    face_roi = frame_bgr[y1:y2, x1:x2]
    if face_roi.size == 0:
        return None

    roi = cv2.GaussianBlur(face_roi, (3, 3), 0)
    h, w = roi.shape[:2]
    valid = np.ones((h, w), dtype=np.uint8) * 255
    if face_roi_mask is not None and face_roi_mask.shape[:2] == valid.shape[:2]:
        valid = np.where(face_roi_mask > 0, 255, 0).astype(np.uint8)

    bruise_mask = _build_ultra_sensitive_bruise_mask(roi, valid)
    cheek_prior = _build_cheek_prior_map(h, w)
    if float(min_score_hint) > 0.0:
        bruise_mask = cv2.dilate(bruise_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
        bruise_mask = _refine_bruise_mask(bruise_mask, valid)

    contours, _ = cv2.findContours(
        bruise_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    valid_pixels = max(int(np.count_nonzero(valid > 0)), 1)
    contour = _pick_best_bruise_contour(
        list(contours),
        bruise_mask,
        valid_pixels=valid_pixels,
        cheek_prior=cheek_prior,
    )
    if contour is None:
        return None

    area = float(cv2.contourArea(contour))
    if area < max(1.0, valid_pixels * 0.0002):
        return None

    rx, ry, rw, rh = cv2.boundingRect(contour)
    if rw < 1 or rh < 1:
        return None
    bx1 = x1 + rx
    by1 = y1 + ry
    bx2 = x1 + rx + rw
    by2 = y1 + ry + rh

    # Expande levemente o bbox para ficar visível no vídeo.
    expand = 2
    sbx1 = max(x1, bx1 - expand)
    sby1 = max(y1, by1 - expand)
    sbx2 = min(x2, bx2 + expand)
    sby2 = min(y2, by2 + expand)
    return (sbx1, sby1, sbx2, sby2)


def _estimate_face_visual_aggression(
    frame_bgr: np.ndarray,
    face_payload: dict[str, Any],
    *,
    patient_roi_mask: np.ndarray | None = None,
    blood_score_threshold: float = 0.85,
    bruise_score_threshold: float = 0.06,
    bruise_min_pixels: int = 12,
) -> dict[str, float | bool]:
    h, w = frame_bgr.shape[:2]
    x = int(face_payload.get("x", 0) or 0)
    y = int(face_payload.get("y", 0) or 0)
    fw = int(face_payload.get("w", 0) or 0)
    fh = int(face_payload.get("h", 0) or 0)
    clipped = _clip_bbox_to_frame((x, y, x + fw, y + fh), w, h)
    if clipped is None:
        return {
            "emotion_stress_score": 0.0,
            "facial_asymmetry_score": 0.0,
            "blood_score": 0.0,
            "blood_detected": False,
            "aggression_score": 0.0,
        }

    x1, y1, x2, y2 = clipped
    face_roi = frame_bgr[y1:y2, x1:x2]
    face_roi_mask = (
        patient_roi_mask[y1:y2, x1:x2]
        if patient_roi_mask is not None
        else None
    )
    emotions = face_payload.get("emotions") or {}
    fear = float(emotions.get("fear", 0.0) or 0.0) / 100.0
    angry = float(emotions.get("angry", 0.0) or 0.0) / 100.0
    sad = float(emotions.get("sad", 0.0) or 0.0) / 100.0
    emotion_stress = max(0.0, min(1.0, fear * 0.5 + angry * 0.3 + sad * 0.2))

    asymmetry = _estimate_face_asymmetry(face_roi)
    blood_score, blood_detected = _estimate_face_blood_score(
        face_roi,
        face_roi_mask=face_roi_mask,
        blood_score_threshold=blood_score_threshold,
    )
    bruise_score, bruise_detected = _estimate_face_bruise_score(
        face_roi,
        face_roi_mask=face_roi_mask,
        bruise_score_threshold=bruise_score_threshold,
        bruise_min_pixels=bruise_min_pixels,
    )
    aggression_score = max(
        0.0,
        min(1.0, emotion_stress * 0.5 + asymmetry * 0.25 + blood_score * 0.25),
    )
    return {
        "emotion_stress_score": float(round(emotion_stress, 4)),
        "facial_asymmetry_score": float(round(asymmetry, 4)),
        "blood_score": float(round(blood_score, 4)),
        "blood_detected": bool(blood_detected),
        "bruise_score": float(round(bruise_score, 4)),
        "bruise_detected": bool(bruise_detected),
        "aggression_score": float(round(aggression_score, 4)),
    }


def _build_video_writer(
    output_video_path: str,
    fps: float,
    frame_size: tuple[int, int],
) -> tuple[cv2.VideoWriter, str]:
    codec_candidates = ("avc1", "H264", "mp4v")
    for codec in codec_candidates:
        writer = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*codec),
            fps,
            frame_size,
        )
        if writer.isOpened():
            print(f"[VideoWriter] Codec selecionado: {codec}")
            return writer, codec
        writer.release()

    raise RuntimeError(
        "Falha ao criar VideoWriter com codecs suportados "
        f"(tentados: {', '.join(codec_candidates)}): {output_video_path}"
    )


def _get_ffmpeg_executable() -> str | None:
    binary = shutil.which("ffmpeg")
    if binary:
        return binary
    try:
        import imageio_ffmpeg                

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return None


def _transcode_video_to_h264(video_path: str) -> bool:
    ffmpeg_bin = _get_ffmpeg_executable()
    if not ffmpeg_bin:
        print("[VideoWriter] ffmpeg indisponível; mantendo vídeo no codec original.")
        return False

    src = Path(video_path)
    if not src.is_file() or src.stat().st_size <= 0:
        print("[VideoWriter] vídeo inválido para transcodificação H.264.")
        return False

    tmp = src.with_name(f"{src.stem}.h264.tmp{src.suffix}")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or (not tmp.is_file()) or tmp.stat().st_size <= 0:
        print("[VideoWriter] falha ao transcodificar para H.264; mantendo original.")
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return False

    tmp.replace(src)
    print("[VideoWriter] vídeo transcodificado para H.264 com sucesso.")
    return True


def _format_video_timestamp(timestamp_sec: float) -> str:
    delta = timedelta(seconds=max(timestamp_sec, 0.0))
    total_seconds = int(delta.total_seconds())
    millis = int(round((delta.total_seconds() - total_seconds) * 1000.0))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _normalized_name_set(values: list[str] | None, fallback: set[str]) -> set[str]:
    if not values:
        return fallback
    normalized = {v.strip().lower() for v in values if v.strip()}
    return normalized or fallback


def _class_group(
    class_name: str,
    instrument_names: set[str],
    bleeding_names: set[str],
) -> str:
    normalized = class_name.strip().lower()
    if normalized in bleeding_names or any(
        token in normalized for token in ("bleeding", "sangramento", "blood")
    ):
        return "bleeding"
    if normalized in instrument_names:
        return "instrumento"
    if any(
        token in normalized
        for token in (
            "forceps",
            "scissors",
            "hemostat",
            "needle",
            "instrument",
            "auxiliary",
            "trocar",
            "grasper",
            "clip",
        )
    ):
        return "instrumento"
    return "other"


def _is_positive_temporal_label(task: str, label: str) -> bool:
    normalized = (label or "").strip().lower()
    if not normalized:
        return False
    if task == "bleeding":
        return normalized in {"bleeding", "blood", "sangramento"}
    if task == "smoke":
        return normalized in {"smoke", "fumaça", "fumaca"}
    return False


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0
    return int((inter_x2 - inter_x1) * (inter_y2 - inter_y1))


def _instrument_inside_bleeding(
    instrument_bbox: tuple[int, int, int, int],
    bleeding_bbox: tuple[int, int, int, int],
    min_overlap_ratio: float,
) -> bool:
    ix1, iy1, ix2, iy2 = instrument_bbox
    cx = (ix1 + ix2) / 2.0
    cy = (iy1 + iy2) / 2.0

    bx1, by1, bx2, by2 = bleeding_bbox
    center_inside = bx1 <= cx <= bx2 and by1 <= cy <= by2
    if center_inside:
        return True

    inter = _intersection_area(instrument_bbox, bleeding_bbox)
    instrument_area = max((ix2 - ix1) * (iy2 - iy1), 1)
    overlap = inter / float(instrument_area)
    return overlap >= min_overlap_ratio


def _bbox_area_ratio(
    bbox: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> float:
    x1, y1, x2, y2 = bbox
    area = max(0, x2 - x1) * max(0, y2 - y1)
    denom = max(frame_width * frame_height, 1)
    return float(area) / float(denom)


def _instrument_temporal_key(
    class_name: str,
    bbox: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> str:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
                                                                                 
    bin_x = int((cx / max(frame_width, 1)) * 10.0)
    bin_y = int((cy / max(frame_height, 1)) * 10.0)
    return f"{class_name.lower()}:{bin_x}:{bin_y}"


def run_pipeline(
    video_path: str | os.PathLike[str],
    output_video_path: str | os.PathLike[str],
    output_csv_path: str | os.PathLike[str] | None = None,
    *,
    pain_every_n_frames: int = 3,
    pain_scale: float = 0.5,
    smooth_window: int = 10,
    yolo_every_n_frames: int = 2,
    yolo_conf: float = 0.20,
    yolo_iou: float = 0.5,
    yolo_imgsz: int = 640,
    yolo_scale: float = 1.0,
    yolo_model_path: str | os.PathLike[str] | None = None,
    yolo_class_filter: list[str] | None = None,
    yolo_instrument_csv_path: str | os.PathLike[str] | None = None,
    bleeding_class_names: list[str] | None = None,
    instrument_class_names: list[str] | None = None,
    enable_spectral_filter: bool = True,
    min_overlap_for_risk: float = 0.15,
    spectral_min_rg_ratio: float = 1.12,
    spectral_min_rb_ratio: float = 1.10,
    spectral_min_red_mean: float = 55.0,
    spectral_min_pixels: int = 120,
    enable_temporal_events: bool = False,
    temporal_events_model_root: str | os.PathLike[str] | None = None,
    temporal_events_every_n_frames: int = 5,
    temporal_events_conf_threshold: float = 0.55,
    vision_every_n_frames: int = 3,
    fear_alert_threshold: float = 0.70,
    curvature_alert_threshold: float = 35.0,
    enable_patient_signals: bool = True,
    enable_instrument_detection: bool = True,
    highlight_visual_blood_boxes: bool = False,
    visual_blood_history_len: int = 4,
    visual_blood_persist_threshold: float = 0.85,
    visual_blood_score_threshold: float = 0.85,
    visual_bruise_score_threshold: float = 0.06,
    visual_bruise_min_pixels: int = 12,
    enable_bruise_detection: bool = True,
    instrument_min_confidence: float = 0.35,
    instrument_min_streak: int = 1,
    register_only_red_instruments: bool = False,
) -> dict[str, Any]:
    video_path = str(video_path)
    output_video_path = str(output_video_path)
    if output_csv_path is None:
        output_csv_path = str(
            Path(output_video_path).with_name("video_events.csv")
        )
    else:
        output_csv_path = str(output_csv_path)

    if yolo_instrument_csv_path is None:
        yolo_instrument_csv_path = str(
            Path(output_video_path).with_name("video_instrument_events.csv")
        )
    else:
        yolo_instrument_csv_path = str(yolo_instrument_csv_path)

    (
        visual_blood_persist_threshold,
        visual_blood_score_threshold,
        visual_bruise_score_threshold,
        visual_bruise_min_pixels,
    ) = _resolve_patient_visual_thresholds(
        visual_blood_persist_threshold=visual_blood_persist_threshold,
        visual_blood_score_threshold=visual_blood_score_threshold,
        visual_bruise_score_threshold=visual_bruise_score_threshold,
        visual_bruise_min_pixels=visual_bruise_min_pixels,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = float(fps) if float(fps) > 0 else 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    required_instrument_frames = max(int(instrument_min_streak), 1)

    selected_codec = "unknown"
    try:
        out, selected_codec = _build_video_writer(output_video_path, fps, (width, height))
    except RuntimeError:
        cap.release()
        raise

    pain_history: deque[float] = deque(maxlen=max(smooth_window, 1))
    visual_aggression_history: deque[float] = deque(maxlen=15)
    visual_blood_history: deque[int] = deque(maxlen=max(int(visual_blood_history_len), 1))
    instrument_streaks: dict[str, int] = {}
    instrument_last_seen_frame: dict[str, int] = {}
    last_pain_payload: dict[str, Any] = {"score": None, "faces": []}
    vision_runtime = get_vision_runtime()
    last_emotions_payload: dict[str, Any] = {
        "available": False,
        "face_detected": False,
        "emotions": _build_default_emotions_payload(),
        "fear_probability": 0.0,
        "error": "",
    }
    last_pose_dl_payload: dict[str, Any] = {
        "available": False,
        "pose_visible": False,
        "left_hip_angle": 180.0,
        "right_hip_angle": 180.0,
        "curvature": 0.0,
        "error": "",
    }
    last_decision_payload: dict[str, Any] = {
        "fear_probability": 0.0,
        "curvature": 0.0,
        "fear_threshold": float(fear_alert_threshold),
        "curvature_threshold": float(curvature_alert_threshold),
        "critical_alert": False,
    }

    yolo_detector = (
        get_default_detector(yolo_model_path)
        if bool(enable_instrument_detection)
        else None
    )
    last_yolo_payload: dict[str, Any] = {"detections": [], "names": {}}
    spectral_cfg = SpectralConfig(
        min_rg_ratio=float(spectral_min_rg_ratio),
        min_rb_ratio=float(spectral_min_rb_ratio),
        min_red_mean=float(spectral_min_red_mean),
        min_pixels=max(int(spectral_min_pixels), 1),
    )
    bleeding_name_set = _normalized_name_set(
        bleeding_class_names,
        fallback={"bleeding", "sangramento", "blood"},
    )
    instrument_name_set = _normalized_name_set(
        instrument_class_names,
        fallback={"instrumento", "instrument", "instrumentos"},
    )
    temporal_model_root_path = (
        Path(temporal_events_model_root)
        if temporal_events_model_root is not None
        else resolve_temporal_events_root()
    )
    temporal_inferencer: TemporalEventsInferencer | None = None
    last_temporal_predictions: dict[str, dict[str, Any]] = {
        "action": {"label": "", "confidence": 0.0, "class_id": -1, "available": False},
        "bleeding": {"label": "", "confidence": 0.0, "class_id": -1, "available": False},
        "smoke": {"label": "", "confidence": 0.0, "class_id": -1, "available": False},
    }
    if enable_temporal_events:
        try:
            temporal_inferencer = TemporalEventsInferencer(
                model_root=temporal_model_root_path
            )
            if not temporal_inferencer.has_models:
                temporal_inferencer = None
                print(
                    "[pipeline_video] Aviso: inferencia temporal habilitada, "
                    f"mas nenhum best.pt foi encontrado em {temporal_model_root_path}."
                )
        except Exception as exc:
            temporal_inferencer = None
            print(f"[pipeline_video] Aviso: falha ao carregar modelos temporais: {exc}")

    csv_rows: list[dict[str, Any]] = []
    instrument_rows: list[dict[str, Any]] = []
    frame_idx = 0

    try:
        with tqdm(total=total_frames, desc="Pipeline Fase 3") as bar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_sec = frame_idx / float(fps) if fps else 0.0

                if enable_patient_signals and frame_idx % max(int(vision_every_n_frames), 1) == 0:
                    preprocessed = preprocess_frame_node(frame)
                    last_emotions_payload = emotions_deepface_node(
                        preprocessed, vision_runtime
                    )
                    last_pose_dl_payload = pose_mediapipe_node(
                        preprocessed, vision_runtime
                    )
                    last_decision_payload = decision_alert_node(
                        last_emotions_payload,
                        last_pose_dl_payload,
                        fear_threshold=float(fear_alert_threshold),
                        curvature_threshold=float(curvature_alert_threshold),
                    )

                if temporal_inferencer is not None:
                    temporal_inferencer.update(frame)
                    if frame_idx % max(temporal_events_every_n_frames, 1) == 0:
                        last_temporal_predictions = temporal_inferencer.predict()

                if enable_patient_signals:
                    pose_result = process_frame_bgr(frame)
                    annotated = annotate_frame_with_pose(frame, pose_result)
                    if frame_idx % max(pain_every_n_frames, 1) == 0:
                        last_pain_payload = analyze_pain_in_frame(
                            frame, scale=pain_scale
                        )
                    pain_payload = last_pain_payload
                    pain_score = pain_payload.get("score")
                    if not bool(last_emotions_payload.get("face_detected", False)):
                        fallback_emotions = _fallback_emotions_from_pain_payload(
                            pain_payload
                        )
                        if bool(fallback_emotions.get("face_detected", False)):
                            last_emotions_payload = fallback_emotions
                    if not bool(last_pose_dl_payload.get("pose_visible", False)):
                        fallback_pose = _fallback_pose_payload_from_pose_result(
                            pose_result,
                            curvature_threshold=float(curvature_alert_threshold),
                        )
                        if bool(fallback_pose.get("pose_visible", False)):
                            last_pose_dl_payload = fallback_pose
                    last_decision_payload = decision_alert_node(
                        last_emotions_payload,
                        last_pose_dl_payload,
                        fear_threshold=float(fear_alert_threshold),
                        curvature_threshold=float(curvature_alert_threshold),
                    )
                else:
                    pose_result = {
                        "behavior_score": 0.0,
                        "flags": {
                            "pose_visible": False,
                            "left_arm_bent": False,
                            "right_arm_bent": False,
                            "curved_posture": False,
                            "asymmetric_shoulders": False,
                        },
                    }
                    annotated = frame.copy()
                    pain_payload = {"score": None, "faces": []}
                    pain_score = None
                    last_emotions_payload = {
                        "available": False,
                        "face_detected": False,
                        "emotions": _build_default_emotions_payload(),
                        "fear_probability": 0.0,
                        "error": "disabled_in_laparoscopy_mode",
                    }
                    last_pose_dl_payload = {
                        "available": False,
                        "pose_visible": False,
                        "left_hip_angle": 180.0,
                        "right_hip_angle": 180.0,
                        "curvature": 0.0,
                        "error": "disabled_in_laparoscopy_mode",
                    }
                    last_decision_payload = {
                        "fear_probability": 0.0,
                        "curvature": 0.0,
                        "fear_threshold": float(fear_alert_threshold),
                        "curvature_threshold": float(curvature_alert_threshold),
                        "critical_alert": False,
                    }

                is_yolo_infer_frame = bool(enable_instrument_detection) and (
                    frame_idx % max(yolo_every_n_frames, 1) == 0
                )
                if is_yolo_infer_frame:
                    last_yolo_payload = detect_instruments_in_frame(
                        frame,
                        detector=yolo_detector,
                        conf=yolo_conf,
                        iou=yolo_iou,
                        imgsz=yolo_imgsz,
                        scale=yolo_scale,
                        class_filter=yolo_class_filter,
                    )
                yolo_payload = last_yolo_payload
                raw_detections = yolo_payload.get("detections", [])
                detections: list[dict[str, Any]] = []
                bleeding_detections: list[dict[str, Any]] = []
                instrument_detections: list[dict[str, Any]] = []

                for det in raw_detections:
                    det_copy = dict(det)
                    class_name = str(det_copy.get("class_name", ""))
                    class_group = _class_group(
                        class_name=class_name,
                        instrument_names=instrument_name_set,
                        bleeding_names=bleeding_name_set,
                    )
                    det_copy["class_group"] = class_group
                    det_copy["risk_alert"] = False
                    det_copy["spectral_confirmed"] = ""
                    det_copy["rg_ratio"] = ""
                    det_copy["rb_ratio"] = ""
                    det_copy["red_mean"] = ""
                    det_copy["pixel_count"] = ""

                    if class_group == "bleeding":
                        if enable_spectral_filter:
                            spectral = evaluate_bleeding_roi(
                                frame_bgr=frame,
                                bbox=tuple(det_copy["bbox"]),
                                config=spectral_cfg,
                            )
                            det_copy["spectral_confirmed"] = int(spectral.is_blood_like)
                            det_copy["rg_ratio"] = round(float(spectral.rg_ratio), 4)
                            det_copy["rb_ratio"] = round(float(spectral.rb_ratio), 4)
                            det_copy["red_mean"] = round(float(spectral.red_mean), 2)
                            det_copy["pixel_count"] = int(spectral.pixel_count)
                            if not spectral.is_blood_like:
                                continue
                        bleeding_detections.append(det_copy)
                        detections.append(det_copy)
                    elif class_group == "instrumento":
                        # Mantém a bbox de instrumentos visível no vídeo.
                        detections.append(det_copy)
                        # Registra instrumento apenas quando a classe mapeia para bbox vermelho.
                        if register_only_red_instruments and not is_red_bbox_class_id(
                            int(det_copy.get("class_id", -1)),
                            str(det_copy.get("class_name", "")),
                        ):
                            continue
                                                                                
                                                                
                                                        
                                                                               
                        conf = float(det_copy.get("confidence", 0.0) or 0.0)
                        area_ratio = _bbox_area_ratio(tuple(det_copy["bbox"]), width, height)
                        keep_instrument = (
                            conf >= float(instrument_min_confidence)
                            and 0.003 <= area_ratio <= 0.45
                        )
                        if keep_instrument:
                            key = _instrument_temporal_key(
                                class_name=class_name,
                                bbox=tuple(det_copy["bbox"]),
                                frame_width=width,
                                frame_height=height,
                            )
                            prev_frame = instrument_last_seen_frame.get(key, -9999)
                            if prev_frame == frame_idx - 1:
                                instrument_streaks[key] = instrument_streaks.get(key, 0) + 1
                            else:
                                instrument_streaks[key] = 1
                            instrument_last_seen_frame[key] = frame_idx
                            keep_instrument = (
                                instrument_streaks.get(key, 0) >= required_instrument_frames
                            )

                        if not keep_instrument:
                            continue
                        instrument_detections.append(det_copy)

                if is_yolo_infer_frame:
                                                                        
                                                              
                    last_yolo_payload = {
                        "detections": detections,
                        "names": yolo_payload.get("names", {}),
                    }

                if pain_score is not None:
                    pain_history.append(float(pain_score))
                pain_smoothed = (
                    float(np.mean(pain_history)) if pain_history else 0.0
                )

                if enable_patient_signals:
                    patient_roi_mask = _build_patient_roi_mask(
                        frame.shape,
                        pose_result.get("landmarks", []),
                    )
                    face_visual_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
                    for face in pain_payload.get("faces", []):
                        payload = _estimate_face_visual_aggression(
                            frame,
                            face,
                            patient_roi_mask=patient_roi_mask,
                            blood_score_threshold=float(visual_blood_score_threshold),
                            bruise_score_threshold=(
                                float(visual_bruise_score_threshold)
                                if enable_bruise_detection
                                else 1.0
                            ),
                            bruise_min_pixels=(
                                int(visual_bruise_min_pixels)
                                if enable_bruise_detection
                                else 99999
                            ),
                        )
                        x = int(face.get("x", 0) or 0)
                        y = int(face.get("y", 0) or 0)
                        w = int(face.get("w", 0) or 0)
                        h = int(face.get("h", 0) or 0)
                        face_roi_mask = (
                            patient_roi_mask[y : y + h, x : x + w]
                            if (
                                patient_roi_mask is not None
                                and y >= 0
                                and x >= 0
                                and (y + h) <= patient_roi_mask.shape[0]
                                and (x + w) <= patient_roi_mask.shape[1]
                            )
                            else None
                        )
                        blood_bbox = None
                        if bool(payload.get("blood_detected", False)):
                            blood_bbox = _locate_face_blood_bbox(
                                frame,
                                (x, y, x + w, y + h),
                                face_roi_mask=face_roi_mask,
                            )
                            if blood_bbox is None:
                                # Consistência: só confirma sangue quando também localiza mancha.
                                payload["blood_detected"] = False
                        payload["blood_bbox"] = blood_bbox
                        if enable_bruise_detection:
                            bruise_score_hint = float(
                                payload.get("bruise_score", 0.0) or 0.0
                            )
                            bruise_bbox = _locate_face_bruise_bbox(
                                frame,
                                (x, y, x + w, y + h),
                                face_roi_mask=face_roi_mask,
                                min_score_hint=bruise_score_hint,
                            )
                            if bruise_bbox is not None:
                                payload["bruise_detected"] = True
                            payload["bruise_bbox"] = bruise_bbox
                        else:
                            payload["bruise_score"] = 0.0
                            payload["bruise_detected"] = False
                            payload["bruise_bbox"] = None
                        face_visual_pairs.append((face, payload))
                    visual_payloads = [payload for _, payload in face_visual_pairs]
                    frame_visual_aggression = max(
                        (float(item.get("aggression_score", 0.0) or 0.0) for item in visual_payloads),
                        default=0.0,
                    )
                    frame_visual_blood_score = max(
                        (float(item.get("blood_score", 0.0) or 0.0) for item in visual_payloads),
                        default=0.0,
                    )
                    frame_visual_blood_detected = any(
                        bool(item.get("blood_detected", False)) for item in visual_payloads
                    )
                    frame_visual_bruise_score = max(
                        (float(item.get("bruise_score", 0.0) or 0.0) for item in visual_payloads),
                        default=0.0,
                    )
                    frame_visual_bruise_detected = (
                        enable_bruise_detection
                        and any(
                            bool(item.get("bruise_detected", False))
                            or item.get("bruise_bbox") is not None
                            or float(item.get("bruise_score", 0.0) or 0.0)
                            >= float(visual_bruise_score_threshold)
                            for item in visual_payloads
                        )
                    )
                    visual_aggression_history.append(frame_visual_aggression)
                    visual_blood_history.append(1 if frame_visual_blood_detected else 0)
                    visual_aggression_persisted = float(np.mean(visual_aggression_history))
                    visual_blood_persisted = (
                        float(np.mean(visual_blood_history))
                        >= float(visual_blood_persist_threshold)
                    )
                else:
                    patient_roi_mask = None
                    face_visual_pairs = []
                    frame_visual_aggression = 0.0
                    frame_visual_blood_score = 0.0
                    frame_visual_blood_detected = False
                    frame_visual_bruise_score = 0.0
                    frame_visual_bruise_detected = False
                    visual_aggression_persisted = 0.0
                    visual_blood_persisted = False

                if enable_patient_signals:
                    for idx, face in enumerate(pain_payload.get("faces", [])):
                        x, y, w, h = face["x"], face["y"], face["w"], face["h"]
                        color = _pain_color(pain_smoothed)
                        label = f"{_pain_label(pain_smoothed)} ({pain_smoothed:.1f}%)"
                        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(
                            annotated,
                            label,
                            (x, max(20, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2,
                        )
                        if bool(highlight_visual_blood_boxes):
                            blood_payload = (
                                face_visual_pairs[idx][1]
                                if idx < len(face_visual_pairs)
                                else {}
                            )
                            blood_detected = bool(
                                blood_payload.get("blood_detected", False)
                            )
                            blood_score = float(
                                blood_payload.get("blood_score", 0.0) or 0.0
                            )
                            if blood_detected:
                                blood_bbox = blood_payload.get("blood_bbox")
                                if blood_bbox is not None:
                                    bx1, by1, bx2, by2 = blood_bbox
                                    cv2.rectangle(
                                        annotated,
                                        (bx1, by1),
                                        (bx2, by2),
                                        (0, 0, 255),
                                        3,
                                    )
                                    cv2.putText(
                                        annotated,
                                        f"SANGUE VISUAL ({blood_score:.2f})",
                                        (bx1, max(20, by1 - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.55,
                                        (0, 0, 255),
                                        2,
                                    )
                            if enable_bruise_detection:
                                bruise_score = float(
                                    blood_payload.get("bruise_score", 0.0) or 0.0
                                )
                                bruise_bbox = blood_payload.get("bruise_bbox")
                                show_bruise_box = (
                                    bruise_bbox is not None
                                    or bool(blood_payload.get("bruise_detected", False))
                                    or bruise_score >= float(visual_bruise_score_threshold)
                                )
                                if show_bruise_box and bruise_bbox is not None:
                                    hx1, hy1, hx2, hy2 = bruise_bbox
                                    cv2.rectangle(
                                        annotated,
                                        (hx1, hy1),
                                        (hx2, hy2),
                                        (255, 0, 255),
                                        3,
                                    )
                                    cv2.putText(
                                        annotated,
                                        f"HEMATOMA VISUAL ({bruise_score:.2f})",
                                        (hx1, max(20, hy1 - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.55,
                                        (255, 0, 255),
                                        2,
                                    )

                annotated = annotate_frame_with_detections(annotated, detections)

                temporal_action_pred = last_temporal_predictions.get("action", {})
                temporal_bleeding_pred = last_temporal_predictions.get("bleeding", {})
                temporal_smoke_pred = last_temporal_predictions.get("smoke", {})
                temporal_bleeding_positive = (
                    bool(temporal_bleeding_pred.get("available"))
                    and float(temporal_bleeding_pred.get("confidence", 0.0) or 0.0)
                    >= float(temporal_events_conf_threshold)
                    and _is_positive_temporal_label(
                        "bleeding", str(temporal_bleeding_pred.get("label", ""))
                    )
                )
                temporal_smoke_positive = (
                    bool(temporal_smoke_pred.get("available"))
                    and float(temporal_smoke_pred.get("confidence", 0.0) or 0.0)
                    >= float(temporal_events_conf_threshold)
                    and _is_positive_temporal_label(
                        "smoke", str(temporal_smoke_pred.get("label", ""))
                    )
                )

                risk_alert_active = False
                for instrument in instrument_detections:
                    for bleeding in bleeding_detections:
                        if _instrument_inside_bleeding(
                            instrument_bbox=tuple(instrument["bbox"]),
                            bleeding_bbox=tuple(bleeding["bbox"]),
                            min_overlap_ratio=float(min_overlap_for_risk),
                        ):
                            risk_alert_active = True
                            instrument["risk_alert"] = True
                            bleeding["risk_alert"] = True

                temporal_risk_flag = temporal_bleeding_positive and bool(
                    instrument_detections
                )
                if temporal_risk_flag:
                    risk_alert_active = True
                vision_critical_alert = bool(
                    enable_patient_signals
                    and last_decision_payload.get("critical_alert", False)
                )
                if vision_critical_alert:
                    risk_alert_active = True

                if risk_alert_active:
                    if vision_critical_alert:
                        cv2.putText(
                            annotated,
                            "ALERTA CRITICO: fear alta + curvatura corporal",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )
                    else:
                        cv2.putText(
                            annotated,
                            "ALERTA RISCO: evento de sangramento + instrumento",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )
                elif (
                    temporal_bleeding_positive
                    or temporal_smoke_positive
                    or visual_aggression_persisted >= 0.65
                    or visual_blood_persisted
                ):
                    cv2.putText(
                        annotated,
                        "Eventos detectados",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                fear_prob = float(last_decision_payload.get("fear_probability", 0.0) or 0.0)
                curvature_deg = float(last_decision_payload.get("curvature", 0.0) or 0.0)
                emotions_payload = last_emotions_payload.get("emotions", {})
                if enable_patient_signals:
                    cv2.putText(
                        annotated,
                        f"fear={fear_prob:.2f} | curv={curvature_deg:.1f}deg",
                        (10, 84),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 220, 120),
                        1,
                    )

                if bool(temporal_action_pred.get("available")):
                    action_label = str(temporal_action_pred.get("label", "") or "-")
                    action_conf = float(temporal_action_pred.get("confidence", 0.0) or 0.0)
                    cv2.putText(
                        annotated,
                        f"Temporal action: {action_label} ({action_conf:.2f})",
                        (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 220, 120),
                        1,
                    )

                cv2.putText(
                    annotated,
                    f"t={timestamp_sec:5.2f}s | frame {frame_idx}",
                    (10, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )
                out.write(annotated)

                flags = pose_result["flags"]

                instruments_seen = sorted(
                    {str(d["class_name"]).strip() for d in instrument_detections if str(d["class_name"]).strip()}
                )
                top_conf = (
                    max(
                        (float(d["confidence"]) for d in instrument_detections),
                        default=0.0,
                    )
                    if instrument_detections
                    else 0.0
                )

                csv_rows.append(
                    {
                        "frame_idx": frame_idx,
                        "timestamp_sec": round(timestamp_sec, 3),
                        "pain_proxy": (
                            round(float(pain_score), 3)
                            if pain_score is not None
                            else ""
                        ),
                        "pain_smoothed": round(pain_smoothed, 3),
                        "behavior_score": round(pose_result["behavior_score"], 3),
                        "flag_pose_visible": _bool_to_int(flags.get("pose_visible")),
                        "flag_left_arm_bent": _bool_to_int(
                            flags.get("left_arm_bent")
                        ),
                        "flag_right_arm_bent": _bool_to_int(
                            flags.get("right_arm_bent")
                        ),
                        "flag_curved_posture": _bool_to_int(
                            flags.get("curved_posture")
                        ),
                        "flag_asymmetric_shoulders": _bool_to_int(
                            flags.get("asymmetric_shoulders")
                        ),
                        "n_faces": len(pain_payload.get("faces", [])),
                        "n_instruments": len(instruments_seen),
                        "instruments_seen": "|".join(instruments_seen),
                        "instrument_top_conf": round(top_conf, 3),
                        "visual_aggression_score": round(frame_visual_aggression, 4),
                        "visual_aggression_persisted": round(visual_aggression_persisted, 4),
                        "score_sangue_identificado": round(
                            frame_visual_blood_score, 4
                        ),
                        "flag_sangue_identificado": int(frame_visual_blood_detected),
                        "score_equimose_identificada": round(
                            frame_visual_bruise_score, 4
                        ),
                        "flag_equimose_identificada": int(frame_visual_bruise_detected),
                        "emotion_angry_prob": round(
                            float(emotions_payload.get("angry", 0.0) or 0.0), 4
                        ),
                        "emotion_disgust_prob": round(
                            float(emotions_payload.get("disgust", 0.0) or 0.0), 4
                        ),
                        "emotion_fear_prob": round(
                            float(last_decision_payload.get("fear_probability", 0.0) or 0.0),
                            4,
                        ),
                        "emotion_happy_prob": round(
                            float(emotions_payload.get("happy", 0.0) or 0.0), 4
                        ),
                        "emotion_neutral_prob": round(
                            float(emotions_payload.get("neutral", 0.0) or 0.0), 4
                        ),
                        "emotion_sad_prob": round(
                            float(emotions_payload.get("sad", 0.0) or 0.0), 4
                        ),
                        "emotion_face_detected": int(
                            bool(last_emotions_payload.get("face_detected", False))
                        ),
                        "pose_curvature_deg": round(
                            float(last_decision_payload.get("curvature", 0.0) or 0.0),
                            3,
                        ),
                        "vision_critical_alert": int(
                            bool(last_decision_payload.get("critical_alert", False))
                        ),
                        "temporal_action_label": str(
                            temporal_action_pred.get("label", "")
                        ),
                        "temporal_action_conf": round(
                            float(temporal_action_pred.get("confidence", 0.0) or 0.0), 4
                        ),
                        "temporal_bleeding_label": str(
                            temporal_bleeding_pred.get("label", "")
                        ),
                        "temporal_bleeding_conf": round(
                            float(temporal_bleeding_pred.get("confidence", 0.0) or 0.0), 4
                        ),
                        "temporal_bleeding_positive": int(temporal_bleeding_positive),
                        "temporal_smoke_label": str(temporal_smoke_pred.get("label", "")),
                        "temporal_smoke_conf": round(
                            float(temporal_smoke_pred.get("confidence", 0.0) or 0.0), 4
                        ),
                        "temporal_smoke_positive": int(temporal_smoke_positive),
                    }
                )

                for det in detections:
                    if (
                        register_only_red_instruments
                        and str(det.get("class_group", "")) == "instrumento"
                        and not is_red_bbox_class_id(
                            int(det.get("class_id", -1)),
                            str(det.get("class_name", "")),
                        )
                    ):
                        continue
                    x1, y1, x2, y2 = det["bbox"]
                    instrument_rows.append(
                        {
                            "frame_idx": frame_idx,
                            "timestamp_sec": round(timestamp_sec, 3),
                            "timestamp": _format_video_timestamp(timestamp_sec),
                            "class_id": int(det["class_id"]),
                            "class_name": str(det["class_name"]),
                            "class_group": str(det.get("class_group", "other")),
                            "confidence": round(float(det["confidence"]), 4),
                            "spectral_confirmed": det.get("spectral_confirmed", ""),
                            "rg_ratio": det.get("rg_ratio", ""),
                            "rb_ratio": det.get("rb_ratio", ""),
                            "red_mean": det.get("red_mean", ""),
                            "pixel_count": det.get("pixel_count", ""),
                            "risk_alert": int(bool(det.get("risk_alert", False))),
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2),
                        }
                    )

                frame_idx += 1
                bar.update(1)
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    if selected_codec.lower() == "mp4v":
        _transcode_video_to_h264(output_video_path)

    with open(output_csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(csv_rows)

    with open(
        yolo_instrument_csv_path, "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=INSTRUMENT_EVENT_COLUMNS)
        writer.writeheader()
        writer.writerows(instrument_rows)

    return {
        "video_path": output_video_path,
        "csv_path": output_csv_path,
        "instrument_csv_path": yolo_instrument_csv_path,
        "frames_processed": frame_idx,
        "instrument_detections": len(instrument_rows),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fase 3: pipeline de vídeo (pose + pain proxy) -> MP4 anotado + CSV."
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        default=str(PROJECT_ROOT / "exame_paciente_01.mp4"),
        help="Vídeo MP4 de entrada.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(PROJECT_ROOT / "data" / "videos" / "exame_paciente_01_fase3.mp4"),
        help="MP4 anotado de saída.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help=(
            "CSV de saída (default: mesmo diretório do MP4 com nome "
            "video_events.csv)."
        ),
    )
    parser.add_argument(
        "--pain-every",
        type=int,
        default=3,
        help="Rodar DeepFace 1 a cada N frames (default: 3).",
    )
    parser.add_argument(
        "--pain-scale",
        type=float,
        default=0.5,
        help="Fator de downscale aplicado ao frame antes do DeepFace.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=10,
        help="Janela da média móvel do score de dor.",
    )
    parser.add_argument(
        "--yolo-every",
        type=int,
        default=2,
        help="Rodar YOLO 1 a cada N frames (default: 2).",
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.20,
        help="Confiança mínima para considerar uma detecção (default: 0.20).",
    )
    parser.add_argument(
        "--yolo-iou",
        type=float,
        default=0.5,
        help="IoU para NMS do YOLO (default: 0.5).",
    )
    parser.add_argument(
        "--yolo-imgsz",
        type=int,
        default=960,
        help="Tamanho de entrada do YOLO (default: 960).",
    )
    parser.add_argument(
        "--yolo-scale",
        type=float,
        default=1.0,
        help="Downscale opcional aplicado antes do YOLO (default: 1.0).",
    )
    parser.add_argument(
        "--yolo-model",
        default=None,
        help=(
            "Caminho para um .pt do YOLO. Default: hf_models/deployment/best.pt "
            "(ou models/yolo_instruments/weights/best.pt), senão yolov8n.pt (COCO)."
        ),
    )
    parser.add_argument(
        "--yolo-classes",
        default=None,
        help=(
            'Lista de classes a manter, separadas por vírgula '
            '(ex.: "scissors,knife"). Útil no fallback COCO.'
        ),
    )
    parser.add_argument(
        "--instrument-csv",
        default=None,
        help=(
            "CSV de eventos por detecção. Default: "
            "video_instrument_events.csv ao lado do MP4 de saída."
        ),
    )
    parser.add_argument(
        "--bleeding-classes",
        default="bleeding,sangramento,blood",
        help="Nomes de classe mapeados para bleeding, separados por virgula.",
    )
    parser.add_argument(
        "--instrument-classes",
        default="instrumento,instrument,instrumentos",
        help="Nomes de classe mapeados para instrumento, separados por virgula.",
    )
    parser.add_argument(
        "--disable-spectral-filter",
        action="store_true",
        help="Desabilita validacao espectral para deteccoes de bleeding.",
    )
    parser.add_argument(
        "--risk-overlap",
        type=float,
        default=0.15,
        help="Sobreposicao minima (0..1) para flag de risco espacial.",
    )
    parser.add_argument(
        "--spectral-min-rg",
        type=float,
        default=1.12,
        help="Limiar minimo para razao R/G no filtro espectral.",
    )
    parser.add_argument(
        "--spectral-min-rb",
        type=float,
        default=1.10,
        help="Limiar minimo para razao R/B no filtro espectral.",
    )
    parser.add_argument(
        "--spectral-min-red",
        type=float,
        default=55.0,
        help="Limiar minimo para media do canal vermelho.",
    )
    parser.add_argument(
        "--spectral-min-pixels",
        type=int,
        default=120,
        help="Area minima de ROI para o filtro espectral.",
    )
    parser.add_argument(
        "--enable-temporal-events",
        action="store_true",
        help="Habilita inferencia temporal supervisionada (action/bleeding/smoke).",
    )
    parser.add_argument(
        "--temporal-events-root",
        default=None,
        help=(
            "Diretorio raiz com subpastas action/bleeding/smoke contendo best.pt. "
            "Default: hf_models/action_events/gynsurg_action_3sec_round2."
        ),
    )
    parser.add_argument(
        "--temporal-events-every",
        type=int,
        default=3,
        help="Executa inferencia temporal a cada N frames (default: 3).",
    )
    parser.add_argument(
        "--temporal-events-threshold",
        type=float,
        default=0.45,
        help="Confianca minima para considerar evento temporal positivo.",
    )
    parser.add_argument(
        "--vision-every",
        type=int,
        default=3,
        help="Executa nós de visão (DeepFace+MediaPipe) a cada N frames.",
    )
    parser.add_argument(
        "--fear-threshold",
        type=float,
        default=0.70,
        help="Limiar de fear (0..1) para alerta crítico de visão.",
    )
    parser.add_argument(
        "--curvature-threshold",
        type=float,
        default=35.0,
        help="Limiar de curvatura (graus) para alerta crítico de visão.",
    )
    parser.add_argument(
        "--instrument-min-conf",
        type=float,
        default=0.35,
        help="Confiança mínima para confirmar instrumento após filtros.",
    )
    parser.add_argument(
        "--instrument-min-streak",
        type=int,
        default=2,
        help="Persistência mínima em frames consecutivos para confirmar instrumento.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if args.csv:
        csv_dir = os.path.dirname(args.csv)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
    if args.instrument_csv:
        inst_dir = os.path.dirname(args.instrument_csv)
        if inst_dir:
            os.makedirs(inst_dir, exist_ok=True)

    class_filter: list[str] | None = None
    if args.yolo_classes:
        class_filter = [
            c.strip() for c in args.yolo_classes.split(",") if c.strip()
        ]
    bleeding_class_names = [
        c.strip() for c in str(args.bleeding_classes).split(",") if c.strip()
    ]
    instrument_class_names = [
        c.strip() for c in str(args.instrument_classes).split(",") if c.strip()
    ]

    try:
        summary = run_pipeline(
            args.input,
            args.output,
            args.csv,
            pain_every_n_frames=args.pain_every,
            pain_scale=args.pain_scale,
            smooth_window=args.smooth_window,
            yolo_every_n_frames=args.yolo_every,
            yolo_conf=args.yolo_conf,
            yolo_iou=args.yolo_iou,
            yolo_imgsz=args.yolo_imgsz,
            yolo_scale=args.yolo_scale,
            yolo_model_path=args.yolo_model,
            yolo_class_filter=class_filter,
            yolo_instrument_csv_path=args.instrument_csv,
            bleeding_class_names=bleeding_class_names,
            instrument_class_names=instrument_class_names,
            enable_spectral_filter=not args.disable_spectral_filter,
            min_overlap_for_risk=args.risk_overlap,
            spectral_min_rg_ratio=args.spectral_min_rg,
            spectral_min_rb_ratio=args.spectral_min_rb,
            spectral_min_red_mean=args.spectral_min_red,
            spectral_min_pixels=args.spectral_min_pixels,
            enable_temporal_events=args.enable_temporal_events,
            temporal_events_model_root=args.temporal_events_root,
            temporal_events_every_n_frames=args.temporal_events_every,
            temporal_events_conf_threshold=args.temporal_events_threshold,
            vision_every_n_frames=args.vision_every,
            fear_alert_threshold=args.fear_threshold,
            curvature_alert_threshold=args.curvature_threshold,
            instrument_min_confidence=args.instrument_min_conf,
            instrument_min_streak=args.instrument_min_streak,
        )
    except FileNotFoundError as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1

    print("Concluído.")
    print(f"Vídeo:              {summary['video_path']}")
    print(f"CSV (frames):       {summary['csv_path']}")
    print(f"CSV (instrumentos): {summary['instrument_csv_path']}")
    print(f"Frames processados:    {summary['frames_processed']}")
    print(f"Detecções gravadas:    {summary['instrument_detections']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
