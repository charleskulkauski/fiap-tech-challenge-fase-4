from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np

from src.hf_assets import resolve_default_yolo_model, resolve_yolo_weight

_MODULE_DIR = Path(__file__).resolve().parent
_WEIGHTS_DIR = _MODULE_DIR / "weights"
_LOCAL_BEST = _WEIGHTS_DIR / "best.pt"
_FALLBACK_MODEL = "yolov8n.pt"

                                                            
_LOADED_DETECTORS: dict[str, Any] = {}

                                                             
_BOX_COLORS: list[tuple[int, int, int]] = [
    (0, 200, 255),
    (255, 120, 0),
    (0, 255, 120),
    (255, 0, 200),
    (120, 0, 255),
    (255, 255, 0),
    (0, 0, 255),
    (200, 200, 200),
]
_RED_BBOX_COLOR: tuple[int, int, int] = (0, 0, 255)
_RED_BBOX_NAME_HINTS: tuple[str, ...] = (
    "instrument_v",
    "auxiliary_v",
    "instrumento",
    "instrument",
    "hemostat",
    "forceps",
    "scissors",
    "needle",
    "hilt",
)


def _color_for_class(class_id: int) -> tuple[int, int, int]:
    return _BOX_COLORS[class_id % len(_BOX_COLORS)]


def is_red_bbox_class_id(class_id: int, class_name: str | None = None) -> bool:
    # Em alguns datasets, a semântica de "instrumento cirúrgico relevante"
    # está no nome da classe (instrument_v*, auxiliary_v*, forceps etc.) e
    # não em um class_id fixo. Mantemos fallback por cor para compatibilidade.
    normalized = str(class_name or "").strip().lower()
    if normalized and any(token in normalized for token in _RED_BBOX_NAME_HINTS):
        return True
    return _color_for_class(int(class_id)) == _RED_BBOX_COLOR


def _resolve_weights_path(model_path: Optional[str | os.PathLike[str]]) -> str:
    if model_path:
        return str(model_path)

    env_path = os.environ.get("YOLO_INSTRUMENTS_WEIGHTS")
    if env_path:
        return env_path

    hf_weight = resolve_default_yolo_model()
    if hf_weight is not None:
        return str(hf_weight)

    legacy = resolve_yolo_weight("best.pt")
    if legacy is not None:
        return str(legacy)

    if _LOCAL_BEST.is_file() and _LOCAL_BEST.stat().st_size > 0:
        return str(_LOCAL_BEST)

    return _FALLBACK_MODEL


def get_default_detector(
    model_path: Optional[str | os.PathLike[str]] = None,
) -> Any:
    from ultralytics import YOLO

    resolved = _resolve_weights_path(model_path)
    detector = _LOADED_DETECTORS.get(resolved)
    if detector is None:
        detector = YOLO(resolved)
        _LOADED_DETECTORS[resolved] = detector
    return detector


def _normalize_class_filter(
    class_filter: Optional[Iterable[str]],
    names: dict[int, str] | list[str],
) -> Optional[set[str]]:
    if class_filter is None:
        return None
    if isinstance(names, dict):
        valid_names = {str(v).lower() for v in names.values()}
    else:
        valid_names = {str(v).lower() for v in names}
    requested = {str(c).lower() for c in class_filter}
    return requested & valid_names if requested else None


def detect_instruments_in_frame(
    frame_bgr: np.ndarray,
    *,
    detector: Optional[Any] = None,
    model_path: Optional[str | os.PathLike[str]] = None,
    conf: float = 0.35,
    iou: float = 0.5,
    imgsz: int = 640,
    scale: float = 1.0,
    class_filter: Optional[Iterable[str]] = None,
    max_det: int = 50,
) -> dict[str, Any]:
    if frame_bgr is None or frame_bgr.size == 0:
        return {"detections": [], "names": {}}

    if detector is None:
        detector = get_default_detector(model_path)

    h, w = frame_bgr.shape[:2]
    if 0.0 < scale < 1.0 and min(h, w) > 100:
        small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
        inv_scale = 1.0 / scale
    else:
        small = frame_bgr
        inv_scale = 1.0

    results = detector.predict(
        source=small,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        max_det=max_det,
        verbose=False,
    )

    detections: list[dict[str, Any]] = []
    if not results:
        return {"detections": detections, "names": {}}

    result = results[0]
    names = result.names if hasattr(result, "names") else {}
    name_filter = _normalize_class_filter(class_filter, names)

    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.shape[0] == 0:
        return {"detections": detections, "names": dict(names) if names else {}}

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), c, cls_id in zip(xyxy, confs, clss):
        class_name = str(names.get(int(cls_id), str(int(cls_id))))
        if name_filter is not None and class_name.lower() not in name_filter:
            continue
        bx1 = int(round(float(x1) * inv_scale))
        by1 = int(round(float(y1) * inv_scale))
        bx2 = int(round(float(x2) * inv_scale))
        by2 = int(round(float(y2) * inv_scale))
        bx1 = max(0, min(bx1, w - 1))
        by1 = max(0, min(by1, h - 1))
        bx2 = max(0, min(bx2, w - 1))
        by2 = max(0, min(by2, h - 1))
        if bx2 <= bx1 or by2 <= by1:
            continue
        detections.append(
            {
                "class_id": int(cls_id),
                "class_name": class_name,
                "confidence": float(c),
                "bbox": (bx1, by1, bx2, by2),
            }
        )

    return {"detections": detections, "names": dict(names) if names else {}}


def annotate_frame_with_detections(
    frame_bgr: np.ndarray,
    detections_payload: dict[str, Any] | list[dict[str, Any]],
) -> np.ndarray:
    if isinstance(detections_payload, dict):
        detections = detections_payload.get("detections", [])
    else:
        detections = detections_payload or []

    out = frame_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = _color_for_class(int(det.get("class_id", 0)))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"{det['class_name']} {det['confidence']:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        tx, ty = x1, max(0, y1 - 6)
        cv2.rectangle(
            out,
            (tx, ty - th - baseline),
            (tx + tw + 4, ty + baseline),
            color,
            thickness=-1,
        )
        cv2.putText(
            out,
            label,
            (tx + 2, ty - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return out
