
from __future__ import annotations

import math
import os
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
_MODEL_PATH = Path(__file__).resolve().parent / "pose_landmarker_lite.task"

                                                               
_DEFAULT_DETECTOR: Optional[mp_vision.PoseLandmarker] = None

                                                                        
_POSE_CONNECTIONS: list[tuple[int, int]] = [
    (c.start, c.end)
    for c in mp_vision.PoseLandmarksConnections.POSE_LANDMARKS
]


def _ensure_model() -> str:
    if _MODEL_PATH.is_file() and _MODEL_PATH.stat().st_size > 0:
        return str(_MODEL_PATH)
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[pose] Baixando modelo PoseLandmarker para {_MODEL_PATH} ...")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return str(_MODEL_PATH)


def _get_default_detector() -> mp_vision.PoseLandmarker:
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None:
        options = mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_ensure_model()),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _DEFAULT_DETECTOR = mp_vision.PoseLandmarker.create_from_options(options)
    return _DEFAULT_DETECTOR


def _angle_deg(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    nba = math.hypot(*ba)
    nbc = math.hypot(*bc)
    if nba == 0 or nbc == 0:
        return float("nan")
    cos_t = max(-1.0, min(1.0, dot / (nba * nbc)))
    return math.degrees(math.acos(cos_t))


def _extract_pose_flags(landmarks: list[dict]) -> tuple[dict, float]:

    def pt(idx: int) -> tuple[float, float] | None:
        if idx >= len(landmarks) or landmarks[idx] is None:
            return None
        if landmarks[idx]["visibility"] < 0.4:
            return None
        return (landmarks[idx]["x"], landmarks[idx]["y"])

    LSH, RSH = pt(11), pt(12)
    LEL, REL = pt(13), pt(14)
    LWR, RWR = pt(15), pt(16)
    LHP, RHP = pt(23), pt(24)

    flags = {
        "pose_visible": False,
        "left_arm_bent": False,
        "right_arm_bent": False,
        "curved_posture": False,
        "asymmetric_shoulders": False,
    }
    score_parts: list[float] = []

    if LSH and LEL and LWR:
        ang_l = _angle_deg(LSH, LEL, LWR)
        if not math.isnan(ang_l):
            flags["left_arm_bent"] = ang_l < 70.0
            score_parts.append(max(0.0, (90.0 - ang_l) / 90.0) * 25.0)

    if RSH and REL and RWR:
        ang_r = _angle_deg(RSH, REL, RWR)
        if not math.isnan(ang_r):
            flags["right_arm_bent"] = ang_r < 70.0
            score_parts.append(max(0.0, (90.0 - ang_r) / 90.0) * 25.0)

    if LSH and RSH and LHP and RHP:
        sh_mid = ((LSH[0] + RSH[0]) / 2.0, (LSH[1] + RSH[1]) / 2.0)
        hp_mid = ((LHP[0] + RHP[0]) / 2.0, (LHP[1] + RHP[1]) / 2.0)
        dx = sh_mid[0] - hp_mid[0]
        dy = sh_mid[1] - hp_mid[1]
        if abs(dy) > 1e-6:
            ang_trunk = math.degrees(math.atan2(abs(dx), abs(dy)))
            flags["curved_posture"] = ang_trunk > 25.0
            score_parts.append(min(ang_trunk / 45.0, 1.0) * 25.0)

    if LSH and RSH:
        diff_y = abs(LSH[1] - RSH[1])
        flags["asymmetric_shoulders"] = diff_y > 0.07
        score_parts.append(min(diff_y / 0.15, 1.0) * 25.0)

    if any([LSH, RSH, LHP, RHP]):
        flags["pose_visible"] = True

    behavior_score = float(min(100.0, sum(score_parts))) if score_parts else 0.0
    return flags, behavior_score


def process_frame_bgr(
    frame_bgr: np.ndarray,
    pose_detector: Optional[mp_vision.PoseLandmarker] = None,
) -> dict:
    if pose_detector is None:
        pose_detector = _get_default_detector()

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    detection = pose_detector.detect(mp_image)

    landmarks: list[dict] = []
    pose_groups = getattr(detection, "pose_landmarks", []) or []
    if pose_groups:
        for lm in pose_groups[0]:
            visibility = getattr(lm, "visibility", 1.0)
            if visibility is None:
                visibility = 1.0
            landmarks.append(
                {
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(getattr(lm, "z", 0.0) or 0.0),
                    "visibility": float(visibility),
                }
            )

    if landmarks:
        flags, behavior_score = _extract_pose_flags(landmarks)
    else:
        flags = {
            "pose_visible": False,
            "left_arm_bent": False,
            "right_arm_bent": False,
            "curved_posture": False,
            "asymmetric_shoulders": False,
        }
        behavior_score = 0.0

    return {
        "landmarks": landmarks,
        "flags": flags,
        "behavior_score": behavior_score,
    }


def _draw_landmarks(frame: np.ndarray, landmarks: list[dict]) -> None:
    h, w = frame.shape[:2]
    pixel_points: list[tuple[int, int] | None] = []
    for lm in landmarks:
        if lm.get("visibility", 0.0) < 0.3:
            pixel_points.append(None)
            continue
        x = int(round(lm["x"] * w))
        y = int(round(lm["y"] * h))
        if 0 <= x < w and 0 <= y < h:
            pixel_points.append((x, y))
        else:
            pixel_points.append(None)

              
    for start, end in _POSE_CONNECTIONS:
        if start >= len(pixel_points) or end >= len(pixel_points):
            continue
        p1, p2 = pixel_points[start], pixel_points[end]
        if p1 is None or p2 is None:
            continue
        cv2.line(frame, p1, p2, (0, 200, 0), 2)

            
    for p in pixel_points:
        if p is None:
            continue
        cv2.circle(frame, p, 3, (0, 0, 255), -1)


def annotate_frame_with_pose(frame_bgr: np.ndarray, pose_result: dict) -> np.ndarray:
    out = frame_bgr.copy()
    landmarks = pose_result.get("landmarks") or []
    if landmarks:
        _draw_landmarks(out, landmarks)

    flags = pose_result.get("flags", {})
    score = float(pose_result.get("behavior_score", 0.0))
    lines = [f"Pose score: {score:.1f}"]
    for name, active in flags.items():
        if active and name != "pose_visible":
            lines.append(f"- {name}")

    y0 = 25
    for i, text in enumerate(lines):
        cv2.putText(
            out,
            text,
            (10, y0 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 255),
            2,
        )
    return out


def detect_pose(video_path: str, output_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        print(f"Erro ao criar o VideoWriter em: {output_path}")
        return

    try:
        for _ in tqdm(range(total_frames), desc="Processando video (pose)"):
            ret, frame = cap.read()
            if not ret:
                break
            result = process_frame_bgr(frame)
            annotated = annotate_frame_with_pose(frame, result)
            out.write(annotated)
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    input_video_path = os.path.join(project_root, "exame_paciente_01.mp4")
    output_video_path = os.path.join(project_root, "video_pose_processado.mp4")
    detect_pose(input_video_path, output_video_path)


if __name__ == "__main__":
    main()
