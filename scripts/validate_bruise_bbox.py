"""Valida bruise_score vs bruise_bbox em frames com rosto (caso mais recente)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.pose_detect.pose_detection_video import process_frame_bgr
from models.score_pain import analyze_pain_in_frame
from src.pipeline_video import (
    _build_patient_roi_mask,
    _build_ultra_sensitive_bruise_mask,
    _estimate_face_visual_aggression,
    _locate_face_bruise_bbox,
)

SAMPLE_EVERY = 15
MAX_FRAMES = 900


def _latest_case_with_video() -> tuple[str, Path]:
    demo = PROJECT_ROOT / "data" / "demo"
    best: tuple[str, str, Path] | None = None
    for case_dir in demo.iterdir():
        if not case_dir.is_dir():
            continue
        status_path = case_dir / "status.json"
        video_path = case_dir / "inputs" / "video_input.mp4"
        if not status_path.is_file() or not video_path.is_file():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        updated = str(status.get("updated_at", ""))
        if best is None or updated > best[0]:
            best = (updated, case_dir.name, video_path)
    if best is None:
        raise FileNotFoundError("Nenhum caso com video_input.mp4 em data/demo")
    return best[1], best[2]


def main() -> int:
    case_id, video_path = _latest_case_with_video()
    out_dir = PROJECT_ROOT / "data" / "demo" / case_id / "outputs" / "bruise_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "bruise_bbox_validation.log"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERRO: não abriu vídeo: {video_path}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    lines: list[str] = []
    lines.append(f"case_id={case_id}")
    lines.append(f"video={video_path}")
    lines.append(f"fps={fps:.2f} total_frames={total}")
    lines.append("frame_idx | n_faces | face | bruise_score | bruise_detected | bruise_bbox | mask_px")
    lines.append("-" * 95)

    stats = {
        "frames_sampled": 0,
        "frames_with_face": 0,
        "bbox_none_score_pos": 0,
        "bbox_ok_score_low": 0,
        "both_ok": 0,
    }
    debug_saved = False

    frame_idx = 0
    while frame_idx < MAX_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % SAMPLE_EVERY != 0:
            frame_idx += 1
            continue

        stats["frames_sampled"] += 1
        pain = analyze_pain_in_frame(frame)
        faces = pain.get("faces") or []
        n_faces = len(faces)

        if n_faces == 0:
            lines.append(f"{frame_idx:6d} | 0 | - | - | - | - | -")
            frame_idx += 1
            continue

        stats["frames_with_face"] += 1
        pose = process_frame_bgr(frame)
        patient_roi_mask = _build_patient_roi_mask(
            frame.shape, pose.get("landmarks", [])
        )

        for fi, face in enumerate(faces):
            payload = _estimate_face_visual_aggression(
                frame,
                face,
                patient_roi_mask=patient_roi_mask,
                blood_score_threshold=0.70,
            )
            x = int(face.get("x", 0) or 0)
            y = int(face.get("y", 0) or 0)
            w = int(face.get("w", 0) or 0)
            h = int(face.get("h", 0) or 0)
            face_roi_mask = None
            if (
                patient_roi_mask is not None
                and y >= 0
                and x >= 0
                and (y + h) <= patient_roi_mask.shape[0]
                and (x + w) <= patient_roi_mask.shape[1]
            ):
                face_roi_mask = patient_roi_mask[y : y + h, x : x + w]

            bruise_score = float(payload.get("bruise_score", 0.0) or 0.0)
            bruise_detected = bool(payload.get("bruise_detected", False))
            bruise_bbox = _locate_face_bruise_bbox(
                frame,
                (x, y, x + w, y + h),
                face_roi_mask=face_roi_mask,
                min_score_hint=bruise_score,
            )

            mask_px = 0
            if h > 0 and w > 0:
                roi = frame[y : y + h, x : x + w]
                valid = np.ones((h, w), dtype=np.uint8) * 255
                if face_roi_mask is not None:
                    valid = np.where(face_roi_mask > 0, 255, 0).astype(np.uint8)
                mask_px = int(
                    np.count_nonzero(_build_ultra_sensitive_bruise_mask(roi, valid))
                )

            bbox_str = "None" if bruise_bbox is None else str(bruise_bbox)
            lines.append(
                f"{frame_idx:6d} | {n_faces} | {fi} | {bruise_score:.4f} | "
                f"{int(bruise_detected)} | {bbox_str} | {mask_px}"
            )

            if bruise_bbox is None and bruise_score >= 0.01:
                stats["bbox_none_score_pos"] += 1
            elif bruise_bbox is not None and bruise_score < 0.01:
                stats["bbox_ok_score_low"] += 1
            elif bruise_bbox is not None:
                stats["both_ok"] += 1

            if not debug_saved and bruise_bbox is not None:
                dbg = frame.copy()
                hx1, hy1, hx2, hy2 = bruise_bbox
                cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(dbg, (hx1, hy1), (hx2, hy2), (255, 0, 255), 3)
                cv2.putText(
                    dbg,
                    f"score={bruise_score:.3f} mask_px={mask_px}",
                    (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 0, 255),
                    2,
                )
                out_img = out_dir / f"frame_{frame_idx:05d}_face{fi}_bruise_ok.jpg"
                cv2.imwrite(str(out_img), dbg)
                debug_saved = True

            if not debug_saved and mask_px > 0 and bruise_bbox is None:
                dbg = frame.copy()
                cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(
                    dbg,
                    f"mask_px={mask_px} bbox=None score={bruise_score:.3f}",
                    (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                )
                out_img = out_dir / f"frame_{frame_idx:05d}_face{fi}_mask_no_bbox.jpg"
                cv2.imwrite(str(out_img), dbg)
                debug_saved = True

        frame_idx += 1

    cap.release()
    lines.append("-" * 95)
    lines.append(f"stats={json.dumps(stats, ensure_ascii=False)}")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Caso: {case_id}")
    print(f"Vídeo: {video_path}")
    print(f"Log: {log_path}")
    print(json.dumps(stats, indent=2))
    print("\n--- primeiras linhas com rosto ---")
    for line in lines:
        if "|" in line and "face |" not in line and "---" not in line:
            parts = line.split("|")
            if len(parts) > 2 and parts[1].strip() not in ("0", "-", " n_faces "):
                print(line)
        if line.startswith("stats="):
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
