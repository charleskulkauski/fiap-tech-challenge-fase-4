
from __future__ import annotations

import os
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deepface import DeepFace
from tqdm import tqdm


def calcular_score_dor(emocoes: dict[str, float]) -> float:
    score = (
        emocoes.get("sad", 0.0) * 0.45
        + emocoes.get("angry", 0.0) * 0.45
        + emocoes.get("fear", 0.0) * 0.30
    )
    return float(score)


def analyze_pain_in_frame(
    frame_bgr: np.ndarray,
    scale: float = 0.5,
) -> dict[str, Any]:
    h, w = frame_bgr.shape[:2]
    if 0.0 < scale < 1.0 and min(h, w) > 100:
        small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
        inv_scale = 1.0 / scale
    else:
        small = frame_bgr
        inv_scale = 1.0

    try:
        results = DeepFace.analyze(
            small,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )
    except Exception:
        return {"score": None, "faces": []}

    if isinstance(results, dict):
        results = [results]

    faces_out: list[dict[str, Any]] = []
    best_score: float | None = None
    for face in results:
        region = face.get("region") or {}
        fw = int(round(region.get("w", 0) * inv_scale))
        fh = int(round(region.get("h", 0) * inv_scale))
        if fw <= 0 or fh <= 0:
            continue
        x = int(round(region.get("x", 0) * inv_scale))
        y = int(round(region.get("y", 0) * inv_scale))
        emotions = face.get("emotion") or {}
        score = calcular_score_dor(emotions)
        faces_out.append(
            {
                "x": max(0, x),
                "y": max(0, y),
                "w": fw,
                "h": fh,
                "score": score,
                "emotions": emotions,
            }
        )
        if best_score is None or score > best_score:
            best_score = score

    return {"score": best_score, "faces": faces_out}


def processar_arquivo_gravado(video_path: str, output_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro: Não foi possível abrir o arquivo.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    historico_dor: list[float] = []

    try:
        for _ in tqdm(range(total_frames), desc="Analisando Arquivo"):
            ret, frame = cap.read()
            if not ret:
                break

            payload = analyze_pain_in_frame(frame, scale=1.0)
            faces = payload["faces"]

            if faces:
                                                                
                face = faces[0]
                historico_dor.append(face["score"])
                media_estavel = float(np.mean(historico_dor[-10:]))

                if media_estavel > 40:
                    status, cor = "DOR SEVERA", (0, 0, 255)
                elif media_estavel > 25:
                    status, cor = "DESCONFORTO MODERADO", (0, 165, 255)
                else:
                    status, cor = "ESTAVEL", (0, 255, 0)

                x, y, w, h = face["x"], face["y"], face["w"], face["h"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), cor, 2)
                cv2.putText(
                    frame,
                    f"{status} ({media_estavel:.1f}%)",
                    (x, max(15, y - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cor,
                    2,
                )
            else:
                historico_dor.append(0.0)

            out.write(frame)
    finally:
        cap.release()
        out.release()

    pd.DataFrame(historico_dor, columns=["pain_score"]).to_csv(
        output_path.replace(".mp4", "_relatorio.csv"), index=False
    )

    plt.figure(figsize=(10, 4))
    plt.plot(historico_dor, color="red")
    plt.title("Variação da Intensidade de Dor ao Longo do Vídeo")
    plt.xlabel("Frames")
    plt.ylabel("Score de Dor %")
    plt.savefig(output_path.replace(".mp4", "_grafico.png"))
    plt.close()

    print(f"\nConcluído! Vídeo, CSV e Gráfico em: {os.path.dirname(output_path)}")
