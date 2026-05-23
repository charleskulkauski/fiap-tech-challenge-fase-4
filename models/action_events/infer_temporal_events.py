from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import cv2
import torch
import torch.nn as nn


class Tiny3DClassifier(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        z = z.flatten(1)
        return self.head(z)


def _default_state() -> dict[str, Any]:
    return {"label": "", "confidence": 0.0, "class_id": -1, "available": False}


class TemporalEventsInferencer:

    def __init__(
        self,
        model_root: str | Path,
        *,
        tasks: tuple[str, ...] = ("action", "bleeding", "smoke"),
        device: str | None = None,
    ) -> None:
        self.model_root = Path(model_root).expanduser().resolve()
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tasks = tasks
        self.models: dict[str, dict[str, Any]] = {}
        self.max_frames_per_clip = 1
        self._frame_buffer: deque[Any] = deque(maxlen=1)
        self.last_predictions: dict[str, dict[str, Any]] = {
            task: _default_state() for task in tasks
        }
        self._load_models()

    def _load_models(self) -> None:
        for task in self.tasks:
            ckpt_path = self.model_root / task / "best.pt"
            if not ckpt_path.is_file():
                continue
            checkpoint = torch.load(str(ckpt_path), map_location=self.device)
            class_names = [str(x) for x in checkpoint.get("class_names", [])]
            if not class_names:
                continue
            frames_per_clip = int(checkpoint.get("frames_per_clip", 16))
            image_size = int(checkpoint.get("image_size", 112))
            model = Tiny3DClassifier(n_classes=len(class_names)).to(self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            self.models[task] = {
                "model": model,
                "class_names": class_names,
                "frames_per_clip": frames_per_clip,
                "image_size": image_size,
            }
            self.max_frames_per_clip = max(self.max_frames_per_clip, frames_per_clip)
        self._frame_buffer = deque(maxlen=max(self.max_frames_per_clip, 1))

    @property
    def has_models(self) -> bool:
        return bool(self.models)

    def update(self, frame_bgr: Any) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._frame_buffer.append(rgb)

    def _build_clip(self, frames_per_clip: int, image_size: int) -> torch.Tensor:
        if not self._frame_buffer:
            return torch.zeros(3, frames_per_clip, image_size, image_size)
        frames = list(self._frame_buffer)
        if len(frames) >= frames_per_clip:
            idx = torch.linspace(0, len(frames) - 1, steps=frames_per_clip).long().tolist()
            selected = [frames[int(i)] for i in idx]
        else:
            selected = frames[:]
            while len(selected) < frames_per_clip:
                selected.append(frames[-1])

        tensors: list[torch.Tensor] = []
        for frame_rgb in selected:
            resized = cv2.resize(frame_rgb, (image_size, image_size))
            t = torch.from_numpy(resized).float() / 255.0
            t = t.permute(2, 0, 1)
            tensors.append(t)
        return torch.stack(tensors, dim=1)

    def predict(self) -> dict[str, dict[str, Any]]:
        predictions: dict[str, dict[str, Any]] = {
            task: _default_state() for task in self.tasks
        }
        if not self.models:
            self.last_predictions = predictions
            return predictions

        with torch.no_grad():
            for task in self.tasks:
                payload = self.models.get(task)
                if payload is None:
                    continue
                clip = self._build_clip(
                    frames_per_clip=int(payload["frames_per_clip"]),
                    image_size=int(payload["image_size"]),
                )
                logits = payload["model"](clip.unsqueeze(0).to(self.device))
                probs = torch.softmax(logits[0], dim=0)
                class_id = int(torch.argmax(probs).item())
                confidence = float(probs[class_id].item())
                class_names = payload["class_names"]
                label = class_names[class_id] if 0 <= class_id < len(class_names) else ""
                predictions[task] = {
                    "label": str(label),
                    "confidence": confidence,
                    "class_id": class_id,
                    "available": True,
                }
        self.last_predictions = predictions
        return predictions
