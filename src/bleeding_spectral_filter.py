
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SpectralConfig:
    min_rg_ratio: float = 1.12
    min_rb_ratio: float = 1.10
    min_red_mean: float = 55.0
    min_pixels: int = 120
    eps: float = 1e-6


@dataclass(frozen=True)
class SpectralDecision:
    is_blood_like: bool
    rg_ratio: float
    rb_ratio: float
    red_mean: float
    pixel_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_blood_like": self.is_blood_like,
            "rg_ratio": self.rg_ratio,
            "rb_ratio": self.rb_ratio,
            "red_mean": self.red_mean,
            "pixel_count": self.pixel_count,
        }


def evaluate_bleeding_roi(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    config: SpectralConfig | None = None,
) -> SpectralDecision:
    if config is None:
        config = SpectralConfig()
    if frame_bgr is None or frame_bgr.size == 0:
        return SpectralDecision(False, 0.0, 0.0, 0.0, 0)

    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))

    if x2 <= x1 or y2 <= y1:
        return SpectralDecision(False, 0.0, 0.0, 0.0, 0)

    roi = frame_bgr[y1:y2, x1:x2]
    pixel_count = int(roi.shape[0] * roi.shape[1])
    if pixel_count < config.min_pixels:
        return SpectralDecision(False, 0.0, 0.0, 0.0, pixel_count)

                                
    blue = roi[:, :, 0].astype(np.float32)
    green = roi[:, :, 1].astype(np.float32)
    red = roi[:, :, 2].astype(np.float32)

    red_mean = float(np.mean(red))
    green_mean = float(np.mean(green))
    blue_mean = float(np.mean(blue))

    rg_ratio = red_mean / (green_mean + config.eps)
    rb_ratio = red_mean / (blue_mean + config.eps)

    is_blood_like = bool(
        rg_ratio >= config.min_rg_ratio
        and rb_ratio >= config.min_rb_ratio
        and red_mean >= config.min_red_mean
    )

    return SpectralDecision(
        is_blood_like=is_blood_like,
        rg_ratio=float(rg_ratio),
        rb_ratio=float(rb_ratio),
        red_mean=red_mean,
        pixel_count=pixel_count,
    )

