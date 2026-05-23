
from .detect_instruments import (
    annotate_frame_with_detections,
    detect_instruments_in_frame,
    get_default_detector,
    is_red_bbox_class_id,
)

__all__ = [
    "annotate_frame_with_detections",
    "detect_instruments_in_frame",
    "get_default_detector",
    "is_red_bbox_class_id",
]
