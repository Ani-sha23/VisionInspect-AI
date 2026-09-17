"""Stage 4: per-image multi-label classification derived from detections.

No separate trained classifier: a class is 'present' when the detector produced
at least one surviving instance of it. This keeps classification and
localisation consistent by construction (see docs/adr/ADR-001.md).
"""

from __future__ import annotations

from typing import Dict, List

from .schemas import Defect


def present_classes(defects: List[Defect], min_confidence: float = 0.0) -> List[int]:
    classes = {d.class_id for d in defects if d.confidence >= min_confidence and d.class_id > 0}
    return sorted(classes)


def per_class_area(defects: List[Defect]) -> Dict[int, int]:
    areas: Dict[int, int] = {}
    for defect in defects:
        areas[defect.class_id] = areas.get(defect.class_id, 0) + int(defect.area_px)
    return dict(sorted(areas.items()))


def class_confidence(defects: List[Defect]) -> Dict[int, float]:
    """Highest-confidence instance per class, used as the image-level class score."""
    scores: Dict[int, float] = {}
    for defect in defects:
        scores[defect.class_id] = max(scores.get(defect.class_id, 0.0), float(defect.confidence))
    return dict(sorted(scores.items()))
