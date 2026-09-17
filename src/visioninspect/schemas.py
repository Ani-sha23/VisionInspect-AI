"""Dataclasses that define the contract between pipeline stages.

Every stage consumes and returns one of these types, so a stage can be swapped
out (for example, the classical baseline detector for the trained U-Net) without
any other stage changing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np

CLASS_NAMES: Dict[int, str] = {
    1: "class_1",
    2: "class_2",
    3: "class_3",
    4: "class_4",
}


@dataclass
class BBox:
    """Axis-aligned bounding box in pixel coordinates of the original image."""

    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return int(self.w * self.h)

    def as_xyxy(self) -> tuple:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass
class Defect:
    """One connected defect region produced by the detection stage."""

    class_id: int
    class_name: str
    bbox: BBox
    area_px: int
    confidence: float
    mask: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "bbox": self.bbox.to_dict(),
            "area_px": int(self.area_px),
            "confidence": round(float(self.confidence), 4),
        }


@dataclass
class SeverityResult:
    """Output of the severity scoring stage."""

    score: float
    label: str
    total_defect_area_px: int
    area_fraction: float
    defect_count: int
    per_class_area_px: Dict[int, int] = field(default_factory=dict)
    calibrated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(float(self.score), 2),
            "label": self.label,
            "total_defect_area_px": int(self.total_defect_area_px),
            "area_fraction": round(float(self.area_fraction), 6),
            "defect_count": int(self.defect_count),
            "per_class_area_px": {int(k): int(v) for k, v in self.per_class_area_px.items()},
            "calibrated": bool(self.calibrated),
        }


@dataclass
class QualityDecision:
    """Output of the quality decision engine."""

    decision: str  # ACCEPT | REVIEW | REJECT
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"decision": self.decision, "reasons": list(self.reasons)}


@dataclass
class InspectionResult:
    """Full result for a single inspected image."""

    image_id: str
    image_path: str
    height: int
    width: int
    defects: List[Defect] = field(default_factory=list)
    present_classes: List[int] = field(default_factory=list)
    severity: Optional[SeverityResult] = None
    quality: Optional[QualityDecision] = None
    backend: str = "unknown"  # "unet" | "baseline"
    elapsed_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "height": self.height,
            "width": self.width,
            "backend": self.backend,
            "elapsed_ms": round(float(self.elapsed_ms), 2),
            "present_classes": list(self.present_classes),
            "defects": [d.to_dict() for d in self.defects],
            "severity": self.severity.to_dict() if self.severity else None,
            "quality": self.quality.to_dict() if self.quality else None,
            "warnings": list(self.warnings),
        }
