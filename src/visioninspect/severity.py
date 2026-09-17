"""Stage 5: severity scoring.

score = 100 * class_factor * (w_area * area_term + w_count * count_term)

  area_term  = min(1, defect_area_fraction / area_scale)
  count_term = min(1, defect_count / count_saturation)
  class_factor = max class weight present, normalised by the largest configured weight

The weights and scales are configuration, not constants in code, and the
thresholds that turn a score into a band are meant to come from a percentile
analysis of the training set (scripts/calibrate_severity.py). Until that has
been run, `severity.calibrated: false` marks every result as UNCALIBRATED so no
report can quietly present provisional bands as measured ones.
"""

from __future__ import annotations

from typing import List

from .classifier import per_class_area
from .config_loader import Config, class_weights
from .schemas import Defect, SeverityResult


def compute_severity(defects: List[Defect], image_h: int, image_w: int, cfg: Config) -> SeverityResult:
    image_area = max(1, int(image_h) * int(image_w))
    total_area = int(sum(d.area_px for d in defects))
    area_fraction = total_area / image_area
    count = len(defects)

    if count == 0:
        return SeverityResult(
            score=0.0,
            label="NONE",
            total_defect_area_px=0,
            area_fraction=0.0,
            defect_count=0,
            per_class_area_px={},
            calibrated=bool(cfg.get("severity.calibrated")),
        )

    weights = class_weights(cfg)
    max_weight = max(weights.values())
    present = [weights.get(d.class_id, 1.0) for d in defects]
    class_factor = max(present) / max_weight if max_weight else 1.0

    area_term = min(1.0, area_fraction / float(cfg.get("severity.area_scale")))
    count_term = min(1.0, count / float(cfg.get("severity.count_saturation")))
    raw = float(cfg.get("severity.weight_area")) * area_term + float(cfg.get("severity.weight_count")) * count_term
    score = 100.0 * class_factor * raw

    return SeverityResult(
        score=round(score, 2),
        label=score_to_label(score, cfg),
        total_defect_area_px=total_area,
        area_fraction=area_fraction,
        defect_count=count,
        per_class_area_px=per_class_area(defects),
        calibrated=bool(cfg.get("severity.calibrated")),
    )


def score_to_label(score: float, cfg: Config) -> str:
    thresholds = cfg.get("severity.thresholds")
    if score <= 0:
        return "NONE"
    if score <= float(thresholds["minor_max"]):
        return "MINOR"
    if score <= float(thresholds["moderate_max"]):
        return "MODERATE"
    if score <= float(thresholds["severe_max"]):
        return "SEVERE"
    return "CRITICAL"
