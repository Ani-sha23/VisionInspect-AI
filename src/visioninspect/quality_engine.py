"""Stage 6: quality decision.

Turns a severity score plus the set of defect classes into one of three
actionable outcomes, and records why. The reasons list is the audit trail: an
inspector should never see a REJECT without being told what triggered it.
"""

from __future__ import annotations

from typing import List

from .config_loader import Config
from .schemas import Defect, QualityDecision, SeverityResult

ACCEPT = "ACCEPT"
REVIEW = "REVIEW"
REJECT = "REJECT"


def decide(severity: SeverityResult, defects: List[Defect], cfg: Config) -> QualityDecision:
    reasons: List[str] = []
    critical_classes = {int(c) for c in cfg.get("quality.critical_classes")}
    critical_area_fraction = float(cfg.get("quality.critical_area_fraction"))
    accept_max = float(cfg.get("quality.accept_max_score"))
    reject_min = float(cfg.get("quality.reject_min_score"))

    image_area_present = severity.area_fraction > 0
    present = {d.class_id for d in defects}
    critical_present = sorted(present & critical_classes)

    decision = ACCEPT

    if severity.defect_count == 0:
        reasons.append("No defect regions detected.")
        return QualityDecision(ACCEPT, reasons)

    reasons.append(
        f"{severity.defect_count} defect region(s) covering "
        f"{severity.area_fraction * 100:.2f}% of the surface; severity score {severity.score:.1f}."
    )

    if severity.score >= reject_min:
        decision = REJECT
        reasons.append(f"Severity score {severity.score:.1f} >= reject threshold {reject_min:.1f}.")
    elif severity.score > accept_max:
        decision = REVIEW
        reasons.append(f"Severity score {severity.score:.1f} exceeds accept threshold {accept_max:.1f}.")

    if critical_present:
        critical_area = sum(d.area_px for d in defects if d.class_id in critical_classes)
        total_pixels = severity.total_defect_area_px / severity.area_fraction if image_area_present else 1
        fraction = critical_area / max(1.0, total_pixels)
        if fraction >= critical_area_fraction:
            decision = REJECT
            reasons.append(
                f"Critical class {critical_present} covers {fraction * 100:.2f}% of the surface "
                f"(limit {critical_area_fraction * 100:.2f}%)."
            )
        elif decision == ACCEPT:
            decision = REVIEW
            reasons.append(f"Critical class {critical_present} present; manual review required.")

    if 0 in present and decision == ACCEPT:
        decision = REVIEW
        reasons.append("Unclassified regions from the baseline detector cannot be auto-accepted.")

    if not severity.calibrated:
        reasons.append("Severity bands are UNCALIBRATED; decision is provisional.")

    return QualityDecision(decision, reasons)
