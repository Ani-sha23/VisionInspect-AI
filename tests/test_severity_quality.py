import pytest

from visioninspect.quality_engine import ACCEPT, REJECT, REVIEW, decide
from visioninspect.schemas import BBox, Defect
from visioninspect.severity import compute_severity, score_to_label


def make_defect(class_id, area, x=0, y=0):
    side = max(1, int(area ** 0.5))
    return Defect(class_id, f"class_{class_id}", BBox(x, y, side, side), area, 0.9)


def test_no_defects_scores_zero_and_accepts(cfg):
    severity = compute_severity([], 256, 1600, cfg)
    assert severity.score == 0.0 and severity.label == "NONE"
    assert decide(severity, [], cfg).decision == ACCEPT


def test_score_increases_with_defect_area(cfg):
    small = compute_severity([make_defect(1, 500)], 256, 1600, cfg)
    large = compute_severity([make_defect(1, 50_000)], 256, 1600, cfg)
    assert large.score > small.score


def test_heavier_class_scores_higher_for_equal_area(cfg):
    light = compute_severity([make_defect(1, 20_000)], 256, 1600, cfg)
    heavy = compute_severity([make_defect(4, 20_000)], 256, 1600, cfg)
    assert heavy.score > light.score


def test_score_is_bounded_at_100(cfg):
    huge = [make_defect(4, 256 * 1600) for _ in range(20)]
    assert compute_severity(huge, 256, 1600, cfg).score <= 100.0


def test_labels_follow_configured_bands(cfg):
    assert score_to_label(0.0, cfg) == "NONE"
    assert score_to_label(1.0, cfg) == "MINOR"
    assert score_to_label(99.0, cfg) == "CRITICAL"


def test_critical_class_never_silently_accepted(cfg):
    defects = [make_defect(4, 300)]
    severity = compute_severity(defects, 256, 1600, cfg)
    assert decide(severity, defects, cfg).decision in {REVIEW, REJECT}


def test_large_defect_is_rejected_with_reasons(cfg):
    defects = [make_defect(4, 120_000)]
    severity = compute_severity(defects, 256, 1600, cfg)
    decision = decide(severity, defects, cfg)
    assert decision.decision == REJECT
    assert decision.reasons


def test_uncalibrated_config_is_flagged_in_reasons(cfg):
    defects = [make_defect(1, 5_000)]
    severity = compute_severity(defects, 256, 1600, cfg)
    reasons = " ".join(decide(severity, defects, cfg).reasons)
    if not severity.calibrated:
        assert "UNCALIBRATED" in reasons
