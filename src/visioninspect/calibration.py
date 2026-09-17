"""Derive severity band edges from the training data instead of guessing them.

Every training image is scored with the current severity formula; the band edges
are then placed at configurable percentiles of the resulting score distribution
over defective images. This makes 'MINOR / MODERATE / SEVERE' mean something
measurable: a SEVERE sheet is worse than (by default) 90% of defective sheets in
the training set.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from .config_loader import Config, class_weights
from .dataset import build_image_index, load_annotations
from .exceptions import DataError
from .logger import get_logger
from .rle import rle_to_mask

logger = get_logger(__name__)


def score_from_masks(
    class_areas: Dict[int, int], image_area: int, count: int, cfg: Config
) -> float:
    if not class_areas or count == 0:
        return 0.0
    weights = class_weights(cfg)
    max_weight = max(weights.values())
    class_factor = max(weights.get(c, 1.0) for c in class_areas) / max_weight
    area_fraction = sum(class_areas.values()) / max(1, image_area)
    area_term = min(1.0, area_fraction / float(cfg.get("severity.area_scale")))
    count_term = min(1.0, count / float(cfg.get("severity.count_saturation")))
    raw = float(cfg.get("severity.weight_area")) * area_term + float(cfg.get("severity.weight_count")) * count_term
    return 100.0 * class_factor * raw


def collect_scores(
    cfg: Config,
    split_csv: Optional[str] = None,
    limit: Optional[int] = None,
    image_shape=(256, 1600),
) -> List[float]:
    index = build_image_index(load_annotations(cfg.get("data.train_csv")))
    if split_csv and os.path.isfile(split_csv):
        image_ids = pd.read_csv(split_csv)["ImageId"].astype(str).tolist()
    else:
        image_ids = sorted(index)
    if limit:
        image_ids = image_ids[:limit]
    if not image_ids:
        raise DataError("No image ids available for calibration.")

    height, width = image_shape
    scores: List[float] = []
    for image_id in image_ids:
        classes = index.get(image_id, {})
        if not classes:
            continue
        areas = {int(c): int(rle_to_mask(rle, (height, width)).sum()) for c, rle in classes.items()}
        scores.append(score_from_masks(areas, height * width, len(areas), cfg))
    return scores


def derive_thresholds(scores: List[float], percentiles=(50, 80, 95)) -> Dict[str, float]:
    if not scores:
        raise DataError("Cannot calibrate on an empty score distribution.")
    array = np.asarray(scores, dtype=np.float64)
    minor, moderate, severe = (float(np.percentile(array, p)) for p in percentiles)
    # Guarantee strictly increasing edges even on a degenerate distribution.
    moderate = max(moderate, minor + 0.1)
    severe = max(severe, moderate + 0.1)
    return {
        "minor_max": round(minor, 2),
        "moderate_max": round(moderate, 2),
        "severe_max": round(severe, 2),
    }


def apply_to_config(thresholds: Dict[str, float], config_path: str) -> str:
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    data["severity"]["thresholds"] = thresholds
    data["severity"]["calibrated"] = True
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    logger.info("Wrote calibrated thresholds %s to %s", thresholds, config_path)
    return config_path


def describe(scores: List[float], thresholds: Dict[str, float]) -> str:
    array = np.asarray(scores, dtype=np.float64)
    return "\n".join([
        "# Severity calibration (measured)",
        "",
        f"- Defective images scored: **{array.size}**",
        f"- Score range: {array.min():.2f} – {array.max():.2f}",
        f"- Mean / median: {array.mean():.2f} / {np.median(array):.2f}",
        "",
        "| Band | Upper edge | Percentile |",
        "|---|---|---|",
        f"| MINOR | {thresholds['minor_max']} | 50th |",
        f"| MODERATE | {thresholds['moderate_max']} | 80th |",
        f"| SEVERE | {thresholds['severe_max']} | 95th |",
        "| CRITICAL | above SEVERE | top 5% |",
        "",
    ])
