"""Classical-CV fallback localiser.

This stage exists so the pipeline is demonstrably end-to-end runnable with zero
trained weights. It finds *anomalous regions*, it does not name them: every
defect it emits carries class_id 0 ("unclassified"). Treating its output as
class-labelled would be dishonest, so the schema keeps the distinction explicit.

Method: CLAHE-enhanced grayscale -> median blur -> adaptive threshold (local
illumination invariance) -> morphological open/close -> connected components
filtered by area.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from .config_loader import Config
from .schemas import BBox, Defect


def detect_regions(image_bgr: np.ndarray, cfg: Config) -> List[Defect]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, int(cfg.get("baseline.blur_kernel")))

    binary = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=int(cfg.get("baseline.block_size")),
        C=float(cfg.get("baseline.c_constant")),
    )

    k = int(cfg.get("baseline.morph_kernel"))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    min_area = int(cfg.get("baseline.min_area"))
    count, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)

    defects: List[Defect] = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        component = (labels == label).astype(np.uint8)
        # Confidence is a heuristic: how strongly the region deviates from the
        # image median intensity, squashed into (0, 1). It is not a probability.
        region_mean = float(gray[component > 0].mean())
        deviation = abs(region_mean - float(np.median(gray))) / 255.0
        confidence = float(min(0.99, 0.30 + deviation * 2.0))
        defects.append(
            Defect(
                class_id=0,
                class_name="unclassified",
                bbox=BBox(int(x), int(y), int(w), int(h)),
                area_px=int(area),
                confidence=confidence,
                mask=component,
            )
        )
    defects.sort(key=lambda d: d.area_px, reverse=True)
    return defects
