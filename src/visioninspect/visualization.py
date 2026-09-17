"""Stage 7: visual output. Draws mask overlays, boxes, and a decision banner."""

from __future__ import annotations

import os
from typing import List, Optional

import cv2
import numpy as np

from .schemas import Defect, InspectionResult

# BGR colours, one per class id (0 = unclassified baseline region).
CLASS_COLORS = {
    0: (200, 200, 200),
    1: (0, 200, 255),
    2: (0, 255, 0),
    3: (255, 140, 0),
    4: (0, 0, 255),
}

DECISION_COLORS = {"ACCEPT": (0, 170, 0), "REVIEW": (0, 200, 255), "REJECT": (0, 0, 220)}


def draw_overlay(image_bgr: np.ndarray, defects: List[Defect], alpha: float = 0.45) -> np.ndarray:
    canvas = image_bgr.copy()
    tint = canvas.copy()

    for defect in defects:
        color = CLASS_COLORS.get(defect.class_id, (255, 255, 255))
        if defect.mask is not None:
            mask = defect.mask
            if mask.shape[:2] != canvas.shape[:2]:
                mask = cv2.resize(mask, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST)
            tint[mask > 0] = color

    blended = cv2.addWeighted(tint, alpha, canvas, 1 - alpha, 0)

    for defect in defects:
        color = CLASS_COLORS.get(defect.class_id, (255, 255, 255))
        x, y, w, h = defect.bbox.x, defect.bbox.y, defect.bbox.w, defect.bbox.h
        cv2.rectangle(blended, (x, y), (x + w, y + h), color, 2)
        label = f"{defect.class_name} {defect.confidence:.2f}"
        cv2.putText(blended, label, (x, max(14, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return blended


def add_banner(image_bgr: np.ndarray, result: InspectionResult) -> np.ndarray:
    decision = result.quality.decision if result.quality else "UNKNOWN"
    severity = result.severity
    color = DECISION_COLORS.get(decision, (120, 120, 120))
    banner_height = 46
    banner = np.full((banner_height, image_bgr.shape[1], 3), 32, dtype=np.uint8)
    cv2.rectangle(banner, (0, 0), (10, banner_height), color, -1)

    text = f"{result.image_id}  |  {decision}"
    if severity:
        text += f"  |  severity {severity.score:.1f} ({severity.label})  |  {severity.defect_count} region(s)"
    text += f"  |  backend: {result.backend}"
    cv2.putText(banner, text, (20, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    return np.vstack([banner, image_bgr])


def save_visualization(
    image_bgr: np.ndarray,
    result: InspectionResult,
    out_dir: str,
    alpha: float = 0.45,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    overlay = draw_overlay(image_bgr, result.defects, alpha)
    framed = add_banner(overlay, result)
    out_path = os.path.join(out_dir, f"{os.path.splitext(result.image_id)[0]}_annotated.png")
    if not cv2.imwrite(out_path, framed):
        raise IOError(f"Could not write visualization to {out_path}")
    return out_path


def plot_class_distribution(counts: dict, out_path: str, title: str = "Defect class distribution") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    keys = [str(k) for k in counts.keys()]
    values = list(counts.values())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(keys, values, color="#3b7dd8")
    ax.set_xlabel("Defect class")
    ax.set_ylabel("Annotation count")
    ax.set_title(title)
    for index, value in enumerate(values):
        ax.text(index, value, str(value), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
