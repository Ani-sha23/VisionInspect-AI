"""Segmentation and classification metrics used by the evaluation stage."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def dice_coefficient(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    """Dice on binary masks. An empty prediction on an empty target scores 1.0,
    which is the Severstal convention (correctly calling a clean image clean)."""
    pred = (np.asarray(pred) > 0).astype(np.float64)
    target = (np.asarray(target) > 0).astype(np.float64)
    denominator = pred.sum() + target.sum()
    if denominator == 0:
        return 1.0
    return float((2.0 * (pred * target).sum() + eps) / (denominator + eps))


def iou(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    pred = (np.asarray(pred) > 0)
    target = (np.asarray(target) > 0)
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 1.0
    return float((np.logical_and(pred, target).sum() + eps) / (union + eps))


def confusion_counts(pred_flags: List[int], true_flags: List[int]) -> Dict[str, int]:
    """Binary TP/FP/FN/TN for one class across a set of images."""
    tp = fp = fn = tn = 0
    for p, t in zip(pred_flags, true_flags):
        if p and t:
            tp += 1
        elif p and not t:
            fp += 1
        elif not p and t:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def precision_recall_f1(counts: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def aggregate_dice(per_image: List[float]) -> Dict[str, float]:
    if not per_image:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    array = np.asarray(per_image, dtype=np.float64)
    return {"mean": round(float(array.mean()), 4), "std": round(float(array.std()), 4), "n": int(array.size)}
