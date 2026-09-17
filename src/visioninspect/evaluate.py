"""Quantitative evaluation against ground-truth masks on a held-out split."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config_loader import Config
from .dataset import build_image_index, load_annotations
from .detector import DefectDetector
from .exceptions import DataError
from .logger import get_logger
from .metrics import aggregate_dice, confusion_counts, dice_coefficient, iou, precision_recall_f1
from .preprocessing import load_image, preprocess
from .rle import build_multilabel_mask

logger = get_logger(__name__)


def _predicted_stack(defects, num_classes: int, shape) -> np.ndarray:
    """Rasterise predicted defects back into a (C, H, W) binary stack."""
    height, width = shape
    stack = np.zeros((num_classes, height, width), dtype=np.uint8)
    for defect in defects:
        if defect.mask is None or defect.class_id < 1:
            continue
        channel = defect.class_id - 1
        if 0 <= channel < num_classes:
            stack[channel] = np.maximum(stack[channel], (defect.mask > 0).astype(np.uint8))
    return stack


def _class_agnostic(defects, shape) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for defect in defects:
        if defect.mask is not None:
            mask = np.maximum(mask, (defect.mask > 0).astype(np.uint8))
    return mask


def evaluate_split(
    cfg: Config,
    split_csv: str,
    backend: str = "auto",
    limit: Optional[int] = None,
) -> Dict:
    if not os.path.isfile(split_csv):
        raise DataError(f"Split file not found: {split_csv}. Run `prepare-data` first.")

    image_ids = pd.read_csv(split_csv)["ImageId"].astype(str).tolist()
    if limit:
        image_ids = image_ids[:limit]

    index = build_image_index(load_annotations(cfg.get("data.train_csv")))
    images_dir = cfg.get("data.images_dir")
    num_classes = int(cfg.get("model.num_classes"))

    detector = DefectDetector(cfg, backend=backend)
    per_class_dice: Dict[int, List[float]] = {c: [] for c in range(1, num_classes + 1)}
    per_class_iou: Dict[int, List[float]] = {c: [] for c in range(1, num_classes + 1)}
    agnostic_dice: List[float] = []
    pred_flags: Dict[int, List[int]] = {c: [] for c in range(1, num_classes + 1)}
    true_flags: Dict[int, List[int]] = {c: [] for c in range(1, num_classes + 1)}

    for position, image_id in enumerate(image_ids, start=1):
        path = os.path.join(images_dir, image_id)
        image = load_image(path)
        enhanced, tensor = preprocess(image, cfg)
        defects = detector.detect(enhanced, tensor)

        shape = (image.shape[0], image.shape[1])
        truth = build_multilabel_mask(index.get(image_id, {}), shape, num_classes)
        prediction = _predicted_stack(defects, num_classes, shape)

        for class_id in range(1, num_classes + 1):
            channel = class_id - 1
            per_class_dice[class_id].append(dice_coefficient(prediction[channel], truth[channel]))
            per_class_iou[class_id].append(iou(prediction[channel], truth[channel]))
            pred_flags[class_id].append(int(prediction[channel].any()))
            true_flags[class_id].append(int(truth[channel].any()))

        agnostic_dice.append(dice_coefficient(_class_agnostic(defects, shape), truth.max(axis=0)))

        if position % 25 == 0:
            logger.info("Evaluated %d/%d images", position, len(image_ids))

    classification = {}
    for class_id in range(1, num_classes + 1):
        counts = confusion_counts(pred_flags[class_id], true_flags[class_id])
        classification[class_id] = {**counts, **precision_recall_f1(counts)}

    macro_f1 = round(float(np.mean([v["f1"] for v in classification.values()])), 4)

    return {
        "backend": detector.backend,
        "split_csv": split_csv,
        "images_evaluated": len(image_ids),
        "per_class_dice": {c: aggregate_dice(v) for c, v in per_class_dice.items()},
        "per_class_iou": {c: aggregate_dice(v) for c, v in per_class_iou.items()},
        "class_agnostic_dice": aggregate_dice(agnostic_dice),
        "classification": classification,
        "macro_f1": macro_f1,
        "note": (
            "The baseline backend emits unclassified regions, so its per-class figures are 0 by "
            "construction. For that backend read class_agnostic_dice, which measures localisation only."
            if detector.backend == "baseline" else
            "Per-class Dice is computed at the configured mask_threshold on full-resolution masks."
        ),
    }


def write_evaluation(results: Dict, out_dir: str = os.path.join("docs", "results")) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "evaluation.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    lines = [
        "# Evaluation results (measured)",
        "",
        f"- Backend: `{results['backend']}`",
        f"- Split: `{results['split_csv']}`",
        f"- Images evaluated: **{results['images_evaluated']}**",
        f"- Macro F1 (image-level classification): **{results['macro_f1']}**",
        f"- Class-agnostic Dice: **{results['class_agnostic_dice']['mean']}** "
        f"(std {results['class_agnostic_dice']['std']})",
        "",
        "## Segmentation, per class",
        "",
        "| Class | Mean Dice | Std | Mean IoU |",
        "|---|---|---|---|",
    ]
    for class_id, stats in results["per_class_dice"].items():
        iou_stats = results["per_class_iou"][class_id]
        lines.append(f"| {class_id} | {stats['mean']} | {stats['std']} | {iou_stats['mean']} |")

    lines += ["", "## Image-level classification, per class", "",
              "| Class | TP | FP | FN | TN | Precision | Recall | F1 |", "|---|---|---|---|---|---|---|---|"]
    for class_id, stats in results["classification"].items():
        lines.append(
            f"| {class_id} | {stats['tp']} | {stats['fp']} | {stats['fn']} | {stats['tn']} | "
            f"{stats['precision']} | {stats['recall']} | {stats['f1']} |"
        )

    lines += ["", f"> {results['note']}", ""]
    md_path = os.path.join(out_dir, "evaluation.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return {"json": json_path, "markdown": md_path}
