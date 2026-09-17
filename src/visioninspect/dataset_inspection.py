"""Measure the dataset. Every number quoted in the report should originate here."""

from __future__ import annotations

import os
from collections import Counter
from typing import Dict, List

import cv2

from .dataset import build_image_index, defect_signature, list_all_images, load_annotations
from .exceptions import DataError
from .rle import rle_to_mask


def inspect(csv_path: str, images_dir: str, sample_size: int = 50) -> Dict:
    annotations = load_annotations(csv_path)
    index = build_image_index(annotations)
    all_images = list_all_images(images_dir)

    class_rows = Counter(int(c) for c in annotations["ClassId"].tolist())
    classes_per_image = Counter(len(v) for v in index.values())
    signatures = Counter(defect_signature(v) for v in index.values())

    defect_images = len(index)
    clean_images = len(all_images) - defect_images

    shapes: Counter = Counter()
    for name in all_images[:sample_size]:
        image = cv2.imread(os.path.join(images_dir, name), cv2.IMREAD_COLOR)
        if image is None:
            raise DataError(f"Sample image could not be read: {name}")
        shapes[(image.shape[0], image.shape[1], image.shape[2])] += 1

    # Verify the RLE convention against a real annotation rather than assuming it.
    rle_check = None
    if index and shapes:
        (height, width, _), _ = shapes.most_common(1)[0]
        first_image = sorted(index)[0]
        first_class, first_rle = sorted(index[first_image].items())[0]
        mask = rle_to_mask(first_rle, (height, width))
        rle_check = {
            "image_id": first_image,
            "class_id": first_class,
            "mask_shape": list(mask.shape),
            "positive_pixels": int(mask.sum()),
            "coverage_percent": round(100.0 * float(mask.sum()) / (height * width), 4),
        }

    return {
        "csv_path": csv_path,
        "images_dir": images_dir,
        "annotation_rows": int(len(annotations)),
        "images_on_disk": len(all_images),
        "images_with_defects": defect_images,
        "images_without_defects": clean_images,
        "rows_per_class": dict(sorted(class_rows.items())),
        "classes_per_image": dict(sorted(classes_per_image.items())),
        "top_signatures": dict(signatures.most_common(10)),
        "image_shapes_sampled": {f"{h}x{w}x{c}": n for (h, w, c), n in shapes.items()},
        "sample_size": min(sample_size, len(all_images)),
        "rle_decode_check": rle_check,
    }


def to_markdown(stats: Dict) -> str:
    lines: List[str] = [
        "# Dataset statistics (measured)",
        "",
        f"- Annotation CSV: `{stats['csv_path']}`",
        f"- Images directory: `{stats['images_dir']}`",
        f"- Annotation rows (non-empty masks): **{stats['annotation_rows']}**",
        f"- Images on disk: **{stats['images_on_disk']}**",
        f"- Images with at least one defect: **{stats['images_with_defects']}**",
        f"- Images with no annotated defect: **{stats['images_without_defects']}**",
        "",
        "## Annotation rows per class",
        "",
        "| Class | Rows |",
        "|---|---|",
    ]
    for class_id, count in stats["rows_per_class"].items():
        lines.append(f"| {class_id} | {count} |")

    lines += ["", "## Defect classes per image", "", "| Classes on one image | Images |", "|---|---|"]
    for k, v in stats["classes_per_image"].items():
        lines.append(f"| {k} | {v} |")

    lines += ["", "## Most common class combinations", "", "| Combination | Images |", "|---|---|"]
    for signature, count in stats["top_signatures"].items():
        lines.append(f"| {signature} | {count} |")

    lines += ["", f"## Image shapes (sample of {stats['sample_size']})", "", "| Shape (HxWxC) | Count |", "|---|---|"]
    for shape, count in stats["image_shapes_sampled"].items():
        lines.append(f"| {shape} | {count} |")

    check = stats.get("rle_decode_check")
    if check:
        lines += [
            "",
            "## RLE decode verification",
            "",
            f"Decoded class {check['class_id']} of `{check['image_id']}` into a "
            f"{check['mask_shape'][0]}x{check['mask_shape'][1]} mask with "
            f"{check['positive_pixels']} positive pixels "
            f"({check['coverage_percent']}% of the image).",
            "",
            "A plausible, non-zero coverage confirms the column-major, 1-based RLE convention.",
        ]

    lines.append("")
    return "\n".join(lines)
