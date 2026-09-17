"""Severstal annotation handling and leakage-safe splitting.

Kaggle has shipped two CSV layouts for this competition:
  A) ImageId_ClassId ("0002cc93b.jpg_1"), EncodedPixels
  B) ImageId, ClassId, EncodedPixels
Both are normalised here to layout B so nothing downstream has to care.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Tuple

import pandas as pd

from .exceptions import DataError


def load_annotations(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise DataError(f"Annotation CSV not found: {csv_path}")
    frame = pd.read_csv(csv_path)

    if {"ImageId", "ClassId", "EncodedPixels"}.issubset(frame.columns):
        normalised = frame[["ImageId", "ClassId", "EncodedPixels"]].copy()
    elif {"ImageId_ClassId", "EncodedPixels"}.issubset(frame.columns):
        split = frame["ImageId_ClassId"].astype(str).str.rsplit("_", n=1, expand=True)
        normalised = pd.DataFrame({
            "ImageId": split[0],
            "ClassId": split[1],
            "EncodedPixels": frame["EncodedPixels"],
        })
    else:
        raise DataError(
            f"Unrecognised CSV schema in {csv_path}. Columns found: {list(frame.columns)}. "
            "Expected either (ImageId, ClassId, EncodedPixels) or (ImageId_ClassId, EncodedPixels)."
        )

    normalised["ClassId"] = pd.to_numeric(normalised["ClassId"], errors="coerce").astype("Int64")
    if normalised["ClassId"].isna().any():
        raise DataError("Some rows have a non-numeric ClassId after normalisation.")
    # Rows with no mask are 'this image does not have this class'; drop them.
    normalised = normalised[normalised["EncodedPixels"].notna()]
    normalised = normalised[normalised["EncodedPixels"].astype(str).str.strip() != ""]
    return normalised.reset_index(drop=True)


def build_image_index(annotations: pd.DataFrame) -> Dict[str, Dict[int, str]]:
    """image_id -> {class_id: rle}. One image may carry several classes."""
    index: Dict[str, Dict[int, str]] = {}
    for row in annotations.itertuples(index=False):
        index.setdefault(row.ImageId, {})[int(row.ClassId)] = str(row.EncodedPixels)
    return index


def defect_signature(classes: Dict[int, str]) -> str:
    """Stable label describing which classes an image carries, used for stratification."""
    return "-".join(str(c) for c in sorted(classes)) or "none"


def list_all_images(images_dir: str) -> List[str]:
    if not os.path.isdir(images_dir):
        raise DataError(f"Images directory not found: {images_dir}")
    return sorted(n for n in os.listdir(images_dir) if n.lower().endswith((".jpg", ".jpeg", ".png")))


def make_splits(
    image_ids: List[str],
    signatures: Dict[str, str],
    val_fraction: float,
    test_fraction: float,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Stratified split over *unique image ids*.

    Splitting on annotation rows would place two annotations of the same image in
    different splits, leaking test pixels into training. Grouping by ImageId
    first makes that impossible.
    """
    if not 0 <= val_fraction < 1 or not 0 <= test_fraction < 1:
        raise DataError("Split fractions must lie in [0, 1).")
    if val_fraction + test_fraction >= 1:
        raise DataError("val_fraction + test_fraction must be below 1.")

    buckets: Dict[str, List[str]] = {}
    for image_id in image_ids:
        buckets.setdefault(signatures.get(image_id, "none"), []).append(image_id)

    rng = random.Random(seed)
    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    for signature, members in sorted(buckets.items()):
        members = sorted(members)
        rng.shuffle(members)
        total = len(members)
        n_val = int(round(total * val_fraction))
        n_test = int(round(total * test_fraction))
        # Tiny strata: keep at least one member in train.
        n_val = min(n_val, max(0, total - 1))
        n_test = min(n_test, max(0, total - n_val - 1))
        splits["val"].extend(members[:n_val])
        splits["test"].extend(members[n_val:n_val + n_test])
        splits["train"].extend(members[n_val + n_test:])

    for name in splits:
        splits[name] = sorted(splits[name])

    overlap = (set(splits["train"]) & set(splits["val"])) | (set(splits["train"]) & set(splits["test"])) \
        | (set(splits["val"]) & set(splits["test"]))
    if overlap:
        raise DataError(f"Split leakage detected for {len(overlap)} image ids.")
    return splits


def write_splits(splits: Dict[str, List[str]], index: Dict[str, Dict[int, str]], out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    written: Dict[str, str] = {}
    for name, ids in splits.items():
        rows = []
        for image_id in ids:
            classes = index.get(image_id, {})
            rows.append({
                "ImageId": image_id,
                "classes": defect_signature(classes),
                "has_defect": int(bool(classes)),
            })
        path = os.path.join(out_dir, f"{name}.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        written[name] = path
    return written
