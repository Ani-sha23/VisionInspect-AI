"""Run-length encoding helpers for the Severstal annotation format.

Severstal encodes masks column-major (Fortran order) with 1-based start indices,
as pairs of "start length" values separated by spaces.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .exceptions import DataError


def rle_to_mask(rle: Optional[str], shape: Tuple[int, int]) -> np.ndarray:
    """Decode an RLE string into a uint8 mask of the given (height, width)."""
    height, width = shape
    mask = np.zeros(height * width, dtype=np.uint8)
    if rle is None or (isinstance(rle, float) and np.isnan(rle)) or str(rle).strip() == "":
        return mask.reshape((height, width), order="F")

    parts = str(rle).split()
    if len(parts) % 2 != 0:
        raise DataError(f"RLE string has an odd number of tokens ({len(parts)}); expected start/length pairs.")

    try:
        numbers = [int(p) for p in parts]
    except ValueError as exc:
        raise DataError(f"RLE string contains a non-integer token: {exc}") from exc

    starts = np.asarray(numbers[0::2], dtype=np.int64) - 1  # 1-based -> 0-based
    lengths = np.asarray(numbers[1::2], dtype=np.int64)
    ends = starts + lengths

    if starts.min() < 0 or ends.max() > mask.size:
        raise DataError(
            f"RLE runs fall outside an image of shape {shape}: max end {ends.max()} > {mask.size}."
        )

    for start, end in zip(starts, ends):
        mask[start:end] = 1
    return mask.reshape((height, width), order="F")


def mask_to_rle(mask: np.ndarray) -> str:
    """Encode a binary mask back to the Severstal RLE string format."""
    pixels = np.asarray(mask, dtype=np.uint8).flatten(order="F")
    padded = np.concatenate([[0], pixels, [0]])
    changes = np.where(padded[1:] != padded[:-1])[0] + 1
    runs: List[int] = []
    for start, end in zip(changes[0::2], changes[1::2]):
        runs.extend([int(start), int(end - start)])
    return " ".join(str(v) for v in runs)


def build_multilabel_mask(rles: dict, shape: Tuple[int, int], num_classes: int = 4) -> np.ndarray:
    """Stack per-class RLEs into a (num_classes, H, W) uint8 mask.

    `rles` maps class_id (1-based) to an RLE string; missing classes become zeros.
    """
    height, width = shape
    stacked = np.zeros((num_classes, height, width), dtype=np.uint8)
    for class_id, rle in rles.items():
        index = int(class_id) - 1
        if not 0 <= index < num_classes:
            raise DataError(f"Class id {class_id} outside the expected range 1..{num_classes}.")
        stacked[index] = rle_to_mask(rle, shape)
    return stacked
