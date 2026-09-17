"""Stage 1-2: image loading, validation, and preprocessing."""

from __future__ import annotations

import os
from typing import Tuple

import cv2
import numpy as np

from .config_loader import Config
from .exceptions import ImageValidationError
from .logger import get_logger

logger = get_logger(__name__)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_image(path: str) -> np.ndarray:
    """Read an image from disk as BGR uint8, raising a typed error on any failure."""
    if not os.path.isfile(path):
        raise ImageValidationError(f"Image not found: {path}")

    extension = os.path.splitext(path)[1].lower()
    if extension not in VALID_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported extension '{extension}' for {path}. Supported: {sorted(VALID_EXTENSIONS)}"
        )

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageValidationError(f"Image could not be decoded (corrupt or empty file): {path}")
    return image


def validate_image(image: np.ndarray, min_side: int = 32) -> None:
    """Reject images that would break downstream stages."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ImageValidationError(f"Expected a 3-channel BGR image, got shape {image.shape}.")
    height, width = image.shape[:2]
    if height < min_side or width < min_side:
        raise ImageValidationError(f"Image too small: {width}x{height}, minimum side is {min_side}px.")
    if image.dtype != np.uint8:
        raise ImageValidationError(f"Expected uint8 pixels, got {image.dtype}.")
    if float(image.std()) < 1e-6:
        raise ImageValidationError("Image is a single flat colour; it carries no inspectable surface.")


def apply_clahe(image: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation on the L channel of LAB.

    Steel surface images are low-contrast and unevenly lit; CLAHE lifts local
    contrast without amplifying global illumination differences the way plain
    histogram equalisation does.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid), int(tile_grid)))
    lightness = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[0] == height and image.shape[1] == width:
        return image
    interpolation = cv2.INTER_AREA if image.shape[0] > height else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def normalize(image: np.ndarray, mean, std) -> np.ndarray:
    """Scale BGR uint8 to a float32 CHW tensor in RGB order, mean/std normalised."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.transpose(rgb, (2, 0, 1))


def preprocess(image: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Run the full preprocessing chain.

    Returns (display_image, tensor) where display_image is the enhanced BGR image
    used for visualisation and tensor is the CHW float32 array fed to the model.
    """
    validate_image(image)
    enhanced = image
    if cfg.get("preprocessing.clahe.enabled"):
        enhanced = apply_clahe(
            enhanced,
            cfg.get("preprocessing.clahe.clip_limit"),
            cfg.get("preprocessing.clahe.tile_grid"),
        )
    resized = resize(
        enhanced,
        int(cfg.get("preprocessing.target_height")),
        int(cfg.get("preprocessing.target_width")),
    )
    tensor = normalize(
        resized,
        cfg.get("preprocessing.normalize.mean"),
        cfg.get("preprocessing.normalize.std"),
    )
    return enhanced, tensor
