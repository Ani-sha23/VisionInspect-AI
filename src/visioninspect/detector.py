"""Stage 3: detection. Turns an image into a list of Defect objects.

Two interchangeable backends behind one interface:
  * "unet"     - trained multi-label segmentation model (requires weights + torch)
  * "baseline" - classical CV localiser (no training, no torch)

Both return List[Defect] in original-image pixel coordinates, so every later
stage is backend-agnostic.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import baseline_detector
from .config_loader import Config
from .exceptions import ModelError
from .logger import get_logger
from .model import load_checkpoint, torch_available
from .schemas import CLASS_NAMES, BBox, Defect

logger = get_logger(__name__)


def masks_to_defects(
    prob_maps: np.ndarray,
    threshold: float,
    min_area: int,
    original_shape: Tuple[int, int],
) -> List[Defect]:
    """Convert (C, H, W) per-class probability maps into discrete defect instances.

    This is the step that makes one segmentation model serve three jobs:
    connected components give localisation (boxes), the channel index gives the
    class label, and the set of channels that fire gives the per-image
    multi-label classification.
    """
    height, width = original_shape
    defects: List[Defect] = []

    for channel in range(prob_maps.shape[0]):
        class_id = channel + 1
        prob = prob_maps[channel]
        if prob.shape != (height, width):
            prob = cv2.resize(prob, (width, height), interpolation=cv2.INTER_LINEAR)
        binary = (prob >= threshold).astype(np.uint8)
        if binary.sum() == 0:
            continue

        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            if area < min_area:
                continue
            component = (labels == label).astype(np.uint8)
            confidence = float(prob[component > 0].mean())
            defects.append(
                Defect(
                    class_id=class_id,
                    class_name=CLASS_NAMES.get(class_id, f"class_{class_id}"),
                    bbox=BBox(int(x), int(y), int(w), int(h)),
                    area_px=int(area),
                    confidence=confidence,
                    mask=component,
                )
            )

    defects.sort(key=lambda d: d.area_px, reverse=True)
    return defects


class DefectDetector:
    """Facade that picks a backend once and reuses it across images."""

    def __init__(self, cfg: Config, backend: str = "auto") -> None:
        self.cfg = cfg
        self.model = None
        self.device = None
        self.backend = self._select_backend(backend)

    def _select_backend(self, requested: str) -> str:
        if requested == "baseline":
            return "baseline"

        weights = self.cfg.get("model.weights")
        try:
            if not torch_available():
                raise ModelError("PyTorch is not installed.")
            self.model, self.device = load_checkpoint(
                weights, self.cfg.section("model"), self.cfg.get("model.device")
            )
            logger.info("Detection backend: U-Net (%s) on %s", weights, self.device)
            return "unet"
        except ModelError as exc:
            if requested == "unet":
                raise
            logger.warning("Falling back to the classical baseline detector: %s", exc)
            return "baseline"

    def detect(self, image_bgr: np.ndarray, tensor: Optional[np.ndarray]) -> List[Defect]:
        if self.backend == "baseline":
            return baseline_detector.detect_regions(image_bgr, self.cfg)

        import torch  # local import: only reachable when the unet backend loaded

        with torch.no_grad():
            batch = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
            logits = self.model(batch)
            prob_maps = torch.sigmoid(logits)[0].cpu().numpy()

        return masks_to_defects(
            prob_maps,
            float(self.cfg.get("model.mask_threshold")),
            int(self.cfg.get("model.min_component_area")),
            (image_bgr.shape[0], image_bgr.shape[1]),
        )
