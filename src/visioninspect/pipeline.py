"""Stage orchestration: image path in, InspectionResult out.

    load -> validate -> preprocess -> detect -> classify -> severity
         -> quality decision -> (optional) visualisation

The orchestrator owns error handling: a failure on one image in a batch is
recorded and skipped, it never aborts the whole run.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

from .classifier import present_classes
from .config_loader import Config
from .detector import DefectDetector
from .exceptions import ImageValidationError, PipelineError
from .logger import get_logger
from .preprocessing import load_image, preprocess
from .quality_engine import decide
from .schemas import InspectionResult
from .severity import compute_severity
from .visualization import save_visualization

logger = get_logger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class InspectionPipeline:
    def __init__(self, cfg: Config, backend: str = "auto", save_visuals: Optional[bool] = None) -> None:
        self.cfg = cfg
        self.detector = DefectDetector(cfg, backend=backend)
        self.save_visuals = cfg.get("report.save_overlay") if save_visuals is None else save_visuals

    def run_image(self, image_path: str) -> InspectionResult:
        started = time.perf_counter()
        image = load_image(image_path)
        enhanced, tensor = preprocess(image, self.cfg)

        defects = self.detector.detect(enhanced, tensor)
        severity = compute_severity(defects, image.shape[0], image.shape[1], self.cfg)
        quality = decide(severity, defects, self.cfg)

        result = InspectionResult(
            image_id=os.path.basename(image_path),
            image_path=os.path.abspath(image_path),
            height=int(image.shape[0]),
            width=int(image.shape[1]),
            defects=defects,
            present_classes=present_classes(defects),
            severity=severity,
            quality=quality,
            backend=self.detector.backend,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        if not severity.calibrated:
            result.warnings.append("severity.calibrated is false: bands are provisional.")

        if self.save_visuals:
            path = save_visualization(
                enhanced, result, self.cfg.get("report.figures_dir"), float(self.cfg.get("report.overlay_alpha"))
            )
            logger.info("Annotated image written to %s", path)

        return result

    def run_batch(self, paths: List[str]) -> Tuple[List[InspectionResult], List[Dict[str, str]]]:
        results: List[InspectionResult] = []
        failures: List[Dict[str, str]] = []
        for path in paths:
            try:
                results.append(self.run_image(path))
            except ImageValidationError as exc:
                logger.warning("Skipping %s: %s", path, exc)
                failures.append({"path": path, "error": str(exc)})
            except Exception as exc:  # unexpected: record and continue the batch
                logger.exception("Unhandled failure on %s", path)
                failures.append({"path": path, "error": f"{type(exc).__name__}: {exc}"})
        return results, failures


def collect_images(target: str, limit: Optional[int] = None) -> List[str]:
    """Expand a file or directory path into a sorted list of image paths."""
    if os.path.isfile(target):
        return [target]
    if not os.path.isdir(target):
        raise PipelineError(f"Input path does not exist: {target}")
    paths = sorted(
        os.path.join(target, name)
        for name in os.listdir(target)
        if name.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not paths:
        raise PipelineError(f"No images with extensions {IMAGE_EXTENSIONS} found in {target}")
    return paths[:limit] if limit else paths
