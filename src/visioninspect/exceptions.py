"""Exception hierarchy for VisionInspect-AI."""


class VisionInspectError(Exception):
    """Base class for every error raised by this project."""


class ConfigError(VisionInspectError):
    """Configuration file missing, malformed, or uncalibrated where calibration is required."""


class DataError(VisionInspectError):
    """Dataset files missing, unreadable, or inconsistent with the expected schema."""


class ImageValidationError(VisionInspectError):
    """An input image failed validation (unreadable, wrong mode, wrong size, corrupt)."""


class ModelError(VisionInspectError):
    """Model weights missing or the deep backend is unavailable."""


class PipelineError(VisionInspectError):
    """A stage of the inspection pipeline failed."""
