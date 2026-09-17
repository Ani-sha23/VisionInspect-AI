"""Loading, validating, and accessing configuration."""

from __future__ import annotations

import os
from typing import Any, Dict

import yaml

from .exceptions import ConfigError

DEFAULT_CONFIG_PATH = os.path.join("configs", "config.yaml")

REQUIRED_SECTIONS = (
    "project", "data", "preprocessing", "model", "baseline",
    "severity", "quality", "report", "logging",
)


class Config:
    """Thin dict wrapper with dotted-path access and validation."""

    def __init__(self, data: Dict[str, Any], path: str = DEFAULT_CONFIG_PATH) -> None:
        self._data = data
        self.path = path

    def get(self, dotted: str, default: Any = "__raise__") -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default == "__raise__":
                    raise ConfigError(f"Missing configuration key: '{dotted}' in {self.path}")
                return default
            node = node[part]
        return node

    def section(self, name: str) -> Dict[str, Any]:
        value = self.get(name)
        if not isinstance(value, dict):
            raise ConfigError(f"Configuration section '{name}' must be a mapping.")
        return value

    def as_dict(self) -> Dict[str, Any]:
        return self._data

    def __contains__(self, key: str) -> bool:
        return self.get(key, None) is not None


def _validate(cfg: Config) -> None:
    for section in REQUIRED_SECTIONS:
        cfg.section(section)

    if cfg.get("baseline.block_size") % 2 == 0:
        raise ConfigError("baseline.block_size must be odd (OpenCV adaptive threshold requirement).")

    weights = cfg.get("severity.class_weights")
    if not isinstance(weights, dict) or not weights:
        raise ConfigError("severity.class_weights must be a non-empty mapping of class_id -> weight.")
    for key, value in weights.items():
        try:
            int(key)
            float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"Invalid severity.class_weights entry {key!r}: {value!r}") from exc

    th = cfg.get("severity.thresholds")
    ordered = [th.get("minor_max"), th.get("moderate_max"), th.get("severe_max")]
    if any(v is None for v in ordered):
        raise ConfigError(
            "severity.thresholds has an unset band edge. Run scripts/calibrate_severity.py "
            "or fill the values in configs/config.yaml."
        )
    if not ordered == sorted(ordered):
        raise ConfigError(f"severity.thresholds must increase: got {ordered}.")

    if cfg.get("quality.accept_max_score") > cfg.get("quality.reject_min_score"):
        raise ConfigError("quality.accept_max_score cannot exceed quality.reject_min_score.")


def load_config(path: str = DEFAULT_CONFIG_PATH) -> Config:
    """Read a YAML config from disk and validate it.

    Raises ConfigError on anything that would silently produce wrong results later.
    """
    if not os.path.isfile(path):
        raise ConfigError(f"Config file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Config root of {path} must be a mapping.")

    cfg = Config(raw, path)
    _validate(cfg)
    return cfg


def class_weights(cfg: Config) -> Dict[int, float]:
    """Return severity class weights with integer keys (YAML may parse them as strings)."""
    return {int(k): float(v) for k, v in cfg.get("severity.class_weights").items()}
