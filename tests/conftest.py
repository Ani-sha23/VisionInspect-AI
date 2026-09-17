import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from visioninspect.config_loader import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    root = os.path.join(os.path.dirname(__file__), "..")
    return load_config(os.path.join(root, "configs", "config.yaml"))


@pytest.fixture
def synthetic_image():
    """A 256x400 grey surface with one dark rectangle the baseline detector should find."""
    image = np.full((256, 400, 3), 120, dtype=np.uint8)
    rng = np.random.default_rng(0)
    image = np.clip(image + rng.normal(0, 4, image.shape), 0, 255).astype(np.uint8)
    image[100:140, 150:260] = 40
    return image
