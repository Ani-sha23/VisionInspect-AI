import numpy as np
import pytest

from visioninspect.exceptions import ImageValidationError
from visioninspect.preprocessing import load_image, normalize, preprocess, resize, validate_image


def test_validate_rejects_grayscale():
    with pytest.raises(ImageValidationError):
        validate_image(np.zeros((50, 50), dtype=np.uint8))


def test_validate_rejects_flat_image():
    with pytest.raises(ImageValidationError):
        validate_image(np.full((64, 64, 3), 200, dtype=np.uint8))


def test_validate_rejects_tiny_image(synthetic_image):
    with pytest.raises(ImageValidationError):
        validate_image(synthetic_image[:8, :8])


def test_load_image_missing_file():
    with pytest.raises(ImageValidationError):
        load_image("does/not/exist.jpg")


def test_resize_is_a_noop_at_target_size(synthetic_image):
    out = resize(synthetic_image, synthetic_image.shape[0], synthetic_image.shape[1])
    assert out.shape == synthetic_image.shape


def test_normalize_returns_chw_float(synthetic_image):
    tensor = normalize(synthetic_image, [0.5] * 3, [0.5] * 3)
    assert tensor.dtype == np.float32
    assert tensor.shape == (3, synthetic_image.shape[0], synthetic_image.shape[1])


def test_preprocess_matches_configured_target(cfg, synthetic_image):
    _, tensor = preprocess(synthetic_image, cfg)
    assert tensor.shape == (
        3,
        cfg.get("preprocessing.target_height"),
        cfg.get("preprocessing.target_width"),
    )
