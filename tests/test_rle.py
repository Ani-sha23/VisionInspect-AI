import numpy as np
import pytest

from visioninspect.exceptions import DataError
from visioninspect.rle import build_multilabel_mask, mask_to_rle, rle_to_mask


def test_empty_rle_gives_empty_mask():
    mask = rle_to_mask(None, (4, 5))
    assert mask.shape == (4, 5)
    assert mask.sum() == 0


def test_decode_is_column_major_and_one_based():
    # In a 3x3 image, start=1 length=3 covers the whole first column (Fortran order).
    mask = rle_to_mask("1 3", (3, 3))
    assert mask[:, 0].sum() == 3
    assert mask[:, 1:].sum() == 0


def test_roundtrip_encode_decode():
    original = np.zeros((10, 12), dtype=np.uint8)
    original[2:6, 3:8] = 1
    original[8, 1] = 1
    recovered = rle_to_mask(mask_to_rle(original), original.shape)
    assert np.array_equal(original, recovered)


def test_odd_token_count_is_rejected():
    with pytest.raises(DataError):
        rle_to_mask("1 3 5", (4, 4))


def test_out_of_range_run_is_rejected():
    with pytest.raises(DataError):
        rle_to_mask("1 500", (4, 4))


def test_build_multilabel_mask_stacks_classes():
    stacked = build_multilabel_mask({1: "1 3", 3: "4 3"}, (3, 3), num_classes=4)
    assert stacked.shape == (4, 3, 3)
    assert stacked[0].sum() == 3
    assert stacked[2].sum() == 3
    assert stacked[1].sum() == 0 and stacked[3].sum() == 0
