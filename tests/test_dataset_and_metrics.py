import numpy as np
import pandas as pd
import pytest

from visioninspect.dataset import build_image_index, defect_signature, load_annotations, make_splits
from visioninspect.exceptions import DataError
from visioninspect.metrics import confusion_counts, dice_coefficient, iou, precision_recall_f1


def write_csv(tmp_path, frame, name="train.csv"):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return str(path)


def test_loads_three_column_schema(tmp_path):
    frame = pd.DataFrame({"ImageId": ["a.jpg", "a.jpg", "b.jpg"], "ClassId": [1, 3, 2],
                          "EncodedPixels": ["1 3", "4 3", "1 2"]})
    annotations = load_annotations(write_csv(tmp_path, frame))
    assert len(annotations) == 3
    index = build_image_index(annotations)
    assert set(index["a.jpg"]) == {1, 3}


def test_loads_legacy_two_column_schema(tmp_path):
    frame = pd.DataFrame({"ImageId_ClassId": ["a.jpg_1", "b.jpg_4"], "EncodedPixels": ["1 3", "2 2"]})
    index = build_image_index(load_annotations(write_csv(tmp_path, frame, "legacy.csv")))
    assert set(index) == {"a.jpg", "b.jpg"} and 4 in index["b.jpg"]


def test_unknown_schema_raises(tmp_path):
    frame = pd.DataFrame({"foo": [1], "bar": [2]})
    with pytest.raises(DataError):
        load_annotations(write_csv(tmp_path, frame, "weird.csv"))


def test_signature_is_order_independent():
    assert defect_signature({3: "x", 1: "y"}) == "1-3"
    assert defect_signature({}) == "none"


def test_splits_do_not_leak_image_ids():
    ids = [f"img_{i}.jpg" for i in range(200)]
    signatures = {i: ("1" if index % 3 else "none") for index, i in enumerate(ids)}
    splits = make_splits(ids, signatures, 0.15, 0.15, seed=7)
    assert sum(len(v) for v in splits.values()) == len(ids)
    assert not set(splits["train"]) & set(splits["val"])
    assert not set(splits["val"]) & set(splits["test"])


def test_splits_are_reproducible():
    ids = [f"img_{i}.jpg" for i in range(50)]
    signatures = {i: "1" for i in ids}
    assert make_splits(ids, signatures, 0.2, 0.2, seed=1) == make_splits(ids, signatures, 0.2, 0.2, seed=1)


def test_invalid_fractions_raise():
    with pytest.raises(DataError):
        make_splits(["a.jpg"], {"a.jpg": "1"}, 0.6, 0.5)


def test_dice_and_iou_edge_cases():
    empty = np.zeros((10, 10), dtype=np.uint8)
    full = np.ones((10, 10), dtype=np.uint8)
    assert dice_coefficient(empty, empty) == 1.0
    assert dice_coefficient(full, full) == pytest.approx(1.0, abs=1e-4)
    assert dice_coefficient(full, empty) == pytest.approx(0.0, abs=1e-4)
    assert iou(empty, empty) == 1.0


def test_dice_of_half_overlap():
    a = np.zeros((10, 10), dtype=np.uint8); a[:, :5] = 1
    b = np.zeros((10, 10), dtype=np.uint8); b[:, 3:8] = 1
    assert 0.3 < dice_coefficient(a, b) < 0.5


def test_precision_recall_f1():
    counts = confusion_counts([1, 1, 0, 0], [1, 0, 1, 0])
    assert counts == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    scores = precision_recall_f1(counts)
    assert scores["precision"] == 0.5 and scores["recall"] == 0.5 and scores["f1"] == 0.5
