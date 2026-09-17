import numpy as np

from visioninspect.baseline_detector import detect_regions
from visioninspect.detector import masks_to_defects


def test_baseline_finds_planted_region(cfg, synthetic_image):
    defects = detect_regions(synthetic_image, cfg)
    assert defects, "expected at least one region"
    biggest = defects[0]
    assert biggest.class_id == 0 and biggest.class_name == "unclassified"
    x1, y1, x2, y2 = biggest.bbox.as_xyxy()
    assert x1 <= 160 and x2 >= 250 and y1 <= 110 and y2 >= 130


def test_baseline_on_clean_surface_finds_little(cfg):
    rng = np.random.default_rng(3)
    clean = np.clip(np.full((256, 400, 3), 120) + rng.normal(0, 3, (256, 400, 3)), 0, 255).astype(np.uint8)
    assert len(detect_regions(clean, cfg)) <= 2


def test_masks_to_defects_assigns_class_from_channel():
    prob = np.zeros((4, 64, 64), dtype=np.float32)
    prob[2, 10:40, 10:40] = 0.9          # channel 2 -> class id 3
    defects = masks_to_defects(prob, threshold=0.5, min_area=100, original_shape=(64, 64))
    assert len(defects) == 1
    assert defects[0].class_id == 3
    assert defects[0].area_px == 900
    assert defects[0].confidence > 0.8


def test_masks_to_defects_drops_small_components():
    prob = np.zeros((4, 64, 64), dtype=np.float32)
    prob[0, 0:3, 0:3] = 0.99
    assert masks_to_defects(prob, 0.5, min_area=100, original_shape=(64, 64)) == []


def test_masks_to_defects_separates_two_blobs():
    prob = np.zeros((4, 64, 64), dtype=np.float32)
    prob[0, 5:20, 5:20] = 0.8
    prob[0, 40:60, 40:60] = 0.8
    assert len(masks_to_defects(prob, 0.5, 50, (64, 64))) == 2
