import os

import cv2
import numpy as np
import pytest
import yaml

from visioninspect.config_loader import Config, load_config
from visioninspect.exceptions import ConfigError, PipelineError
from visioninspect.pipeline import InspectionPipeline, collect_images
from visioninspect.report_generator import summarize, write_csv, write_json, write_markdown


def test_missing_config_file_raises():
    with pytest.raises(ConfigError):
        load_config("configs/definitely_not_here.yaml")


def test_missing_key_raises(cfg):
    with pytest.raises(ConfigError):
        cfg.get("model.does_not_exist")


def test_even_block_size_is_rejected(tmp_path, cfg):
    data = dict(cfg.as_dict())
    data["baseline"] = {**data["baseline"], "block_size": 50}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError):
        load_config(str(path))


def test_unordered_thresholds_are_rejected(tmp_path, cfg):
    data = dict(cfg.as_dict())
    data["severity"] = {**data["severity"], "thresholds": {"minor_max": 80, "moderate_max": 40, "severe_max": 10}}
    path = tmp_path / "bad2.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError):
        load_config(str(path))


def test_collect_images_on_empty_directory(tmp_path):
    with pytest.raises(PipelineError):
        collect_images(str(tmp_path))


def test_end_to_end_baseline_pipeline(cfg, synthetic_image, tmp_path):
    image_path = tmp_path / "sheet.jpg"
    cv2.imwrite(str(image_path), synthetic_image)

    pipeline = InspectionPipeline(cfg, backend="baseline", save_visuals=False)
    result = pipeline.run_image(str(image_path))

    assert result.backend == "baseline"
    assert result.height == synthetic_image.shape[0]
    assert result.quality.decision in {"ACCEPT", "REVIEW", "REJECT"}
    assert result.severity is not None
    assert result.elapsed_ms > 0
    assert len(result.defects) >= 1, "the planted dark rectangle should be localised"


def test_batch_records_failures_without_aborting(cfg, synthetic_image, tmp_path):
    good = tmp_path / "good.jpg"
    cv2.imwrite(str(good), synthetic_image)
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")

    pipeline = InspectionPipeline(cfg, backend="baseline", save_visuals=False)
    results, failures = pipeline.run_batch([str(good), str(broken)])
    assert len(results) == 1 and len(failures) == 1


def test_reports_are_written(cfg, synthetic_image, tmp_path):
    image_path = tmp_path / "sheet.jpg"
    cv2.imwrite(str(image_path), synthetic_image)
    pipeline = InspectionPipeline(cfg, backend="baseline", save_visuals=False)
    results, _ = pipeline.run_batch([str(image_path)])

    json_path = write_json(results, str(tmp_path / "out" / "results.json"))
    csv_path = write_csv(results, str(tmp_path / "out" / "results.csv"))
    md_path = write_markdown(results, str(tmp_path / "out" / "report.md"))
    for path in (json_path, csv_path, md_path):
        assert os.path.isfile(path) and os.path.getsize(path) > 0

    stats = summarize(results)
    assert stats["images"] == 1
