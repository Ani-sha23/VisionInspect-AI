"""Command-line interface. Every capability of the project is reachable from here.

    python -m visioninspect <command> [options]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from . import __version__
from .config_loader import DEFAULT_CONFIG_PATH, load_config
from .exceptions import VisionInspectError
from .logger import get_logger, setup_logging

logger = get_logger("visioninspect.cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visioninspect",
        description="Automated surface-defect inspection for steel sheet images.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    parser.add_argument("--log-level", default=None, help="Override logging level (DEBUG/INFO/WARNING)")
    parser.add_argument("--version", action="version", version=f"VisionInspect-AI {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect-data", help="Measure the dataset and write docs/results/dataset_stats.md")
    p_inspect.add_argument("--out", default=os.path.join("docs", "results", "dataset_stats.md"))
    p_inspect.add_argument("--sample-size", type=int, default=50, help="Images to sample for shape checking")

    p_prepare = sub.add_parser("prepare-data", help="Create leakage-safe train/val/test splits")
    p_prepare.add_argument("--out-dir", default=os.path.join("data", "processed"))

    p_cal = sub.add_parser("calibrate", help="Derive severity band edges from the training distribution")
    p_cal.add_argument("--split", default=os.path.join("data", "processed", "train.csv"))
    p_cal.add_argument("--limit", type=int, default=None)
    p_cal.add_argument("--apply", action="store_true", help="Write the thresholds back into the config file")
    p_cal.add_argument("--out", default=os.path.join("docs", "results", "severity_calibration.md"))

    p_train = sub.add_parser("train", help="Train the U-Net (requires PyTorch and, realistically, a GPU)")
    p_train.add_argument("--splits-dir", default=os.path.join("data", "processed"))
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--limit", type=int, default=None, help="Cap images per split (smoke testing)")

    p_run = sub.add_parser("run", help="Inspect an image or a directory of images")
    p_run.add_argument("--input", required=True, help="Image file or directory")
    p_run.add_argument("--backend", choices=["auto", "unet", "baseline"], default="auto")
    p_run.add_argument("--limit", type=int, default=None)
    p_run.add_argument("--out-dir", default=None, help="Overrides report.output_dir")
    p_run.add_argument("--no-visuals", action="store_true", help="Skip writing annotated images")
    p_run.add_argument("--json-only", action="store_true", help="Print JSON to stdout and write no files")

    p_eval = sub.add_parser("evaluate", help="Score a split against ground-truth masks")
    p_eval.add_argument("--split", default=os.path.join("data", "processed", "test.csv"))
    p_eval.add_argument("--backend", choices=["auto", "unet", "baseline"], default="auto")
    p_eval.add_argument("--limit", type=int, default=None)
    p_eval.add_argument("--out-dir", default=os.path.join("docs", "results"))

    return parser


def cmd_inspect_data(args, cfg) -> int:
    from .dataset_inspection import inspect, to_markdown

    stats = inspect(cfg.get("data.train_csv"), cfg.get("data.images_dir"), args.sample_size)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(to_markdown(stats))

    try:
        from .visualization import plot_class_distribution
        figure = plot_class_distribution(
            stats["rows_per_class"], os.path.join(cfg.get("report.figures_dir"), "class_distribution.png")
        )
        print(f"Class distribution figure: {figure}")
    except Exception as exc:  # plotting is a nicety, not a requirement
        logger.warning("Could not render the class-distribution figure: %s", exc)

    print(json.dumps({k: v for k, v in stats.items() if k != "rle_decode_check"}, indent=2))
    print(f"\nWritten: {args.out}")
    return 0


def cmd_prepare_data(args, cfg) -> int:
    from .dataset import (build_image_index, defect_signature, list_all_images,
                          load_annotations, make_splits, write_splits)

    index = build_image_index(load_annotations(cfg.get("data.train_csv")))
    image_ids = list_all_images(cfg.get("data.images_dir"))
    signatures = {image_id: defect_signature(index.get(image_id, {})) for image_id in image_ids}
    splits = make_splits(
        image_ids,
        signatures,
        float(cfg.get("data.split.val_fraction")),
        float(cfg.get("data.split.test_fraction")),
        int(cfg.get("project.seed")),
    )
    written = write_splits(splits, index, args.out_dir)
    for name, path in written.items():
        print(f"{name:5s}: {len(splits[name]):6d} images -> {path}")
    return 0


def cmd_calibrate(args, cfg) -> int:
    from .calibration import apply_to_config, collect_scores, derive_thresholds, describe

    shape = (int(cfg.get("preprocessing.target_height")), int(cfg.get("preprocessing.target_width")))
    scores = collect_scores(cfg, args.split, args.limit, image_shape=shape)
    thresholds = derive_thresholds(scores)
    report = describe(scores, thresholds)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(report)

    if args.apply:
        apply_to_config(thresholds, cfg.path)
        print(f"Config updated: {cfg.path} (severity.calibrated = true)")
    else:
        print("Run again with --apply to write these thresholds into the config.")
    return 0


def cmd_train(args, cfg) -> int:
    from .train import train

    summary = train(cfg, args.splits_dir, args.epochs, args.limit)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_run(args, cfg) -> int:
    from .pipeline import InspectionPipeline, collect_images
    from .report_generator import summarize, write_csv, write_json, write_markdown

    paths = collect_images(args.input, args.limit)
    pipeline = InspectionPipeline(cfg, backend=args.backend, save_visuals=not (args.no_visuals or args.json_only))
    results, failures = pipeline.run_batch(paths)

    if args.json_only:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return 0 if results else 1

    out_dir = args.out_dir or cfg.get("report.output_dir")
    meta = {"backend": pipeline.detector.backend, "config": cfg.path, "failures": failures}
    json_path = write_json(results, os.path.join(out_dir, "results.json"), meta)
    csv_path = write_csv(results, os.path.join(out_dir, "results.csv"))
    md_path = write_markdown(results, os.path.join(out_dir, "report.md"), failures)

    print(json.dumps(summarize(results), indent=2))
    print(f"\nJSON report     : {json_path}")
    print(f"CSV report      : {csv_path}")
    print(f"Markdown report : {md_path}")
    if not args.no_visuals:
        print(f"Annotated images: {cfg.get('report.figures_dir')}")
    if failures:
        print(f"\n{len(failures)} input(s) skipped; see the report for details.")
    return 0 if results else 1


def cmd_evaluate(args, cfg) -> int:
    from .evaluate import evaluate_split, write_evaluation

    results = evaluate_split(cfg, args.split, args.backend, args.limit)
    paths = write_evaluation(results, args.out_dir)
    print(json.dumps(results, indent=2))
    print(f"\nWritten: {paths['markdown']} and {paths['json']}")
    return 0


COMMANDS = {
    "inspect-data": cmd_inspect_data,
    "prepare-data": cmd_prepare_data,
    "calibrate": cmd_calibrate,
    "train": cmd_train,
    "run": cmd_run,
    "evaluate": cmd_evaluate,
}


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        setup_logging(args.log_level or cfg.get("logging.level"), cfg.get("logging.file"))
        return COMMANDS[args.command](args, cfg)
    except VisionInspectError as exc:
        setup_logging("INFO", None)
        logger.error("%s: %s", type(exc).__name__, exc)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
