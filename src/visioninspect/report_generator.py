"""Stage 8: reporting. Writes machine-readable JSON/CSV and a human-readable Markdown summary."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Dict, List

from .schemas import InspectionResult


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_json(results: List[InspectionResult], out_path: str, meta: Dict | None = None) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = {
        "generated_at": _timestamp(),
        "meta": meta or {},
        "image_count": len(results),
        "results": [r.to_dict() for r in results],
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return out_path


def write_csv(results: List[InspectionResult], out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    columns = [
        "image_id", "backend", "decision", "severity_score", "severity_label",
        "defect_count", "area_fraction", "present_classes", "elapsed_ms",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for r in results:
            writer.writerow([
                r.image_id,
                r.backend,
                r.quality.decision if r.quality else "",
                f"{r.severity.score:.2f}" if r.severity else "",
                r.severity.label if r.severity else "",
                r.severity.defect_count if r.severity else 0,
                f"{r.severity.area_fraction:.6f}" if r.severity else "",
                "|".join(str(c) for c in r.present_classes),
                f"{r.elapsed_ms:.1f}",
            ])
    return out_path


def summarize(results: List[InspectionResult]) -> Dict:
    decisions: Dict[str, int] = {}
    class_counts: Dict[int, int] = {}
    total_ms = 0.0
    for r in results:
        if r.quality:
            decisions[r.quality.decision] = decisions.get(r.quality.decision, 0) + 1
        for class_id in r.present_classes:
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        total_ms += r.elapsed_ms
    return {
        "images": len(results),
        "decisions": dict(sorted(decisions.items())),
        "images_per_class": dict(sorted(class_counts.items())),
        "mean_latency_ms": round(total_ms / len(results), 2) if results else 0.0,
    }


def write_markdown(results: List[InspectionResult], out_path: str, failures: List[Dict] | None = None) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    stats = summarize(results)
    backend = results[0].backend if results else "n/a"
    uncalibrated = any(not r.severity.calibrated for r in results if r.severity)

    lines = [
        "# VisionInspect-AI — Inspection Report",
        "",
        f"- Generated: {_timestamp()}",
        f"- Images inspected: {stats['images']}",
        f"- Detection backend: `{backend}`",
        f"- Mean latency: {stats['mean_latency_ms']} ms/image",
        "",
    ]
    if uncalibrated:
        lines += [
            "> **UNCALIBRATED RUN.** Severity band edges have not been derived from a percentile",
            "> analysis of the training set, so severity labels and decisions below are provisional.",
            "",
        ]

    lines += ["## Decision summary", "", "| Decision | Images |", "|---|---|"]
    for decision, count in stats["decisions"].items():
        lines.append(f"| {decision} | {count} |")

    lines += ["", "## Defect classes observed", "", "| Class | Images containing it |", "|---|---|"]
    if stats["images_per_class"]:
        for class_id, count in stats["images_per_class"].items():
            lines.append(f"| {class_id} | {count} |")
    else:
        lines.append("| — | none |")

    lines += ["", "## Per-image results", "",
              "| Image | Decision | Severity | Label | Regions | Area % | Classes |", "|---|---|---|---|---|---|---|"]
    for r in results:
        s = r.severity
        lines.append(
            f"| {r.image_id} | {r.quality.decision if r.quality else '—'} | "
            f"{s.score:.1f} | {s.label} | {s.defect_count} | {s.area_fraction * 100:.2f} | "
            f"{','.join(str(c) for c in r.present_classes) or '—'} |"
        )

    if failures:
        lines += ["", "## Skipped inputs", "", "| Path | Error |", "|---|---|"]
        for failure in failures:
            lines.append(f"| {failure['path']} | {failure['error']} |")

    lines.append("")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return out_path
