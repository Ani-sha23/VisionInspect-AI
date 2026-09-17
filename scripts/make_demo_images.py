"""Generate synthetic steel-surface images so the pipeline can be demonstrated
without downloading the 6 GB Severstal dataset.

These are NOT real defects and must never be used to report accuracy. They exist
purely so an evaluator can run the CLI end to end on a clean checkout.

    python scripts/make_demo_images.py --out samples --count 6
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np


def make_surface(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    base = rng.normal(118, 9, (height, width)).astype(np.float32)
    # Rolled steel shows horizontal banding; simulate it with a low-frequency sinusoid.
    columns = np.arange(width, dtype=np.float32)
    band = 6.0 * np.sin(columns / 37.0) + 4.0 * np.sin(columns / 11.0)
    base += band[None, :]
    base = cv2.GaussianBlur(base, (0, 0), 1.2)
    return np.clip(base, 0, 255).astype(np.uint8)


def add_scratch(canvas: np.ndarray, rng: np.random.Generator) -> None:
    height, width = canvas.shape[:2]
    x1, y1 = rng.integers(0, width // 2), rng.integers(0, height)
    x2, y2 = rng.integers(width // 2, width), rng.integers(0, height)
    cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), int(rng.integers(30, 70)),
             thickness=int(rng.integers(2, 6)))


def add_patch(canvas: np.ndarray, rng: np.random.Generator) -> None:
    height, width = canvas.shape[:2]
    cx, cy = int(rng.integers(40, width - 40)), int(rng.integers(20, height - 20))
    axes = (int(rng.integers(15, 60)), int(rng.integers(8, 30)))
    cv2.ellipse(canvas, (cx, cy), axes, float(rng.integers(0, 180)), 0, 360,
                int(rng.integers(40, 90)), -1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="samples")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for i in range(args.count):
        gray = make_surface(args.height, args.width, rng)
        defects = 0 if i == 0 else int(rng.integers(1, 4))
        for _ in range(defects):
            if rng.random() < 0.5:
                add_scratch(gray, rng)
            else:
                add_patch(gray, rng)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        name = f"demo_{i:02d}_{'clean' if defects == 0 else f'{defects}def'}.jpg"
        cv2.imwrite(os.path.join(args.out, name), image)
        print(f"wrote {name} ({defects} synthetic defect(s))")

    print(f"\n{args.count} demo images in {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
