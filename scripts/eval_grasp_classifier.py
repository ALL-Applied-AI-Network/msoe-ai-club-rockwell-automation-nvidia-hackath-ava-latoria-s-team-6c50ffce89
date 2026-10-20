#!/usr/bin/env python3
"""Confusion matrix and latency for grasp-quality model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import cv2
except ImportError:
    raise SystemExit("pip install opencv-python") from None

from stacking_recovery.vision.grasp_quality_model import GraspQualityModel, load_grasp_config
from stacking_recovery.data.grasp_dataset_store import resolve_data_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Dataset root (default: GRASP_DATA_ROOT or repo/data/grasp_quality)",
    )
    ap.add_argument("--max", type=int, default=500)
    args = ap.parse_args()
    args.data = resolve_data_root(ROOT, args.data)
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    model = GraspQualityModel()
    model.load()
    correct = 0
    confusion: Counter[tuple[str, str]] = Counter()
    latencies = []
    for row in manifest[: args.max]:
        img = cv2.imread(str(args.data / row["path"]))
        if img is None:
            continue
        t0 = time.perf_counter()
        pred = model.predict(img)
        latencies.append(time.perf_counter() - t0)
        truth = row["label"]
        confusion[(truth, pred.class_name)] += 1
        if pred.class_name == truth:
            correct += 1
    n = min(len(manifest), args.max)
    print(f"accuracy={correct / n:.3f} n={n}")
    if latencies:
        ms = 1000 * sum(latencies) / len(latencies)
        print(f"mean_latency_ms={ms:.1f}")
    for (truth, pred), count in sorted(confusion.items()):
        if truth != pred:
            print(f"  {truth} -> {pred}: {count}")


if __name__ == "__main__":
    main()
