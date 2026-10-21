#!/usr/bin/env python3
"""Confusion matrix and latency for grasp-quality model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import cv2
except ImportError:
    raise SystemExit("pip install opencv-python") from None

from stacking_recovery.vision.grasp_quality_model import (
    GraspQualityModel,
    configured_classes,
    load_grasp_config,
    normalize_grasp_label,
)
from stacking_recovery.data.grasp_dataset_store import resolve_data_root
from stacking_recovery.data.evaluation import classification_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Dataset root (default: GRASP_DATA_ROOT or repo/data/grasp_quality)",
    )
    ap.add_argument("--max", type=int, default=500)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "grasp_model.yaml")
    ap.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    ap.add_argument("--source", action="append", default=[], help="Filter by source; repeatable")
    ap.add_argument(
        "--observation-source",
        default=None,
        help="Filter by observation format, e.g. table_before_after",
    )
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    args.data = resolve_data_root(ROOT, args.data)
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    cfg = load_grasp_config(args.config)
    classes = configured_classes(cfg)
    model = GraspQualityModel(cfg)
    model.load()
    selected = [
        row
        for row in manifest
        if (args.split == "all" or row.get("split") == args.split)
        and (not args.source or row.get("source") in args.source)
        and (
            not args.observation_source
            or row.get("observation_source") == args.observation_source
        )
    ][: args.max]
    if not selected:
        raise SystemExit(
            f"No samples matched split={args.split!r} source={args.source or 'any'}."
        )
    pairs: list[tuple[str, str]] = []
    latencies = []
    skipped = 0
    for row in selected:
        img = cv2.imread(str(args.data / row["path"]))
        if img is None:
            skipped += 1
            continue
        t0 = time.perf_counter()
        pred = model.predict(
            img, observation_source=row.get("observation_source")
        )
        latencies.append(time.perf_counter() - t0)
        truth = normalize_grasp_label(row["label"], cfg)
        pairs.append((truth, pred.class_name))
    report = classification_report(pairs, classes)
    report.update(
        {
            "split": args.split,
            "sources": args.source,
            "skipped_images": skipped,
            "classes": classes,
            "label_mode": cfg.get("label_mode", "detailed"),
        }
    )
    print(
        f"accuracy={report['accuracy']:.3f} n={report['n']} "
        f"unsafe_good_rate={report['unsafe_good_rate']:.3f}"
    )
    if latencies:
        ms = 1000 * sum(latencies) / len(latencies)
        print(f"mean_latency_ms={ms:.1f}")
    for label, metrics in report["per_class"].items():
        if metrics["support"]:
            print(
                f"  {label}: precision={metrics['precision']:.3f} "
                f"recall={metrics['recall']:.3f} f1={metrics['f1']:.3f} "
                f"n={metrics['support']}"
            )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
