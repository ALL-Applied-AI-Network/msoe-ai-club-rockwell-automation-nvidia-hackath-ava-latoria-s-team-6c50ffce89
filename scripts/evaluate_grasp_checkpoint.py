#!/usr/bin/env python3
"""Evaluate one frozen before/after checkpoint on a separate dataset."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import yaml
    from torch.utils.data import DataLoader
except ImportError:
    raise SystemExit("Install ML dependencies: pip install -e '.[ml]'") from None

from benchmark_grasp_models import TablePairDataset, metrics, predict
from plot_grasp_benchmark import plot_confusion, plot_per_class
from stacking_recovery.vision.pair_models import ModelSpec, build_pair_model


def wilson_interval(successes: int, total: int, z: float = 1.959964) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total**2)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def scenario_name(row: dict) -> str:
    return str(
        row.get("campaign_scenario")
        or row.get("condition_id")
        or row.get("scenario")
        or "unknown"
    )


def plot_scenarios(by_scenario: dict[str, dict], output: Path) -> None:
    names = sorted(by_scenario)
    x = np.arange(len(names))
    accuracy = [by_scenario[name]["accuracy"] for name in names]
    unsafe = [by_scenario[name]["unsafe_good_rate"] for name in names]
    fig, ax = plt.subplots(figsize=(max(8, 1.7 * len(names)), 4.8))
    width = 0.36
    ax.bar(x - width / 2, accuracy, width, label="Accuracy", color="#2563eb")
    ax.bar(x + width / 2, unsafe, width, label="Unsafe GOOD rate", color="#dc2626")
    ax.set_xticks(x, [name.replace("_", "\n") for name in names])
    ax.set(title="Frozen model by fresh-test scenario", ylabel="Score", ylim=(0, 1.05))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "scenario_breakdown.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    pair_mode = "siamese" if config.get("table_pair_mode") == "siamese_rgb_pair" else "delta"
    spec = ModelSpec(str(config.get("backbone", "mobilenet_v3_small")), pair_mode, pretrained=False)
    input_size = int(config.get("input_size", 224))
    threshold = float(config.get("thresholds", {}).get("tau_good", 0.5))

    rows = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    selected = rows if args.split == "all" else [row for row in rows if row.get("split") == args.split]
    if not selected:
        raise SystemExit(f"No {args.split!r} rows in {args.data}")

    dataset = TablePairDataset(
        selected,
        args.data,
        pair_mode,
        input_size,
        augment=False,
        seed=0,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_pair_model(spec, 2).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    truth, probability, latency = predict(model, loader, device)
    report = metrics(truth, probability, threshold)
    correct = round(report["accuracy"] * report["n"])
    report["accuracy_wilson_95"] = wilson_interval(correct, report["n"])

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(selected):
        grouped[scenario_name(row)].append(index)
    by_scenario: dict[str, dict] = {}
    for name, indices in sorted(grouped.items()):
        scenario_report = metrics(
            [truth[index] for index in indices],
            [probability[index] for index in indices],
            threshold,
        )
        scenario_correct = round(scenario_report["accuracy"] * scenario_report["n"])
        scenario_report["accuracy_wilson_95"] = wilson_interval(
            scenario_correct, scenario_report["n"]
        )
        by_scenario[name] = scenario_report

    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "name": "frozen_resnet18_siamese_external_test",
        "classes": ["GOOD", "SHIFTED"],
        "dataset": str(args.data),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "threshold_source": "frozen_validation_threshold",
        "threshold_good": threshold,
        "test_latency_ms": latency,
        "test": report,
        "by_scenario": by_scenario,
    }
    (args.output / "external_test_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    plot_confusion(result, args.output)
    plot_per_class(result, args.output)
    plot_scenarios(by_scenario, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
