#!/usr/bin/env python3
"""Combine dataset, classifier, and Isaac RL evidence into one go/no-go report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_FAILURE_SCENARIOS = (
    "missed_grasp",
    "shifted_cube",
    "occlusion",
    "placement_offset",
)


def binary_label(label: object) -> str:
    return "GOOD" if label == "GOOD" else "FAILURE"


def audit_dataset(data_root: Path, target_test_per_class: int) -> dict[str, Any]:
    manifest = data_root / "manifest.json"
    if not manifest.is_file():
        return {"status": "missing", "manifest": str(manifest)}
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    test_rows = [row for row in rows if row.get("split") == "test"]
    counts = Counter(binary_label(row.get("label")) for row in test_rows)
    paths = [str(row.get("path")) for row in test_rows]
    seeds = [row.get("seed") for row in test_rows if isinstance(row.get("seed"), int)]
    missing = {
        label: max(0, target_test_per_class - counts[label])
        for label in ("GOOD", "FAILURE")
    }
    complete = not any(missing.values())
    return {
        "status": "pass" if complete else "collecting",
        "manifest": str(manifest),
        "test_samples": len(test_rows),
        "target_test_samples": 2 * target_test_per_class,
        "counts": dict(sorted(counts.items())),
        "missing": missing,
        "unique_seeds": len(set(seeds)),
        "duplicate_seed_count": len(seeds) - len(set(seeds)),
        "duplicate_path_count": len(paths) - len(set(paths)),
        "collection_complete": complete,
    }


def classifier_status(path: Path | None, required_samples: int) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"status": "pending", "reason": "external classifier metrics not generated"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("test", {})
    n = int(metrics.get("n", 0))
    accuracy = float(metrics.get("accuracy", 0.0))
    unsafe = float(metrics.get("unsafe_good_rate", 1.0))
    passed = n >= required_samples and accuracy >= 0.95 and unsafe <= 0.02
    return {
        "status": "pass" if passed else "fail",
        "metrics": str(path),
        "n": n,
        "accuracy": accuracy,
        "unsafe_good_rate": unsafe,
        "requirements": {
            "minimum_samples": required_samples,
            "minimum_accuracy": 0.95,
            "maximum_unsafe_good_rate": 0.02,
        },
    }


def rl_status(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"status": "pending", "reason": "matched Isaac RL comparison not run"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    acceptance = dict(payload.get("acceptance") or {})
    return {
        "status": acceptance.get("status", "pending"),
        "metrics": str(path),
        "acceptance": acceptance,
        "aggregate": payload.get("aggregate", {}),
        "paired_rl_comparison": payload.get("paired_rl_comparison", {}),
    }


def failure_handling_status(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"status": "pending", "reason": "intentional-failure evidence not generated"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    total = int(payload.get("total_trials", 0))
    counts = {
        scenario: int((payload.get("scenario_counts") or {}).get(scenario, 0))
        for scenario in REQUIRED_FAILURE_SCENARIOS
    }
    safe_outcomes = int(payload.get("safe_outcomes", 0))
    unsafe_carries = int(payload.get("unsafe_carries", 0))
    ignored_failures = int(payload.get("ignored_verification_failures", 0))
    passed = (
        total >= 20
        and all(count >= 1 for count in counts.values())
        and safe_outcomes == total
        and unsafe_carries == 0
        and ignored_failures == 0
    )
    return {
        "status": "pass" if passed else "fail",
        "metrics": str(path),
        "total_trials": total,
        "scenario_counts": counts,
        "safe_outcomes": safe_outcomes,
        "unsafe_carries": unsafe_carries,
        "ignored_verification_failures": ignored_failures,
        "requirements": {
            "minimum_trials": 20,
            "required_scenarios": list(REQUIRED_FAILURE_SCENARIOS),
            "all_outcomes_must_be_safe": True,
            "maximum_unsafe_carries": 0,
            "maximum_ignored_verification_failures": 0,
        },
    }


def repeatability_status(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"status": "pending", "reason": "clean-restart evidence not generated"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    clean_restarts = int(payload.get("clean_restarts", 0))
    successful_restarts = int(payload.get("successful_restarts", 0))
    identical_plan = bool(payload.get("identical_plan", False))
    passed = clean_restarts >= 3 and successful_restarts >= 3 and identical_plan
    return {
        "status": "pass" if passed else "fail",
        "metrics": str(path),
        "clean_restarts": clean_restarts,
        "successful_restarts": successful_restarts,
        "identical_plan": identical_plan,
        "requirements": {
            "minimum_clean_restarts": 3,
            "minimum_successful_restarts": 3,
            "identical_plan_required": True,
        },
    }


def build_report(
    data_root: Path,
    classifier_metrics: Path | None,
    rl_metrics: Path | None,
    target_test_per_class: int,
    failure_metrics: Path | None = None,
    repeatability_metrics: Path | None = None,
) -> dict[str, Any]:
    dataset = audit_dataset(data_root, target_test_per_class)
    classifier = classifier_status(classifier_metrics, 2 * target_test_per_class)
    rl = rl_status(rl_metrics)
    failures = failure_handling_status(failure_metrics)
    repeatability = repeatability_status(repeatability_metrics)
    statuses = [
        dataset["status"],
        classifier["status"],
        rl["status"],
        failures["status"],
        repeatability["status"],
    ]
    if all(status == "pass" for status in statuses):
        overall = "go"
    elif "fail" in statuses:
        overall = "no_go"
    else:
        overall = "pending"
    return {
        "overall": overall,
        "dataset": dataset,
        "classifier": classifier,
        "rl_digital_twin": rl,
        "failure_handling": failures,
        "repeatability": repeatability,
        "rule": "GO only when all five documented lab-readiness gates pass.",
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    dataset = report["dataset"]
    classifier = report["classifier"]
    rl = report["rl_digital_twin"]
    failures = report["failure_handling"]
    repeatability = report["repeatability"]
    lines = [
        "# Lab Readiness Scorecard",
        "",
        f"**Overall: {str(report['overall']).upper()}**",
        "",
        f"- Dataset: {dataset['status']} ({dataset.get('test_samples', 0)}/{dataset.get('target_test_samples', 0)} test pairs)",
        f"- Frozen classifier: {classifier['status']}",
        f"- Matched Isaac RL comparison: {rl['status']}",
        f"- Intentional failure handling: {failures['status']}",
        f"- Clean-restart repeatability: {repeatability['status']}",
        "",
        "A GO requires all five documented gates to pass. Lightweight RL training reward is diagnostic only.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--classifier-metrics", type=Path)
    parser.add_argument("--rl-metrics", type=Path)
    parser.add_argument("--failure-metrics", type=Path)
    parser.add_argument("--repeatability-metrics", type=Path)
    parser.add_argument("--target-test-per-class", type=int, default=250)
    parser.add_argument("--output", type=Path, default=Path("results/lab_readiness"))
    args = parser.parse_args()
    if args.target_test_per_class < 1:
        parser.error("--target-test-per-class must be positive")
    report = build_report(
        args.data,
        args.classifier_metrics,
        args.rl_metrics,
        args.target_test_per_class,
        args.failure_metrics,
        args.repeatability_metrics,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "readiness.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(report, args.output / "README.md")
    print(json.dumps(report, indent=2))
    return 0 if report["overall"] == "go" else 2


if __name__ == "__main__":
    raise SystemExit(main())
