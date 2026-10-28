#!/usr/bin/env python3
"""Create a plain-language go/no-go scorecard for grasp-and-sort trials."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def build_scorecard(
    reports: list[dict[str, Any]],
    *,
    min_pick_success: float = 0.90,
    placement_tolerance_mm: float = 20.0,
    min_placement_success: float = 0.90,
    max_mean_cycle_s: float = 30.0,
    min_recovery_success: float = 0.80,
) -> dict[str, Any]:
    picks = [pick for report in reports for pick in report.get("picks", [])]
    attempts = sum(1 + int(p.get("closed_loop_grasp_attempts", 0)) for p in picks)
    successful_picks = sum(bool(p.get("pickup_verified")) for p in picks)
    placement_errors = [
        float(p["placement_error_mm"])
        for p in picks
        if p.get("placement_error_mm") is not None
    ]
    accurate_placements = sum(
        bool(p.get("pickup_verified"))
        and p.get("placement_error_mm") is not None
        and float(p["placement_error_mm"]) <= placement_tolerance_mm
        for p in picks
    )
    cycle_times = [
        float(p["cycle_time_s"])
        for p in picks
        if p.get("cycle_time_s") is not None
    ]
    recovery_picks = [
        p
        for p in picks
        if int(p.get("closed_loop_grasp_attempts", 0)) > 0
        or int(p.get("closed_loop_placement_attempts", 0)) > 0
    ]
    recovered = sum(bool(p.get("placement_verified")) for p in recovery_picks)

    pick_rate = successful_picks / attempts if attempts else 0.0
    placement_rate = accurate_placements / len(picks) if picks else 0.0
    recovery_rate = recovered / len(recovery_picks) if recovery_picks else None
    mean_cycle = statistics.fmean(cycle_times) if cycle_times else None
    checks = {
        "enough_trials": len(picks) >= 20,
        "pick_success": pick_rate >= min_pick_success,
        "placement_success": placement_rate >= min_placement_success,
        "cycle_time": mean_cycle is not None and mean_cycle <= max_mean_cycle_s,
        "recovery": recovery_rate is not None and recovery_rate >= min_recovery_success,
    }
    return {
        "decision": "GO_TO_STACKING" if all(checks.values()) else "KEEP_TESTING_SORT",
        "checks": checks,
        "runs": len(reports),
        "planned_picks": len(picks),
        "pick_attempts": attempts,
        "successful_picks": successful_picks,
        "pick_success_per_attempt": pick_rate,
        "placement_success_rate": placement_rate,
        "placement_error_mean_mm": statistics.fmean(placement_errors) if placement_errors else None,
        "placement_error_p95_mm": _percentile(placement_errors, 0.95),
        "mean_cycle_time_s": mean_cycle,
        "recovery_cases": len(recovery_picks),
        "successful_recoveries": recovered,
        "recovery_success_rate": recovery_rate,
        "thresholds": {
            "minimum_trials": 20,
            "pick_success_per_attempt": min_pick_success,
            "placement_tolerance_mm": placement_tolerance_mm,
            "placement_success_rate": min_placement_success,
            "maximum_mean_cycle_time_s": max_mean_cycle_s,
            "recovery_success_rate": min_recovery_success,
        },
    }


def markdown(score: dict[str, Any]) -> str:
    def pct(value: float | None) -> str:
        return "not measured" if value is None else f"{value:.1%}"

    def num(value: float | None, suffix: str) -> str:
        return "not measured" if value is None else f"{value:.2f}{suffix}"

    lines = [
        "# Grasp-and-sort readiness",
        "",
        f"Decision: **{score['decision']}**",
        "",
        f"- Successful picks per attempt: {score['successful_picks']}/{score['pick_attempts']} ({pct(score['pick_success_per_attempt'])})",
        f"- Accurate placements: {pct(score['placement_success_rate'])}",
        f"- Placement error: mean {num(score['placement_error_mean_mm'], ' mm')}, p95 {num(score['placement_error_p95_mm'], ' mm')}",
        f"- Mean cycle time: {num(score['mean_cycle_time_s'], ' s')}",
        f"- Recovery after a failed attempt: {score['successful_recoveries']}/{score['recovery_cases']} ({pct(score['recovery_success_rate'])})",
        f"- Trials: {score['planned_picks']} picks across {score['runs']} runs",
        "",
        "## Gate checks",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'NOT YET'} — {name.replace('_', ' ')}"
        for name, passed in score["checks"].items()
    )
    lines.extend([
        "",
        "Move to stacking only after every check passes. Keep the classifier as a safety gate and use RL only for bounded placement/recovery adjustments.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    paths = sorted(args.results_dir.rglob("execution_report.json"))
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    score = build_scorecard(reports)
    output = args.output_dir or args.results_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "grasp_sort_scorecard.json").write_text(
        json.dumps(score, indent=2) + "\n", encoding="utf-8"
    )
    (output / "grasp_sort_scorecard.md").write_text(markdown(score), encoding="utf-8")
    print(markdown(score))


if __name__ == "__main__":
    main()
