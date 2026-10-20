"""Aggregate execution_report.json and manipulation JSONL."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def summarize_run(run_dir: Path) -> dict[str, Any]:
    report_path = run_dir / "execution_report.json"
    events_path = run_dir / "manipulation.jsonl"
    legacy_path = run_dir / "recovery.jsonl"
    report = {}
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    events = _read_jsonl(events_path) or _read_jsonl(legacy_path)
    grasp_checks = [e for e in events if e.get("action") == "grasp_check"]
    first_grasp_ok = (
        grasp_checks[0].get("seated") is True if grasp_checks else None
    )
    regrasps = sum(1 for e in events if e.get("action") == "regrasp")
    recoveries = sum(
        1
        for i, e in enumerate(events)
        if e.get("action") == "regrasp"
        and i + 1 < len(events)
        and events[i + 1].get("outcome") == "success"
    )
    failures = Counter(
        e.get("reason")
        for e in events
        if e.get("outcome") == "failed" or e.get("action") == "safe_stop"
    )
    durations = [e["duration_s"] for e in events if e.get("duration_s") is not None]
    attempt_indices = [e.get("attempt_index", 0) for e in grasp_checks]
    comparisons = Counter()
    for pick in report.get("events", []):
        grasp = pick.get("closed_loop_grasp")
        physics = pick.get("pickup_verified")
        if not isinstance(grasp, dict) or not isinstance(physics, bool):
            continue
        predicted_good = grasp.get("seated") is True
        if predicted_good and physics:
            comparisons["true_positive"] += 1
        elif predicted_good and not physics:
            comparisons["false_positive"] += 1
        elif not predicted_good and physics:
            comparisons["false_negative"] += 1
        else:
            comparisons["true_negative"] += 1
    fp_denominator = comparisons["false_positive"] + comparisons["true_negative"]
    fn_denominator = comparisons["false_negative"] + comparisons["true_positive"]
    return {
        "run_dir": str(run_dir),
        "task_complete": report.get("task_complete"),
        "verified_pickups": report.get("verified_pickups"),
        "verified_placements": report.get("verified_placements"),
        "first_grasp_success": first_grasp_ok,
        "regrasp_count": regrasps,
        "recovery_success_count": recoveries,
        "mean_retries": (
            sum(attempt_indices) / len(attempt_indices) if attempt_indices else 0.0
        ),
        "mean_duration_s": sum(durations) / len(durations) if durations else None,
        "failure_reasons": dict(failures),
        "grasp_confusion": dict(comparisons),
        "false_positive_rate": (
            comparisons["false_positive"] / fp_denominator if fp_denominator else None
        ),
        "false_negative_rate": (
            comparisons["false_negative"] / fn_denominator if fn_denominator else None
        ),
    }


def summarize_suite(run_dirs: list[Path]) -> dict[str, Any]:
    rows = [summarize_run(p) for p in run_dirs]
    n = len(rows)
    complete = sum(1 for r in rows if r.get("task_complete") is True)
    first_ok = [r["first_grasp_success"] for r in rows if r["first_grasp_success"] is not None]
    confusion = Counter()
    for row in rows:
        confusion.update(row.get("grasp_confusion", {}))
    fp_denominator = confusion["false_positive"] + confusion["true_negative"]
    fn_denominator = confusion["false_negative"] + confusion["true_positive"]
    return {
        "n_runs": n,
        "tower_completion_rate": complete / n if n else 0.0,
        "first_grasp_success_rate": sum(first_ok) / len(first_ok) if first_ok else None,
        "total_recoveries": sum(r.get("recovery_success_count", 0) for r in rows),
        "grasp_confusion": dict(confusion),
        "false_positive_rate": (
            confusion["false_positive"] / fp_denominator if fp_denominator else None
        ),
        "false_negative_rate": (
            confusion["false_negative"] / fn_denominator if fn_denominator else None
        ),
        "runs": rows,
    }


def format_summary_markdown(suite: dict[str, Any]) -> str:
    lines = [
        "# Benchmark summary",
        "",
        f"Runs: {suite.get('n_runs', 0)} | "
        f"Tower completion: {suite.get('tower_completion_rate', 0):.1%} | "
        f"First grasp: {suite.get('first_grasp_success_rate')}",
        f"AI false-positive rate: {suite.get('false_positive_rate')} | "
        f"AI false-negative rate: {suite.get('false_negative_rate')}",
        "",
        "| run | task_complete | first_grasp | regrasps | failures |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in suite.get("runs", []):
        lines.append(
            f"| {Path(row['run_dir']).name} | {row.get('task_complete')} | "
            f"{row.get('first_grasp_success')} | {row.get('regrasp_count')} | "
            f"{row.get('failure_reasons')} |"
        )
    lines.append("")
    return "\n".join(lines)
