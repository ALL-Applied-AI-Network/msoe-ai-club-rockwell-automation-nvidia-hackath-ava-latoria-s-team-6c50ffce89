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
    picks = report.get("picks", report.get("events", []))
    if not isinstance(picks, list):
        picks = []
    for pick in picks:
        if not isinstance(pick, dict):
            continue
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
    first_failure = None
    for index, pick in enumerate(picks):
        if not isinstance(pick, dict):
            continue
        phase = None
        result = None
        if pick.get("pickup_verified") is False:
            phase = "pickup"
            result = pick.get("closed_loop_grasp")
        elif pick.get("placement_verified") is False:
            phase = "placement"
            result = pick.get("closed_loop_place")
        if phase:
            first_failure = {
                "pick_index": index,
                "phase": phase,
                "detection_id": pick.get("detection_id"),
                "stack_level": pick.get("stack_level"),
                "reason": result.get("reason") if isinstance(result, dict) else None,
            }
            break
    return {
        "run_dir": str(run_dir),
        "report_found": report_path.exists(),
        "observed_picks": len(picks),
        "task_complete": report.get("task_complete"),
        "verified_pickups": report.get("verified_pickups"),
        "verified_placements": report.get("verified_placements"),
        "pick_attempts": sum(
            1 + int(pick.get("closed_loop_grasp_attempts", 0)) for pick in picks
        ),
        "placement_errors_mm": [
            float(pick["placement_error_mm"])
            for pick in picks
            if pick.get("placement_error_mm") is not None
        ],
        "cycle_times_s": [
            float(pick["cycle_time_s"])
            for pick in picks
            if pick.get("cycle_time_s") is not None
        ],
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
        "abort_reason": report.get("abort_reason"),
        "first_failure": first_failure,
    }


def diagnose_run(summary: dict[str, Any]) -> list[str]:
    """Turn run metrics into short, actionable next steps."""
    if not summary.get("report_found"):
        return [
            "No execution_report.json was found. Check the simulator log for an "
            "early startup, asset, camera, or IK failure before tuning the model."
        ]
    if summary.get("task_complete") is True:
        return [
            "Run completed successfully; repeat it across randomized seeds before "
            "changing control logic."
        ]

    advice: list[str] = []
    abort = str(summary.get("abort_reason") or "").lower()
    confusion = summary.get("grasp_confusion") or {}
    if confusion.get("false_positive", 0):
        advice.append(
            "The verifier accepted at least one grasp that physics says failed. Use "
            "ai_guarded or rules for the demo, inspect the matching wrist frame, and "
            "raise the GOOD threshold only after validation."
        )
    if confusion.get("false_negative", 0):
        advice.append(
            "The verifier rejected at least one physically successful grasp. Review "
            "that wrist frame for crop or label problems before retraining."
        )
    if "closed-loop grasp stop" in abort and not confusion.get("false_negative", 0):
        advice.append(
            "Grasp verification exhausted its retries. Inspect the saved grasp result, "
            "jaw gap, wrist crop, and the first failure reason before changing thresholds."
        )
    pickups = summary.get("verified_pickups")
    placements = summary.get("verified_placements")
    observed = summary.get("observed_picks")
    if (
        isinstance(pickups, int)
        and isinstance(observed, int)
        and observed > 0
        and pickups < observed
        and not advice
    ):
        advice.append(
            "At least one attempted pickup failed; inspect wrist alignment, jaw gap, "
            "friction, and the first failed pick in execution_report.json."
        )
    if isinstance(placements, int) and isinstance(pickups, int) and placements < pickups:
        advice.append(
            "At least one lifted cube was not placed correctly; inspect placement XY/Z, "
            "yaw, tilt, and stack-support checks."
        )
    if "ik failed" in abort:
        advice.append(
            "Motion planning failed before the task could finish. Check the commanded "
            "pose, offsets, and reachability before changing vision or recovery logic."
        )
    if not advice:
        advice.append(
            "The run is incomplete; inspect abort_reason and the first failed pick before "
            "changing thresholds."
        )
    return advice


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
