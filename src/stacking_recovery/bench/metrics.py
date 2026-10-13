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
    regrasps = sum(1 for e in events if e.get("action") == "regrasp")
    grasp_checks = [e for e in events if e.get("action") == "grasp_check"]
    first_grasp_ok = (
        grasp_checks[0].get("seated") is True if grasp_checks else None
    )
    failures = Counter(
        e.get("reason")
        for e in events
        if e.get("outcome") == "failed" or e.get("action") == "safe_stop"
    )
    durations = [e["duration_s"] for e in events if e.get("duration_s") is not None]
    return {
        "run_dir": str(run_dir),
        "task_complete": report.get("task_complete"),
        "verified_pickups": report.get("verified_pickups"),
        "verified_placements": report.get("verified_placements"),
        "first_grasp_success": first_grasp_ok,
        "regrasp_count": regrasps,
        "mean_duration_s": sum(durations) / len(durations) if durations else None,
        "failure_reasons": dict(failures),
    }


def summarize_suite(run_dirs: list[Path]) -> dict[str, Any]:
    rows = [summarize_run(p) for p in run_dirs]
    n = len(rows)
    complete = sum(1 for r in rows if r.get("task_complete") is True)
    first_ok = [r["first_grasp_success"] for r in rows if r["first_grasp_success"] is not None]
    return {
        "n_runs": n,
        "tower_completion_rate": complete / n if n else 0.0,
        "first_grasp_success_rate": sum(first_ok) / len(first_ok) if first_ok else None,
        "runs": rows,
    }
