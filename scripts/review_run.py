#!/usr/bin/env python3
"""Print a teammate-friendly review of one Isaac Sim run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stacking_recovery.bench.metrics import diagnose_run, summarize_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="Run directory with execution_report.json")
    args = parser.parse_args()
    summary = summarize_run(args.run)
    print(f"Run: {args.run}")
    print(f"Complete: {summary['task_complete']}")
    print(
        "Verified pickups / placements: "
        f"{summary['verified_pickups']} / {summary['verified_placements']}"
    )
    print(f"Abort reason: {summary['abort_reason'] or 'none'}")
    first_failure = summary.get("first_failure")
    if first_failure:
        where = first_failure.get("detection_id")
        where = (
            f"detection {where}"
            if where is not None
            else f"pick {first_failure['pick_index']}"
        )
        reason = first_failure.get("reason") or "not recorded"
        print(f"First failure: {first_failure['phase']} at {where} ({reason})")
    print(f"AI vs physics: {summary['grasp_confusion'] or 'no comparable samples'}")
    print("What to change next:")
    for item in diagnose_run(summary):
        print(f"- {item}")


if __name__ == "__main__":
    main()
