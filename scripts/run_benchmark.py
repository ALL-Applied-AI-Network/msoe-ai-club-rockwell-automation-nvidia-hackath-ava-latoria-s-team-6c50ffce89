#!/usr/bin/env python3
"""List or execute perturbation trials (Isaac invocation TODO)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stacking_recovery.bench.perturbations import load_trials, trial_to_overrides


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trials-dir",
        type=Path,
        default=ROOT / "configs" / "trials",
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Run fanuc starter pipeline per trial (not implemented)",
    )
    args = ap.parse_args()
    trials = load_trials(args.trials_dir)
    print(f"Found {len(trials)} trials under {args.trials_dir}")
    for t in trials:
        ov = trial_to_overrides(t)
        print(
            f"  {ov['condition_id']} seed={ov['seed']} "
            f"policy={ov['policy']} finger={ov['finger']}"
        )
    if args.execute:
        raise SystemExit(
            "Wire to data/.../fanuc_er4ia_sim/run_sort_pipeline.sh "
            "with RUN_ROOT=results/runs/<trial_id>"
        )


if __name__ == "__main__":
    main()
