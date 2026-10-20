#!/usr/bin/env python3
"""List perturbation trials; optionally write benchmark summary from results/runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stacking_recovery.bench.metrics import format_summary_markdown, summarize_suite
from stacking_recovery.bench.perturbations import load_trials, trial_to_overrides


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trials-dir",
        type=Path,
        default=ROOT / "configs" / "trials",
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "runs",
    )
    ap.add_argument(
        "--write-summary",
        action="store_true",
        help="Aggregate results/runs/*/execution_report.json into summary.md",
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Run fanuc starter pipeline per trial (requires ISAAC/COSMOS)",
    )
    ap.add_argument(
        "--verification",
        choices=("rules", "ai", "ai_guarded", "stub"),
        default="rules",
        help=(
            "Closed-loop grasp checker used by executed trials; ai_guarded "
            "requires AI and deterministic rules to agree"
        ),
    )
    ap.add_argument(
        "--max-trials",
        type=int,
        help="Execute only the first N sorted trials (useful for a smoke test)",
    )
    ap.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running later trials if one simulator run fails",
    )
    args = ap.parse_args()
    trials = load_trials(args.trials_dir)
    print(f"Found {len(trials)} trials under {args.trials_dir}")
    for t in trials[:20]:
        ov = trial_to_overrides(t)
        print(
            f"  {ov['condition_id']} seed={ov['seed']} "
            f"policy={ov['policy']} finger={ov['finger']}"
        )
    if len(trials) > 20:
        print(f"  ... and {len(trials) - 20} more")
    if args.execute:
        sim = ROOT / "data" / "fanuc_er4ia_hackathon_starter_v05 (1)" / "fanuc_er4ia_sim"
        selected = trials[: args.max_trials] if args.max_trials else trials
        args.results_dir.mkdir(parents=True, exist_ok=True)
        failures: list[str] = []
        for trial in selected:
            overrides = trial_to_overrides(trial)
            trial_id = (
                f"{overrides['condition_id']}_{overrides['policy']}_"
                f"{overrides['finger']}_seed{overrides['seed']}"
            )
            run_dir = (args.results_dir / trial_id).resolve()
            if run_dir.exists() and any(run_dir.iterdir()):
                raise SystemExit(
                    f"Run directory is not empty: {run_dir}. "
                    "Move it aside or choose a different --results-dir."
                )
            env = os.environ.copy()
            trial_path = next(
                path
                for path in sorted(args.trials_dir.glob("*.json"))
                if json.loads(path.read_text(encoding="utf-8")) == trial
            )
            env["TRIAL_CONFIG"] = str(trial_path.resolve())
            env["CLOSED_LOOP"] = (
                "1" if overrides["policy"] == "closed_loop" else "0"
            )
            env["VERIFICATION"] = args.verification
            print(f"Running {trial_id} -> {run_dir}", flush=True)
            result = subprocess.run(
                ["bash", str(sim / "run_sort_pipeline.sh"), str(run_dir)],
                cwd=sim,
                env=env,
                check=False,
            )
            if result.returncode:
                failures.append(trial_id)
                if not args.continue_on_error:
                    raise SystemExit(result.returncode)
        if failures:
            print(f"Failed trials: {', '.join(failures)}", file=sys.stderr)
    if args.write_summary or args.execute:
        run_dirs = sorted(
            p
            for p in args.results_dir.iterdir()
            if p.is_dir() and (p / "execution_report.json").is_file()
        )
        suite = summarize_suite(run_dirs)
        md = format_summary_markdown(suite)
        out = ROOT / "results" / "benchmarks" / "summary.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
