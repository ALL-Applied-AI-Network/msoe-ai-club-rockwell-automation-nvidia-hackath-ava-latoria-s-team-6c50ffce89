#!/usr/bin/env python3
"""Run Isaac wrist collection sequentially across prepared run directories."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stacking_recovery.data.grasp_dataset_store import resolve_data_root


def find_runs(explicit: list[Path], runs_dir: Path | None, pattern: str) -> list[Path]:
    candidates = [path.resolve() for path in explicit]
    if runs_dir:
        candidates.extend(path.parent.resolve() for path in runs_dir.glob(pattern))
    unique = {path for path in candidates if (path / "sort_plan.json").is_file()}
    return sorted(unique)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--run", type=Path, action="append", default=[])
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument(
        "--glob",
        default="*/sort_plan.json",
        help="Pattern below --runs-dir (default: */sort_plan.json)",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--trial", type=Path, default=None)
    parser.add_argument(
        "--isaac",
        type=Path,
        default=Path(os.environ.get("ISAAC", "/isaac-sim")),
    )
    parser.add_argument("--physics-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    runs_dir = args.runs_dir.resolve() if args.runs_dir else None
    runs = find_runs(args.run, runs_dir, args.glob)
    if not runs:
        print("No run directories containing sort_plan.json were found.", file=sys.stderr)
        return 2

    out = resolve_data_root(repo, args.out)
    collector = repo / "scripts" / "sim" / "collect_wrist_dataset.py"
    command_prefix = [str(args.isaac / "python.sh"), str(collector), "--repo", str(repo)]
    failures = 0
    for index, run in enumerate(runs, start=1):
        cmd = [*command_prefix, "--run", str(run), "--out", str(out), "--isaac", str(args.isaac)]
        if args.trial:
            cmd.extend(["--trial", str(args.trial)])
        if args.physics_only:
            cmd.append("--physics-only")
        print(f"[{index}/{len(runs)}] {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(cmd, check=False)
        if result.returncode:
            failures += 1
            if not args.continue_on_error:
                return result.returncode
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
