#!/usr/bin/env python3
"""
Batch Isaac wrist dataset collection via sort_scene.py --collect-dataset.

Requires ISAAC python and an existing run directory (capture + perceive + plan).

Example (on Brev):
  export ISAAC=/isaacsim
  export REPO=/path/to/team-repo
  $ISAAC/python.sh $REPO/scripts/sim/collect_wrist_dataset.py \\
    --repo $REPO \\
    --run $REPO/results/runs/my_capture \\
    --trial configs/trials/pick_xy_shift_closed_loop_seed0.json
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stacking_recovery.data.grasp_dataset_store import resolve_data_root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True, help="Existing sim run with sort_plan.json")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Dataset root (default: repo/data/grasp_quality)",
    )
    ap.add_argument("--trial", type=Path, default=None, help="configs/trials/*.json")
    ap.add_argument("--isaac", type=Path, default=Path(os.environ.get("ISAAC", "/isaacsim")))
    ap.add_argument("--physics-only", action="store_true")
    args = ap.parse_args()

    repo = args.repo.resolve()
    run = args.run.resolve()
    out = resolve_data_root(repo, args.out)
    sim_dir = repo / "data" / "fanuc_er4ia_hackathon_starter_v05 (1)" / "fanuc_er4ia_sim"
    sort_scene = sim_dir / "sort_scene.py"
    isaac_py = args.isaac / "python.sh"

    if not sort_scene.is_file():
        print(f"Missing {sort_scene}", file=sys.stderr)
        return 2
    if not isaac_py.is_file():
        print(
            f"Isaac not found at {isaac_py}. Set ISAAC= or run sort_scene manually.",
            file=sys.stderr,
        )
        return 2
    if not (run / "sort_plan.json").is_file():
        print(f"Run missing sort_plan.json: {run}", file=sys.stderr)
        return 2

    cmd = [
        str(isaac_py),
        str(sort_scene),
        "--run",
        str(run),
        "--collect-dataset",
        str(out),
    ]
    if args.trial:
        cmd.extend(["--trial-config", str((repo / args.trial).resolve() if not args.trial.is_absolute() else args.trial)])
    if args.physics_only:
        cmd.append("--physics-only")

    print("Running:", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
