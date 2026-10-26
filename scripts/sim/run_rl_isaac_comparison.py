#!/usr/bin/env python3
"""Compare open-loop, deterministic, and RL stacking on identical Isaac seeds."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def build_jobs(seeds: list[int], offset_m: float) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    for seed in seeds:
        angle = math.radians((seed * 137.508) % 360.0)
        offset = [round(offset_m * math.cos(angle), 6), round(offset_m * math.sin(angle), 6)]
        for policy in ("open_loop", "closed_loop", "rl"):
            jobs.append(
                {
                    "condition_id": "placement_shift",
                    "seed": seed,
                    "policy": policy,
                    "finger": "stock_aa_scallop",
                    "place_xy_offset_m": offset,
                    "notes": "same hidden placement error for all policy modes",
                }
            )
    return jobs


def summarize(report_paths: list[Path]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for path in report_paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        picks = list(report.get("picks") or [])
        rows.append(
            {
                "policy": report.get("policy"),
                "task_complete": bool(report.get("task_complete")),
                "verified_pickups": int(report.get("verified_pickups", 0)),
                "verified_placements": int(report.get("verified_placements", 0)),
                "grasp_retries": sum(
                    int(p.get("closed_loop_grasp_attempts", 0)) for p in picks
                ),
                "placement_retries": sum(
                    int(p.get("closed_loop_placement_attempts", 0)) for p in picks
                ),
                "visual_gate_proceeds": sum(
                    (p.get("rl_visual_verification") or {}).get("gate", {}).get("action")
                    == "proceed"
                    for p in picks
                ),
                "visual_gate_stops": sum(
                    (p.get("rl_visual_verification") or {}).get("gate", {}).get("action")
                    == "safe_stop"
                    for p in picks
                ),
                "abort_reason": report.get("abort_reason"),
                "report": str(path),
            }
        )
    policies = sorted({str(row["policy"]) for row in rows})
    aggregate = {
        policy: {
            "runs": sum(row["policy"] == policy for row in rows),
            "completion_rate": (
                sum(row["policy"] == policy and row["task_complete"] for row in rows)
                / max(1, sum(row["policy"] == policy for row in rows))
            ),
            "verified_placements": sum(
                int(row["verified_placements"])
                for row in rows
                if row["policy"] == policy
            ),
            "visual_gate_stops": sum(
                int(row["visual_gate_stops"])
                for row in rows
                if row["policy"] == policy
            ),
        }
        for policy in policies
    }
    return {"aggregate": aggregate, "runs": rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/rl_isaac_comparison"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("models/rl_stacking_qtable.json")
    )
    parser.add_argument(
        "--vision-config",
        type=Path,
        default=Path("configs/grasp_model_resnet_siamese.yaml"),
        help="pickup classifier used to gate RL placements",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[310, 311, 312, 313, 314])
    parser.add_argument("--offset-m", type=float, default=0.008)
    parser.add_argument(
        "--max-picks",
        type=int,
        default=6,
        help="picks per run; six is a complete hackathon tower",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="write matched trial files without launching Isaac Sim",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run = args.run.resolve()
    out = args.out.resolve()
    checkpoint = args.checkpoint.resolve()
    vision_config = args.vision_config.resolve()
    if not (run / "sort_plan.json").is_file() or not (run / "scene_state.json").is_file():
        raise SystemExit("--run must contain sort_plan.json and scene_state.json")
    if not checkpoint.is_file():
        raise SystemExit(f"missing RL checkpoint: {checkpoint}")
    if not vision_config.is_file():
        raise SystemExit(f"missing RL vision config: {vision_config}")

    trials_dir = out / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args.seeds, args.offset_m)
    prepared: list[dict[str, object]] = []
    for job in jobs:
        policy = str(job["policy"])
        seed = int(job["seed"])
        trial_path = trials_dir / f"seed{seed}-{policy}.json"
        trial_path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
        prepared.append({**job, "trial_path": str(trial_path)})
    (out / "comparison_plan.json").write_text(
        json.dumps(prepared, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(prepared)} matched Isaac trials in {trials_dir}")
    if args.prepare_only:
        return 0

    from stacking_recovery.vision.grasp_quality_model import (
        GraspQualityModel,
        load_grasp_config,
    )

    configured_model = GraspQualityModel(load_grasp_config(vision_config)).model_path
    if not configured_model.is_file() or configured_model.stat().st_size < 1_000_000:
        raise SystemExit(
            f"classifier weights are missing or still an LFS pointer: {configured_model}; "
            "run 'git lfs pull' before starting Isaac"
        )

    sim_dir = ROOT / "data" / "fanuc_er4ia_hackathon_starter_v05 (1)" / "fanuc_er4ia_sim"
    sys.path.insert(0, str(sim_dir))
    import sort_scene  # noqa: E402
    from isaacsim import SimulationApp  # noqa: E402

    report_paths: list[Path] = []
    app = SimulationApp({"headless": True})
    try:
        for job in prepared:
            policy = str(job["policy"])
            seed = int(job["seed"])
            job_dir = out / "runs" / f"seed{seed}-{policy}"
            frames = job_dir / "frames"
            job_dir.mkdir(parents=True, exist_ok=True)
            argv = [
                "--run",
                str(run),
                "--out",
                str(frames),
                "--trial-config",
                str(job["trial_path"]),
                "--max-picks",
                str(args.max_picks),
            ]
            if policy == "rl":
                argv.extend(
                    [
                        "--rl-checkpoint",
                        str(checkpoint),
                        "--rl-vision-config",
                        str(vision_config),
                    ]
                )
            sort_scene.main(argv, simulation_app=app)
            source_report = run / "execution_report.json"
            destination = job_dir / "execution_report.json"
            shutil.copy2(source_report, destination)
            report_paths.append(destination)
    finally:
        app.close()

    summary = summarize(report_paths)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
