#!/usr/bin/env python3
"""Compare open-loop, deterministic, and RL stacking on identical Isaac seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def wilson_interval(successes: int, total: int, z: float = 1.959964) -> list[float] | None:
    """Return a 95% Wilson interval without requiring scipy."""
    if total < 1:
        return None
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return [max(0.0, (centre - margin) / denominator), min(1.0, (centre + margin) / denominator)]


def trial_seed(path: Path, report: dict[str, object]) -> int | None:
    """Read a comparison seed from the report or its seedNNN-policy folder."""
    raw = report.get("comparison_seed")
    if raw is not None:
        return int(raw)
    match = re.search(r"seed(\d+)-", str(path))
    return int(match.group(1)) if match else None


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


def validate_stack_plan(plan: dict[str, object]) -> None:
    """Reject sort-only plans before an expensive Isaac comparison starts."""

    if plan.get("task") != "cube_stack" or plan.get("stack_ready") is not True:
        raise ValueError(
            "--run must be a stack-ready cube_stack plan, not a sorting plan"
        )
    picks = plan.get("picks")
    if not isinstance(picks, list) or len(picks) < 6:
        raise ValueError("--run must contain all six tower picks")
    levels = [pick.get("stack_level") for pick in picks if isinstance(pick, dict)]
    if sorted(levels) != list(range(6)):
        raise ValueError(
            "--run must contain exactly one stack level from 0 through 5"
        )


def summarize(report_paths: list[Path]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for path in report_paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        picks = list(report.get("picks") or [])
        rows.append(
            {
                "policy": report.get("policy"),
                "seed": trial_seed(path, report),
                "task": report.get("task"),
                "task_complete": bool(report.get("task_complete")),
                "verified_pickups": int(report.get("verified_pickups", 0)),
                "verified_placements": int(report.get("verified_placements", 0)),
                "full_tower_complete": report.get("task") == "cube_stack"
                and bool(report.get("task_complete"))
                and int(report.get("verified_placements", 0)) >= 6,
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
    aggregate: dict[str, dict[str, object]] = {}
    for policy in policies:
        selected = [row for row in rows if row["policy"] == policy]
        runs = len(selected)
        completions = sum(bool(row["full_tower_complete"]) for row in selected)
        placements = sum(int(row["verified_placements"]) for row in selected)
        aborts = Counter(str(row["abort_reason"] or "none") for row in selected)
        aggregate[policy] = {
            "runs": runs,
            "completions": completions,
            "completion_rate": completions / runs if runs else None,
            "completion_rate_95ci": wilson_interval(completions, runs),
            "verified_placements": placements,
            "mean_verified_placements": placements / runs if runs else None,
            "grasp_retries": sum(int(row["grasp_retries"]) for row in selected),
            "placement_retries": sum(int(row["placement_retries"]) for row in selected),
            "visual_gate_proceeds": sum(int(row["visual_gate_proceeds"]) for row in selected),
            "visual_gate_stops": sum(int(row["visual_gate_stops"]) for row in selected),
            "abort_reasons": dict(sorted(aborts.items())),
        }

    by_seed: dict[int, dict[str, dict[str, object]]] = {}
    for row in rows:
        if row["seed"] is not None:
            by_seed.setdefault(int(row["seed"]), {})[str(row["policy"])] = row
    paired: dict[str, dict[str, int]] = {}
    for opponent in ("open_loop", "closed_loop"):
        score = {"wins": 0, "losses": 0, "ties": 0, "matched_seeds": 0}
        for seed_rows in by_seed.values():
            if "rl" not in seed_rows or opponent not in seed_rows:
                continue
            score["matched_seeds"] += 1
            rl = seed_rows["rl"]
            other = seed_rows[opponent]
            rl_score = (bool(rl["full_tower_complete"]), int(rl["verified_placements"]))
            other_score = (bool(other["full_tower_complete"]), int(other["verified_placements"]))
            if rl_score > other_score:
                score["wins"] += 1
            elif rl_score < other_score:
                score["losses"] += 1
            else:
                score["ties"] += 1
        paired[f"rl_vs_{opponent}"] = score

    rl_stats = aggregate.get("rl", {})
    expected_policies = ("open_loop", "closed_loop", "rl")
    enough_runs = all(
        int(aggregate.get(policy, {}).get("runs", 0)) >= 20
        for policy in expected_policies
    )
    rl_completions = int(rl_stats.get("completions", 0))
    acceptance = {
        "status": (
            "pass"
            if enough_runs and rl_completions >= 18
            else "fail" if enough_runs else "insufficient_evidence"
        ),
        "requires": "20 matched full six-cube trials per policy and at least 18/20 RL completions",
        "has_20_runs_per_policy": enough_runs,
        "rl_has_at_least_18_completions": rl_completions >= 18,
        "note": "This is the digital-twin gate; lightweight Q-learning metrics are diagnostic only.",
    }
    return {
        "scope": "matched full-tower Isaac Sim evaluation",
        "aggregate": aggregate,
        "paired_rl_comparison": paired,
        "acceptance": acceptance,
        "runs": rows,
    }


def write_metrics_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "seed", "policy", "task_complete", "full_tower_complete", "verified_pickups",
        "verified_placements", "grasp_retries", "placement_retries",
        "visual_gate_proceeds", "visual_gate_stops", "abort_reason", "report",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_svg(aggregate: dict[str, dict[str, object]], path: Path) -> None:
    """Write a dependency-free completion-rate chart for reports/slides."""
    labels = [p for p in ("open_loop", "closed_loop", "rl") if p in aggregate]
    width, height = 720, 430
    left, top, plot_h = 90, 50, 280
    bar_w, gap = 120, 70
    colours = {"open_loop": "#9ca3af", "closed_loop": "#3b82f6", "rl": "#22c55e"}
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="360" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">Matched Isaac full-tower completion</text>',
    ]
    for tick in range(0, 101, 20):
        y = top + plot_h - plot_h * tick / 100
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="680" y2="{y:.1f}" stroke="#e5e7eb"/>')
        elements.append(f'<text x="78" y="{y + 5:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{tick}%</text>')
    for index, label in enumerate(labels):
        rate = float(aggregate[label].get("completion_rate") or 0.0)
        runs = int(aggregate[label].get("runs", 0))
        completions = int(aggregate[label].get("completions", 0))
        x = left + 65 + index * (bar_w + gap)
        bar_h = plot_h * rate
        y = top + plot_h - bar_h
        elements.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{colours[label]}" rx="4"/>')
        elements.append(f'<text x="{x + bar_w / 2}" y="{max(45, y - 8):.1f}" text-anchor="middle" font-family="sans-serif" font-size="14">{rate:.1%} ({completions}/{runs})</text>')
        elements.append(f'<text x="{x + bar_w / 2}" y="355" text-anchor="middle" font-family="sans-serif" font-size="14">{label.replace("_", " ")}</text>')
    elements.append('<text x="360" y="405" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#4b5563">Use 20 identical seeds per policy; lightweight training reward is not a lab-readiness metric.</text>')
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


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
    parser.add_argument(
        "--placement-config",
        type=Path,
        default=Path("configs/placement_verification.json"),
        help="calibrated post-place camera gate",
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
    placement_config = args.placement_config.resolve()
    if not (run / "sort_plan.json").is_file() or not (run / "scene_state.json").is_file():
        raise SystemExit("--run must contain sort_plan.json and scene_state.json")
    try:
        validate_stack_plan(
            json.loads((run / "sort_plan.json").read_text(encoding="utf-8"))
        )
    except (ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    if not checkpoint.is_file():
        raise SystemExit(f"missing RL checkpoint: {checkpoint}")
    if not vision_config.is_file():
        raise SystemExit(f"missing RL vision config: {vision_config}")
    if not placement_config.is_file():
        raise SystemExit(f"missing placement config: {placement_config}")

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
                "--verification",
                "stub",
                "--placement-config",
                str(placement_config),
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
            copied = json.loads(destination.read_text(encoding="utf-8"))
            copied["comparison_seed"] = seed
            copied["comparison_policy"] = policy
            destination.write_text(json.dumps(copied, indent=2) + "\n", encoding="utf-8")
            report_paths.append(destination)

        # SimulationApp.close() may terminate the interpreter under Isaac's
        # fast-shutdown mode. Persist completed acceptance evidence first.
        summary = summarize(report_paths)
        (out / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        write_metrics_csv(summary["runs"], out / "trial_metrics.csv")
        write_comparison_svg(summary["aggregate"], out / "completion_rate.svg")
        print(json.dumps(summary["aggregate"], indent=2))
        print(json.dumps(summary["acceptance"], indent=2))
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
