#!/usr/bin/env python3
"""Run several balanced table-data trials inside one Isaac Sim process."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stacking_recovery.data.collection_campaign import (  # noqa: E402
    CampaignTargets,
    campaign_progress,
    next_collection_attempt,
)
from stacking_recovery.data.grasp_dataset_store import load_manifest  # noqa: E402


def _attempted_seeds(trials_dir: Path) -> set[int]:
    seeds: set[int] = set()
    for path in trials_dir.glob("*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get("seed")
            if isinstance(value, int):
                seeds.add(value)
        except (OSError, json.JSONDecodeError):
            continue
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-train", type=int, required=True)
    parser.add_argument("--target-val", type=int, required=True)
    parser.add_argument("--target-test", type=int, required=True)
    parser.add_argument("--max-attempts", type=int, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    run = args.run.resolve()
    out = args.out.resolve()
    targets = CampaignTargets(args.target_train, args.target_val, args.target_test)
    campaign_dir = out / "campaign"
    trials_dir = campaign_dir / "trials"
    runs_dir = campaign_dir / "runs"
    trials_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    attempted_seeds = _attempted_seeds(trials_dir)

    sim_dir = (
        repo
        / "data"
        / "fanuc_er4ia_hackathon_starter_v05 (1)"
        / "fanuc_er4ia_sim"
    )
    sys.path.insert(0, str(sim_dir))
    import sort_scene  # noqa: E402
    from isaacsim import SimulationApp  # noqa: E402

    app = SimulationApp({"headless": True})
    attempts = 0
    try:
        while attempts < args.max_attempts:
            rows = load_manifest(out / "manifest.json")
            attempt = next_collection_attempt(
                rows, targets, attempted_seeds=attempted_seeds
            )
            if attempt is None:
                print("BALANCED COLLECTION COMPLETE", flush=True)
                print(json.dumps(campaign_progress(rows, targets), indent=2), flush=True)
                return 0

            attempt_id = f"{len(rows):04d}-seed{attempt.seed}-{attempt.scenario}"
            trial_path = trials_dir / f"{attempt_id}.json"
            frame_dir = runs_dir / attempt_id
            trial_path.write_text(
                json.dumps(attempt.trial, indent=2) + "\n", encoding="utf-8"
            )
            attempted_seeds.add(attempt.seed)
            print(
                f"[batch {attempts + 1}/{args.max_attempts}] need {attempt.split} "
                f"{attempt.target_label}: {attempt.scenario} seed={attempt.seed} "
                f"pick={attempt.pick_index}",
                flush=True,
            )
            sort_scene.main(
                [
                    "--run",
                    str(run),
                    "--collect-dataset",
                    str(out),
                    "--trial-config",
                    str(trial_path),
                    "--pick-index",
                    str(attempt.pick_index),
                    "--out",
                    str(frame_dir),
                ],
                simulation_app=app,
            )
            attempts += 1
            new_rows = load_manifest(out / "manifest.json")
            if len(new_rows) <= len(rows):
                print(
                    "Attempt produced no valid image; continuing with a fresh seed.",
                    flush=True,
                )
    finally:
        app.close()

    rows = load_manifest(out / "manifest.json")
    print(json.dumps(campaign_progress(rows, targets), indent=2), flush=True)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
