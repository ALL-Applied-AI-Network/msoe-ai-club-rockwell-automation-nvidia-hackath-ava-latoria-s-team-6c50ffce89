#!/usr/bin/env python3
"""Collect a quota-balanced Isaac table-camera dataset, one verified attempt at a time."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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


def _print_progress(rows: list[dict], targets: CampaignTargets) -> None:
    print(json.dumps(campaign_progress(rows, targets), indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--isaac",
        type=Path,
        default=Path(os.environ.get("ISAAC", "/isaac-sim")),
    )
    parser.add_argument("--target-train", type=int, default=100)
    parser.add_argument("--target-val", type=int, default=25)
    parser.add_argument("--target-test", type=int, default=25)
    parser.add_argument("--max-attempts", type=int, default=500)
    parser.add_argument(
        "--seed-min",
        type=int,
        default=0,
        help="Never use a seed below this value (useful for independent evaluation campaigns)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Reuse one Isaac process for this many fully reset trials",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    run = args.run.resolve()
    out = args.out.resolve()
    targets = CampaignTargets(args.target_train, args.target_val, args.target_test)
    if min(targets.train, targets.val, targets.test) <= 0:
        parser.error("all split targets must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not (run / "sort_plan.json").is_file():
        parser.error(f"prepared run missing sort_plan.json: {run}")

    sort_scene = (
        repo
        / "data"
        / "fanuc_er4ia_hackathon_starter_v05 (1)"
        / "fanuc_er4ia_sim"
        / "sort_scene.py"
    )
    isaac_python = args.isaac / "python.sh"
    if not args.dry_run and not isaac_python.is_file():
        parser.error(f"Isaac Python not found: {isaac_python}")
    if not sort_scene.is_file():
        parser.error(f"sort_scene.py not found: {sort_scene}")

    if args.batch_size > 1:
        batch_worker = repo / "scripts" / "sim" / "run_table_collection_batch.py"
        if not batch_worker.is_file():
            parser.error(f"batch worker not found: {batch_worker}")
        attempts = 0
        while attempts < args.max_attempts:
            rows = load_manifest(out / "manifest.json")
            if next_collection_attempt(rows, targets, seed_min=args.seed_min) is None:
                print("BALANCED COLLECTION COMPLETE", flush=True)
                _print_progress(rows, targets)
                return 0
            batch_attempts = min(args.batch_size, args.max_attempts - attempts)
            command = [
                str(isaac_python),
                str(batch_worker),
                "--repo",
                str(repo),
                "--run",
                str(run),
                "--out",
                str(out),
                "--target-train",
                str(args.target_train),
                "--target-val",
                str(args.target_val),
                "--target-test",
                str(args.target_test),
                "--max-attempts",
                str(batch_attempts),
                "--seed-min",
                str(args.seed_min),
            ]
            if args.dry_run:
                print(" ".join(command), flush=True)
                return 0
            result = subprocess.run(command, check=False)
            attempts += batch_attempts
            if result.returncode not in (0, 3):
                print(
                    f"Isaac batch failed with exit code {result.returncode}",
                    file=sys.stderr,
                )
                return result.returncode
            if result.returncode == 0:
                return 0
        print(f"Stopped after max_attempts={args.max_attempts} before quotas were met.")
        _print_progress(load_manifest(out / "manifest.json"), targets)
        return 3

    out.mkdir(parents=True, exist_ok=True)
    campaign_dir = out / "campaign"
    trials_dir = campaign_dir / "trials"
    runs_dir = campaign_dir / "runs"
    trials_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    attempted_seeds: set[int] = set()
    for previous_trial in trials_dir.glob("*.json"):
        try:
            value = json.loads(previous_trial.read_text(encoding="utf-8")).get("seed")
            if isinstance(value, int):
                attempted_seeds.add(value)
        except (OSError, json.JSONDecodeError):
            # A partial trial file from an interrupted write is ignored; the
            # new file name includes the seed and will not overwrite it.
            continue

    attempts = 0
    while attempts < args.max_attempts:
        rows = load_manifest(out / "manifest.json")
        attempt = next_collection_attempt(
            rows,
            targets,
            attempted_seeds=attempted_seeds,
            seed_min=args.seed_min,
        )
        if attempt is None:
            print("BALANCED COLLECTION COMPLETE", flush=True)
            _print_progress(rows, targets)
            return 0

        attempt_id = f"{len(rows):04d}-seed{attempt.seed}-{attempt.scenario}"
        trial_path = trials_dir / f"{attempt_id}.json"
        frame_dir = runs_dir / attempt_id
        trial_path.write_text(
            json.dumps(attempt.trial, indent=2) + "\n", encoding="utf-8"
        )
        attempted_seeds.add(attempt.seed)
        command = [
            str(isaac_python),
            str(sort_scene),
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
        ]
        print(
            f"[{attempts + 1}/{args.max_attempts}] need {attempt.split} "
            f"{attempt.target_label}: {attempt.scenario} seed={attempt.seed} "
            f"pick={attempt.pick_index}",
            flush=True,
        )
        if args.dry_run:
            print(" ".join(command), flush=True)
            return 0

        result = subprocess.run(command, check=False)
        attempts += 1
        if result.returncode:
            print(f"Isaac attempt failed with exit code {result.returncode}", file=sys.stderr)
            return result.returncode
        new_rows = load_manifest(out / "manifest.json")
        if len(new_rows) <= len(rows):
            print(
                "Attempt produced no valid image; continuing with a fresh seed.",
                flush=True,
            )
        if attempts % 10 == 0:
            _print_progress(new_rows, targets)

    print(f"Stopped after max_attempts={args.max_attempts} before quotas were met.")
    _print_progress(load_manifest(out / "manifest.json"), targets)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
