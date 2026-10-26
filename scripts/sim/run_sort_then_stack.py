#!/usr/bin/env python3
"""Run cube sorting followed by stacking and write per-grab metrics.

The sort report becomes the initial state for stacking.  Isaac physics is the
ground truth for every pickup; the table-camera Siamese classifier is scored
against that truth when visual verification is enabled.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SIM_DIR = ROOT / "data" / "fanuc_er4ia_hackathon_starter_v05 (1)" / "fanuc_er4ia_sim"
sys.path.insert(0, str(SIM_DIR))

import scene_config as C  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def prepare_sort_stage(source: Path, destination: Path, *, use_rl: bool) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("scene_state.json", "sort_plan.json", "ground_truth.json"):
        src = source / name
        if src.is_file():
            shutil.copy2(src, destination / name)
    plan = read_json(destination / "sort_plan.json")
    if use_rl:
        plan["closed_loop_policy"] = "rl"
    write_json(destination / "sort_plan.json", plan)


def _cube_colours(sort_plan: dict[str, Any], sort_report: dict[str, Any]) -> dict[str, str]:
    planned = {int(p["detection_id"]): p for p in sort_plan.get("picks", [])}
    colours: dict[str, str] = {}
    for event in sort_report.get("picks", []):
        tracked = event.get("tracked_object")
        pick = planned.get(int(event.get("detection_id", -1)))
        if tracked and pick and pick.get("category") == "cube" and pick.get("colour"):
            colours[str(tracked)] = str(pick["colour"])
    return colours


def build_stack_stage(sort_dir: Path, stack_dir: Path, *, use_rl: bool) -> dict[str, Any]:
    """Carry the sorted physical poses forward and build a six-cube tower plan."""
    sort_plan = read_json(sort_dir / "sort_plan.json")
    report = read_json(sort_dir / "execution_report.json")
    if not report.get("task_complete"):
        raise ValueError("sorting did not complete; refusing to start stacking")

    after = report.get("after") or {}
    colours = _cube_colours(sort_plan, report)
    by_colour: dict[str, list[str]] = defaultdict(list)
    for name, colour in colours.items():
        if name in after:
            by_colour[colour].append(name)
    for names in by_colour.values():
        names.sort()

    required = Counter(C.CUBE_STACK_COLOURS)
    available = Counter({colour: len(names) for colour, names in by_colour.items()})
    if any(available[colour] < count for colour, count in required.items()):
        raise ValueError(
            f"need sorted cubes {dict(required)}, found {dict(available)}; "
            "check that the input plan contains six coloured cubes"
        )

    objects = {
        name: {
            "position": list(pose["position"]),
            "orientation": list(pose["orientation"]),
        }
        for name, pose in after.items()
    }
    picks: list[dict[str, Any]] = []
    used: Counter[str] = Counter()
    for level, colour in enumerate(C.CUBE_STACK_COLOURS):
        name = by_colour[colour][used[colour]]
        used[colour] += 1
        pose = after[name]
        xy = [float(v) for v in pose["centre"][:2]]
        yaw = float(pose.get("yaw_deg", 0.0))
        theta = math.radians(yaw)
        picks.append(
            {
                "detection_id": level,
                "category": "cube",
                "colour": colour,
                "source_object": name,
                "xy": xy,
                "observation_xy": xy,
                "yaw_deg": yaw,
                "cosmos_yaw_deg": yaw,
                "pose_confidence": 1.0,
                "spread_dir": [-math.sin(theta), math.cos(theta), 0.0],
                "pick_z": C.SURFACE_DZ + C.CUBE_TIP_CLEARANCE,
                "jaw_half": C.CUBE_EDGE / 2,
                "eligible": True,
                "skip_reason": None,
                "stack_level": level,
                "place_xy": list(C.CUBE_STACK_XY),
                "place_z": C.SURFACE_DZ + level * C.CUBE_EDGE + C.CUBE_TIP_CLEARANCE,
                "expected_centre_z": C.SURFACE_DZ + (level + 0.5) * C.CUBE_EDGE,
                "place_spread_dir": [0.0, 1.0, 0.0],
                "place_yaw_deg": 0.0,
            }
        )

    plan: dict[str, Any] = {
        "source": "Cosmos3-Nano",
        "task": "cube_stack",
        "stack_ready": True,
        "stack_colours_bottom_to_top": list(C.CUBE_STACK_COLOURS),
        "open_gap_m": 2 * C.JAW_OPEN,
        "picks": picks,
        "observations": picks,
        "skipped": [],
    }
    if use_rl:
        plan["closed_loop_policy"] = "rl"
    stack_dir.mkdir(parents=True, exist_ok=True)
    write_json(stack_dir / "scene_state.json", {"objects": objects})
    write_json(stack_dir / "sort_plan.json", plan)
    return plan


def classifier_row(stage: str, index: int, event: dict[str, Any]) -> dict[str, Any]:
    visual = event.get("rl_visual_verification") or {}
    predictions = visual.get("predictions") or []
    prediction = predictions[-1] if predictions else {}
    predicted_class = prediction.get("class")
    predicted_good = predicted_class == "GOOD" if predicted_class is not None else None
    physics_good = bool(event.get("pickup_verified"))
    return {
        "stage": stage,
        "grab": index + 1,
        "detection_id": event.get("detection_id"),
        "tracked_object": event.get("tracked_object"),
        "predicted_class": predicted_class,
        "confidence": prediction.get("confidence"),
        "probabilities": prediction.get("probabilities"),
        "physics_pickup_success": physics_good,
        "classifier_correct": (
            predicted_good == physics_good if predicted_good is not None else None
        ),
        "pickup_verified": physics_good,
        "placement_verified": bool(event.get("placement_verified")),
        "placement_error_mm": event.get("placement_error_mm"),
        "cycle_time_s": event.get("cycle_time_s"),
        "visual_gate": (visual.get("gate") or {}).get("action"),
    }


def summarize(sort_report: dict[str, Any], stack_report: dict[str, Any]) -> dict[str, Any]:
    rows = [
        classifier_row(stage, index, event)
        for stage, report in (("sort", sort_report), ("stack", stack_report))
        for index, event in enumerate(report.get("picks", []))
    ]
    scored = [r for r in rows if r["classifier_correct"] is not None]
    tp = sum(r["predicted_class"] == "GOOD" and r["physics_pickup_success"] for r in scored)
    fp = sum(r["predicted_class"] == "GOOD" and not r["physics_pickup_success"] for r in scored)
    fn = sum(r["predicted_class"] != "GOOD" and r["physics_pickup_success"] for r in scored)
    tn = sum(r["predicted_class"] != "GOOD" and not r["physics_pickup_success"] for r in scored)
    return {
        "full_process_complete": bool(sort_report.get("task_complete"))
        and bool(stack_report.get("task_complete")),
        "sort_complete": bool(sort_report.get("task_complete")),
        "stack_complete": bool(stack_report.get("task_complete")),
        "sort_pickups": sort_report.get("verified_pickups"),
        "sort_placements": sort_report.get("verified_placements"),
        "stack_pickups": stack_report.get("verified_pickups"),
        "stack_placements": stack_report.get("verified_placements"),
        "classifier": {
            "scored_grabs": len(scored),
            "correct": sum(r["classifier_correct"] is True for r in scored),
            "accuracy": (
                sum(r["classifier_correct"] is True for r in scored) / len(scored)
                if scored
                else None
            ),
            "confusion": {
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
            },
            "note": (
                "Run intentional failed grasps as well as nominal grabs before treating "
                "accuracy as a robust safety metric."
            ),
        },
        "grabs": rows,
    }


def markdown(summary: dict[str, Any]) -> str:
    classifier = summary["classifier"]
    accuracy = (
        "not scored"
        if classifier["accuracy"] is None
        else f"{classifier['accuracy']:.1%} ({classifier['correct']}/{classifier['scored_grabs']})"
    )
    lines = [
        "# Sort then stack end-to-end test",
        "",
        f"Overall: **{'PASS' if summary['full_process_complete'] else 'FAIL'}**",
        "",
        f"- Sort: {summary['sort_pickups']} pickups, {summary['sort_placements']} placements",
        f"- Stack: {summary['stack_pickups']} pickups, {summary['stack_placements']} placements",
        f"- Classifier accuracy against Isaac physics: {accuracy}",
        "",
        "## Every grab",
        "",
        "| stage | grab | object | prediction | confidence | physics pickup | correct | placement | error mm |",
        "| --- | ---: | --- | --- | ---: | --- | --- | --- | ---: |",
    ]
    for row in summary["grabs"]:
        confidence = "-" if row["confidence"] is None else f"{float(row['confidence']):.3f}"
        error = "-" if row["placement_error_mm"] is None else f"{float(row['placement_error_mm']):.3f}"
        lines.append(
            f"| {row['stage']} | {row['grab']} | {row['tracked_object'] or '-'} | "
            f"{row['predicted_class'] or '-'} | {confidence} | "
            f"{'PASS' if row['physics_pickup_success'] else 'FAIL'} | "
            f"{('-' if row['classifier_correct'] is None else ('yes' if row['classifier_correct'] else 'no'))} | "
            f"{'PASS' if row['placement_verified'] else 'FAIL'} | {error} |"
        )
    lines.extend(["", classifier["note"], ""])
    return "\n".join(lines)


def run_stage(
    sort_scene: Any,
    app: Any,
    run_dir: Path,
    frames_dir: Path,
    *,
    physics_only: bool,
    checkpoint: Path,
    vision_config: Path,
) -> dict[str, Any]:
    argv = ["--run", str(run_dir), "--out", str(frames_dir)]
    if physics_only:
        argv.append("--physics-only")
    else:
        argv.extend(["--verification", "stub", "--rl-vision-config", str(vision_config)])
    argv.extend(["--rl-checkpoint", str(checkpoint)])
    sort_scene.main(argv, simulation_app=app)
    return read_json(run_dir / "execution_report.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="six-cube sort run")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--physics-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/rl_stacking_qtable.json")
    parser.add_argument(
        "--vision-config",
        type=Path,
        default=ROOT / "configs/grasp_model_resnet_siamese.yaml",
    )
    args = parser.parse_args()
    source = args.run.resolve()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output is not empty: {out}")
    if not args.checkpoint.resolve().is_file():
        raise SystemExit(f"missing RL checkpoint: {args.checkpoint.resolve()}")
    if not args.physics_only and not args.vision_config.resolve().is_file():
        raise SystemExit(f"missing classifier config: {args.vision_config.resolve()}")

    sort_dir = out / "01_sort"
    stack_dir = out / "02_stack"
    prepare_sort_stage(source, sort_dir, use_rl=True)

    import sort_scene  # noqa: E402
    from isaacsim import SimulationApp  # noqa: E402

    app = SimulationApp({"headless": True})
    try:
        sort_report = run_stage(
            sort_scene,
            app,
            sort_dir,
            sort_dir / "frames",
            physics_only=args.physics_only,
            checkpoint=args.checkpoint.resolve(),
            vision_config=args.vision_config.resolve(),
        )
        build_stack_stage(sort_dir, stack_dir, use_rl=True)
        stack_report = run_stage(
            sort_scene,
            app,
            stack_dir,
            stack_dir / "frames",
            physics_only=args.physics_only,
            checkpoint=args.checkpoint.resolve(),
            vision_config=args.vision_config.resolve(),
        )
        # Write the report before closing Kit.  Isaac may terminate the Python
        # process as part of shutdown, especially in the Brev launchable.
        summary = summarize(sort_report, stack_report)
        write_json(out / "end_to_end_metrics.json", summary)
        (out / "end_to_end_metrics.md").write_text(markdown(summary), encoding="utf-8")
        print(markdown(summary), flush=True)
        exit_code = 0 if summary["full_process_complete"] else 1
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
