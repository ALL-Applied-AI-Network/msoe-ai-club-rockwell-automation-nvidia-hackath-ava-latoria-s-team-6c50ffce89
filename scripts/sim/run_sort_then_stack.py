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
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
SIM_DIR = ROOT / "data" / "fanuc_er4ia_hackathon_starter_v05 (1)" / "fanuc_er4ia_sim"
# Isaac's bundled Python does not necessarily have this repository installed
# as an editable package.  Make both the application package and simulator
# helpers importable before integrated-demo preflight runs.
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SIM_DIR))

import scene_config as C  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class LiveStatus:
    """Publish a small, atomic JSON snapshot for a React/TypeScript UI."""

    def __init__(self, out: Path, *, mode: str, total_grabs: int = 12) -> None:
        self.out = out
        self.path = out / "live_status.json"
        self.mode = mode
        self.total_grabs = total_grabs
        self.phase = "starting"
        self.state = "running"
        self.error: str | None = None
        self.summary: dict[str, Any] | None = None
        self._stop = threading.Event()
        self._write_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._last_signature: tuple[Any, ...] | None = None

    def _events(self, stage: str) -> list[dict[str, Any]]:
        path = self.out / ("01_sort" if stage == "sort" else "02_stack") / "execution_events.json"
        try:
            value = read_json(path)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return []
        return value if isinstance(value, list) else []

    @staticmethod
    def _ui_event(stage: str, index: int, event: dict[str, Any]) -> dict[str, Any]:
        row = classifier_row(stage, index, event)
        visual = event.get("rl_visual_verification") or {}
        quality = visual.get("quality") or {}
        rl = event.get("rl_placement_decision") or {}
        raw_actions = rl.get("actions") or []
        if isinstance(raw_actions, str):
            actions = [raw_actions]
        elif isinstance(raw_actions, (list, tuple)):
            actions = [str(action) for action in raw_actions]
        else:
            actions = []
        latest_action = rl.get("action") or (actions[-1] if actions else None)
        return {
            **row,
            "camera_quality": {
                "valid": quality.get("valid"),
                "reason": quality.get("reason"),
            },
            "rl": {
                # Keep the singular field for the existing dashboard while
                # also publishing the complete policy trace emitted by the
                # placement adapter.
                "action": latest_action,
                "actions": actions,
                "safe_stop": bool(event.get("rl_safe_stop") or rl.get("safe_stop")),
                "grasp_retries": int(event.get("closed_loop_grasp_attempts", 0)),
                "placement_retries": int(event.get("closed_loop_placement_attempts", 0)),
            },
        }

    def payload(self) -> dict[str, Any]:
        sort_events = self._events("sort")
        stack_events = self._events("stack")
        events = [
            *(self._ui_event("sort", index, event) for index, event in enumerate(sort_events)),
            *(self._ui_event("stack", index, event) for index, event in enumerate(stack_events)),
        ]
        scored = [event for event in events if event["classifier_correct"] is not None]
        placed = sum(bool(event["placement_verified"]) for event in events)
        completed = len(events)
        return {
            "schema_version": "1.0",
            "run_id": self.out.name,
            "updated_at": utc_now(),
            "state": self.state,
            "phase": self.phase,
            "mode": self.mode,
            "progress": {
                "completed_grabs": completed,
                "total_grabs": self.total_grabs,
                "percent": round(100 * completed / self.total_grabs, 1),
                "sort_completed": len(sort_events),
                "stack_completed": len(stack_events),
            },
            "current": events[-1] if events else None,
            "metrics": {
                "successful_pickups": sum(bool(event["pickup_verified"]) for event in events),
                "verified_placements": placed,
                "classifier_scored": len(scored),
                "classifier_correct": sum(event["classifier_correct"] is True for event in scored),
                "classifier_accuracy": (
                    sum(event["classifier_correct"] is True for event in scored) / len(scored)
                    if scored
                    else None
                ),
                "classifier_coverage": len(scored) / completed if completed else None,
            },
            "events": events,
            "summary": self.summary,
            "error": self.error,
            "artifacts": {
                "metrics_json": "end_to_end_metrics.json",
                "metrics_markdown": "end_to_end_metrics.md",
                "sort_report": "01_sort/execution_report.json",
                "stack_report": "02_stack/execution_report.json",
            },
        }

    def publish(self, *, force: bool = False) -> None:
        # The monitor thread and phase transitions can publish at nearly the
        # same instant. Serialize them so they never share the temporary file.
        with self._write_lock:
            payload = self.payload()
            signature = (
                payload["state"],
                payload["phase"],
                payload["progress"]["completed_grabs"],
                payload["metrics"]["classifier_scored"],
                self.error,
            )
            if force or signature != self._last_signature:
                write_json(self.path, payload)
                self._last_signature = signature

    def start(self) -> None:
        self.publish(force=True)

        def monitor() -> None:
            while not self._stop.wait(0.5):
                self.publish()

        self._thread = threading.Thread(target=monitor, name="live-status", daemon=True)
        self._thread.start()

    def set_phase(self, phase: str) -> None:
        self.phase = phase
        self.publish(force=True)

    def finish(self, summary: dict[str, Any]) -> None:
        self.summary = summary
        self.state = "completed" if summary.get("full_process_complete") else "failed"
        self.phase = "complete"
        self._stop.set()
        self.publish(force=True)

    def fail(self, error: BaseException) -> None:
        self.state = "failed"
        self.phase = "error"
        self.error = f"{type(error).__name__}: {error}"
        self._stop.set()
        self.publish(force=True)


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
        "tower_disturbance_verified": event.get("tower_disturbance_verified"),
        "max_existing_layer_shift_mm": event.get("max_existing_layer_shift_mm"),
        "max_existing_layer_yaw_change_deg": event.get(
            "max_existing_layer_yaw_change_deg"
        ),
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
    stack_rows = [row for row in rows if row["stage"] == "stack"]
    disturbance_rows = [
        row for row in stack_rows if row["tower_disturbance_verified"] is not None
    ]
    return {
        "full_process_complete": bool(sort_report.get("task_complete"))
        and bool(stack_report.get("task_complete")),
        "sort_complete": bool(sort_report.get("task_complete")),
        "stack_complete": bool(stack_report.get("task_complete")),
        "sort_pickups": sort_report.get("verified_pickups"),
        "sort_placements": sort_report.get("verified_placements"),
        "stack_pickups": stack_report.get("verified_pickups"),
        "stack_placements": stack_report.get("verified_placements"),
        "tower_safety": {
            "placements_checked": len(disturbance_rows),
            "all_existing_layers_stable": bool(disturbance_rows)
            and all(row["tower_disturbance_verified"] for row in disturbance_rows),
            "max_existing_layer_shift_mm": max(
                (
                    float(row["max_existing_layer_shift_mm"])
                    for row in disturbance_rows
                    if row["max_existing_layer_shift_mm"] is not None
                ),
                default=None,
            ),
            "max_existing_layer_yaw_change_deg": max(
                (
                    float(row["max_existing_layer_yaw_change_deg"])
                    for row in disturbance_rows
                    if row["max_existing_layer_yaw_change_deg"] is not None
                ),
                default=None,
            ),
        },
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
    tower = summary["tower_safety"]
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
        (
            "- Existing tower stayed stable: "
            f"{'PASS' if tower['all_existing_layers_stable'] else 'FAIL'} "
            f"(max shift {tower['max_existing_layer_shift_mm'] or 0:.3f} mm, "
            f"max yaw change {tower['max_existing_layer_yaw_change_deg'] or 0:.3f} deg)"
        ),
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


def stage_argv(
    run_dir: Path,
    frames_dir: Path,
    *,
    physics_only: bool,
    fast: bool,
    checkpoint: Path,
    vision_config: Path,
    integrated: bool = False,
    placement_config: Path | None = None,
) -> list[str]:
    argv = ["--run", str(run_dir), "--out", str(frames_dir)]
    if placement_config is not None:
        argv.extend(["--placement-config", str(placement_config)])
    if fast:
        argv.extend(["--sim-speedup", "1.5"])
    if physics_only:
        argv.append("--physics-only")
    elif integrated:
        # The validated before/after table-camera classifier is the real
        # fail-closed pickup gate. Do not also apply the legacy single-frame
        # wrist colour rule: it can veto a physically successful grasp before
        # the trained model sees it. Physics, learned vision and placement
        # checks remain active; RL may adjust only the bounded XY target.
        argv.extend(
            [
                "--verification",
                "stub",
                "--rl-vision-config",
                str(vision_config),
            ]
        )
    else:
        argv.extend(
            [
                "--verification",
                "stub",
                "--rl-vision-config",
                str(vision_config),
                "--rl-vision-observe-only",
            ]
        )
    argv.extend(["--rl-checkpoint", str(checkpoint)])
    return argv


def run_stage(
    sort_scene: Any,
    app: Any,
    run_dir: Path,
    frames_dir: Path,
    *,
    physics_only: bool,
    fast: bool,
    checkpoint: Path,
    vision_config: Path,
    integrated: bool = False,
    placement_config: Path | None = None,
) -> dict[str, Any]:
    argv = stage_argv(
        run_dir,
        frames_dir,
        physics_only=physics_only,
        fast=fast,
        checkpoint=checkpoint,
        vision_config=vision_config,
        integrated=integrated,
        placement_config=placement_config,
    )
    sort_scene.main(argv, simulation_app=app)
    return read_json(run_dir / "execution_report.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="six-cube sort run")
    parser.add_argument("--out", type=Path, required=True)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--physics-only", action="store_true")
    mode_group.add_argument(
        "--integrated-demo",
        action="store_true",
        help=(
            "run rendered classifier gating, bounded recovery and RL placement "
            "in one fail-closed end-to-end demonstration"
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "run the digital-twin motion at 1.5x while preserving paths, "
            "classifier checks, physics verification and metrics"
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/rl_stacking_qtable.json")
    parser.add_argument(
        "--vision-config",
        type=Path,
        default=ROOT / "configs/grasp_model_resnet_siamese.yaml",
    )
    parser.add_argument(
        "--placement-config",
        type=Path,
        default=ROOT / "configs/placement_verification.json",
        help="calibrated camera geometry used by the post-place safety gate",
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
    if not args.placement_config.resolve().is_file():
        raise SystemExit(f"missing placement config: {args.placement_config.resolve()}")
    if args.integrated_demo:
        from stacking_recovery.preflight import passed as preflight_passed
        from stacking_recovery.preflight import run_preflight

        checks = run_preflight(
            ROOT,
            rl_path=args.checkpoint,
            vision_config=args.vision_config,
            placement_config=args.placement_config,
            run_dir=source,
        )
        if not preflight_passed(checks):
            failures = [
                f"- {check.name}: {check.detail}"
                + (f" ({check.fix})" if check.fix else "")
                for check in checks
                if not check.passed
            ]
            raise SystemExit("integrated-demo preflight failed:\n" + "\n".join(failures))

    sort_dir = out / "01_sort"
    stack_dir = out / "02_stack"
    mode = (
        "physics"
        if args.physics_only
        else ("integrated" if args.integrated_demo else ("fast" if args.fast else "demo"))
    )
    live = LiveStatus(out, mode=mode)
    live.start()
    # Benchmark/demo mode preserves the independent visual scorecard.  Physics
    # mode and the explicit integrated demo both enable the bounded RL policy.
    use_rl = bool(args.physics_only or args.integrated_demo)
    prepare_sort_stage(source, sort_dir, use_rl=use_rl)

    import sort_scene  # noqa: E402
    from isaacsim import SimulationApp  # noqa: E402

    app = SimulationApp({"headless": True})
    try:
        live.set_phase("sort")
        sort_report = run_stage(
            sort_scene,
            app,
            sort_dir,
            sort_dir / "frames",
            physics_only=args.physics_only,
            fast=args.fast,
            checkpoint=args.checkpoint.resolve(),
            vision_config=args.vision_config.resolve(),
            integrated=args.integrated_demo,
            placement_config=args.placement_config.resolve(),
        )
        build_stack_stage(sort_dir, stack_dir, use_rl=use_rl)
        live.set_phase("stack")
        stack_report = run_stage(
            sort_scene,
            app,
            stack_dir,
            stack_dir / "frames",
            physics_only=args.physics_only,
            fast=args.fast,
            checkpoint=args.checkpoint.resolve(),
            vision_config=args.vision_config.resolve(),
            integrated=args.integrated_demo,
            placement_config=args.placement_config.resolve(),
        )
        # Write the report before closing Kit.  Isaac may terminate the Python
        # process as part of shutdown, especially in the Brev launchable.
        summary = summarize(sort_report, stack_report)
        write_json(out / "end_to_end_metrics.json", summary)
        (out / "end_to_end_metrics.md").write_text(markdown(summary), encoding="utf-8")
        live.finish(summary)
        print(markdown(summary), flush=True)
        exit_code = 0 if summary["full_process_complete"] else 1
    except BaseException as error:
        live.fail(error)
        raise
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
