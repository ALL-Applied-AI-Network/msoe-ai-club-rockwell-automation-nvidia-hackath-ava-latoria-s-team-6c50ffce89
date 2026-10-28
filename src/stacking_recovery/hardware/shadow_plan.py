"""Generate an auditable no-motion command transcript from a manipulation plan."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

from stacking_recovery.hardware.adapter import PoseTarget, ShadowRobotAdapter


def _finite_pair(pick: dict[str, Any], key: str) -> tuple[float, float]:
    value = pick.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{key} must contain two coordinates")
    pair = (float(value[0]), float(value[1]))
    if not all(math.isfinite(v) for v in pair):
        raise ValueError(f"{key} coordinates must be finite")
    return pair


def _finite_scalar(pick: dict[str, Any], key: str) -> float:
    try:
        value = float(pick[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{key} must be a finite number") from error
    if not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number")
    return value


def execute_shadow_plan(
    plan: dict[str, Any],
    adapter: ShadowRobotAdapter,
    *,
    clearance_m: float = 0.05,
    speed_scale: float = 0.10,
    frame_id: str = "robot_base",
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> dict[str, Any]:
    """Record approach/pick/lift/place/retreat commands for every eligible pick.

    The orientation default is an explicit shadow-only placeholder.  It must
    not be interpreted as a qualified FANUC tool orientation.
    """

    if not math.isfinite(clearance_m) or not 0.0 < clearance_m <= 0.25:
        raise ValueError("clearance_m must be finite and in (0, 0.25]")
    if not math.isfinite(speed_scale) or not 0.0 < speed_scale <= 0.25:
        raise ValueError("speed_scale must be finite and in (0, 0.25]")
    picks = plan.get("picks")
    if not isinstance(picks, list) or not picks:
        raise ValueError("plan must contain at least one pick")

    phases: list[str] = []

    def move(phase: str, xyz: tuple[float, float, float]) -> None:
        adapter.move_to_pose(
            PoseTarget(xyz, orientation_xyzw, frame_id=frame_id),
            speed_scale=speed_scale,
        )
        phases.append(phase)

    for index, raw_pick in enumerate(picks):
        if not isinstance(raw_pick, dict):
            raise ValueError(f"pick {index} must be an object")
        if raw_pick.get("eligible") is False:
            continue
        pick_xy = _finite_pair(raw_pick, "xy")
        place_xy = _finite_pair(raw_pick, "place_xy")
        pick_z = _finite_scalar(raw_pick, "pick_z")
        place_z = _finite_scalar(raw_pick, "place_z")
        if pick_z < 0 or place_z < 0:
            raise ValueError("pick_z and place_z must be non-negative")

        move("approach_pick", (pick_xy[0], pick_xy[1], pick_z + clearance_m))
        adapter.open_gripper()
        phases.append("open_before_pick")
        move("descend_pick", (pick_xy[0], pick_xy[1], pick_z))
        adapter.close_gripper()
        phases.append("close_on_object")
        move("lift", (pick_xy[0], pick_xy[1], pick_z + clearance_m))
        move(
            "approach_place",
            (place_xy[0], place_xy[1], place_z + clearance_m),
        )
        move("descend_place", (place_xy[0], place_xy[1], place_z))
        adapter.open_gripper()
        phases.append("release")
        move("retreat", (place_xy[0], place_xy[1], place_z + clearance_m))

    commands = []
    for phase, command in zip(phases, adapter.commands, strict=True):
        commands.append(
            {
                "index": len(commands),
                "phase": phase,
                "command": command.command,
                "payload": command.payload,
            }
        )
    return {
        "schema_version": "1.0",
        "mode": "shadow",
        "motion_sent": False,
        "orientation_source": "placeholder_shadow_only",
        "frame_id": frame_id,
        "clearance_m": clearance_m,
        "speed_scale": speed_scale,
        "planned_picks": len(picks),
        "commands": commands,
        "final_status": asdict(adapter.get_status()),
    }
