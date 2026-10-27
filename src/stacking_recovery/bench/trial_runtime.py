"""Apply perturbation trial JSON to plan picks and scene state."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_trial_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_pick_overrides(pick: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    pick = copy.deepcopy(pick)
    # Preserve the perceived object location for camera crops. Motion
    # perturbations change the commanded grasp, not where the cube appears.
    pick.setdefault("observation_xy", copy.deepcopy(pick.get("xy")))
    yaw_off = float(overrides.get("yaw_offset_deg", 0) or 0)
    if yaw_off and "yaw_deg" in pick:
        yaw = float(pick["yaw_deg"]) + yaw_off
        pick["yaw_deg"] = yaw
        theta = np.deg2rad(yaw)
        pick["spread_dir"] = [
            -float(np.sin(theta)),
            float(np.cos(theta)),
            0.0,
        ]
    xy_off = overrides.get("pick_xy_offset_m")
    if xy_off and len(xy_off) == 2:
        xy = np.array(pick["xy"], float) + np.array(xy_off, float)
        pick["xy"] = xy.tolist()
    return pick


def apply_plan_overrides(plan: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    plan = copy.deepcopy(plan)
    plan["picks"] = [
        apply_pick_overrides(p, overrides)
        for p in plan.get("picks", [])
    ]
    plan["closed_loop_policy"] = overrides.get("policy", "open_loop")
    plan["finger_id"] = overrides.get("finger", "stock_aa_scallop")
    plan["trial_condition"] = overrides.get("condition_id")
    plan["trial_seed"] = overrides.get("seed")
    plan["trial_yaw_offset_deg"] = float(overrides.get("yaw_offset_deg", 0) or 0)
    plan["trial_pick_xy_offset_m"] = list(
        overrides.get("pick_xy_offset_m") or [0.0, 0.0]
    )
    plan["trial_place_xy_offset_m"] = list(
        overrides.get("place_xy_offset_m") or [0.0, 0.0]
    )
    plan["campaign_target"] = overrides.get("campaign_target")
    plan["campaign_scenario"] = overrides.get("campaign_scenario")
    return plan


def apply_scene_friction(overrides: dict[str, Any]) -> float | None:
    return overrides.get("tip_static_friction")


def finger_scallop_radius(finger_id: str, repo_root: Path) -> float | None:
    path = repo_root / "configs" / "finger" / f"{finger_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("contact_face") == "flat":
        return None
    return data.get("scallop_radius_m")
