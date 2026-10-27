#!/usr/bin/env python3
"""Create a simulation-only collection plan from Isaac ground truth.

This deliberately bypasses perception only for generating labeled synthetic
training data. The executor rejects this plan unless --collect-dataset is set.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SIM_DIR = (
    ROOT
    / "data"
    / "fanuc_er4ia_hackathon_starter_v05 (1)"
    / "fanuc_er4ia_sim"
)
sys.path.insert(0, str(SIM_DIR))

import scene_config as C  # noqa: E402


def build_collection_plan() -> dict:
    cubes = []
    for detection_id, ((x, y, colour), yaw) in enumerate(
        zip(C.SORT_CUBES, C.SORT_CUBE_YAWS_DEG)
    ):
        theta = math.radians(float(yaw))
        cubes.append(
            {
                "detection_id": detection_id,
                "category": "cube",
                "colour": colour,
                "xy": [float(x), float(y)],
                "yaw_deg": float(yaw),
                "cosmos_yaw_deg": float(yaw),
                "pose_confidence": 1.0,
                "spread_dir": [-math.sin(theta), math.cos(theta), 0.0],
                "pick_z": C.SURFACE_DZ + C.CUBE_TIP_CLEARANCE,
                "jaw_half": C.CUBE_EDGE / 2,
                "eligible": True,
                "skip_reason": None,
            }
        )

    remaining = sorted(cubes, key=lambda cube: (cube["xy"][0], cube["xy"][1]))
    picks = []
    for level, colour in enumerate(C.CUBE_STACK_COLOURS):
        cube = next(cube for cube in remaining if cube["colour"] == colour)
        remaining.remove(cube)
        picks.append(
            {
                **cube,
                "stack_level": level,
                "place_xy": list(C.CUBE_STACK_XY),
                "place_z": C.SURFACE_DZ
                + level * C.CUBE_EDGE
                + C.CUBE_TIP_CLEARANCE,
                "expected_centre_z": C.SURFACE_DZ
                + (level + 0.5) * C.CUBE_EDGE,
                "place_spread_dir": [0.0, 1.0, 0.0],
                "place_yaw_deg": 0.0,
            }
        )

    return {
        "source": "Isaac-sim-ground-truth-data-collection-only",
        "data_collection_only": True,
        "gripper_model": C.GRIPPER_MODEL,
        "task": "cube_stack",
        "stack_ready": True,
        "stack_colours_bottom_to_top": list(C.CUBE_STACK_COLOURS),
        "open_gap_m": 2 * C.JAW_OPEN,
        "picks": picks,
        "observations": cubes,
        "skipped": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if not (run / "scene_state.json").is_file():
        print(f"Missing scene_state.json in {run}; run capture_scene.py first.", file=sys.stderr)
        return 2
    out = run / "sort_plan.json"
    out.write_text(json.dumps(build_collection_plan(), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote guarded simulation-data plan: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
