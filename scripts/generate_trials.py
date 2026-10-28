#!/usr/bin/env python3
"""Expand configs/benchmark.yaml into configs/trials/*.json (2×2 matrix)."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "trials"


def main() -> None:
    core = ["nominal", "yaw_error", "pick_xy_shift", "low_friction"]
    extended = ["partial_occlusion", "touching_cubes", "distractor", "cube_tilted"]
    policies = ["open_loop", "closed_loop"]
    fingers = ["stock_aa_scallop", "cube_flat_pad"]
    OUT.mkdir(parents=True, exist_ok=True)
    count = 0
    for cond in core:
        for seed in range(10):
            for policy, finger in product(policies, fingers):
                trial = {
                    "condition_id": cond,
                    "seed": seed,
                    "policy": policy,
                    "finger": finger,
                }
                if cond == "yaw_error":
                    trial["yaw_offset_deg"] = 8.0 + seed % 8
                if cond == "pick_xy_shift":
                    trial["pick_xy_offset_m"] = [0.002 + 0.0003 * seed, 0.0]
                if cond == "low_friction":
                    trial["tip_static_friction"] = 0.7 + 0.05 * (seed % 3)
                name = f"{cond}_{policy}_{finger}_seed{seed}.json"
                (OUT / name).write_text(json.dumps(trial, indent=2) + "\n", encoding="utf-8")
                count += 1
    for cond in extended:
        for seed in range(5):
            for policy in policies:
                trial = {
                    "condition_id": cond,
                    "seed": seed,
                    "policy": policy,
                    "finger": "stock_aa_scallop",
                }
                if cond == "cube_tilted":
                    trial["tilt_deg"] = 15.0 + seed
                if cond == "partial_occlusion":
                    trial["occluding_object"] = "cap"
                if cond == "touching_cubes":
                    trial["touching_cube_colours"] = ["yellow", "blue"]
                name = f"{cond}_{policy}_seed{seed}.json"
                (OUT / name).write_text(json.dumps(trial, indent=2) + "\n", encoding="utf-8")
                count += 1
    print(f"Wrote {count} trials to {OUT}")


if __name__ == "__main__":
    main()
