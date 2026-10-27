"""Conservative cube-tower motion and disturbance checks."""

import numpy as np

import scene_config as C


def _square_yaw(degrees):
    return (float(degrees) + 45) % 90 - 45


def measure_stack_disturbance(before, after, existing_entries):
    """Measure motion of tower layers that existed before the current place."""
    shifts_mm = []
    yaw_changes_deg = []
    tilt_changes_deg = []
    missing = []
    for entry in existing_entries:
        name = entry.get("tracked_object")
        if not name or name not in before or name not in after:
            missing.append(name)
            continue
        before_pose = before[name]
        after_pose = after[name]
        shifts_mm.append(
            float(
                np.linalg.norm(
                    np.asarray(after_pose["centre"], float)
                    - np.asarray(before_pose["centre"], float)
                )
                * 1000
            )
        )
        yaw_changes_deg.append(
            abs(_square_yaw(after_pose["yaw_deg"] - before_pose["yaw_deg"]))
        )
        tilt_changes_deg.append(
            abs(
                float(after_pose.get("tilt_deg", 0))
                - float(before_pose.get("tilt_deg", 0))
            )
        )
    max_shift = max(shifts_mm, default=0.0)
    max_yaw = max(yaw_changes_deg, default=0.0)
    max_tilt = max(tilt_changes_deg, default=0.0)
    return {
        "existing_layers_checked": len(shifts_mm),
        "missing_existing_layers": missing,
        "max_existing_layer_shift_mm": max_shift,
        "max_existing_layer_yaw_change_deg": max_yaw,
        "max_existing_layer_tilt_change_deg": max_tilt,
        "tower_disturbance_verified": bool(
            not missing
            and max_shift < C.CUBE_STACK_MAX_EXISTING_SHIFT_M * 1000
            and max_yaw < C.CUBE_STACK_MAX_EXISTING_YAW_CHANGE_DEG
            and max_tilt < C.CUBE_STACK_MAX_EXISTING_TILT_CHANGE_DEG
        ),
    }


def segments(pick, home):
    """Build a normal transfer or a tower-aware two-stage placement path."""
    p = np.array([*pick["xy"], pick["pick_z"]])
    d = np.array([*pick.get("command_place_xy", pick["place_xy"]), pick["place_z"]])
    up, above = p + [0, 0, 0.13], d + [0, 0, 0.13]
    careful_stack = "stack_level" in pick
    preplace = d + [0, 0, C.CUBE_STACK_FINAL_APPROACH_M] if careful_stack else d
    spread = pick.get("spread_dir")
    dest_spread = pick.get("place_spread_dir", spread)
    shut = C.CUBE_JAW_HALF_TARGET if pick["category"] == "cube" else C.JAW_CLOSED
    path = [
        ("approach", home, up, 1.2, C.JAW_OPEN, C.JAW_OPEN, spread),
        ("descend", up, p, 0.8, C.JAW_OPEN, C.JAW_OPEN, spread),
        ("close", p, p, 0.6, C.JAW_OPEN, shut, spread),
        ("dwell", p, p, 0.5, shut, shut, spread),
        ("lift", p, up, 0.8, shut, shut, spread),
        ("carry", up, above, 1.4, shut, shut, dest_spread),
    ]
    if careful_stack:
        path.extend(
            [
                ("prelower", above, preplace, 0.75, shut, shut, dest_spread),
                ("final_lower", preplace, d, 0.75, shut, shut, dest_spread),
                ("release", d, d, 0.8, shut, C.JAW_OPEN, dest_spread),
                ("settle", d, d, 0.9, C.JAW_OPEN, C.JAW_OPEN, dest_spread),
                ("retreat", d, above, 1.1, C.JAW_OPEN, C.JAW_OPEN, dest_spread),
            ]
        )
    else:
        path.extend(
            [
                ("lower", above, d, 0.8, shut, shut, dest_spread),
                ("release", d, d, 0.6, shut, C.JAW_OPEN, dest_spread),
                ("settle", d, d, 0.6, C.JAW_OPEN, C.JAW_OPEN, dest_spread),
                ("retreat", d, above, 0.8, C.JAW_OPEN, C.JAW_OPEN, dest_spread),
            ]
        )
    path.append(("home", above, home, 1.1, C.JAW_OPEN, C.JAW_OPEN, None))
    return path
