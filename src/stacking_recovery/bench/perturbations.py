"""Load perturbation trials from configs/trials/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_TRIALS_DIR = (
    Path(__file__).resolve().parents[3] / "configs" / "trials"
)


def load_trial(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_trials(trials_dir: Path | None = None) -> list[dict[str, Any]]:
    root = trials_dir or DEFAULT_TRIALS_DIR
    return [load_trial(p) for p in sorted(root.glob("*.json"))]


def trial_to_overrides(trial: dict[str, Any]) -> dict[str, Any]:
    """Map trial JSON to scene/runtime overrides (wired in integration later)."""
    overrides: dict[str, Any] = {
        "condition_id": trial["condition_id"],
        "seed": trial["seed"],
        "policy": trial["policy"],
        "finger": trial.get("finger", "stock_aa_scallop"),
    }
    for key in (
        "yaw_offset_deg",
        "pick_xy_offset_m",
        "tip_static_friction",
        "touching_cube_colours",
        "occluding_object",
        "tilt_deg",
    ):
        if key in trial:
            overrides[key] = trial[key]
    return overrides
