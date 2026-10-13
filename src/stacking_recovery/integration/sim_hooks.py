"""
Isaac Sim integration — call from sort_scene.py after close / after place.

TODO:
  1. Capture wrist frame via starter capture_scene / in-sim camera path.
  2. Instantiate RecoveryPolicy with grasp_check / place_check (not stubs).
  3. On REGRASP, open jaws and re-run table perception for this cube only.
  4. Write manipulation.jsonl under the run directory.

Starter path (relative to repo root):
  data/fanuc_er4ia_hackathon_starter_v05 (1)/fanuc_er4ia_sim/sort_scene.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from stacking_recovery.control.recovery_policy import (
    ManipulationContext,
    RecoveryDecision,
    RecoveryPolicy,
)


def make_policy(run_dir: Path, use_closed_loop: bool) -> RecoveryPolicy:
    """Factory used when wiring the sim executor."""
    if not use_closed_loop:
        # Open-loop baseline: policy never called; hook returns PROCEED always.
        return RecoveryPolicy(log_path=None)
    log = run_dir / "manipulation.jsonl"
    from stacking_recovery.vision.grasp_check import grasp_check
    from stacking_recovery.vision.place_check import place_check

    return RecoveryPolicy(
        log_path=log,
        grasp_check_fn=grasp_check,
        place_check_fn=place_check,
    )


def hook_after_grasp_close(
    policy: RecoveryPolicy,
    ctx: ManipulationContext,
    wrist_image_bgr: np.ndarray,
    *,
    closed_loop_enabled: bool,
) -> RecoveryDecision:
    if not closed_loop_enabled:
        return RecoveryDecision.PROCEED
    decision, _ = policy.after_grasp_close(ctx, wrist_image_bgr)
    return decision


def hook_after_place(
    policy: RecoveryPolicy,
    ctx: ManipulationContext,
    verify_image_bgr: np.ndarray,
    *,
    closed_loop_enabled: bool,
) -> RecoveryDecision:
    if not closed_loop_enabled:
        return RecoveryDecision.PROCEED
    decision, _ = policy.after_place(ctx, verify_image_bgr)
    return decision


def load_finger_params(config_name: str) -> dict[str, Any]:
    """Read configs/finger/{config_name}.json for gripper_geometry updates."""
    root = Path(__file__).resolve().parents[3] / "configs" / "finger"
    path = root / f"{config_name}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    import json

    return json.loads(path.read_text(encoding="utf-8"))
