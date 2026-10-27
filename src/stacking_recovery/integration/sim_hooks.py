"""Isaac Sim integration helpers (import from sort_scene via closed_loop_support)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from stacking_recovery.control.recovery_policy import (
    ManipulationContext,
    RecoveryDecision,
    RecoveryPolicy,
)


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def build_grasp_recorder(
    repo_root: Path,
    *,
    collect_dataset_dir: Path | None = None,
    grasp_feedback: bool = False,
) -> "GraspFeedbackRecorder | None":
    """Configure dataset recorder for sort_scene (see docs/GRASP_DATA_AND_FEEDBACK.md)."""
    if not collect_dataset_dir and not grasp_feedback:
        return None
    from stacking_recovery.data.grasp_feedback import GraspFeedbackRecorder
    from stacking_recovery.vision.grasp_quality_model import load_grasp_config

    cfg = load_grasp_config(repo_root / "configs" / "grasp_model.yaml")
    fb = dict(cfg.get("feedback") or {})
    fb["enabled"] = True
    if collect_dataset_dir:
        fb["mode"] = "collect_all"
        fb["data_root"] = str(collect_dataset_dir.resolve())
    else:
        fb["mode"] = "feedback"
    cfg["feedback"] = fb
    return GraspFeedbackRecorder.from_config(repo_root, cfg)


def make_policy(
    run_dir: Path,
    use_closed_loop: bool,
    verification: str = "rules",
) -> RecoveryPolicy:
    if not use_closed_loop:
        return RecoveryPolicy(log_path=None)
    from stacking_recovery.vision.grasp_check_factory import get_grasp_check_fn
    from stacking_recovery.vision.place_check import place_check

    mode = (
        verification
        if verification in ("rules", "ai", "ai_guarded", "stub")
        else "rules"
    )
    return RecoveryPolicy(
        log_path=run_dir / "manipulation.jsonl",
        grasp_check_fn=get_grasp_check_fn(mode),  # type: ignore[arg-type]
        place_check_fn=place_check,
    )


def apply_finger_config(finger_id: str) -> dict[str, Any]:
    path = repo_root_from_here() / "configs" / "finger" / f"{finger_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def apply_finger_to_scene_config(scene_config_module: Any, finger_id: str) -> None:
    """Mutate imported scene_config before cell.build()."""
    params = apply_finger_config(finger_id)
    if params.get("soft_tip_static_friction") is not None:
        scene_config_module.SOFT_TIP_STATIC_FRIC = float(
            params["soft_tip_static_friction"]
        )
    if params.get("contact_face") == "flat":
        scene_config_module.SCALLOP_R = 0.001


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
