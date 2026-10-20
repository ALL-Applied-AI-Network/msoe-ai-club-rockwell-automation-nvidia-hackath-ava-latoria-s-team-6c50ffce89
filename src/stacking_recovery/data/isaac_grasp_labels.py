"""Weak grasp labels from sim physics (data generation only — not for runtime cheating)."""

from __future__ import annotations

from typing import Any, Optional

from stacking_recovery.types import FailureReason, GraspCheckResult
from stacking_recovery.vision.grasp_labels import GRASP_CLASSES


def failure_reason_to_class(reason: FailureReason) -> str:
    mapping = {
        FailureReason.OK: "GOOD",
        FailureReason.EMPTY_GRASP: "EMPTY_GRASP",
        FailureReason.MISALIGNED_CUBE: "MISALIGNED",
        FailureReason.TILTED_OBJECT: "ROTATED",
        FailureReason.PLACEMENT_FAILURE: "PLACEMENT_FAILURE",
        FailureReason.LOW_CONFIDENCE: "UNCERTAIN",
        FailureReason.OCCLUSION: "UNCERTAIN",
        FailureReason.UNKNOWN: "UNCERTAIN",
    }
    return mapping.get(reason, "UNCERTAIN")


def grasp_result_to_class(result: GraspCheckResult) -> str:
    if result.predicted_class:
        return result.predicted_class
    if result.seated:
        return "GOOD"
    return failure_reason_to_class(result.reason)


def label_from_physics_event(
    *,
    pickup_verified: bool,
    perceived_yaw_deg: Optional[float] = None,
    jaw_gap_mm: Optional[float] = None,
    closed_loop_grasp: Optional[dict[str, Any]] = None,
) -> str:
    """
    Isaac / lab weak labels after lift (design doc §4.1 simplified).

    Uses physics pickup first; refines failure mode from jaw gap and yaw when available.
    """
    if pickup_verified:
        return "GOOD"
    yaw = perceived_yaw_deg
    if yaw is not None and abs(yaw) > 13.5:
        return "ROTATED"
    if jaw_gap_mm is not None and jaw_gap_mm < 2.0:
        return "EMPTY_GRASP"
    if closed_loop_grasp:
        yaw_err = closed_loop_grasp.get("yaw_error_deg")
        if isinstance(yaw_err, (int, float)) and yaw_err == yaw_err and yaw_err > 13.5:
            return "ROTATED"
        reason = closed_loop_grasp.get("reason")
        if reason == FailureReason.EMPTY_GRASP.value:
            return "EMPTY_GRASP"
        if reason in (
            FailureReason.MISALIGNED_CUBE.value,
            FailureReason.TILTED_OBJECT.value,
        ):
            return "MISALIGNED"
    return "SHIFTED"


def should_add_feedback_sample(
    *,
    physics_label: str,
    grasp_result: GraspCheckResult,
    pickup_verified: bool,
    record_bad_grasps: bool = True,
    record_mismatch: bool = True,
) -> tuple[bool, str]:
    """
    Decide whether to append a runtime feedback sample and which label to use.

    Ground truth priority: physics after lift > verifier reason on failure.
    """
    predicted = grasp_result_to_class(grasp_result)
    if record_mismatch and physics_label in GRASP_CLASSES:
        if physics_label != predicted:
            gt = physics_label
            if not pickup_verified and gt == "GOOD":
                gt = predicted if predicted != "GOOD" else "SHIFTED"
            elif pickup_verified and not grasp_result.seated:
                gt = "GOOD"
            return True, gt
        if pickup_verified and not grasp_result.seated:
            return True, "GOOD"
        if not pickup_verified and grasp_result.seated:
            return True, physics_label if physics_label != "GOOD" else "SHIFTED"
    if record_bad_grasps and not grasp_result.seated:
        label = physics_label if physics_label != "GOOD" else grasp_result_to_class(grasp_result)
        if label == "GOOD" and not pickup_verified:
            label = physics_label
        return True, label
    return False, physics_label
