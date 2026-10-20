"""Select rule-based, ML, or guarded-ML grasp verification."""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

from stacking_recovery.types import GraspCheckResult
from stacking_recovery.vision.grasp_check import grasp_check as grasp_check_rules

VerificationMode = Literal["rules", "ai", "ai_guarded", "stub"]


def build_guarded_ai_check(
    ai_check: Callable[[np.ndarray], GraspCheckResult],
    rules_check: Callable[[np.ndarray], GraspCheckResult] = grasp_check_rules,
) -> Callable[[np.ndarray], GraspCheckResult]:
    """Require independent AI and geometry checks to agree before proceeding.

    The learned model is trained primarily on synthetic wrist frames.  Until
    real/Isaac holdout results justify trusting it alone, this conservative
    composition prevents a confident AI false-positive from moving the arm.
    """

    def check(image_bgr: np.ndarray) -> GraspCheckResult:
        ai = ai_check(image_bgr)
        rules = rules_check(image_bgr)
        agreed_good = ai.seated and rules.seated
        if agreed_good:
            return GraspCheckResult(
                seated=True,
                yaw_error_deg=rules.yaw_error_deg,
                confidence=min(ai.confidence, rules.confidence),
                reason=rules.reason,
                predicted_class=ai.predicted_class,
                verification="ai_guarded",
            )

        # Fail closed. Prefer the AI's class-specific recovery when it rejects
        # the grasp; otherwise expose the independent rule failure that vetoed
        # an AI GOOD prediction.
        rejected = ai if not ai.seated else rules
        yaw = rules.yaw_error_deg if rules.yaw_error_deg == rules.yaw_error_deg else rejected.yaw_error_deg
        return GraspCheckResult(
            seated=False,
            yaw_error_deg=yaw,
            confidence=min(ai.confidence, rules.confidence),
            reason=rejected.reason,
            predicted_class=ai.predicted_class,
            verification="ai_guarded",
        )

    return check


def get_grasp_check_fn(mode: VerificationMode) -> Callable[[np.ndarray], GraspCheckResult]:
    if mode == "rules":
        return grasp_check_rules
    if mode == "ai":
        from stacking_recovery.vision.grasp_quality_model import grasp_check_ml

        return grasp_check_ml
    if mode == "ai_guarded":
        from stacking_recovery.vision.grasp_quality_model import grasp_check_ml

        return build_guarded_ai_check(grasp_check_ml)
    from stacking_recovery.vision.grasp_check import grasp_check_stub

    return grasp_check_stub
