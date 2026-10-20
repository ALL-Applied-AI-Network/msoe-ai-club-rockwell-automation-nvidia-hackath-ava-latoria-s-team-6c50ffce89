"""Select rule-based vs ML grasp verification."""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

from stacking_recovery.types import GraspCheckResult
from stacking_recovery.vision.grasp_check import grasp_check as grasp_check_rules

VerificationMode = Literal["rules", "ai", "stub"]


def get_grasp_check_fn(mode: VerificationMode) -> Callable[[np.ndarray], GraspCheckResult]:
    if mode == "rules":
        return grasp_check_rules
    if mode == "ai":
        from stacking_recovery.vision.grasp_quality_model import grasp_check_ml

        return grasp_check_ml
    from stacking_recovery.vision.grasp_check import grasp_check_stub

    return grasp_check_stub
