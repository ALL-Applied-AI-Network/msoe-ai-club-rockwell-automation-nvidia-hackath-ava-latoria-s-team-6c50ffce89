"""Verify cube on tower after release (wrist or table view)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from stacking_recovery.types import CONFIDENCE_MIN, FailureReason, PlaceCheckResult


def place_check(image_bgr: np.ndarray, stack_level: int) -> PlaceCheckResult:
    """
    TODO: detect expected colour/geometry at stack site for `stack_level`.

    Week 1 stub: same center-ROI heuristic as grasp_check.
    """
    _ = stack_level
    if image_bgr.ndim != 3:
        return PlaceCheckResult(
            placed=False, confidence=0.0, reason=FailureReason.LOW_CONFIDENCE
        )
    h, w = image_bgr.shape[:2]
    roi = image_bgr[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
    fill = float((roi.mean(axis=2) > 40).mean()) if roi.size else 0.0
    confidence = min(0.99, 0.4 + fill)
    if confidence < CONFIDENCE_MIN:
        return PlaceCheckResult(
            placed=False,
            confidence=confidence,
            reason=FailureReason.PLACEMENT_FAILURE,
        )
    return PlaceCheckResult(
        placed=True, confidence=confidence, reason=FailureReason.OK
    )


def place_check_stub(
    image_bgr: np.ndarray,
    stack_level: int,
    *,
    scripted: Optional[PlaceCheckResult] = None,
) -> PlaceCheckResult:
    _ = image_bgr, stack_level
    if scripted is not None:
        return scripted
    return PlaceCheckResult(placed=True, confidence=1.0, reason=FailureReason.STUB)
