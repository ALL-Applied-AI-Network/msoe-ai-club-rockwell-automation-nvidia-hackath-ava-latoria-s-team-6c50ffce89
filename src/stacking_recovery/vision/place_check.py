"""Verify cube on tower after release."""

from __future__ import annotations

from typing import Optional

import numpy as np

from stacking_recovery.types import CONFIDENCE_MIN, FailureReason, PlaceCheckResult

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore


def place_check(image_bgr: np.ndarray, stack_level: int) -> PlaceCheckResult:
    """Look for saturated colour blob in upper-center ROI (stack region)."""
    _ = stack_level
    if image_bgr.ndim != 3:
        return PlaceCheckResult(
            placed=False, confidence=0.0, reason=FailureReason.LOW_CONFIDENCE
        )
    h, w = image_bgr.shape[:2]
    roi = image_bgr[int(h * 0.25) : int(h * 0.75), int(w * 0.3) : int(w * 0.7)]
    if roi.size == 0:
        return PlaceCheckResult(
            placed=False, confidence=0.0, reason=FailureReason.PLACEMENT_FAILURE
        )
    if cv2 is not None:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        fill = float((hsv[:, :, 1] > 30).mean())
    else:
        fill = float((roi.mean(axis=2) > 40).mean())
    confidence = min(0.99, 0.35 + fill)
    if confidence < CONFIDENCE_MIN:
        return PlaceCheckResult(
            placed=False,
            confidence=confidence,
            reason=FailureReason.PLACEMENT_FAILURE,
        )
    return PlaceCheckResult(placed=True, confidence=confidence, reason=FailureReason.OK)


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


def place_check_with_physics(
    image_bgr: np.ndarray,
    stack_level: int,
    *,
    physics_verified: bool,
) -> PlaceCheckResult:
    """Combine vision with starter placement_verified when sim reports geometry."""
    vision = place_check(image_bgr, stack_level)
    if not physics_verified:
        return PlaceCheckResult(
            placed=False,
            confidence=min(vision.confidence, 0.5),
            reason=FailureReason.PLACEMENT_FAILURE,
        )
    if not vision.placed:
        return vision
    return PlaceCheckResult(
        placed=True,
        confidence=vision.confidence,
        reason=FailureReason.OK,
    )
