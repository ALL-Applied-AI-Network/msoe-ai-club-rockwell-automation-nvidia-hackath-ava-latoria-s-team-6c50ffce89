"""Wrist-camera grasp verification after jaw close."""

from __future__ import annotations

from typing import Optional

import numpy as np

from stacking_recovery.types import (
    CONFIDENCE_MIN,
    FailureReason,
    GraspCheckResult,
)

MAX_YAW_DEG = 13.5


def grasp_check(image_bgr: np.ndarray) -> GraspCheckResult:
    """
    Classical vision on 1280×720 wrist crop (TODO: colour blob + edge yaw).

    Cosmos stays on the table camera; this must be fast enough for retries.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        return GraspCheckResult(
            seated=False,
            yaw_error_deg=float("nan"),
            confidence=0.0,
            reason=FailureReason.LOW_CONFIDENCE,
        )
    h, w = image_bgr.shape[:2]
    cy, cx = h // 2, w // 2
    rh, rw = h // 8, w // 8
    roi = image_bgr[cy - rh : cy + rh, cx - rw : cx + rw]
    if roi.size == 0:
        return GraspCheckResult(
            seated=False,
            yaw_error_deg=float("nan"),
            confidence=0.0,
            reason=FailureReason.EMPTY_GRASP,
        )
    gray = roi.mean(axis=2)
    fill = float((gray > 40).mean())
    if fill < 0.15:
        return GraspCheckResult(
            seated=False,
            yaw_error_deg=float("nan"),
            confidence=0.5,
            reason=FailureReason.EMPTY_GRASP,
        )
    confidence = min(0.99, 0.5 + fill)
    yaw = 0.0
    if confidence < CONFIDENCE_MIN:
        return GraspCheckResult(
            seated=False,
            yaw_error_deg=yaw,
            confidence=confidence,
            reason=FailureReason.LOW_CONFIDENCE,
        )
    seated = yaw <= MAX_YAW_DEG
    return GraspCheckResult(
        seated=seated,
        yaw_error_deg=yaw,
        confidence=confidence,
        reason=FailureReason.OK if seated else FailureReason.MISALIGNED_CUBE,
    )


def grasp_check_stub(
    image_bgr: np.ndarray,
    *,
    scripted: Optional[GraspCheckResult] = None,
) -> GraspCheckResult:
    """Drive the recovery policy in tests before real vision exists."""
    _ = image_bgr
    if scripted is not None:
        return scripted
    return GraspCheckResult(
        seated=True,
        yaw_error_deg=0.0,
        confidence=1.0,
        reason=FailureReason.STUB,
    )
