"""Wrist-camera grasp verification after jaw close (API: seat_check / grasp_check)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from stacking_recovery.types import (
    CONFIDENCE_MIN,
    FailureReason,
    GraspCheckResult,
)

MAX_YAW_DEG = 13.5

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore


def _estimate_square_yaw_deg(roi_bgr: np.ndarray) -> tuple[float, float]:
    """Return (yaw_error_deg modulo 45, confidence) from largest quadrilateral."""
    if cv2 is None:
        return 0.0, 0.55
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best_yaw = 0.0
    best_score = 0.0
    for cnt in contours:
        if cv2.contourArea(cnt) < 80:
            continue
        rect = cv2.minAreaRect(cnt)
        (_, _), (w, h), angle = rect
        if w < 1 or h < 1:
            continue
        side = min(w, h)
        long_side = max(w, h)
        if long_side / side > 2.5:
            continue
        yaw = abs(angle)
        if yaw > 45:
            yaw = 90 - yaw
        score = min(1.0, cv2.contourArea(cnt) / (roi_bgr.shape[0] * roi_bgr.shape[1]))
        if score > best_score:
            best_score = score
            best_yaw = float(yaw)
    conf = 0.45 + 0.5 * best_score if best_score > 0 else 0.35
    return best_yaw, conf


def grasp_check(image_bgr: np.ndarray) -> GraspCheckResult:
    """Classical colour + edge yaw on wrist ROI."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        return GraspCheckResult(
            seated=False,
            yaw_error_deg=float("nan"),
            confidence=0.0,
            reason=FailureReason.LOW_CONFIDENCE,
        )
    h, w = image_bgr.shape[:2]
    cy, cx = h // 2, w // 2
    rh, rw = max(h // 6, 20), max(w // 4, 20)
    y0, y1 = max(0, cy - rh), min(h, cy + rh)
    x0, x1 = max(0, cx - rw), min(w, cx + rw)
    roi = image_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return GraspCheckResult(
            seated=False,
            yaw_error_deg=float("nan"),
            confidence=0.0,
            reason=FailureReason.EMPTY_GRASP,
        )
    if cv2 is not None:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        colour_fill = float((sat > 35).mean())
    else:
        colour_fill = float((roi.mean(axis=2) > 40).mean())
    if colour_fill < 0.12:
        return GraspCheckResult(
            seated=False,
            yaw_error_deg=float("nan"),
            confidence=0.45,
            reason=FailureReason.EMPTY_GRASP,
        )
    yaw, edge_conf = _estimate_square_yaw_deg(roi)
    confidence = min(0.99, 0.35 + 0.4 * colour_fill + 0.25 * edge_conf)
    if confidence < CONFIDENCE_MIN:
        return GraspCheckResult(
            seated=False,
            yaw_error_deg=yaw,
            confidence=confidence,
            reason=FailureReason.LOW_CONFIDENCE,
        )
    if yaw > MAX_YAW_DEG:
        return GraspCheckResult(
            seated=False,
            yaw_error_deg=yaw,
            confidence=confidence,
            reason=FailureReason.MISALIGNED_CUBE,
        )
    return GraspCheckResult(
        seated=True,
        yaw_error_deg=yaw,
        confidence=confidence,
        reason=FailureReason.OK,
    )


seat_check = grasp_check


def grasp_check_stub(
    image_bgr: np.ndarray,
    *,
    scripted: Optional[GraspCheckResult] = None,
) -> GraspCheckResult:
    _ = image_bgr
    if scripted is not None:
        return scripted
    return GraspCheckResult(
        seated=True,
        yaw_error_deg=0.0,
        confidence=1.0,
        reason=FailureReason.STUB,
    )

