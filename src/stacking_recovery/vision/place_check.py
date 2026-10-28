"""Verify cube on tower after release."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from stacking_recovery.types import CONFIDENCE_MIN, FailureReason, PlaceCheckResult

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore


@dataclass(frozen=True)
class PlacementGeometryConfig:
    """Calibratable image-space gate for a cube at the stack site.

    Coordinates are normalized inside ``roi_xyxy``.  Higher stack levels move
    the expected centre and support line upward by ``level_y_step``.
    """

    roi_xyxy: tuple[float, float, float, float] = (0.30, 0.25, 0.70, 0.75)
    expected_center_xy: tuple[float, float] = (0.50, 0.50)
    center_tolerance: float = 0.28
    expected_bottom_y: float = 0.68
    bottom_tolerance: float = 0.30
    level_y_step: float = 0.035
    min_area_fraction: float = 0.015
    max_area_fraction: float = 0.55
    min_aspect_ratio: float = 0.45
    max_aspect_ratio: float = 2.20
    min_support_fraction: float = 0.20
    min_confidence: float = CONFIDENCE_MIN
    min_saturation: int = 30
    min_value: int = 35

    def __post_init__(self) -> None:
        x0, y0, x1, y1 = self.roi_xyxy
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError("roi_xyxy must be normalized and ordered")
        if not all(0 <= value <= 1 for value in self.expected_center_xy):
            raise ValueError("expected_center_xy must be normalized")
        if not 0 < self.center_tolerance <= 1:
            raise ValueError("center_tolerance must be in (0, 1]")
        if not 0 < self.bottom_tolerance <= 1:
            raise ValueError("bottom_tolerance must be in (0, 1]")
        if not 0 < self.min_area_fraction < self.max_area_fraction <= 1:
            raise ValueError("invalid component area limits")
        if not 0 < self.min_aspect_ratio <= self.max_aspect_ratio:
            raise ValueError("invalid aspect-ratio limits")
        if not 0 <= self.min_support_fraction <= 1:
            raise ValueError("min_support_fraction must be in [0, 1]")
        if not 0 < self.min_confidence <= 1:
            raise ValueError("min_confidence must be in (0, 1]")


def load_placement_config(path: str | Path) -> PlacementGeometryConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("roi_xyxy", "expected_center_xy"):
        if key in raw:
            raw[key] = tuple(float(v) for v in raw[key])
    return PlacementGeometryConfig(**raw)


def _target_component(
    mask: np.ndarray,
    stack_level: int,
    expected_center_xy: tuple[float, float],
) -> tuple[float, float, float, float, float, float] | None:
    """Return geometry for the component nearest the expected stack site.

    Saturated cubes that touch vertically form one connected component.  For
    stack levels above zero, treating that whole component as one object makes
    a valid tower look too tall.  Instead, use one component-width of pixels at
    its top as the newly placed cube, then measure whether saturated support is
    present immediately below it.  Selecting by target proximity prevents a
    large, saturated robot link elsewhere in the ROI from being mistaken for
    the placed cube.
    """
    h, w = mask.shape
    components: list[tuple[float, float, float, float, float, float]] = []
    if cv2 is not None:
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        if count <= 1:
            return None
        raw_components = [
            (
                labels == index,
                *(int(v) for v in stats[index]),
                *(float(v) for v in centroids[index]),
            )
            for index in range(1, count)
        ]
    else:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        x, y = int(xs.min()), int(ys.min())
        bw, bh = int(xs.max() - x + 1), int(ys.max() - y + 1)
        area = int(mask.sum())
        cx, cy = float(xs.mean()), float(ys.mean())
        raw_components = [(mask, x, y, bw, bh, area, cx, cy)]

    for component_mask, x, y, bw, bh, area, cx, cy in raw_components:
        support_fraction = 1.0
        if stack_level > 0:
            top_height = max(1, min(bh, bw))
            top = component_mask[y : y + top_height, x : x + bw]
            top_ys, top_xs = np.nonzero(top)
            if len(top_xs) == 0:
                continue
            top_x0, top_x1 = int(top_xs.min()), int(top_xs.max()) + 1
            top_y0, top_y1 = int(top_ys.min()), int(top_ys.max()) + 1
            area = int(len(top_xs))
            cx = x + float(top_xs.mean())
            cy = y + float(top_ys.mean())
            bw = top_x1 - top_x0
            bh = top_y1 - top_y0
            y = y + top_y0

            support_y = y + bh
            support_height = max(2, min(max(2, bh // 4), h - support_y))
            if support_height <= 0:
                support_fraction = 0.0
            else:
                support_x0 = max(0, x + top_x0)
                support_x1 = min(w, support_x0 + bw)
                support = component_mask[
                    support_y : support_y + support_height,
                    support_x0:support_x1,
                ]
                support_fraction = float(support.mean()) if support.size else 0.0

        components.append(
            (
                area / float(h * w),
                cx / max(w - 1, 1),
                cy / max(h - 1, 1),
                bw / max(float(bh), 1.0),
                (y + bh) / float(h),
                support_fraction,
            )
        )

    if not components:
        return None
    expected_x, expected_y = expected_center_xy
    return min(
        components,
        key=lambda item: float(np.hypot(item[1] - expected_x, item[2] - expected_y)),
    )


def place_check(
    image_bgr: np.ndarray,
    stack_level: int,
    *,
    config: PlacementGeometryConfig | None = None,
) -> PlaceCheckResult:
    """Verify a cube-sized coloured component is aligned with the stack site.

    Unlike the old fill-only heuristic, this checks component area, centre,
    shape and the expected support-line location.  The gate remains
    image-space and must be calibrated for the real camera before lab motion.
    """
    cfg = config or PlacementGeometryConfig()
    if image_bgr.ndim != 3:
        return PlaceCheckResult(
            placed=False, confidence=0.0, reason=FailureReason.LOW_CONFIDENCE
        )
    h, w = image_bgr.shape[:2]
    x0, y0, x1, y1 = cfg.roi_xyxy
    roi = image_bgr[int(h * y0) : int(h * y1), int(w * x0) : int(w * x1)]
    if roi.size == 0:
        return PlaceCheckResult(
            placed=False, confidence=0.0, reason=FailureReason.PLACEMENT_FAILURE
        )
    if cv2 is not None:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = (hsv[:, :, 1] >= cfg.min_saturation) & (
            hsv[:, :, 2] >= cfg.min_value
        )
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
    else:
        spread = roi.max(axis=2).astype(np.int16) - roi.min(axis=2).astype(np.int16)
        mask = (spread >= cfg.min_saturation) & (
            roi.mean(axis=2) >= cfg.min_value
        )
    level = max(0, int(stack_level))
    expected_x = cfg.expected_center_xy[0]
    expected_y = float(np.clip(cfg.expected_center_xy[1] - level * cfg.level_y_step, 0, 1))
    component = _target_component(mask, level, (expected_x, expected_y))
    if component is None:
        return PlaceCheckResult(
            placed=False,
            confidence=0.0,
            reason=FailureReason.PLACEMENT_FAILURE,
        )
    area, cx, cy, aspect, bottom, support_fraction = component
    expected_bottom = float(np.clip(cfg.expected_bottom_y - level * cfg.level_y_step, 0, 1))
    center_distance = float(np.hypot(cx - expected_x, cy - expected_y))
    center_score = max(0.0, 1.0 - center_distance / cfg.center_tolerance)
    bottom_score = max(0.0, 1.0 - abs(bottom - expected_bottom) / cfg.bottom_tolerance)
    area_mid = (cfg.min_area_fraction + cfg.max_area_fraction) / 2.0
    area_half = (cfg.max_area_fraction - cfg.min_area_fraction) / 2.0
    area_score = max(0.0, 1.0 - abs(area - area_mid) / max(area_half, 1e-9))
    aspect_mid = (cfg.min_aspect_ratio + cfg.max_aspect_ratio) / 2.0
    aspect_half = (cfg.max_aspect_ratio - cfg.min_aspect_ratio) / 2.0
    aspect_score = max(0.0, 1.0 - abs(aspect - aspect_mid) / max(aspect_half, 1e-9))
    support_score = 1.0 if level == 0 else min(
        1.0, support_fraction / max(cfg.min_support_fraction, 1e-9)
    )
    confidence = min(
        0.99,
        0.22 * center_score
        + 0.22 * bottom_score
        + 0.18 * area_score
        + 0.18 * aspect_score
        + 0.20 * support_score,
    )
    geometry_ok = (
        cfg.min_area_fraction <= area <= cfg.max_area_fraction
        and center_distance <= cfg.center_tolerance
        and cfg.min_aspect_ratio <= aspect <= cfg.max_aspect_ratio
        and abs(bottom - expected_bottom) <= cfg.bottom_tolerance
        and (level == 0 or support_fraction >= cfg.min_support_fraction)
    )
    if not geometry_ok or confidence < cfg.min_confidence:
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
