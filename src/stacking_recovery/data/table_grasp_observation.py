"""Build model-ready before/after observations from the fixed table camera."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObservationQuality:
    valid: bool
    before_std: float
    after_std: float
    mean_abs_change: float
    padded_fraction: float
    reason: str | None = None

    def to_dict(self) -> dict[str, float | bool | str | None]:
        return {
            "valid": self.valid,
            "before_std": self.before_std,
            "after_std": self.after_std,
            "mean_abs_change": self.mean_abs_change,
            "padded_fraction": self.padded_fraction,
            "reason": self.reason,
        }


def _square_crop(
    image_bgr: np.ndarray,
    center_uv: tuple[float, float],
    size: int,
) -> tuple[np.ndarray, float]:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR image with shape HxWx3")
    if size <= 0:
        raise ValueError("Crop size must be positive")

    height, width = image_bgr.shape[:2]
    cx, cy = (int(round(center_uv[0])), int(round(center_uv[1])))
    x0, y0 = cx - size // 2, cy - size // 2
    x1, y1 = x0 + size, y0 + size
    src_x0, src_y0 = max(0, x0), max(0, y0)
    src_x1, src_y1 = min(width, x1), min(height, y1)
    copied = max(0, src_x1 - src_x0) * max(0, src_y1 - src_y0)
    padded_fraction = 1.0 - copied / float(size * size)

    crop = np.zeros((size, size, 3), dtype=image_bgr.dtype)
    if copied:
        dst_x0, dst_y0 = src_x0 - x0, src_y0 - y0
        crop[
            dst_y0 : dst_y0 + (src_y1 - src_y0),
            dst_x0 : dst_x0 + (src_x1 - src_x0),
        ] = image_bgr[src_y0:src_y1, src_x0:src_x1]
    return crop, padded_fraction


def make_before_after_observation(
    before_bgr: np.ndarray,
    after_bgr: np.ndarray,
    center_uv: tuple[float, float],
    *,
    crop_size: int = 256,
    min_std: float = 2.0,
    max_padded_fraction: float = 0.10,
) -> tuple[np.ndarray, ObservationQuality]:
    """Return a horizontal before/after crop and basic capture diagnostics."""
    if before_bgr.shape != after_bgr.shape:
        raise ValueError("Before and after images must have the same shape")
    before, before_pad = _square_crop(before_bgr, center_uv, crop_size)
    after, after_pad = _square_crop(after_bgr, center_uv, crop_size)
    padded = max(before_pad, after_pad)
    before_std = float(np.std(before))
    after_std = float(np.std(after))
    change = float(
        np.mean(np.abs(before.astype(np.float32) - after.astype(np.float32)))
    )

    reason = None
    if padded > max_padded_fraction:
        reason = "crop_out_of_frame"
    elif min(before_std, after_std) < min_std:
        reason = "blank_or_low_detail_crop"
    quality = ObservationQuality(
        valid=reason is None,
        before_std=before_std,
        after_std=after_std,
        mean_abs_change=change,
        padded_fraction=padded,
        reason=reason,
    )
    return np.concatenate([before, after], axis=1), quality
