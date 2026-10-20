"""Training-time augmentations for wrist ROI crops."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore


def default_augment_config() -> dict[str, Any]:
    return {
        "brightness_delta": 0.18,
        "contrast_delta": 0.12,
        "noise_std": 10.0,
        "blur_prob": 0.25,
        "blur_ksize": 5,
        "jpeg_prob": 0.2,
        "jpeg_quality_min": 55,
        "jpeg_quality_max": 92,
        "hflip_prob": 0.0,
    }


def augment_wrist_bgr(
    image_bgr: np.ndarray,
    config: dict[str, Any] | None = None,
    rng: random.Random | None = None,
) -> np.ndarray:
    """Apply random photometric / mild blur augmentations (BGR, H×W×3)."""
    if cv2 is None:
        return image_bgr
    cfg = config or default_augment_config()
    r = rng or random.Random()
    out = image_bgr.copy()

    bdelta = float(cfg.get("brightness_delta", 0.18))
    cdelta = float(cfg.get("contrast_delta", 0.12))
    alpha = 1.0 + r.uniform(-cdelta, cdelta)
    beta = r.uniform(-bdelta, bdelta) * 255.0
    out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    noise_std = float(cfg.get("noise_std", 0))
    if noise_std > 0:
        noise = np.random.default_rng(r.randint(0, 2**31 - 1)).normal(
            0, noise_std, out.shape
        )
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if r.random() < float(cfg.get("blur_prob", 0)):
        k = int(cfg.get("blur_ksize", 5))
        k = k if k % 2 == 1 else k + 1
        out = cv2.GaussianBlur(out, (k, k), 0)

    if r.random() < float(cfg.get("jpeg_prob", 0)):
        qmin = int(cfg.get("jpeg_quality_min", 55))
        qmax = int(cfg.get("jpeg_quality_max", 92))
        ok, buf = cv2.imencode(
            ".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), r.randint(qmin, qmax)]
        )
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    if r.random() < float(cfg.get("hflip_prob", 0)):
        out = cv2.flip(out, 1)

    return out
