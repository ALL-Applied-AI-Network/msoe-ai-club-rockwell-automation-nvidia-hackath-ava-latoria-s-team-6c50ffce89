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
        "gamma_min": 0.85,
        "gamma_max": 1.15,
        "saturation_delta": 0.12,
        "warmth_delta": 0.08,
        "noise_std": 10.0,
        "blur_prob": 0.25,
        "blur_ksize": 5,
        "jpeg_prob": 0.2,
        "jpeg_quality_min": 55,
        "jpeg_quality_max": 92,
        "shadow_prob": 0.2,
        "shadow_strength_max": 0.18,
        "hflip_prob": 0.0,
        "affine_prob": 0.0,
        "rotation_deg": 0.0,
        "translate_frac": 0.0,
        "scale_delta": 0.0,
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

    gamma_min = float(cfg.get("gamma_min", 1.0))
    gamma_max = float(cfg.get("gamma_max", 1.0))
    if gamma_max < gamma_min:
        gamma_min, gamma_max = gamma_max, gamma_min
    gamma = r.uniform(gamma_min, gamma_max)
    if abs(gamma - 1.0) > 1e-6:
        normalized = out.astype(np.float32) / 255.0
        out = np.clip(np.power(normalized, gamma) * 255.0, 0, 255).astype(np.uint8)

    saturation_delta = float(cfg.get("saturation_delta", 0.0))
    if saturation_delta > 0:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= 1.0 + r.uniform(-saturation_delta, saturation_delta)
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    warmth_delta = float(cfg.get("warmth_delta", 0.0))
    if warmth_delta > 0:
        warmth = r.uniform(-warmth_delta, warmth_delta)
        channels = out.astype(np.float32)
        channels[..., 0] *= 1.0 - warmth
        channels[..., 2] *= 1.0 + warmth
        out = np.clip(channels, 0, 255).astype(np.uint8)

    if r.random() < float(cfg.get("shadow_prob", 0.0)):
        strength = r.uniform(0.0, float(cfg.get("shadow_strength_max", 0.0)))
        height, width = out.shape[:2]
        if r.random() < 0.5:
            gradient = np.linspace(1.0 - strength, 1.0, width, dtype=np.float32)[None, :, None]
            if r.random() < 0.5:
                gradient = gradient[:, ::-1, :]
        else:
            gradient = np.linspace(1.0 - strength, 1.0, height, dtype=np.float32)[:, None, None]
            if r.random() < 0.5:
                gradient = gradient[::-1, :, :]
        out = np.clip(out.astype(np.float32) * gradient, 0, 255).astype(np.uint8)

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


def augment_table_pair_bgr(
    image_bgr: np.ndarray,
    config: dict[str, Any] | None = None,
    rng: random.Random | None = None,
) -> np.ndarray:
    """Apply one coherent augmentation to concatenated before/after frames.

    Each half receives the same random photometric and geometric parameters so
    augmentation artifacts cannot masquerade as robot motion.
    """
    if cv2 is None:
        return image_bgr
    height, width = image_bgr.shape[:2]
    if width < 2 or width % 2:
        return augment_wrist_bgr(image_bgr, config, rng=rng)

    cfg = config or default_augment_config()
    r = rng or random.Random()
    midpoint = width // 2
    before = image_bgr[:, :midpoint]
    after = image_bgr[:, midpoint:]
    if before.shape != after.shape:
        return augment_wrist_bgr(image_bgr, cfg, rng=r)

    # Replay one RNG stream so both frames receive identical camera effects.
    shared_seed = r.randint(0, 2**31 - 1)
    before_aug = augment_wrist_bgr(before, cfg, rng=random.Random(shared_seed))
    after_aug = augment_wrist_bgr(after, cfg, rng=random.Random(shared_seed))

    if r.random() < float(cfg.get("affine_prob", 0.0)):
        angle_limit = float(cfg.get("rotation_deg", 0.0))
        angle = r.uniform(-angle_limit, angle_limit)
        scale_delta = float(cfg.get("scale_delta", 0.0))
        scale = 1.0 + r.uniform(-scale_delta, scale_delta)
        translate_frac = float(cfg.get("translate_frac", 0.0))
        tx = r.uniform(-translate_frac, translate_frac) * midpoint
        ty = r.uniform(-translate_frac, translate_frac) * height
        matrix = cv2.getRotationMatrix2D(
            ((midpoint - 1) / 2.0, (height - 1) / 2.0), angle, scale
        )
        matrix[:, 2] += (tx, ty)
        warp_args = {
            "dsize": (midpoint, height),
            "flags": cv2.INTER_LINEAR,
            "borderMode": cv2.BORDER_REFLECT_101,
        }
        before_aug = cv2.warpAffine(before_aug, matrix, **warp_args)
        after_aug = cv2.warpAffine(after_aug, matrix, **warp_args)

    return np.concatenate([before_aug, after_aug], axis=1)
