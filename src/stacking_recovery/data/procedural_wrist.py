"""
Procedural wrist-camera stand-in for offline training until Isaac Sim data exists.

Images mimic: centered cube blob, empty jaws, offset, rotation, partial capture, noise.
Replace with sim/collect_wrist_dataset.py on Brev without changing the training script.
"""

from __future__ import annotations

import random
from typing import Iterator

import numpy as np

from stacking_recovery.vision.grasp_labels import GRASP_CLASSES

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

DEFAULT_SIZE = (720, 1280, 3)


def _blank_wrist(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full(DEFAULT_SIZE, 28, dtype=np.uint8)
    img[:, :, 0] += rng.integers(0, 15, size=img.shape[:2], dtype=np.uint8)
    img[:, :, 1] += rng.integers(0, 12, size=img.shape[:2], dtype=np.uint8)
    img[:, :, 2] += rng.integers(0, 10, size=img.shape[:2], dtype=np.uint8)
    h, w = img.shape[:2]
    # Soft vignette + faint jaw guides (closer to real wrist FOV)
    yy, xx = np.ogrid[:h, :w]
    cx, cy = w // 2, int(h * 0.52)
    dist = np.sqrt(((xx - cx) / (w * 0.55)) ** 2 + ((yy - cy) / (h * 0.45)) ** 2)
    vignette = np.clip(1.0 - 0.35 * dist, 0.55, 1.0)
    img = (img.astype(np.float32) * vignette[:, :, None]).astype(np.uint8)
    if cv2 is not None:
        for xoff in (-int(w * 0.18), int(w * 0.18)):
            cv2.line(
                img,
                (cx + xoff, int(h * 0.22)),
                (cx + xoff, int(h * 0.78)),
                (55, 55, 58),
                2,
            )
    return img


def _draw_square(
    img: np.ndarray,
    center: tuple[int, int],
    size: int,
    angle_deg: float,
    colour_bgr: tuple[int, int, int],
) -> None:
    if cv2 is None:
        cy, cx = center[1], center[0]
        half = size // 2
        y0, y1 = max(0, cy - half), min(img.shape[0], cy + half)
        x0, x1 = max(0, cx - half), min(img.shape[1], cx + half)
        img[y0:y1, x0:x1] = colour_bgr
        return
    rect = ((float(center[0]), float(center[1])), (float(size), float(size)), angle_deg)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(img, box, colour_bgr)


def render_sample(label: str, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    img = _blank_wrist(seed)
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    colours = [
        (45, 160, 200),
        (160, 45, 30),
        (40, 160, 200),
    ]
    colour = colours[rng.randint(0, 2)]

    if label == "EMPTY_GRASP":
        return img
    if label == "GOOD":
        _draw_square(img, (cx, cy), rng.randint(70, 95), rng.uniform(-8, 8), colour)
    elif label == "MISALIGNED":
        _draw_square(
            img,
            (cx + rng.randint(40, 90), cy + rng.randint(-20, 20)),
            rng.randint(70, 95),
            rng.uniform(-5, 5),
            colour,
        )
    elif label == "ROTATED":
        _draw_square(img, (cx, cy), rng.randint(70, 95), rng.uniform(22, 38), colour)
    elif label == "SHIFTED":
        _draw_square(
            img,
            (cx + rng.randint(50, 80), cy),
            rng.randint(70, 95),
            rng.uniform(-5, 5),
            colour,
        )
    elif label == "PLACEMENT_FAILURE":
        _draw_square(img, (cx, cy - 40), rng.randint(50, 70), 0.0, colour)
    elif label == "UNCERTAIN":
        _draw_square(img, (cx, cy), rng.randint(60, 85), rng.uniform(-10, 10), colour)
        if cv2 is not None:
            img = cv2.GaussianBlur(img, (11, 11), 3)
            noise = np.random.default_rng(seed).integers(0, 40, size=img.shape, dtype=np.uint8)
            img = np.clip(img.astype(np.int16) + noise - 15, 0, 255).astype(np.uint8)
    else:
        raise ValueError(label)

    if cv2 is not None and label != "UNCERTAIN":
        from stacking_recovery.data.wrist_augment import augment_wrist_bgr, default_augment_config

        aug_cfg = default_augment_config()
        aug_cfg["blur_prob"] = 0.12
        aug_cfg["jpeg_prob"] = 0.15
        img = augment_wrist_bgr(img, aug_cfg, rng=random.Random(seed + 99991))

    return img


def iter_labeled_samples(
    per_class: int,
    base_seed: int = 0,
) -> Iterator[tuple[np.ndarray, str, int]]:
    for class_idx, label in enumerate(GRASP_CLASSES):
        for i in range(per_class):
            seed = base_seed + class_idx * 10_000 + i
            yield render_sample(label, seed), label, seed
