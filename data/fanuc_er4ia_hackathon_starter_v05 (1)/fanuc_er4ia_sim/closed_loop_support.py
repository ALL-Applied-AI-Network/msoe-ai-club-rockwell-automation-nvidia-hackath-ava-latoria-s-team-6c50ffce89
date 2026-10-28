"""Repo path setup and closed-loop helpers for sort_scene.py."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_stacking_recovery_on_path() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        src = parent / "src"
        if (src / "stacking_recovery").is_dir():
            src_str = str(src)
            if src_str not in sys.path:
                sys.path.insert(0, src_str)
            return parent
    return None


def rgba_to_bgr(rgba) -> "object":
    import cv2
    import numpy as np

    if rgba is None:
        return np.zeros((720, 1280, 3), dtype=np.uint8)
    rgb = rgba[:, :, :3].astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
