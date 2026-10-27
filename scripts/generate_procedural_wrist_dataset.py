#!/usr/bin/env python3
"""Build data/grasp_quality from procedural images (pre-Isaac)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import cv2
except ImportError:
    raise SystemExit("pip install opencv-python") from None

from stacking_recovery.data.procedural_wrist import iter_labeled_samples
from stacking_recovery.vision.grasp_labels import GRASP_CLASSES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "grasp_quality")
    ap.add_argument("--per-class", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = args.out
    manifest = []
    for label in GRASP_CLASSES:
        (out / "images" / "train" / label).mkdir(parents=True, exist_ok=True)
    for img, label, seed in iter_labeled_samples(args.per_class, args.seed):
        rel = f"images/train/{label}/{label}_{seed:06d}.png"
        path = out / rel
        cv2.imwrite(str(path), img)
        manifest.append({"path": rel, "label": label, "seed": seed, "source": "procedural"})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest)} images to {out}")


if __name__ == "__main__":
    main()
