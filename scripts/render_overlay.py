#!/usr/bin/env python3
"""Burn manipulation JSONL fields onto a frame for the demo video."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--frame", type=Path, required=True)
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()
    try:
        import cv2
    except ImportError:
        raise SystemExit("pip install opencv-python") from None
    lines = [
        json.loads(l)
        for l in args.jsonl.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    last = lines[-1] if lines else {}
    img = cv2.imread(str(args.frame))
    if img is None:
        raise SystemExit(f"Could not read {args.frame}")
    y = 30
    for key in ("cube_colour", "action", "seated", "placed", "reason", "outcome"):
        cv2.putText(
            img,
            f"{key}: {last.get(key, '')}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        y += 28
    out = args.output or args.frame.with_name(args.frame.stem + "_overlay.png")
    cv2.imwrite(str(out), img)
    print(out)


if __name__ == "__main__":
    main()
