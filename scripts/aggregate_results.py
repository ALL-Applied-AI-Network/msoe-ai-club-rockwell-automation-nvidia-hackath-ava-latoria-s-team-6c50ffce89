#!/usr/bin/env python3
"""Summarize one or more sim run directories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stacking_recovery.bench.metrics import summarize_run, summarize_suite


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--suite", action="store_true", help="Aggregate suite-level rates")
    args = ap.parse_args()
    if args.suite:
        payload = summarize_suite(args.runs)
    else:
        payload = [summarize_run(p) for p in args.runs]
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
