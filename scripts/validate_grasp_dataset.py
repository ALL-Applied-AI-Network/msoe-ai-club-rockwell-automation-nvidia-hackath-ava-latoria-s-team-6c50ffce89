#!/usr/bin/env python3
"""Validate a local or Brev-hosted grasp dataset before training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stacking_recovery.data.dataset_validation import validate_dataset
from stacking_recovery.data.grasp_dataset_store import resolve_data_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Dataset root (default: GRASP_DATA_ROOT or repo/data/grasp_quality)",
    )
    args = parser.parse_args()
    report = validate_dataset(resolve_data_root(ROOT, args.data))
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
