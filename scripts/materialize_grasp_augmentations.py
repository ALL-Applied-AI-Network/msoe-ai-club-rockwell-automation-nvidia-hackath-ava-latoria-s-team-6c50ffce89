#!/usr/bin/env python3
"""Save augmented training copies without touching the source dataset."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stacking_recovery.data.materialized_augment import (  # noqa: E402
    materialize_augmented_dataset,
)
from stacking_recovery.vision.grasp_quality_model import load_grasp_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--copies-per-train", type=int, default=2)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "grasp_model.yaml"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_grasp_config(args.config)
    rows = materialize_augmented_dataset(
        args.source,
        args.out,
        copies_per_train=args.copies_per_train,
        augmentation_config=config.get("augmentation") or {},
        overwrite=args.overwrite,
    )
    kinds = Counter(row.get("materialized_kind") for row in rows)
    splits = Counter(row.get("split") for row in rows)
    print(
        json.dumps(
            {"total": len(rows), "kinds": dict(kinds), "splits": dict(splits)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
