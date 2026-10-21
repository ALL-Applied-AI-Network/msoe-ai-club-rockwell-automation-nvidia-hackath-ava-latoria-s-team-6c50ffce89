"""Deterministic dataset splits that keep every sample from one seed together."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


VALID_SPLITS = ("train", "val", "test")


def split_for_seed(seed: int) -> str:
    """Return an approximately 70/15/15 split without leaking a seed across sets.

    The permutation spreads small consecutive seed ranges across all three sets.
    Across every 20 seeds it is exactly 14 train, 3 validation, and 3 test.
    """
    bucket = (int(seed) * 11 + 3) % 20
    if bucket < 14:
        return "train"
    if bucket < 17:
        return "val"
    return "test"


def split_manifest_by_seed(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Use explicit splits, or derive them from seed while preserving seed groups."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        explicit = row.get("split")
        if explicit in VALID_SPLITS:
            split = str(explicit)
        elif row.get("seed") is not None:
            split = split_for_seed(int(row["seed"]))
        else:
            # Legacy procedural rows have unique, deterministic integer seeds.
            # Rows without either field stay in train rather than leaking randomly.
            split = "train"
        grouped[split].append(row)
    return grouped["train"], grouped["val"], grouped["test"]
