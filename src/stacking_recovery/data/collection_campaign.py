"""Plan balanced Isaac collection attempts for the binary pickup verifier."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from stacking_recovery.data.dataset_split import VALID_SPLITS, split_for_seed


BINARY_LABELS = ("GOOD", "FAILURE")


def binary_label(label: str) -> str:
    """Normalize detailed grasp labels to the table verifier's two classes."""
    return "GOOD" if label == "GOOD" else "FAILURE"


@dataclass(frozen=True)
class CampaignTargets:
    """Desired final count per binary label for each held-out split."""

    train: int = 100
    val: int = 25
    test: int = 25

    def for_split(self, split: str) -> int:
        if split not in VALID_SPLITS:
            raise ValueError(f"invalid split: {split}")
        return int(getattr(self, split))

    @property
    def total(self) -> int:
        return 2 * (self.train + self.val + self.test)


@dataclass(frozen=True)
class CollectionAttempt:
    target_label: str
    split: str
    seed: int
    pick_index: int
    scenario: str
    trial: dict[str, Any]


def count_binary_by_split(rows: Iterable[dict[str, Any]]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        split = str(row.get("split", "train"))
        label = row.get("label")
        if split in VALID_SPLITS and isinstance(label, str):
            counts[(split, binary_label(label))] += 1
    return counts


def campaign_complete(
    rows: Iterable[dict[str, Any]], targets: CampaignTargets
) -> bool:
    counts = count_binary_by_split(rows)
    return all(
        counts[(split, label)] >= targets.for_split(split)
        for split in VALID_SPLITS
        for label in BINARY_LABELS
    )


def _next_seed_for_split(split: str, used_seeds: set[int]) -> int:
    seed = 0
    while seed in used_seeds or split_for_seed(seed) != split:
        seed += 1
    return seed


def _choose_deficit(
    counts: Counter[tuple[str, str]], targets: CampaignTargets
) -> tuple[str, str] | None:
    deficits: list[tuple[float, int, str, str]] = []
    for split in VALID_SPLITS:
        target = targets.for_split(split)
        for label in BINARY_LABELS:
            missing = max(0, target - counts[(split, label)])
            if missing:
                # Fraction first keeps small validation/test targets from being
                # ignored while the larger training quota is filled.
                deficits.append((missing / target, missing, split, label))
    if not deficits:
        return None
    _, _, split, label = max(deficits)
    return split, label


def _failure_trial(seed: int) -> tuple[str, dict[str, Any]]:
    """Create one reproducible but varied intentional grasp failure."""
    rng = random.Random(seed * 7919 + 17)
    kind = (seed // 2) % 3
    if kind == 0:
        yaw = rng.uniform(16.0, 28.0) * rng.choice((-1.0, 1.0))
        return "rotated", {
            "condition_id": "yaw_error",
            "yaw_offset_deg": round(yaw, 3),
        }

    angle = rng.uniform(0.0, 2.0 * math.pi)
    if kind == 1:
        magnitude = rng.uniform(0.0055, 0.010)
        scenario = "shifted"
    else:
        magnitude = rng.uniform(0.014, 0.022)
        scenario = "empty_grasp"
    return scenario, {
        "condition_id": "pick_xy_shift",
        "pick_xy_offset_m": [
            round(magnitude * math.cos(angle), 6),
            round(magnitude * math.sin(angle), 6),
        ],
    }


def next_collection_attempt(
    rows: list[dict[str, Any]],
    targets: CampaignTargets,
    *,
    attempted_seeds: Iterable[int] = (),
) -> CollectionAttempt | None:
    """Plan the next attempt from actual manifest deficits.

    Outcomes are intentionally not assumed. If a requested failure succeeds,
    the manifest records GOOD and the next call continues filling FAILURE.
    """
    counts = count_binary_by_split(rows)
    deficit = _choose_deficit(counts, targets)
    if deficit is None:
        return None
    split, label = deficit
    used_seeds = {int(seed) for seed in attempted_seeds}
    used_seeds.update({
        int(row["seed"])
        for row in rows
        if isinstance(row.get("seed"), int)
    })
    seed = _next_seed_for_split(split, used_seeds)
    if label == "GOOD":
        scenario = "nominal"
        details: dict[str, Any] = {"condition_id": "nominal"}
    else:
        scenario, details = _failure_trial(seed)
    trial = {
        **details,
        "seed": seed,
        "policy": "open_loop",
        "finger": "stock_aa_scallop",
        "campaign_target": label,
        "campaign_scenario": scenario,
    }
    return CollectionAttempt(
        target_label=label,
        split=split,
        seed=seed,
        pick_index=seed % 6,
        scenario=scenario,
        trial=trial,
    )


def campaign_progress(
    rows: Iterable[dict[str, Any]], targets: CampaignTargets
) -> dict[str, dict[str, dict[str, int]]]:
    counts = count_binary_by_split(rows)
    return {
        split: {
            label: {
                "count": counts[(split, label)],
                "target": targets.for_split(split),
            }
            for label in BINARY_LABELS
        }
        for split in VALID_SPLITS
    }
