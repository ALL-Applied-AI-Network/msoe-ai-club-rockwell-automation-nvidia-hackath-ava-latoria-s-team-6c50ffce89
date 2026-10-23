"""Build a saved, self-contained augmented copy of a grasp dataset."""

from __future__ import annotations

import hashlib
import os
import random
import shutil
from pathlib import Path
from typing import Any

from stacking_recovery.data.grasp_dataset_store import load_manifest, save_manifest
from stacking_recovery.data.wrist_augment import (
    augment_table_pair_bgr,
    augment_wrist_bgr,
)

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore


def _stable_seed(row: dict[str, Any], copy_index: int) -> int:
    identity = str(row.get("sample_id") or row.get("path") or "sample")
    digest = hashlib.sha256(f"{identity}:{copy_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def materialize_augmented_dataset(
    source_root: Path,
    output_root: Path,
    *,
    copies_per_train: int = 2,
    augmentation_config: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Save deterministic augmented training copies in a separate dataset.

    Raw images are hard-linked when possible (copied otherwise), making the
    derived dataset self-contained without modifying the live collection
    manifest. Validation and test rows are copied but never augmented.
    """
    if cv2 is None:
        raise RuntimeError("opencv is required to materialize augmentations")
    if copies_per_train < 0:
        raise ValueError("copies_per_train must be non-negative")

    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if source_root == output_root:
        raise ValueError("output_root must differ from source_root")

    source_rows = load_manifest(source_root / "manifest.json")
    derived_rows: list[dict[str, Any]] = []

    for row in source_rows:
        rel = Path(str(row["path"]))
        source_image = source_root / rel
        if not source_image.is_file():
            raise FileNotFoundError(source_image)

        raw_rel = Path("raw") / rel
        _link_or_copy(source_image, output_root / raw_rel)
        raw_row = dict(row)
        raw_row["path"] = raw_rel.as_posix()
        raw_row["materialized_kind"] = "raw"
        derived_rows.append(raw_row)

        if row.get("split") != "train":
            continue

        image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unable to read image: {source_image}")
        label = str(row.get("label", "UNKNOWN"))
        original_id = str(row.get("sample_id") or rel.stem)
        for copy_index in range(copies_per_train):
            aug_id = f"{original_id}-aug{copy_index + 1:02d}"
            aug_rel = Path("augmented") / "train" / label / f"{aug_id}.png"
            aug_path = output_root / aug_rel
            if overwrite or not aug_path.is_file():
                rng = random.Random(_stable_seed(row, copy_index))
                if row.get("observation_source") == "table_before_after":
                    augmented = augment_table_pair_bgr(
                        image, augmentation_config, rng=rng
                    )
                else:
                    augmented = augment_wrist_bgr(
                        image, augmentation_config, rng=rng
                    )
                aug_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(aug_path), augmented):
                    raise OSError(f"unable to write image: {aug_path}")

            aug_row = dict(row)
            aug_row.update(
                {
                    "path": aug_rel.as_posix(),
                    "sample_id": aug_id,
                    "source": "saved_augmentation",
                    "augmentation_of": original_id,
                    "augmentation_index": copy_index + 1,
                    "materialized_kind": "augmentation",
                }
            )
            derived_rows.append(aug_row)

    save_manifest(output_root / "manifest.json", derived_rows)
    return derived_rows
