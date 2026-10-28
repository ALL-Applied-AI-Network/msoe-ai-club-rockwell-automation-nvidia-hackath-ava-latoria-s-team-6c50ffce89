"""Fast, dependency-free validation for shared grasp datasets."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from stacking_recovery.data.grasp_dataset_store import load_manifest
from stacking_recovery.vision.grasp_labels import GRASP_CLASSES

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def validate_dataset(data_root: Path) -> dict[str, Any]:
    manifest_path = data_root / "manifest.json"
    rows = load_manifest(manifest_path)
    errors: list[str] = []
    warnings: list[str] = []
    labels: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    sample_ids: set[str] = set()
    paths: set[str] = set()
    missing_sample_ids = 0

    if not manifest_path.is_file():
        errors.append(f"missing manifest: {manifest_path}")

    for index, row in enumerate(rows):
        prefix = f"row {index}"
        rel_text = row.get("path")
        label = row.get("label")
        source = row.get("source")
        sample_id = row.get("sample_id")

        if label not in GRASP_CLASSES:
            errors.append(f"{prefix}: invalid label {label!r}")
        else:
            labels[label] += 1
        if not source:
            errors.append(f"{prefix}: missing source")
        else:
            sources[str(source)] += 1

        if sample_id:
            if sample_id in sample_ids:
                errors.append(f"{prefix}: duplicate sample_id {sample_id}")
            sample_ids.add(str(sample_id))
        else:
            missing_sample_ids += 1

        if not isinstance(rel_text, str) or not rel_text:
            errors.append(f"{prefix}: missing path")
            continue
        rel = Path(rel_text)
        if rel.is_absolute() or ".." in rel.parts:
            errors.append(f"{prefix}: unsafe path {rel_text!r}")
            continue
        if rel_text in paths:
            errors.append(f"{prefix}: duplicate path {rel_text}")
        paths.add(rel_text)
        image_path = data_root / rel
        if not image_path.is_file():
            errors.append(f"{prefix}: missing image {rel_text}")
            continue
        with image_path.open("rb") as image_file:
            if image_file.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
                errors.append(f"{prefix}: invalid PNG {rel_text}")

    missing_labels = sorted(set(GRASP_CLASSES) - set(labels))
    if missing_sample_ids:
        warnings.append(f"{missing_sample_ids} rows are missing sample_id")
    if missing_labels:
        warnings.append("labels with no samples: " + ", ".join(missing_labels))

    return {
        "ok": not errors,
        "sample_count": len(rows),
        "labels": dict(sorted(labels.items())),
        "sources": dict(sorted(sources.items())),
        "errors": errors,
        "warnings": warnings,
    }
