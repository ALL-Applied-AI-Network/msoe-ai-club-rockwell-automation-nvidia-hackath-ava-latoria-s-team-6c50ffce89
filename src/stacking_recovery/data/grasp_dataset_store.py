"""Manifest + image storage for grasp-quality training (procedural, Isaac, feedback)."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from stacking_recovery.vision.grasp_labels import GRASP_CLASSES

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None  # type: ignore
    np = None  # type: ignore


DATA_ROOT_ENV = "GRASP_DATA_ROOT"


def default_data_root(repo_root: Path) -> Path:
    """Return the local default or the shared Brev dataset location."""
    configured = os.environ.get(DATA_ROOT_ENV)
    if not configured:
        return (repo_root / "data" / "grasp_quality").resolve()
    root = Path(configured).expanduser()
    if not root.is_absolute():
        root = repo_root / root
    return root.resolve()


def resolve_data_root(repo_root: Path, explicit: Optional[Path] = None) -> Path:
    """Resolve CLI override first, then GRASP_DATA_ROOT, then the repo default."""
    if explicit is None:
        return default_data_root(repo_root)
    root = explicit.expanduser()
    if not root.is_absolute():
        root = repo_root / root
    return root.resolve()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def append_manifest_row(path: Path, row: dict[str, Any]) -> None:
    rows = load_manifest(path)
    rows.append(row)
    save_manifest(path, rows)


@dataclass
class GraspSampleRecord:
    path: str
    label: str
    source: str
    sample_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    meta: dict[str, Any] = field(default_factory=dict)

    def to_manifest_row(self) -> dict[str, Any]:
        row = asdict(self)
        if not row["meta"]:
            del row["meta"]
        else:
            row.update(row.pop("meta"))
        return row

    def validate(self) -> None:
        if self.label not in GRASP_CLASSES:
            raise ValueError(f"Invalid label: {self.label}")


class GraspDatasetStore:
    """Write labeled wrist PNGs under data/grasp_quality/images/train/{label}/."""

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.manifest_path = data_root / "manifest.json"
        self.feedback_log_path = data_root / "feedback_log.jsonl"

    def save_sample(
        self,
        image_bgr: "np.ndarray",
        label: str,
        *,
        source: str,
        meta: Optional[dict[str, Any]] = None,
        split: str = "train",
    ) -> GraspSampleRecord:
        if cv2 is None or np is None:
            raise RuntimeError("opencv required to save grasp samples")
        record = GraspSampleRecord(
            path="",
            label=label,
            source=source,
            meta=dict(meta or {}),
        )
        record.validate()
        fname = f"{source}_{record.sample_id}.png"
        rel = f"images/{split}/{label}/{fname}"
        out = self.data_root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), image_bgr)
        record.path = rel
        row = {
            "path": record.path,
            "label": record.label,
            "source": record.source,
            "sample_id": record.sample_id,
            "created_at": record.created_at,
            **record.meta,
        }
        append_manifest_row(self.manifest_path, row)
        self._log_feedback(row)
        return record

    def _log_feedback(self, row: dict[str, Any]) -> None:
        self.feedback_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.feedback_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def count_by_source(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in load_manifest(self.manifest_path):
            src = row.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        return counts
