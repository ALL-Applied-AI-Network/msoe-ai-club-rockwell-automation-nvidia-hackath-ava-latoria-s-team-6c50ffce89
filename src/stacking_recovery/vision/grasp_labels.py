"""Grasp-quality class names and mapping to recovery / FailureReason."""

from __future__ import annotations

from stacking_recovery.types import FailureReason

GRASP_CLASSES: tuple[str, ...] = (
    "GOOD",
    "EMPTY_GRASP",
    "MISALIGNED",
    "ROTATED",
    "SHIFTED",
    "PLACEMENT_FAILURE",
    "UNCERTAIN",
)

CLASS_TO_INDEX = {name: i for i, name in enumerate(GRASP_CLASSES)}
INDEX_TO_CLASS = {i: name for i, name in enumerate(GRASP_CLASSES)}


def class_to_failure_reason(name: str) -> FailureReason:
    mapping = {
        "GOOD": FailureReason.OK,
        "EMPTY_GRASP": FailureReason.EMPTY_GRASP,
        "MISALIGNED": FailureReason.MISALIGNED_CUBE,
        "ROTATED": FailureReason.MISALIGNED_CUBE,
        "SHIFTED": FailureReason.MISALIGNED_CUBE,
        "PLACEMENT_FAILURE": FailureReason.PLACEMENT_FAILURE,
        "UNCERTAIN": FailureReason.LOW_CONFIDENCE,
    }
    return mapping.get(name, FailureReason.UNKNOWN)


def is_seated_class(name: str) -> bool:
    return name == "GOOD"
