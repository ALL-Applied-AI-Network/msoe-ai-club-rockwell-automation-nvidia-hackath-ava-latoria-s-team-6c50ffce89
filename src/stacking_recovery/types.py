"""Shared result types and failure taxonomy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

class FailureReason(str, Enum):
    OK = "OK"
    EMPTY_GRASP = "EMPTY_GRASP"
    MISALIGNED_CUBE = "MISALIGNED_CUBE"
    OCCLUSION = "OCCLUSION"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    TILTED_OBJECT = "TILTED_OBJECT"
    PLACEMENT_FAILURE = "PLACEMENT_FAILURE"
    UNKNOWN = "UNKNOWN"
    STUB = "STUB"
    MAX_RETRIES = "MAX_RETRIES_EXCEEDED"


CONFIDENCE_MIN = 0.6


@dataclass(frozen=True)
class GraspCheckResult:
    """Wrist-camera grasp verification (plan name: seat_check)."""

    seated: bool
    yaw_error_deg: float
    confidence: float
    reason: FailureReason
    predicted_class: str | None = None
    verification: str = "rules"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["reason"] = self.reason.value
        if d.get("predicted_class") is None:
            del d["predicted_class"]
        return d


SeatCheckResult = GraspCheckResult


@dataclass(frozen=True)
class PlaceCheckResult:
    placed: bool
    confidence: float
    reason: FailureReason

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["reason"] = self.reason.value
        return d
