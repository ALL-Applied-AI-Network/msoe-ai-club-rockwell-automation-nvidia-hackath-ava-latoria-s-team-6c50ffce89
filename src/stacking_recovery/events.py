"""JSONL event log for grasp / placement attempts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

ActionCode = Literal[
    "grasp_close",
    "grasp_check",
    "place_check",
    "regrasp",
    "carry",
    "place",
    "safe_stop",
]

OutcomeCode = Literal["success", "retry", "failed", "in_progress"]


@dataclass
class ManipulationEvent:
    timestamp_utc: str
    cube_colour: str
    stack_level: int
    attempt_index: int
    table_pose_confidence: float
    action: ActionCode
    reason: str
    outcome: OutcomeCode
    seated: Optional[bool] = None
    placed: Optional[bool] = None
    check_confidence: Optional[float] = None
    yaw_error_deg: Optional[float] = None
    duration_s: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def new_event(
    *,
    cube_colour: str,
    stack_level: int,
    attempt_index: int,
    table_pose_confidence: float,
    action: ActionCode,
    reason: str,
    outcome: OutcomeCode,
    seated: Optional[bool] = None,
    placed: Optional[bool] = None,
    check_confidence: Optional[float] = None,
    yaw_error_deg: Optional[float] = None,
    duration_s: Optional[float] = None,
    extra: Optional[dict[str, Any]] = None,
) -> ManipulationEvent:
    return ManipulationEvent(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        cube_colour=cube_colour,
        stack_level=stack_level,
        attempt_index=attempt_index,
        table_pose_confidence=table_pose_confidence,
        action=action,
        reason=reason,
        outcome=outcome,
        seated=seated,
        placed=placed,
        check_confidence=check_confidence,
        yaw_error_deg=yaw_error_deg,
        duration_s=duration_s,
        extra=extra or {},
    )


def append_event(path: Path, event: ManipulationEvent) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event.to_dict()) + "\n")
