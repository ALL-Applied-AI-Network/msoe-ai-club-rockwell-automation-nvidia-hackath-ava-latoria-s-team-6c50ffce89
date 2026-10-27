"""Safe boundary between planning code and a supervised robot driver.

Only shadow and disabled adapters live in this repository.  A real FANUC and
SCHUNK adapter must be supplied and reviewed with the lab mentor; selecting a
real adapter from the checked-in template fails closed.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


class HardwareInterlockError(RuntimeError):
    """Raised when a command would cross an unapproved hardware boundary."""


@dataclass(frozen=True)
class PoseTarget:
    """Robot-agnostic Cartesian target in metres and a unit quaternion."""

    position_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    frame_id: str = "robot_base"

    def __post_init__(self) -> None:
        values = (*self.position_m, *self.orientation_xyzw)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("pose values must be finite")
        norm = math.sqrt(sum(value * value for value in self.orientation_xyzw))
        if not 0.99 <= norm <= 1.01:
            raise ValueError("orientation_xyzw must be a unit quaternion")
        if not self.frame_id.strip():
            raise ValueError("frame_id is required")


@dataclass(frozen=True)
class RobotStatus:
    mode: str
    connected: bool
    motion_enabled: bool
    stopped: bool
    gripper: str
    last_command: str | None = None


@dataclass(frozen=True)
class CommandRecord:
    command: str
    payload: dict[str, object]


@runtime_checkable
class RobotAdapter(Protocol):
    """Minimal interface used by the planner and state machine."""

    def move_to_pose(self, target: PoseTarget, *, speed_scale: float) -> None: ...

    def open_gripper(self) -> None: ...

    def close_gripper(self) -> None: ...

    def stop(self) -> None: ...

    def get_status(self) -> RobotStatus: ...


class ShadowRobotAdapter:
    """Record intended commands without connecting to or moving hardware."""

    def __init__(self) -> None:
        self.commands: list[CommandRecord] = []
        self._stopped = False
        self._gripper = "unknown"

    def _record(self, command: str, payload: dict[str, object] | None = None) -> None:
        if self._stopped and command != "stop":
            raise HardwareInterlockError(
                "shadow adapter is stopped; create a new supervised session to continue"
            )
        self.commands.append(CommandRecord(command, payload or {}))

    def move_to_pose(self, target: PoseTarget, *, speed_scale: float) -> None:
        if not 0.0 < speed_scale <= 0.25:
            raise HardwareInterlockError(
                "shadow qualification limits speed_scale to (0, 0.25]"
            )
        self._record(
            "move_to_pose",
            {"target": asdict(target), "speed_scale": float(speed_scale)},
        )

    def open_gripper(self) -> None:
        self._record("open_gripper")
        self._gripper = "open"

    def close_gripper(self) -> None:
        self._record("close_gripper")
        self._gripper = "closed"

    def stop(self) -> None:
        if not self._stopped:
            self.commands.append(CommandRecord("stop", {}))
        self._stopped = True

    def get_status(self) -> RobotStatus:
        return RobotStatus(
            mode="shadow",
            connected=False,
            motion_enabled=False,
            stopped=self._stopped,
            gripper=self._gripper,
            last_command=self.commands[-1].command if self.commands else None,
        )


class DisabledHardwareAdapter:
    """Fail every command when a real adapter has not been approved."""

    def __init__(self, reason: str = "real hardware adapter is disabled") -> None:
        self.reason = reason
        self._stopped = True

    def _deny(self) -> None:
        raise HardwareInterlockError(self.reason)

    def move_to_pose(self, target: PoseTarget, *, speed_scale: float) -> None:
        _ = target, speed_scale
        self._deny()

    def open_gripper(self) -> None:
        self._deny()

    def close_gripper(self) -> None:
        self._deny()

    def stop(self) -> None:
        self._stopped = True

    def get_status(self) -> RobotStatus:
        return RobotStatus(
            mode="disabled",
            connected=False,
            motion_enabled=False,
            stopped=True,
            gripper="unknown",
            last_command="stop" if self._stopped else None,
        )


def load_hardware_adapter(config_path: str | Path) -> RobotAdapter:
    """Load only checked-in safe modes; real drivers require separate review."""

    path = Path(config_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    mode = str(config.get("mode", "disabled")).strip().lower()
    if mode == "shadow":
        if config.get("motion_enabled") is True:
            raise HardwareInterlockError("shadow mode cannot enable robot motion")
        return ShadowRobotAdapter()
    if mode == "disabled":
        return DisabledHardwareAdapter(str(config.get("reason") or "hardware disabled"))
    raise HardwareInterlockError(
        f"unsupported hardware mode {mode!r}; a mentor-reviewed adapter is required"
    )
