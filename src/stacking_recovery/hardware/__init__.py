"""Robot-neutral, fail-closed hardware interfaces."""

from stacking_recovery.hardware.adapter import (
    CommandRecord,
    DisabledHardwareAdapter,
    HardwareInterlockError,
    PoseTarget,
    RobotAdapter,
    RobotStatus,
    ShadowRobotAdapter,
    load_hardware_adapter,
)
from stacking_recovery.hardware.shadow_plan import execute_shadow_plan

__all__ = [
    "CommandRecord",
    "DisabledHardwareAdapter",
    "HardwareInterlockError",
    "PoseTarget",
    "RobotAdapter",
    "RobotStatus",
    "ShadowRobotAdapter",
    "load_hardware_adapter",
    "execute_shadow_plan",
]
