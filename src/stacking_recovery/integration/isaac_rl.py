"""Bridge the high-level RL policy to bounded Isaac Sim placement commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stacking_recovery.rl.q_learning import QLearningAgent
from stacking_recovery.rl.stacking_env import (
    Action,
    StackingState,
)


@dataclass(frozen=True)
class IsaacPlacementDecision:
    commanded_xy: tuple[float, float]
    injected_offset_m: tuple[float, float]
    learned_correction_m: tuple[float, float]
    actions: tuple[str, ...]
    safe_stop: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "commanded_xy": list(self.commanded_xy),
            "injected_offset_m": list(self.injected_offset_m),
            "learned_correction_m": list(self.learned_correction_m),
            "actions": list(self.actions),
            "safe_stop": self.safe_stop,
        }


class IsaacRLPlacementAdapter:
    """Convert an Isaac placement perturbation into an RL-corrected command.

    The adapter never produces joint targets.  It only adjusts the XY target
    supplied to the existing IK/trajectory code, which continues to enforce
    its normal workspace and convergence checks.
    """

    def __init__(self, checkpoint: str | Path, *, max_decisions: int = 6) -> None:
        self.agent = QLearningAgent.load(checkpoint, seed=0)
        self.max_decisions = max_decisions

    def decide(
        self,
        *,
        target_xy: tuple[float, float] | list[float],
        injected_offset_m: tuple[float, float] | list[float],
        stack_level: int,
        grasp_quality: int = 2,
        stability: int = 2,
    ) -> IsaacPlacementDecision:
        target = np.asarray(target_xy, dtype=float)
        injected = np.asarray(injected_offset_m, dtype=float)
        if target.shape != (2,) or injected.shape != (2,):
            raise ValueError("target_xy and injected_offset_m must contain two values")
        if not np.all(np.isfinite(target)) or not np.all(np.isfinite(injected)):
            raise ValueError("placement coordinates must be finite")

        state = StackingState(
            stack_level=max(0, int(stack_level)),
            offset_x=_offset_bin(injected[0]),
            offset_y=_offset_bin(injected[1]),
            grasp_quality=int(np.clip(grasp_quality, 0, 2)),
            stability=int(np.clip(stability, 0, 2)),
            retries=0,
        )
        correction = np.zeros(2, dtype=float)
        actions: list[str] = []
        safe_stop = False
        for _ in range(self.max_decisions):
            action = self._masked_action(state)
            actions.append(action.name)
            if action == Action.PLACE_CENTER:
                break
            if action == Action.SAFE_STOP:
                safe_stop = True
                break
            if action == Action.RETRY_GRASP:
                state = StackingState(
                    state.stack_level,
                    state.offset_x,
                    state.offset_y,
                    2,
                    state.stability,
                    state.retries + 1,
                )
                continue

            dx, dy = {
                Action.CORRECT_LEFT: (-1, 0),
                Action.CORRECT_RIGHT: (1, 0),
                Action.CORRECT_FORWARD: (0, -1),
                Action.CORRECT_BACKWARD: (0, 1),
            }[action]
            if dx and state.offset_x:
                correction[0] -= injected[0]
            if dy and state.offset_y:
                correction[1] -= injected[1]
            state = StackingState(
                state.stack_level,
                int(np.clip(state.offset_x + dx, -1, 1)),
                int(np.clip(state.offset_y + dy, -1, 1)),
                state.grasp_quality,
                state.stability,
                state.retries,
            )
        else:
            safe_stop = True

        command = target + injected + correction
        return IsaacPlacementDecision(
            commanded_xy=(float(command[0]), float(command[1])),
            injected_offset_m=(float(injected[0]), float(injected[1])),
            learned_correction_m=(float(correction[0]), float(correction[1])),
            actions=tuple(actions),
            safe_stop=safe_stop,
        )

    def _masked_action(self, state: StackingState) -> Action:
        """Choose among physically meaningful actions for this observation.

        Masking prevents an RL-table tie from commanding a correction away
        from an already-centered axis.  The learner still chooses whether and
        in what order to correct, retry, place, or stop.
        """
        # Never let a high Q-value turn a still-observed placement error into
        # a release command.  RL may choose the correction order, but the
        # existing safety layer only exposes PLACE_CENTER after both axes are
        # inside the calibrated deadband.
        allowed = (
            [Action.PLACE_CENTER]
            if state.offset_x == 0 and state.offset_y == 0
            else []
        )
        if state.offset_x < 0:
            allowed.append(Action.CORRECT_RIGHT)
        elif state.offset_x > 0:
            allowed.append(Action.CORRECT_LEFT)
        if state.offset_y < 0:
            allowed.append(Action.CORRECT_BACKWARD)
        elif state.offset_y > 0:
            allowed.append(Action.CORRECT_FORWARD)
        if state.grasp_quality < 2 and state.retries < 2:
            allowed.append(Action.RETRY_GRASP)
        if state.grasp_quality == 0 and state.retries >= 2:
            allowed.append(Action.SAFE_STOP)
        values = self.agent._values(state)  # table lookup; no exploration in Isaac
        return max(allowed, key=lambda action: float(values[int(action)]))


def _offset_bin(value: float, deadband_m: float = 0.0015) -> int:
    if value < -deadband_m:
        return -1
    if value > deadband_m:
        return 1
    return 0
