"""Small domain-randomized stacking task for high-level RL experiments.

This environment intentionally does not send robot joint commands.  The agent
only selects a bounded recovery/placement decision.  Isaac Sim (or the simple
stochastic model below) owns physics, and a separate safety layer can reject
unsafe decisions before they reach any motion planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Action(IntEnum):
    PLACE_CENTER = 0
    CORRECT_LEFT = 1
    CORRECT_RIGHT = 2
    CORRECT_FORWARD = 3
    CORRECT_BACKWARD = 4
    RETRY_GRASP = 5
    SAFE_STOP = 6


@dataclass(frozen=True)
class StackingState:
    """Compact observation available to the decision policy.

    Offsets use -1/0/1 bins.  Grasp quality and stability use 0/1/2 for
    poor/uncertain/good.  This keeps the first experiment interpretable and
    small enough to train during the hackathon.
    """

    stack_level: int
    offset_x: int
    offset_y: int
    grasp_quality: int
    stability: int
    retries: int

    def key(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.stack_level,
            self.offset_x,
            self.offset_y,
            self.grasp_quality,
            self.stability,
            self.retries,
        )


@dataclass(frozen=True)
class StepResult:
    state: StackingState
    reward: float
    terminated: bool
    info: dict[str, object]


class StackingDecisionEnv:
    """Domain-randomized, high-level cube stacking environment.

    It is a fast training stand-in for Isaac Sim.  The same state/action
    contract can be driven by real Isaac observations later; only ``step``
    needs to be replaced by the simulator adapter.
    """

    ACTIONS = tuple(Action)

    def __init__(
        self,
        *,
        cubes: int = 3,
        max_retries: int = 2,
        max_steps: int = 18,
        seed: int | None = None,
    ) -> None:
        if cubes < 1:
            raise ValueError("cubes must be at least 1")
        self.cubes = cubes
        self.max_retries = max_retries
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.state = StackingState(0, 0, 0, 2, 2, 0)
        self.steps = 0

    def reset(self, *, seed: int | None = None) -> StackingState:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.state = self._new_cube_state(stack_level=0, retries=0)
        return self.state

    def step(self, action: Action | int) -> StepResult:
        action = Action(action)
        current = self.state
        self.steps += 1

        if action == Action.SAFE_STOP:
            risky = current.grasp_quality == 0 or current.retries >= self.max_retries
            reward = -3.0 if risky else -35.0
            return StepResult(current, reward, True, {"outcome": "safe_stop"})

        if action == Action.RETRY_GRASP:
            if current.retries >= self.max_retries:
                return StepResult(current, -20.0, True, {"outcome": "retry_limit"})
            quality = int(self.rng.choice([0, 1, 2], p=[0.08, 0.22, 0.70]))
            self.state = StackingState(
                current.stack_level,
                current.offset_x,
                current.offset_y,
                quality,
                current.stability,
                current.retries + 1,
            )
            return self._with_timeout(-1.0, {"outcome": "regrasp"})

        if action != Action.PLACE_CENTER:
            correction_x, correction_y = self._correction(action)
            old_error = abs(current.offset_x) + abs(current.offset_y)
            corrected_x = int(np.clip(current.offset_x + correction_x, -1, 1))
            corrected_y = int(np.clip(current.offset_y + correction_y, -1, 1))
            new_error = abs(corrected_x) + abs(corrected_y)
            self.state = StackingState(
                current.stack_level,
                corrected_x,
                corrected_y,
                current.grasp_quality,
                current.stability,
                current.retries,
            )
            reward = 1.5 if new_error < old_error else -1.5
            return self._with_timeout(reward, {"outcome": "placement_corrected"})

        residual_x = current.offset_x
        residual_y = current.offset_y
        aligned = residual_x == 0 and residual_y == 0

        # Poor grasps should be retried, not placed.  This large penalty makes
        # the learned behavior easy to explain in a demo.
        if current.grasp_quality == 0:
            return StepResult(current, -24.0, True, {"outcome": "dropped_cube"})

        success_p = 0.90 if aligned else 0.24
        success_p += 0.04 * (current.grasp_quality - 1)
        success_p += 0.03 * (current.stability - 1)
        success_p -= 0.05 * current.stack_level
        success_p = float(np.clip(success_p, 0.05, 0.97))

        if self.rng.random() > success_p:
            # A failed placement may still be recoverable when the grasp and
            # stack are stable.  Otherwise terminate as a simulated collapse.
            recoverable = current.stability > 0 and self.rng.random() < 0.55
            if recoverable and current.retries < self.max_retries:
                self.state = StackingState(
                    current.stack_level,
                    int(np.clip(residual_x, -1, 1)),
                    int(np.clip(residual_y, -1, 1)),
                    max(0, current.grasp_quality - 1),
                    max(0, current.stability - 1),
                    current.retries + 1,
                )
                return self._with_timeout(-5.0, {"outcome": "placement_retry"})
            return StepResult(current, -25.0, True, {"outcome": "stack_collapse"})

        next_level = current.stack_level + 1
        reward = 12.0 + 3.0 * current.stack_level
        if next_level >= self.cubes:
            return StepResult(current, reward + 35.0, True, {"outcome": "stack_complete"})

        self.state = self._new_cube_state(stack_level=next_level, retries=0)
        return self._with_timeout(reward, {"outcome": "cube_placed"})

    def _new_cube_state(self, *, stack_level: int, retries: int) -> StackingState:
        offset_choices = np.array([-1, 0, 1])
        return StackingState(
            stack_level=stack_level,
            offset_x=int(self.rng.choice(offset_choices, p=[0.25, 0.50, 0.25])),
            offset_y=int(self.rng.choice(offset_choices, p=[0.25, 0.50, 0.25])),
            grasp_quality=int(self.rng.choice([0, 1, 2], p=[0.10, 0.20, 0.70])),
            stability=int(self.rng.choice([0, 1, 2], p=[0.08, 0.22, 0.70])),
            retries=retries,
        )

    def _with_timeout(self, reward: float, info: dict[str, object]) -> StepResult:
        if self.steps >= self.max_steps:
            return StepResult(self.state, reward - 10.0, True, {"outcome": "timeout"})
        return StepResult(self.state, reward, False, info)

    @staticmethod
    def _correction(action: Action) -> tuple[int, int]:
        return {
            Action.PLACE_CENTER: (0, 0),
            Action.CORRECT_LEFT: (-1, 0),
            Action.CORRECT_RIGHT: (1, 0),
            Action.CORRECT_FORWARD: (0, -1),
            Action.CORRECT_BACKWARD: (0, 1),
        }.get(action, (0, 0))


class SafeRLStackingPolicy:
    """Safety shield around an RL agent's proposed action."""

    def __init__(self, agent: object, *, max_retries: int = 2) -> None:
        self.agent = agent
        self.max_retries = max_retries

    def act(self, state: StackingState) -> Action:
        if state.retries >= self.max_retries and state.grasp_quality < 2:
            return Action.SAFE_STOP
        if state.grasp_quality == 0:
            return Action.RETRY_GRASP
        proposed = Action(self.agent.act(state, explore=False))  # type: ignore[attr-defined]
        if proposed == Action.RETRY_GRASP and state.retries >= self.max_retries:
            return Action.SAFE_STOP
        return proposed
