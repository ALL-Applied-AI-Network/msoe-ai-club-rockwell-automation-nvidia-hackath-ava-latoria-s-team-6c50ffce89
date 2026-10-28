"""Dependency-light tabular Q-learning for the stacking decision task."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from stacking_recovery.rl.stacking_env import Action, StackingDecisionEnv, StackingState


@dataclass(frozen=True)
class TrainingSummary:
    episodes: int
    mean_reward_last_100: float
    success_rate_last_100: float
    visited_states: int


class QLearningAgent:
    def __init__(
        self,
        *,
        learning_rate: float = 0.18,
        discount: float = 0.96,
        epsilon: float = 1.0,
        epsilon_min: float = 0.03,
        epsilon_decay: float = 0.9992,
        seed: int | None = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.q: dict[tuple[int, ...], np.ndarray] = {}

    def _values(self, state: StackingState) -> np.ndarray:
        return self.q.setdefault(state.key(), np.zeros(len(Action), dtype=np.float64))

    def act(self, state: StackingState, *, explore: bool = True) -> Action:
        if explore and self.rng.random() < self.epsilon:
            return Action(int(self.rng.integers(0, len(Action))))
        values = self._values(state)
        best = np.flatnonzero(values == values.max())
        return Action(int(self.rng.choice(best)))

    def update(
        self,
        state: StackingState,
        action: Action,
        reward: float,
        next_state: StackingState,
        terminated: bool,
    ) -> None:
        values = self._values(state)
        target = reward
        if not terminated:
            target += self.discount * float(self._values(next_state).max())
        values[int(action)] += self.learning_rate * (target - values[int(action)])

    def train(self, env: StackingDecisionEnv, episodes: int) -> TrainingSummary:
        if episodes < 1:
            raise ValueError("episodes must be at least 1")
        recent_rewards: list[float] = []
        recent_success: list[float] = []
        for _ in range(episodes):
            state = env.reset()
            episode_reward = 0.0
            success = False
            while True:
                action = self.act(state)
                result = env.step(action)
                self.update(state, action, result.reward, result.state, result.terminated)
                state = result.state
                episode_reward += result.reward
                if result.terminated:
                    success = result.info.get("outcome") == "stack_complete"
                    break
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            recent_rewards.append(episode_reward)
            recent_success.append(float(success))
            recent_rewards = recent_rewards[-100:]
            recent_success = recent_success[-100:]
        return TrainingSummary(
            episodes=episodes,
            mean_reward_last_100=float(np.mean(recent_rewards)),
            success_rate_last_100=float(np.mean(recent_success)),
            visited_states=len(self.q),
        )

    def save(self, path: str | Path, *, metadata: dict[str, object] | None = None) -> None:
        payload = {
            "format": "stacking_recovery.q_table.v1",
            "agent": {
                "learning_rate": self.learning_rate,
                "discount": self.discount,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
            },
            "metadata": metadata or {},
            "q_table": {",".join(map(str, key)): values.tolist() for key, values in self.q.items()},
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, *, seed: int | None = None) -> "QLearningAgent":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "stacking_recovery.q_table.v1":
            raise ValueError("unsupported RL checkpoint format")
        agent = cls(seed=seed, **payload["agent"])
        agent.q = {
            tuple(int(part) for part in key.split(",")): np.asarray(values, dtype=np.float64)
            for key, values in payload["q_table"].items()
        }
        return agent


def evaluate_policy(
    env: StackingDecisionEnv,
    policy: object,
    episodes: int,
    *,
    seeds: Iterable[int] | None = None,
) -> dict[str, float]:
    rewards: list[float] = []
    successes = 0
    outcomes: dict[str, int] = {}
    seed_iter = iter(seeds) if seeds is not None else None
    for index in range(episodes):
        state = env.reset(seed=next(seed_iter) if seed_iter is not None else None)
        total = 0.0
        while True:
            action = Action(policy.act(state))  # type: ignore[attr-defined]
            result = env.step(action)
            total += result.reward
            state = result.state
            if result.terminated:
                outcome = str(result.info.get("outcome"))
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
                successes += int(outcome == "stack_complete")
                break
        rewards.append(total)
    return {
        "episodes": float(episodes),
        "mean_reward": float(np.mean(rewards)),
        "success_rate": successes / episodes,
        "safe_stop_rate": outcomes.get("safe_stop", 0) / episodes,
        "collapse_rate": outcomes.get("stack_collapse", 0) / episodes,
        "drop_rate": outcomes.get("dropped_cube", 0) / episodes,
    }
