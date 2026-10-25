#!/usr/bin/env python3
"""Train and evaluate the simulation-only stacking decision policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stacking_recovery.rl.q_learning import QLearningAgent, evaluate_policy  # noqa: E402
from stacking_recovery.rl.stacking_env import (  # noqa: E402
    Action,
    SafeRLStackingPolicy,
    StackingDecisionEnv,
)


class OpenLoopBaseline:
    """Current-style baseline: retry an empty grasp, otherwise place as-is."""

    def act(self, state):
        if state.grasp_quality == 0:
            return Action.RETRY_GRASP if state.retries < 2 else Action.SAFE_STOP
        return Action.PLACE_CENTER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("models/rl_stacking_qtable.json"))
    parser.add_argument("--metrics", type=Path, default=Path("results/rl_stacking_metrics.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = StackingDecisionEnv(seed=args.seed)
    agent = QLearningAgent(seed=args.seed)
    summary = agent.train(env, args.episodes)

    safe_policy = SafeRLStackingPolicy(agent)
    evaluation = evaluate_policy(
        StackingDecisionEnv(seed=args.seed + 1),
        safe_policy,
        args.eval_episodes,
        seeds=range(100_000, 100_000 + args.eval_episodes),
    )
    baseline = evaluate_policy(
        StackingDecisionEnv(seed=args.seed + 1),
        OpenLoopBaseline(),
        args.eval_episodes,
        seeds=range(100_000, 100_000 + args.eval_episodes),
    )
    metrics = {
        "training": summary.__dict__,
        "rl_evaluation": evaluation,
        "open_loop_baseline": baseline,
        "success_rate_lift": evaluation["success_rate"] - baseline["success_rate"],
        "scope": "simulation-only high-level decisions; no direct joint control",
        "seed": args.seed,
    }
    agent.save(args.out, metadata=metrics)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"checkpoint: {args.out}")
    print(f"metrics: {args.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
