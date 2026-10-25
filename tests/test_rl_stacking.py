import tempfile
import unittest
from pathlib import Path

from stacking_recovery.rl.q_learning import QLearningAgent, evaluate_policy
from stacking_recovery.rl.stacking_env import (
    Action,
    SafeRLStackingPolicy,
    StackingDecisionEnv,
    StackingState,
)


class TestRLStacking(unittest.TestCase):
    def test_environment_is_seeded_and_bounded(self) -> None:
        left = StackingDecisionEnv(seed=7)
        right = StackingDecisionEnv(seed=7)
        self.assertEqual(left.reset(), right.reset())

        state = left.state
        for _ in range(left.max_steps + 1):
            result = left.step(
                Action.RETRY_GRASP if state.retries < 2 else Action.SAFE_STOP
            )
            state = result.state
            if result.terminated:
                break
        self.assertTrue(result.terminated)

    def test_safety_policy_overrides_unsafe_proposals(self) -> None:
        class AlwaysPlace:
            def act(self, state, explore=False):
                return Action.PLACE_CENTER

        policy = SafeRLStackingPolicy(AlwaysPlace())
        poor_grasp = StackingState(0, 0, 0, 0, 2, 0)
        exhausted = StackingState(1, 0, 0, 0, 1, 2)
        self.assertEqual(policy.act(poor_grasp), Action.RETRY_GRASP)
        self.assertEqual(policy.act(exhausted), Action.SAFE_STOP)

    def test_q_table_round_trip(self) -> None:
        env = StackingDecisionEnv(seed=4)
        agent = QLearningAgent(seed=4)
        agent.train(env, 30)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.json"
            agent.save(path)
            loaded = QLearningAgent.load(path, seed=4)
        self.assertEqual(loaded.q.keys(), agent.q.keys())

    def test_training_produces_working_policy(self) -> None:
        agent = QLearningAgent(seed=12)
        agent.train(StackingDecisionEnv(seed=12), 8_000)
        metrics = evaluate_policy(
            StackingDecisionEnv(seed=13),
            SafeRLStackingPolicy(agent),
            300,
            seeds=range(20_000, 20_300),
        )
        self.assertGreaterEqual(metrics["success_rate"], 0.55)
