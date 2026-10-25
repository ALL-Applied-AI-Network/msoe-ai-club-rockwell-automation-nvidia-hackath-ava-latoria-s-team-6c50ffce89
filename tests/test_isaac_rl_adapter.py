import tempfile
import unittest
from pathlib import Path

from stacking_recovery.integration.isaac_rl import IsaacRLPlacementAdapter
from stacking_recovery.rl.q_learning import QLearningAgent
from stacking_recovery.rl.stacking_env import Action, StackingState


class TestIsaacRLPlacementAdapter(unittest.TestCase):
    def _checkpoint(self, root: Path, action: Action) -> Path:
        agent = QLearningAgent(seed=0)
        # Populate every state the adapter can traverse with the desired action.
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                state = StackingState(0, x, y, 2, 2, 0)
                agent._values(state)[int(action)] = 10.0
        path = root / "policy.json"
        agent.save(path)
        return path

    def test_zero_offset_stays_at_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = IsaacRLPlacementAdapter(
                self._checkpoint(Path(tmp), Action.PLACE_CENTER)
            )
            decision = adapter.decide(
                target_xy=[0.4, 0.2],
                injected_offset_m=[0.0, 0.0],
                stack_level=0,
            )
        self.assertEqual(decision.commanded_xy, (0.4, 0.2))
        self.assertFalse(decision.safe_stop)

    def test_learned_correction_is_bounded_target_adjustment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = QLearningAgent(seed=0)
            shifted = StackingState(0, 1, 0, 2, 2, 0)
            centered = StackingState(0, 0, 0, 2, 2, 0)
            agent._values(shifted)[int(Action.CORRECT_LEFT)] = 10.0
            agent._values(centered)[int(Action.PLACE_CENTER)] = 10.0
            path = Path(tmp) / "policy.json"
            agent.save(path)
            decision = IsaacRLPlacementAdapter(path).decide(
                target_xy=[0.4, 0.2],
                injected_offset_m=[0.008, 0.0],
                stack_level=0,
            )
        self.assertAlmostEqual(decision.commanded_xy[0], 0.4)
        self.assertEqual(decision.actions, ("CORRECT_LEFT", "PLACE_CENTER"))
        self.assertEqual(decision.learned_correction_m, (-0.008, 0.0))


if __name__ == "__main__":
    unittest.main()
