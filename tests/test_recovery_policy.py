import tempfile
import unittest
from pathlib import Path

import numpy as np

from stacking_recovery.control.recovery_policy import (
    ManipulationContext,
    RecoveryDecision,
    RecoveryPolicy,
)
from stacking_recovery.types import FailureReason, GraspCheckResult


class TestRecoveryPolicy(unittest.TestCase):
    def test_proceed_when_seated(self):
        policy = RecoveryPolicy()
        ctx = ManipulationContext("yellow", 0, 0.95)
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        decision, _ = policy.after_grasp_close(ctx, img)
        self.assertEqual(decision, RecoveryDecision.PROCEED)

    def test_regrasp_then_stop(self):
        def always_empty(_img):
            return GraspCheckResult(
                seated=False,
                yaw_error_deg=12.0,
                confidence=0.9,
                reason=FailureReason.EMPTY_GRASP,
            )

        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "manipulation.jsonl"
            policy = RecoveryPolicy(log_path=log, grasp_check_fn=always_empty)
            ctx = ManipulationContext("blue", 2, 0.88, grasp_attempt_index=0)
            img = np.zeros((10, 10, 3), dtype=np.uint8)
            self.assertEqual(
                policy.after_grasp_close(ctx, img)[0], RecoveryDecision.REGRASP
            )
            ctx.grasp_attempt_index = 2
            self.assertEqual(
                policy.after_grasp_close(ctx, img)[0], RecoveryDecision.SAFE_STOP
            )
            self.assertGreaterEqual(len(log.read_text().strip().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
