import tempfile
import unittest
from pathlib import Path

import numpy as np

from stacking_recovery.control.recovery_policy import (
    ManipulationContext,
    RecoveryDecision,
    RecoveryPolicy,
)
from stacking_recovery.types import FailureReason, GraspCheckResult, PlaceCheckResult


class TestRecoveryPolicy(unittest.TestCase):
    def test_proceed_when_seated(self):
        policy = RecoveryPolicy()
        ctx = ManipulationContext("yellow", 0, 0.95)
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        decision, _ = policy.after_grasp_close(ctx, img)
        self.assertEqual(decision, RecoveryDecision.PROCEED)

    def test_empty_grasp_reacquires_then_stops(self):
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
                policy.after_grasp_close(ctx, img)[0], RecoveryDecision.REACQUIRE
            )
            ctx.grasp_attempt_index = 2
            self.assertEqual(
                policy.after_grasp_close(ctx, img)[0], RecoveryDecision.SAFE_STOP
            )
            self.assertGreaterEqual(len(log.read_text().strip().splitlines()), 2)

    def test_uncertain_gets_one_second_look_then_regrasp(self):
        def uncertain(_img):
            return GraspCheckResult(
                seated=False,
                yaw_error_deg=float("nan"),
                confidence=0.52,
                reason=FailureReason.LOW_CONFIDENCE,
                predicted_class="UNCERTAIN",
                verification="ai",
            )

        policy = RecoveryPolicy(grasp_check_fn=uncertain)
        ctx = ManipulationContext("blue", 2, 0.88)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        self.assertEqual(
            policy.after_grasp_close(ctx, img)[0], RecoveryDecision.SECOND_LOOK
        )
        ctx.uncertain_look_index = 1
        self.assertEqual(
            policy.after_grasp_close(ctx, img)[0], RecoveryDecision.REGRASP
        )

    def test_misaligned_grasp_regrasps(self):
        def misaligned(_img):
            return GraspCheckResult(
                seated=False,
                yaw_error_deg=12.0,
                confidence=0.9,
                reason=FailureReason.MISALIGNED_CUBE,
                predicted_class="MISALIGNED",
                verification="ai",
            )

        policy = RecoveryPolicy(grasp_check_fn=misaligned)
        ctx = ManipulationContext("red", 1, 0.9)
        decision, _ = policy.after_grasp_close(
            ctx, np.zeros((10, 10, 3), dtype=np.uint8)
        )
        self.assertEqual(decision, RecoveryDecision.REGRASP)

    def test_placement_regrasp_then_stop(self):
        def bad_place(_img, _level):
            return PlaceCheckResult(
                placed=False,
                confidence=0.9,
                reason=FailureReason.PLACEMENT_FAILURE,
            )

        policy = RecoveryPolicy(place_check_fn=bad_place)
        ctx = ManipulationContext("red", 5, 0.92, placement_attempt_index=1)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        decision, _ = policy.after_place(ctx, img)
        self.assertEqual(decision, RecoveryDecision.SAFE_STOP)

    def test_placement_uses_verifier_calibrated_confidence(self):
        def calibrated_good_place(_img, _level):
            return PlaceCheckResult(
                placed=True,
                confidence=0.55,
                reason=FailureReason.OK,
            )

        policy = RecoveryPolicy(place_check_fn=calibrated_good_place)
        ctx = ManipulationContext("red", 4, 0.92)
        decision, result = policy.after_place(
            ctx,
            np.zeros((10, 10, 3), dtype=np.uint8),
        )

        self.assertTrue(result.placed)
        self.assertEqual(decision, RecoveryDecision.PROCEED)


if __name__ == "__main__":
    unittest.main()
