import tempfile
import unittest
from pathlib import Path

import numpy as np

from stacking_recovery.control.recovery_policy import (
    ManipulationContext,
    RecoveryDecision,
    RecoveryPolicy,
    choose_action,
)
from stacking_recovery.types import (
    FailureReason, GraspCheckResult, PlaceCheckResult)


class TestRecoveryPolicy(unittest.TestCase):
    def test_stop_when_recovery_decision_isnt_proceed(self):
        policy = RecoveryPolicy()
        ctx = ManipulationContext("yellow", 0, 0.95)
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        decision, result = policy.after_grasp_close(ctx, img)
        self.assertEqual(decision, RecoveryDecision.SAFE_STOP)

    #this tests the same thing as above but sets the failure reason to
    #ok which is what passes the choose action behavior outline
    def test_proceed_when_successful_grasp_seated(self):
        def successful_grasp(_img):
            return GraspCheckResult(
                seated=True,
                yaw_error_deg=0.0,
                confidence=0.95,
                reason=FailureReason.OK,
            )

        policy = RecoveryPolicy(grasp_check_fn=successful_grasp)

        ctx = ManipulationContext("yellow", 0, 0.95)
        img = np.zeros((720, 1280, 3), dtype=np.uint8)

        decision, result = policy.after_grasp_close(ctx, img)

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

    #Tests for the different failure reasons
    def test_occlusion(self):
        decision = choose_action(
            ok=False,
            confidence=0.9,
            reason=FailureReason.OCCLUSION,
            attempt=0,
            max_retries=2,
        )
        self.assertEqual(decision, RecoveryDecision.REACQUIRE)

    def test_low_confidence(self):
        decision = choose_action(
            ok=False,
            confidence=0.3,
            reason=FailureReason.LOW_CONFIDENCE,
            attempt=0,
            max_retries=2,
        )
        self.assertEqual(decision, RecoveryDecision.RETRY)

    def test_empty_grasp(self):
        decision = choose_action(
            ok=False,
            confidence=0.9,
            reason=FailureReason.EMPTY_GRASP,
            attempt=0,
            max_retries=2,
        )
        self.assertEqual(decision, RecoveryDecision.REGRASP)

    def test_misaligned_cube(self):
        decision = choose_action(
            ok=False,
            confidence=0.9,
            reason=FailureReason.MISALIGNED_CUBE,
            attempt=0,
            max_retries=2,
        )
        self.assertEqual(decision, RecoveryDecision.REGRASP)

    def test_tilted_object(self):
        decision = choose_action(
            ok=False,
            confidence=0.9,
            reason=FailureReason.TILTED_OBJECT,
            attempt=0,
            max_retries=2,
        )
        self.assertEqual(decision, RecoveryDecision.REGRASP)

    def test_placement_failure(self):
        decision = choose_action(
            ok=False,
            confidence=0.9,
            reason=FailureReason.PLACEMENT_FAILURE,
            attempt=0,
            max_retries=1,
        )
        self.assertEqual(decision, RecoveryDecision.RETRY)

    def test_unknown_failure(self):
        decision = choose_action(
            ok=False,
            confidence=0.9,
            reason=FailureReason.UNKNOWN,
            attempt=0,
            max_retries=2,
        )
        self.assertEqual(decision, RecoveryDecision.SAFE_STOP)

    def test_stub_result(self):
        decision = choose_action(
            ok=False,
            confidence=0.9,
            reason=FailureReason.STUB,
            attempt=0,
            max_retries=2,
        )
        self.assertEqual(decision, RecoveryDecision.SAFE_STOP)


    # Tests for retry exhaustion and invalid confidence
    def test_invalid_confidence(self):
        decision = choose_action(
            ok=False,
            confidence=float("nan"),
            reason=FailureReason.EMPTY_GRASP,
            attempt=0,
            max_retries=2,
        )
        self.assertEqual(decision, RecoveryDecision.SAFE_STOP)

    def test_grasp_retry_exhaustion(self):
        decision = choose_action(
            ok=False,
            confidence=0.9,
            reason=FailureReason.EMPTY_GRASP,
            attempt=2,
            max_retries=2,
        )
        self.assertEqual(decision, RecoveryDecision.SAFE_STOP)

    def test_placement_retry_exhaustion(self):
        decision = choose_action(
            ok=False,
            confidence=0.9,
            reason=FailureReason.PLACEMENT_FAILURE,
            attempt=1,
            max_retries=1,
        )
        self.assertEqual(decision, RecoveryDecision.SAFE_STOP)

    #Tests for vision function errors

    def test_grasp_invalid_result(self):
        def invalid_grasp(_img):
            return None

        policy = RecoveryPolicy(grasp_check_fn=invalid_grasp)
        ctx = ManipulationContext("blue", 0, 0.95)
        img = np.zeros((10, 10, 3), dtype=np.uint8)

        decision, result = policy.after_grasp_close(ctx, img)

        self.assertEqual(decision, RecoveryDecision.SAFE_STOP)
        self.assertEqual(result.reason, FailureReason.UNKNOWN)

    def test_place_vision_exception(self):
        def failing_place(_img, _level):
            raise ValueError("Invalid camera image")

        policy = RecoveryPolicy(place_check_fn=failing_place)
        ctx = ManipulationContext("yellow", 1, 0.95)
        img = np.zeros((10, 10, 3), dtype=np.uint8)

        decision, result = policy.after_place(ctx, img)

        self.assertEqual(decision, RecoveryDecision.SAFE_STOP)
        self.assertEqual(result.reason, FailureReason.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
