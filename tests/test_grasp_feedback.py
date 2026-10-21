"""Feedback recorder and physics weak labels."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from stacking_recovery.data.grasp_dataset_store import load_manifest
from stacking_recovery.data.grasp_feedback import GraspFeedbackRecorder
from stacking_recovery.data.isaac_grasp_labels import (
    label_from_physics_event,
    should_add_feedback_sample,
)
from stacking_recovery.data.grasp_dataset_store import GraspDatasetStore
from stacking_recovery.types import FailureReason, GraspCheckResult


class TestPhysicsLabels(unittest.TestCase):
    def test_pickup_good(self) -> None:
        self.assertEqual(label_from_physics_event(pickup_verified=True), "GOOD")

    def test_empty_jaw(self) -> None:
        self.assertEqual(
            label_from_physics_event(pickup_verified=False, jaw_gap_mm=1.0),
            "EMPTY_GRASP",
        )

    def test_absolute_cube_yaw_is_not_a_grasp_error(self) -> None:
        self.assertEqual(
            label_from_physics_event(
                pickup_verified=False,
                perceived_yaw_deg=32.0,
                jaw_gap_mm=5.0,
            ),
            "SHIFTED",
        )

    def test_injected_yaw_error_labels_rotation(self) -> None:
        self.assertEqual(
            label_from_physics_event(
                pickup_verified=False,
                perceived_yaw_deg=32.0,
                yaw_error_deg=14.0,
                jaw_gap_mm=5.0,
            ),
            "ROTATED",
        )


class TestFeedbackRecorder(unittest.TestCase):
    def test_mismatch_saved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rec = GraspFeedbackRecorder(
                GraspDatasetStore(root),
                mode="feedback",
            )
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
            bad = GraspCheckResult(
                seated=True,
                yaw_error_deg=0.0,
                confidence=0.9,
                reason=FailureReason.OK,
                predicted_class="GOOD",
                verification="ai",
            )
            rec.on_dwell(img, bad, detection_id=1)
            path = rec.on_lift(
                {
                    "pickup_verified": False,
                    "jaw_gap_mm": 1.0,
                    "closed_loop_grasp": bad.to_dict(),
                    "detection_id": 1,
                }
            )
            self.assertIsNotNone(path)
            rows = load_manifest(root / "manifest.json")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source"], "feedback")
            self.assertEqual(rows[0]["label"], "EMPTY_GRASP")

    def test_should_add_on_bad_grasp(self) -> None:
        ok, label = should_add_feedback_sample(
            physics_label="GOOD",
            grasp_result=GraspCheckResult(
                seated=False,
                yaw_error_deg=float("nan"),
                confidence=0.4,
                reason=FailureReason.EMPTY_GRASP,
            ),
            pickup_verified=False,
        )
        self.assertTrue(ok)
        self.assertEqual(label, "EMPTY_GRASP")


if __name__ == "__main__":
    unittest.main()
