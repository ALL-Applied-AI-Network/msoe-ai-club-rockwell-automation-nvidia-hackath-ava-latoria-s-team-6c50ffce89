"""ML grasp path: thresholds and procedural smoke (skips if torch missing)."""

from __future__ import annotations

import unittest

import numpy as np

from stacking_recovery.data.procedural_wrist import render_sample
from stacking_recovery.vision.grasp_labels import GRASP_CLASSES
from stacking_recovery.vision.grasp_quality_model import (
    GraspQualityPrediction,
    prediction_to_grasp_result,
)
from stacking_recovery.types import FailureReason, GraspCheckResult
from stacking_recovery.vision.grasp_check_factory import build_guarded_ai_check


class TestPredictionToResult(unittest.TestCase):
    def test_table_pair_mode_exposes_before_after_and_delta(self) -> None:
        from stacking_recovery.vision.grasp_quality_model import crop_for_observation

        before = np.full((8, 8, 3), 20, dtype=np.uint8)
        after = before.copy()
        after[2:6, 2:6] = 80
        pair = np.concatenate([before, after], axis=1)

        encoded = crop_for_observation(
            pair,
            {"y0": 0.0, "y1": 1.0, "x0": 0.0, "x1": 1.0},
            "table_before_after",
            "before_after_delta",
        )

        self.assertEqual(encoded.shape, (8, 8, 3))
        np.testing.assert_array_equal(encoded[..., 2], 20)
        np.testing.assert_array_equal(encoded[..., 1], after[..., 0])
        np.testing.assert_array_equal(
            encoded[..., 0], np.abs(after[..., 0].astype(int) - 20).astype(np.uint8)
        )

    def test_binary_pickup_labels_fold_failure_causes(self) -> None:
        from stacking_recovery.vision.grasp_quality_model import normalize_grasp_label

        cfg = {"label_mode": "pickup_binary", "classes": ["GOOD", "SHIFTED"]}
        self.assertEqual(normalize_grasp_label("GOOD", cfg), "GOOD")
        self.assertEqual(normalize_grasp_label("SHIFTED", cfg), "SHIFTED")
        self.assertEqual(normalize_grasp_label("ROTATED", cfg), "SHIFTED")

    def test_good_high_confidence_seated(self) -> None:
        pred = GraspQualityPrediction("GOOD", 0.9, {"GOOD": 0.9})
        r = prediction_to_grasp_result(pred, {"thresholds": {"tau_good": 0.75, "tau_act": 0.55}})
        self.assertTrue(r.seated)
        self.assertEqual(r.predicted_class, "GOOD")
        self.assertEqual(r.verification, "ai")

    def test_low_confidence_not_seated(self) -> None:
        pred = GraspQualityPrediction("EMPTY_GRASP", 0.4, {})
        r = prediction_to_grasp_result(pred, {"thresholds": {"tau_good": 0.75, "tau_act": 0.55}})
        self.assertFalse(r.seated)
        self.assertEqual(r.reason, FailureReason.LOW_CONFIDENCE)


class TestGuardedAiCheck(unittest.TestCase):
    @staticmethod
    def result(
        seated: bool,
        confidence: float,
        reason: FailureReason,
        *,
        predicted_class: str | None = None,
        yaw: float = 0.0,
    ) -> GraspCheckResult:
        return GraspCheckResult(
            seated=seated,
            yaw_error_deg=yaw,
            confidence=confidence,
            reason=reason,
            predicted_class=predicted_class,
            verification="ai" if predicted_class else "rules",
        )

    def test_proceeds_only_when_ai_and_rules_agree(self) -> None:
        ai = self.result(True, 0.92, FailureReason.OK, predicted_class="GOOD")
        rules = self.result(True, 0.81, FailureReason.OK, yaw=2.5)
        check = build_guarded_ai_check(lambda _: ai, lambda _: rules)
        observed = check(np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertTrue(observed.seated)
        self.assertEqual(observed.verification, "ai_guarded")
        self.assertEqual(observed.confidence, 0.81)
        self.assertEqual(observed.yaw_error_deg, 2.5)

    def test_rules_veto_confident_ai_false_positive(self) -> None:
        ai = self.result(True, 0.97, FailureReason.OK, predicted_class="GOOD")
        rules = self.result(False, 0.78, FailureReason.EMPTY_GRASP, yaw=float("nan"))
        check = build_guarded_ai_check(lambda _: ai, lambda _: rules)
        observed = check(np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertFalse(observed.seated)
        self.assertEqual(observed.reason, FailureReason.EMPTY_GRASP)

    def test_ai_rejection_is_not_overridden_by_rules(self) -> None:
        ai = self.result(
            False,
            0.88,
            FailureReason.MISALIGNED_CUBE,
            predicted_class="SHIFTED",
        )
        rules = self.result(True, 0.84, FailureReason.OK, yaw=1.0)
        check = build_guarded_ai_check(lambda _: ai, lambda _: rules)
        observed = check(np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertFalse(observed.seated)
        self.assertEqual(observed.reason, FailureReason.MISALIGNED_CUBE)


class TestProceduralEndToEnd(unittest.TestCase):
    def test_train_and_predict_smoke(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from pathlib import Path
        import tempfile

        from stacking_recovery.vision.grasp_quality_model import (
            build_mobilenet_v3,
            crop_roi,
            load_grasp_config,
            preprocess,
            save_model_checkpoint,
            GraspQualityModel,
        )

        cfg = load_grasp_config()
        roi = cfg["roi"]
        size = int(cfg["input_size"])
        xs, ys = [], []
        for label in GRASP_CLASSES[:3]:
            img = render_sample(label, seed=7)
            crop = crop_roi(img, roi)
            xs.append(preprocess(crop, size))
            ys.append(GRASP_CLASSES.index(label))
        xb = torch.stack(xs)
        yb = torch.tensor(ys)
        model = build_mobilenet_v3(len(GRASP_CLASSES))
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        for _ in range(30):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.pt"
            save_model_checkpoint(model, path, meta={"smoke": True})
            m = GraspQualityModel({**cfg, "model_path": str(path)})
            pred = m.predict(render_sample("GOOD", 99))
            self.assertIn(pred.class_name, GRASP_CLASSES)


if __name__ == "__main__":
    unittest.main()
