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
from stacking_recovery.types import FailureReason


class TestPredictionToResult(unittest.TestCase):
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
