import unittest
import tempfile
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is optional")
class TestPairModels(unittest.TestCase):
    def test_siamese_forward_shape(self):
        from stacking_recovery.vision.pair_models import ModelSpec, build_pair_model

        model = build_pair_model(
            ModelSpec("mobilenet_v3_small", "siamese", pretrained=False), 2
        )
        output = model(torch.zeros(2, 6, 64, 64))
        self.assertEqual(tuple(output.shape), (2, 2))

    def test_siamese_rejects_non_pair_tensor(self):
        from stacking_recovery.vision.pair_models import ModelSpec, build_pair_model

        model = build_pair_model(
            ModelSpec("mobilenet_v3_small", "siamese", pretrained=False), 2
        )
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 3, 64, 64))

    def test_runtime_loads_siamese_checkpoint(self):
        from stacking_recovery.vision.grasp_quality_model import GraspQualityModel
        from stacking_recovery.vision.pair_models import ModelSpec, build_pair_model

        model = build_pair_model(
            ModelSpec("resnet18", "siamese", pretrained=False), 2
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save(model.state_dict(), checkpoint)
            runtime = GraspQualityModel(
                {
                    "model_path": str(checkpoint),
                    "backbone": "resnet18",
                    "table_pair_mode": "siamese_rgb_pair",
                    "input_size": 32,
                    "classes": ["GOOD", "SHIFTED"],
                    "roi": {"x0": 0.0, "x1": 1.0, "y0": 0.0, "y1": 1.0},
                }
            )
            pair = np.zeros((32, 64, 3), dtype=np.uint8)
            prediction = runtime.predict(
                pair, observation_source="table_before_after"
            )
            self.assertIn(prediction.class_name, {"GOOD", "SHIFTED"})


if __name__ == "__main__":
    unittest.main()
