"""Check that focused Cosmos replies preserve coordinates across colour groups."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import json
import cv2
import numpy as np
from perceive import audit_stack_colours, filter_stack_cube_poses, parse_json


class CubePerceptionTests(unittest.TestCase):
    def test_mixed_coordinate_replies_are_normalized_before_merging(self):
        parsed = {
            "objects": [
                dict(
                    category="cube",
                    colour=colour,
                    center=[x, 500],
                    bbox=[x - 5, 490, x + 5, 510],
                )
                for colour, x in [
                    ("blue", 300),
                    ("blue", 400),
                    ("red", 600),
                    ("red", 700),
                ]
            ]
        }
        reply = {
            "objects": [
                dict(
                    category="cube",
                    colour="yellow",
                    center=[x, 0.8],
                    bbox=[x - 0.01, 0.78, x + 0.01, 0.82],
                )
                for x in (0.2, 0.5)
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img = root / "camera.png"
            cv2.imwrite(str(img), np.zeros((1080, 1920, 3), np.uint8))
            with patch("perceive.ask", return_value=json.dumps(reply)) as ask:
                result = audit_stack_colours(None, None, img, parsed, root)
            self.assertEqual(ask.call_count, 1)
        self.assertEqual(len(result["objects"]), 6)
        self.assertEqual(result["coordinate_convention"], "unit")
        self.assertAlmostEqual(result["objects"][0]["center"][0], 0.3)
        self.assertAlmostEqual(result["objects"][-1]["center"][0], 0.5)
        self.assertTrue(all(0 <= v <= 1 for o in result["objects"] for v in o["bbox"]))

    def test_empty_colour_reply_does_not_fabricate_cubes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img = root / "camera.png"
            cv2.imwrite(str(img), np.zeros((100, 100, 3), np.uint8))
            with patch("perceive.ask", return_value='{"objects":[]}'):
                result = audit_stack_colours(None, None, img, {"objects": []}, root)
        self.assertEqual(result["objects"], [])

    def test_model_can_return_bare_detection_array(self):
        self.assertEqual(
            parse_json('```json\n[{"category":"cube"}]\n```'),
            {"objects": [{"category": "cube"}]},
        )

    def test_safety_filter_rejects_bad_geometry_and_keeps_audit_trail(self):
        good = dict(
            category="cube",
            colour="blue",
            pose_confidence=0.91,
            pose_world={"position_m": [0.3, 0.1, 0.2]},
            image_pose_measurement={"silhouette_iou": 0.93},
        )
        cap_mistake = dict(
            category="cube",
            colour="yellow",
            pose_confidence=0,
            image_pose_measurement={"silhouette_iou": 0},
        )
        result = filter_stack_cube_poses({"objects": [good, cap_mistake]})
        self.assertEqual(result["objects"], [dict(good, perception_eligible=True)])
        self.assertEqual(len(result["rejected_objects"]), 1)
        self.assertEqual(
            result["rejected_objects"][0]["skip_reason"],
            "cube_silhouette_fit_uncertain",
        )
        self.assertEqual(result["perception_safety"]["accepted"], 1)

    def test_safety_filter_fails_closed_when_pose_evidence_is_missing(self):
        result = filter_stack_cube_poses(
            {"objects": [dict(category="cube", colour="red", pose_confidence=0.99)]}
        )
        self.assertEqual(result["objects"], [])
        self.assertEqual(
            result["rejected_objects"][0]["skip_reason"],
            "cube_silhouette_fit_uncertain",
        )


if __name__ == "__main__":
    unittest.main()
