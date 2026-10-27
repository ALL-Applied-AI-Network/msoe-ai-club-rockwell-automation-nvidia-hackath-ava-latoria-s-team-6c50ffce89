import importlib.util
import os
import unittest
from pathlib import Path

import numpy as np


POLICY_PATH = Path(__file__).resolve().parents[1] / "hardware" / "remote_cell" / "policy.py"
SPEC = importlib.util.spec_from_file_location("remote_cell_policy", POLICY_PATH)
assert SPEC and SPEC.loader
POLICY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POLICY)


class TestRemoteCellPolicy(unittest.TestCase):
    def test_color_centers_find_small_red_and_blue_regions(self):
        image = np.zeros((120, 200, 3), dtype=np.uint8)
        image[20:32, 30:44] = (230, 20, 20)
        image[70:84, 150:166] = (15, 30, 220)
        red = POLICY.color_center(image, "red")
        blue = POLICY.color_center(image, "blue")
        self.assertIsNotNone(red)
        self.assertIsNotNone(blue)
        self.assertAlmostEqual(red[0], 36.5, delta=1.0)
        self.assertAlmostEqual(blue[0], 157.5, delta=1.0)

    def test_color_components_ignore_border_clutter(self):
        image = np.zeros((120, 200, 3), dtype=np.uint8)
        image[0:20, 0:70] = (230, 20, 20)
        image[40:54, 50:66] = (230, 20, 20)
        image[75:91, 130:148] = (230, 20, 20)

        components = POLICY.color_components(image, "red")

        self.assertEqual(len(components), 2)
        centers = [component["center"] for component in components]
        self.assertTrue(any(abs(u - 57.5) < 1 and abs(v - 46.5) < 1 for u, v in centers))
        self.assertTrue(any(abs(u - 138.5) < 1 and abs(v - 82.5) < 1 for u, v in centers))

    def test_wrist_color_center_allows_close_cube(self):
        image = np.zeros((120, 200, 3), dtype=np.uint8)
        image[25:75, 65:125] = (230, 20, 20)

        self.assertEqual(POLICY.color_components(image, "red"), [])
        center = POLICY.color_center(image, "red")
        self.assertIsNotNone(center)
        self.assertAlmostEqual(center[0], 94.5, delta=1.0)
        self.assertAlmostEqual(center[1], 49.5, delta=1.0)

    def test_workspace_rejects_out_of_box_and_out_of_reach(self):
        self.assertTrue(POLICY.tcp_is_safe([450, 0, -250, -180, 0, 0, 0]))
        self.assertFalse(POLICY.tcp_is_safe([149, 0, -250, -180, 0, 0, 0]))
        self.assertFalse(POLICY.tcp_is_safe([600, 0, -250, -180, 0, 0, 0]))

    def test_interpolation_is_bounded_and_finishes_at_target(self):
        rows = POLICY.interpolate_tcp(
            [400, 0, -250, -180, 0, 0],
            [430, 20, -280, -180, 0, 0, 1],
            step_mm=10,
        )
        self.assertGreater(len(rows), 1)
        self.assertLessEqual(len(rows), 100)
        self.assertEqual(rows[-1], [430.0, 20.0, -280.0, -180.0, 0.0, 0.0, 1.0])
        self.assertTrue(all(POLICY.tcp_is_safe(row) for row in rows))

    def test_operator_xy_parser_requires_two_finite_values(self):
        self.assertEqual(POLICY.parse_xy("420,-100"), (420.0, -100.0))
        self.assertIsNone(POLICY.parse_xy(None))
        with self.assertRaises(ValueError):
            POLICY.parse_xy("420")
        with self.assertRaises(ValueError):
            POLICY.parse_xy("nan,100")

    def test_operator_fallback_must_be_complete_and_safe(self):
        original = {name: os.environ.get(name) for name in POLICY.FIXED_TARGET_ENV.values()}
        try:
            os.environ["REMOTE_ARM_RED_XY_MM"] = "420,-100"
            os.environ.pop("REMOTE_ARM_BLUE_XY_MM", None)
            with self.assertRaises(ValueError):
                POLICY.Policy()

            os.environ["REMOTE_ARM_BLUE_XY_MM"] = "500,100"
            policy = POLICY.Policy()
            self.assertEqual(policy._fixed_targets["red"], (420.0, -100.0))
            self.assertEqual(policy._fixed_targets["blue"], (500.0, 100.0))
            self.assertTrue(policy._operator_fallback_active)
        finally:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_operator_fallback_replaces_only_an_unsafe_visible_detection(self):
        original = {name: os.environ.get(name) for name in POLICY.FIXED_TARGET_ENV.values()}
        try:
            os.environ["REMOTE_ARM_RED_XY_MM"] = "420,-100"
            os.environ["REMOTE_ARM_BLUE_XY_MM"] = "500,100"
            policy = POLICY.Policy()
            policy._cams = {"table": {"size": (640, 360)}}
            policy._pixel_to_base = lambda *_: np.array([0.9, 0.9, 0.0])
            image = np.zeros((360, 640, 3), dtype=np.uint8)
            image[100:112, 200:212] = (230, 20, 20)

            self.assertEqual(policy._table_xy(image, "red"), (420.0, -100.0))
            self.assertEqual(
                policy._last_detections["red"]["source"],
                "operator_confirmed_fallback",
            )
            self.assertIsNone(
                policy._table_xy(image, "red", allow_operator_fallback=False)
            )
        finally:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_table_detector_checks_all_components_and_uses_safe_one(self):
        original = {name: os.environ.get(name) for name in POLICY.FIXED_TARGET_ENV.values()}
        try:
            for name in POLICY.FIXED_TARGET_ENV.values():
                os.environ.pop(name, None)
            policy = POLICY.Policy()
            policy._cams = {"table": {"size": (640, 360)}}

            def fake_pixel_to_base(_camera, uv):
                # The larger component is intentionally unsafe.  The smaller
                # one is the cube and maps to its expected physical location.
                if uv[0] > 400:
                    return np.array([0.9, 0.9, 0.0])
                return np.array([0.42, -0.10, 0.0])

            policy._pixel_to_base = fake_pixel_to_base
            image = np.zeros((360, 640, 3), dtype=np.uint8)
            image[90:104, 180:194] = (230, 20, 20)
            image[250:280, 500:530] = (230, 20, 20)

            self.assertEqual(policy._table_xy(image, "red"), (420.0, -100.0))
            detection = policy._last_detections["red"]
            self.assertEqual(detection["component_count"], 2)
            self.assertEqual(detection["safe_component_count"], 1)
            self.assertEqual(detection["source"], "calibrated_camera")
        finally:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_operator_anchor_rejects_distant_in_bounds_false_positive(self):
        original = {name: os.environ.get(name) for name in POLICY.FIXED_TARGET_ENV.values()}
        try:
            os.environ["REMOTE_ARM_RED_XY_MM"] = "420,-100"
            os.environ["REMOTE_ARM_BLUE_XY_MM"] = "500,100"
            policy = POLICY.Policy()
            policy._cams = {"table": {"size": (640, 360)}}
            policy._pixel_to_base = lambda *_: np.array([0.25, 0.25, 0.0])
            image = np.zeros((360, 640, 3), dtype=np.uint8)
            image[100:114, 200:214] = (230, 20, 20)

            self.assertEqual(policy._table_xy(image, "red"), (420.0, -100.0))
            self.assertEqual(
                policy._last_detections["red"]["source"],
                "operator_confirmed_fallback",
            )
        finally:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_supervised_fallback_sequence_stays_safe_and_reports_unverified(self):
        original = {name: os.environ.get(name) for name in POLICY.FIXED_TARGET_ENV.values()}
        try:
            os.environ["REMOTE_ARM_RED_XY_MM"] = "420,-100"
            os.environ["REMOTE_ARM_BLUE_XY_MM"] = "500,100"
            policy = POLICY.Policy()
            policy._calibration_error = None
            policy._cams = {"table": {"size": (640, 360)}}
            policy._pixel_to_base = lambda *_: np.array([0.9, 0.9, 0.0])

            table = np.zeros((360, 640, 3), dtype=np.uint8)
            table[100:112, 200:212] = (230, 20, 20)
            table[200:212, 400:412] = (20, 30, 230)
            wrist = np.zeros_like(table)
            wrist[170:186, 310:326] = (230, 20, 20)
            obs = {
                "images": {"table": table, "wrist": wrist},
                "tcp_pose": [300, 0, -90, -180, 0, 0],
                "joints": [0, 0, 0, 0, 0, 0, 0],
            }

            for _ in range(9):
                action = np.asarray(policy.act(obs), dtype=float).reshape(-1, 7)
                self.assertTrue(all(POLICY.tcp_is_safe(row) for row in action))
                obs["tcp_pose"] = action[-1, :6].tolist()
                obs["joints"][-1] = action[-1, 6]

            self.assertEqual(policy.phase, "complete")
            self.assertEqual(policy.result, "completed_unverified")
            self.assertIn("fallback", policy.reason)
        finally:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


if __name__ == "__main__":
    unittest.main()
