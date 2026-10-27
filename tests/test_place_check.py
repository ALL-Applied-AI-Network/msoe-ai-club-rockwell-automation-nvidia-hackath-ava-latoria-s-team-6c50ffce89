import unittest

import numpy as np

from stacking_recovery.types import FailureReason
from stacking_recovery.vision.place_check import (
    PlacementGeometryConfig,
    place_check,
    place_check_with_physics,
)


class TestPlaceCheck(unittest.TestCase):
    def test_dark_frame_fails(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        r = place_check(img, 0)
        self.assertFalse(r.placed)

    def test_physics_gate(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[100:200, 200:400] = (50, 180, 200)
        r = place_check_with_physics(img, 1, physics_verified=False)
        self.assertFalse(r.placed)
        self.assertEqual(r.reason, FailureReason.PLACEMENT_FAILURE)

    def test_centered_cube_sized_blob_passes(self):
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        img[170:230, 170:230] = (30, 180, 220)
        r = place_check(img, 0)
        self.assertTrue(r.placed)
        self.assertGreaterEqual(r.confidence, 0.6)

    def test_off_target_blob_fails_even_when_colour_fill_is_large(self):
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        img[105:190, 125:175] = (30, 180, 220)
        strict = PlacementGeometryConfig(center_tolerance=0.18)
        r = place_check(img, 0, config=strict)
        self.assertFalse(r.placed)

    def test_tiny_noise_component_fails(self):
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        img[198:202, 198:202] = (30, 180, 220)
        r = place_check(img, 0)
        self.assertFalse(r.placed)

    def test_elongated_component_fails_alignment(self):
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        img[190:204, 140:260] = (30, 180, 220)
        r = place_check(img, 0)
        self.assertFalse(r.placed)

    def test_touching_multilevel_tower_checks_top_cube_not_whole_tower(self):
        config = PlacementGeometryConfig(
            roi_xyxy=(0.1, 0.1, 0.9, 0.9),
            expected_center_xy=(0.5, 0.78125),
            center_tolerance=0.08,
            expected_bottom_y=0.8125,
            bottom_tolerance=0.08,
            level_y_step=0.0625,
            min_area_fraction=0.002,
            max_area_fraction=0.02,
            min_aspect_ratio=0.7,
            max_aspect_ratio=1.4,
            min_support_fraction=0.5,
        )
        for level in range(6):
            with self.subTest(level=level):
                img = np.zeros((400, 400, 3), dtype=np.uint8)
                img[280 - 20 * level : 300, 190:210] = (30, 180, 220)
                result = place_check(img, level, config=config)
                self.assertTrue(result.placed, result)

    def test_upper_level_requires_visible_support_below_top_cube(self):
        config = PlacementGeometryConfig(
            roi_xyxy=(0.1, 0.1, 0.9, 0.9),
            expected_center_xy=(0.5, 0.78125),
            center_tolerance=0.08,
            expected_bottom_y=0.8125,
            bottom_tolerance=0.08,
            level_y_step=0.0625,
            min_area_fraction=0.002,
            max_area_fraction=0.02,
            min_aspect_ratio=0.7,
            max_aspect_ratio=1.4,
            min_support_fraction=0.5,
        )
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        img[240:260, 190:210] = (30, 180, 220)
        result = place_check(img, 2, config=config)
        self.assertFalse(result.placed)


if __name__ == "__main__":
    unittest.main()
