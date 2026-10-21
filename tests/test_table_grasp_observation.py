from __future__ import annotations

import unittest

import numpy as np

from stacking_recovery.data.table_grasp_observation import (
    make_before_after_observation,
)


class TestTableGraspObservation(unittest.TestCase):
    def test_pair_is_centered_and_side_by_side(self) -> None:
        before = np.full((100, 120, 3), 30, np.uint8)
        after = before.copy()
        before[45:55, 55:65] = (20, 180, 240)
        after[45:55, 55:65] = (220, 220, 220)
        pair, quality = make_before_after_observation(
            before, after, (60, 50), crop_size=40, min_std=1
        )
        self.assertEqual(pair.shape, (40, 80, 3))
        self.assertTrue(quality.valid)
        self.assertGreater(quality.mean_abs_change, 0)

    def test_out_of_frame_crop_is_rejected(self) -> None:
        image = np.arange(100 * 120 * 3, dtype=np.uint8).reshape(100, 120, 3)
        _, quality = make_before_after_observation(
            image, image.copy(), (1, 1), crop_size=40
        )
        self.assertFalse(quality.valid)
        self.assertEqual(quality.reason, "crop_out_of_frame")

    def test_blank_crop_is_rejected(self) -> None:
        image = np.zeros((100, 120, 3), np.uint8)
        _, quality = make_before_after_observation(
            image, image.copy(), (60, 50), crop_size=40
        )
        self.assertFalse(quality.valid)
        self.assertEqual(quality.reason, "blank_or_low_detail_crop")


if __name__ == "__main__":
    unittest.main()
