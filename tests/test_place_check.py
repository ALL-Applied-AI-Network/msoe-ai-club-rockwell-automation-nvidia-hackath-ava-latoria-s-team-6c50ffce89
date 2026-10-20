import unittest

import numpy as np

from stacking_recovery.types import FailureReason
from stacking_recovery.vision.place_check import place_check, place_check_with_physics


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


if __name__ == "__main__":
    unittest.main()
