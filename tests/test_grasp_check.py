import unittest

import numpy as np

from stacking_recovery.vision.grasp_check import grasp_check


class TestGraspCheck(unittest.TestCase):
    def test_empty_wrist(self):
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        r = grasp_check(img)
        self.assertFalse(r.seated)

    def test_bright_center_passes(self):
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        img[280:440, 480:800] = (200, 180, 50)
        r = grasp_check(img)
        self.assertTrue(r.seated, msg=f"{r.reason} conf={r.confidence}")


if __name__ == "__main__":
    unittest.main()
