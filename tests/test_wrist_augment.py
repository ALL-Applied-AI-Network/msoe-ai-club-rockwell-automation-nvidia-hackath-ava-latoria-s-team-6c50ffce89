import random
import unittest

import numpy as np

from stacking_recovery.data import wrist_augment


@unittest.skipIf(wrist_augment.cv2 is None, "opencv is not installed")
class TestWristAugment(unittest.TestCase):
    def test_augmentation_preserves_image_contract(self):
        image = np.full((32, 48, 3), 128, dtype=np.uint8)
        result = wrist_augment.augment_wrist_bgr(
            image,
            wrist_augment.default_augment_config(),
            rng=random.Random(7),
        )

        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertFalse(np.shares_memory(result, image))

    def test_seeded_augmentation_is_repeatable(self):
        image = np.arange(32 * 48 * 3, dtype=np.uint8).reshape(32, 48, 3)
        config = wrist_augment.default_augment_config()

        first = wrist_augment.augment_wrist_bgr(image, config, rng=random.Random(11))
        second = wrist_augment.augment_wrist_bgr(image, config, rng=random.Random(11))

        np.testing.assert_array_equal(first, second)

    def test_successive_calls_generate_new_variations(self):
        image = np.arange(32 * 48 * 3, dtype=np.uint8).reshape(32, 48, 3)
        config = wrist_augment.default_augment_config()
        rng = random.Random(19)

        first = wrist_augment.augment_wrist_bgr(image, config, rng=rng)
        second = wrist_augment.augment_wrist_bgr(image, config, rng=rng)

        self.assertFalse(np.array_equal(first, second))


if __name__ == "__main__":
    unittest.main()
