import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stacking_recovery.data import materialized_augment


@unittest.skipIf(materialized_augment.cv2 is None, "opencv is not installed")
class TestMaterializedAugment(unittest.TestCase):
    def test_saves_train_copies_but_not_validation_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "derived"
            image = np.full((24, 40, 3), 128, dtype=np.uint8)
            rows = []
            for split in ("train", "val"):
                rel = Path("images") / split / "GOOD" / f"{split}.png"
                path = source / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                materialized_augment.cv2.imwrite(str(path), image)
                rows.append(
                    {
                        "path": rel.as_posix(),
                        "label": "GOOD",
                        "split": split,
                        "sample_id": split,
                        "observation_source": "table_before_after",
                    }
                )
            (source / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")

            derived = materialized_augment.materialize_augmented_dataset(
                source,
                output,
                copies_per_train=2,
                augmentation_config={"brightness_delta": 0.1},
            )

            self.assertEqual(len(derived), 4)
            self.assertEqual(
                sum(row["materialized_kind"] == "augmentation" for row in derived),
                2,
            )
            self.assertEqual(
                sum(
                    row["materialized_kind"] == "augmentation"
                    and row["split"] == "val"
                    for row in derived
                ),
                0,
            )
            for row in derived:
                self.assertTrue((output / row["path"]).is_file())

            second = materialized_augment.materialize_augmented_dataset(
                source,
                output,
                copies_per_train=2,
                augmentation_config={"brightness_delta": 0.1},
            )
            self.assertEqual(derived, second)


if __name__ == "__main__":
    unittest.main()
