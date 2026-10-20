"""Tests for shared local/Brev dataset utilities."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stacking_recovery.data.dataset_validation import PNG_SIGNATURE, validate_dataset
from stacking_recovery.data.grasp_dataset_store import resolve_data_root


class TestDataRoot(unittest.TestCase):
    def test_environment_data_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            with patch.dict(os.environ, {"GRASP_DATA_ROOT": "shared/grasp"}):
                self.assertEqual(resolve_data_root(repo), (repo / "shared/grasp").resolve())

    def test_explicit_data_root_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            with patch.dict(os.environ, {"GRASP_DATA_ROOT": "ignored"}):
                self.assertEqual(resolve_data_root(repo, Path("chosen")), (repo / "chosen").resolve())


class TestDatasetValidation(unittest.TestCase):
    def test_valid_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "images" / "train" / "GOOD" / "one.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(PNG_SIGNATURE + b"test")
            (root / "manifest.json").write_text(
                json.dumps(
                    [
                        {
                            "path": "images/train/GOOD/one.png",
                            "label": "GOOD",
                            "source": "isaac",
                            "sample_id": "one",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            report = validate_dataset(root)
            self.assertTrue(report["ok"])
            self.assertEqual(report["sample_count"], 1)

    def test_missing_image_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(
                json.dumps(
                    [
                        {
                            "path": "images/train/GOOD/missing.png",
                            "label": "GOOD",
                            "source": "isaac",
                            "sample_id": "missing",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            report = validate_dataset(root)
            self.assertFalse(report["ok"])
            self.assertTrue(any("missing image" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
