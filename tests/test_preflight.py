from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from stacking_recovery.preflight import (
    check_model,
    check_rl_checkpoint,
    is_git_lfs_pointer,
    passed,
    run_preflight,
)


class TestDemoPreflight(unittest.TestCase):
    def test_rejects_git_lfs_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.pt"
            model.write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:123\nsize 47940697\n",
                encoding="utf-8",
            )
            self.assertTrue(is_git_lfs_pointer(model))
            result = check_model(model)
            self.assertFalse(result.passed)
            self.assertIn("git lfs pull", result.fix or "")

    def test_accepts_materialized_model_and_q_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "models").mkdir()
            (root / "configs").mkdir()
            model = root / "models" / "grasp_pickup_table_resnet18_siamese_224.pt"
            model.write_bytes(b"weights" * 200_000)
            (root / "models" / "rl_stacking_qtable.json").write_text(
                json.dumps({"format": "stacking_recovery.q_table.v1"}),
                encoding="utf-8",
            )
            (root / "configs" / "grasp_model_resnet_siamese.yaml").write_text(
                "model_path: models/grasp_pickup_table_resnet18_siamese_224.pt\n",
                encoding="utf-8",
            )
            checks = run_preflight(root)
            self.assertTrue(passed(checks), checks)

    def test_validates_model_selected_by_config_and_its_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "models").mkdir()
            (root / "configs").mkdir()
            model = root / "models" / "custom.pt"
            model.write_bytes(b"frozen-weights" * 100_000)
            digest = hashlib.sha256(model.read_bytes()).hexdigest()
            (root / "models" / "rl_stacking_qtable.json").write_text(
                json.dumps({"format": "stacking_recovery.q_table.v1"}),
                encoding="utf-8",
            )
            (root / "configs" / "custom.yaml").write_text(
                f"model_path: models/custom.pt\nmodel_sha256: {digest}\n",
                encoding="utf-8",
            )

            checks = run_preflight(root, vision_config=Path("configs/custom.yaml"))

            self.assertTrue(passed(checks), checks)
            checkpoint = next(
                check for check in checks if check.name == "classifier_checkpoint"
            )
            self.assertIn("SHA-256 verified", checkpoint.detail)
            self.assertIn("custom.pt", checkpoint.detail)

    def test_rejects_hash_mismatch_for_configured_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "models").mkdir()
            (root / "configs").mkdir()
            (root / "models" / "custom.pt").write_bytes(b"weights" * 200_000)
            (root / "models" / "rl_stacking_qtable.json").write_text(
                json.dumps({"format": "stacking_recovery.q_table.v1"}),
                encoding="utf-8",
            )
            (root / "configs" / "custom.yaml").write_text(
                'model_path: models/custom.pt\nmodel_sha256: "'
                + ("0" * 64)
                + '"\n',
                encoding="utf-8",
            )

            checks = run_preflight(root, vision_config=Path("configs/custom.yaml"))

            self.assertFalse(passed(checks))
            checkpoint = next(
                check for check in checks if check.name == "classifier_checkpoint"
            )
            self.assertIn("SHA-256 mismatch", checkpoint.detail)

    def test_explicit_model_must_match_config_selected_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "models").mkdir()
            (root / "configs").mkdir()
            configured = root / "models" / "configured.pt"
            configured.write_bytes(b"configured" * 200_000)
            other = root / "models" / "other.pt"
            other.write_bytes(b"other" * 300_000)
            (root / "models" / "rl_stacking_qtable.json").write_text(
                json.dumps({"format": "stacking_recovery.q_table.v1"}),
                encoding="utf-8",
            )
            (root / "configs" / "custom.yaml").write_text(
                "model_path: models/configured.pt\n", encoding="utf-8"
            )

            checks = run_preflight(
                root,
                model_path=other,
                vision_config=Path("configs/custom.yaml"),
            )

            self.assertFalse(passed(checks))
            alignment = next(
                check
                for check in checks
                if check.name == "classifier_config_alignment"
            )
            self.assertIn("does not match", alignment.detail)

    def test_missing_model_selected_by_config_does_not_fall_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "models").mkdir()
            (root / "configs").mkdir()
            (root / "models" / "grasp_pickup_table_resnet18_siamese_224.pt").write_bytes(
                b"default" * 200_000
            )
            (root / "models" / "rl_stacking_qtable.json").write_text(
                json.dumps({"format": "stacking_recovery.q_table.v1"}),
                encoding="utf-8",
            )
            (root / "configs" / "custom.yaml").write_text(
                "model_path: models/missing.pt\n", encoding="utf-8"
            )

            checks = run_preflight(root, vision_config=Path("configs/custom.yaml"))

            checkpoint = next(
                check for check in checks if check.name == "classifier_checkpoint"
            )
            self.assertFalse(checkpoint.passed)
            self.assertIn("missing.pt", checkpoint.detail)

    def test_rejects_wrong_rl_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rl.json"
            path.write_text('{"format": "unknown"}', encoding="utf-8")
            self.assertFalse(check_rl_checkpoint(path).passed)


if __name__ == "__main__":
    unittest.main()
