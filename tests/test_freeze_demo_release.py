from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "freeze_demo_release.py"
SPEC = importlib.util.spec_from_file_location("freeze_demo_release", SCRIPT)
assert SPEC and SPEC.loader
freeze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(freeze)


class TestFreezeDemoRelease(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "models").mkdir()
        (self.root / "configs").mkdir()
        self.run = self.root / "run"
        self.run.mkdir()
        self.model = self.root / "models" / "model.pt"
        self.model.write_bytes(b"weights" * 200_000)
        self.rl = self.root / "models" / "rl.json"
        self.rl.write_text(
            json.dumps({"format": "stacking_recovery.q_table.v1"}),
            encoding="utf-8",
        )
        self.vision = self.root / "configs" / "vision.yaml"
        self.vision.write_text("model_path: models/model.pt\n", encoding="utf-8")
        self.placement = self.root / "configs" / "placement.json"
        self.placement.write_text("{}\n", encoding="utf-8")
        (self.run / "scene_state.json").write_text("{}\n", encoding="utf-8")
        (self.run / "sort_plan.json").write_text("{}\n", encoding="utf-8")
        self.readiness = self.root / "readiness.json"
        self.readiness.write_text('{"overall":"go"}\n', encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def build(self, **overrides):
        arguments = {
            "root": self.root,
            "run_dir": self.run,
            "model": self.model,
            "rl_checkpoint": self.rl,
            "vision_config": self.vision,
            "placement_config": self.placement,
            "readiness_path": self.readiness,
        }
        arguments.update(overrides)
        return freeze.build_manifest(**arguments)

    def test_qualified_manifest_hashes_every_frozen_input(self) -> None:
        with patch.object(
            freeze,
            "git_snapshot",
            return_value={
                "commit": "a" * 40,
                "branch": "ava-rl",
                "clean": True,
                "changed_paths": [],
            },
        ):
            manifest = self.build()
        self.assertTrue(manifest["qualified"])
        self.assertEqual(manifest["qualification"], "qualified")
        self.assertEqual(set(manifest["files"]), {
            "classifier",
            "rl_checkpoint",
            "vision_config",
            "placement_config",
            "scene_state",
            "sort_plan",
        })
        self.assertEqual(len(manifest["files"]["classifier"]["sha256"]), 64)
        self.assertFalse(manifest["safety"]["connects_to_hardware"])

    def test_dirty_worktree_fails_closed_by_default(self) -> None:
        with patch.object(
            freeze,
            "git_snapshot",
            return_value={
                "commit": "b" * 40,
                "branch": "ava-rl",
                "clean": False,
                "changed_paths": [" M README.md"],
            },
        ):
            with self.assertRaisesRegex(freeze.ReleaseGateError, "worktree is dirty"):
                self.build()

    def test_explicit_development_overrides_are_never_qualified(self) -> None:
        with patch.object(
            freeze,
            "git_snapshot",
            return_value={
                "commit": "c" * 40,
                "branch": "ava-rl",
                "clean": False,
                "changed_paths": ["?? scratch.txt"],
            },
        ):
            manifest = self.build(
                allow_dirty=True,
                skip_readiness=True,
                readiness_path=None,
            )
        self.assertFalse(manifest["qualified"])
        self.assertEqual(manifest["qualification"], "development_only")
        self.assertEqual(manifest["readiness"]["status"], "skipped")

    def test_non_go_readiness_is_rejected(self) -> None:
        self.readiness.write_text('{"overall":"pending"}\n', encoding="utf-8")
        with patch.object(
            freeze,
            "git_snapshot",
            return_value={
                "commit": "d" * 40,
                "branch": "ava-rl",
                "clean": True,
                "changed_paths": [],
            },
        ):
            with self.assertRaisesRegex(freeze.ReleaseGateError, "not GO"):
                self.build()


if __name__ == "__main__":
    unittest.main()
