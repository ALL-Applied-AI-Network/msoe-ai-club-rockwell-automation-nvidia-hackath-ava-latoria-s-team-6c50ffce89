"""Safety guard for simulation-ground-truth data collection plans."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = ROOT / "data" / "fanuc_er4ia_hackathon_starter_v05 (1)" / "fanuc_er4ia_sim"
sys.path.insert(0, str(ROOT / "scripts" / "sim"))
sys.path.insert(0, str(SIM_DIR))

from prepare_collection_run import build_collection_plan  # noqa: E402
from sort_scene import validate_plan  # noqa: E402


class TestCollectionPlan(unittest.TestCase):
    def test_ground_truth_plan_only_allowed_for_collection(self) -> None:
        plan = build_collection_plan()
        with self.assertRaises(ValueError):
            validate_plan(plan)
        validate_plan(plan, allow_collection_ground_truth=True)
        self.assertEqual(len(plan["picks"]), 6)


if __name__ == "__main__":
    unittest.main()
