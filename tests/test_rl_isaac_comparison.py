import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sim" / "run_rl_isaac_comparison.py"
SPEC = importlib.util.spec_from_file_location("run_rl_isaac_comparison", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestRLIsaacComparison(unittest.TestCase):
    def test_jobs_use_identical_offset_for_all_policies(self) -> None:
        jobs = MODULE.build_jobs([17], 0.008)
        self.assertEqual([job["policy"] for job in jobs], ["open_loop", "closed_loop", "rl"])
        self.assertEqual(len({tuple(job["place_xy_offset_m"]) for job in jobs}), 1)
        self.assertEqual({job["seed"] for job in jobs}, {17})

    def test_summary_counts_visual_safety_stops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "execution_report.json"
            report.write_text(
                json.dumps(
                    {
                        "policy": "rl",
                        "task_complete": False,
                        "verified_pickups": 1,
                        "verified_placements": 0,
                        "picks": [
                            {
                                "rl_visual_verification": {
                                    "gate": {"action": "safe_stop"}
                                }
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            summary = MODULE.summarize([report])
        self.assertEqual(summary["aggregate"]["rl"]["visual_gate_stops"], 1)


if __name__ == "__main__":
    unittest.main()
