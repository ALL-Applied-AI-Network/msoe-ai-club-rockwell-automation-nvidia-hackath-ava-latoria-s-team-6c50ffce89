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
    def test_stack_plan_validation_rejects_sort_only_plan(self) -> None:
        with self.assertRaisesRegex(ValueError, "stack-ready cube_stack"):
            MODULE.validate_stack_plan(
                {
                    "task": "sort_cubes_by_color",
                    "stack_ready": False,
                    "picks": [{"stack_level": None}] * 6,
                }
            )

    def test_stack_plan_validation_accepts_complete_tower(self) -> None:
        MODULE.validate_stack_plan(
            {
                "task": "cube_stack",
                "stack_ready": True,
                "picks": [{"stack_level": level} for level in range(6)],
            }
        )

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
                        "task": "cube_stack",
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
        self.assertEqual(summary["acceptance"]["status"], "insufficient_evidence")

    def test_summary_builds_matched_rl_scorecard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for seed in range(20):
                for policy in ("open_loop", "closed_loop", "rl"):
                    run = root / f"seed{seed}-{policy}"
                    run.mkdir()
                    complete = policy == "rl" or (policy == "closed_loop" and seed < 10)
                    report = run / "execution_report.json"
                    report.write_text(
                        json.dumps(
                            {
                                "policy": policy,
                                "task": "cube_stack",
                                "task_complete": complete,
                                "verified_pickups": 6 if complete else 3,
                                "verified_placements": 6 if complete else 3,
                                "picks": [],
                                "abort_reason": None if complete else "tower stopped",
                            }
                        ),
                        encoding="utf-8",
                    )
                    paths.append(report)
            summary = MODULE.summarize(paths)

        self.assertEqual(summary["aggregate"]["rl"]["completion_rate"], 1.0)
        self.assertEqual(summary["aggregate"]["rl"]["completion_rate_95ci"][1], 1.0)
        self.assertEqual(summary["paired_rl_comparison"]["rl_vs_open_loop"]["wins"], 20)
        self.assertEqual(summary["acceptance"]["status"], "pass")

    def test_sort_report_cannot_count_as_full_tower(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "execution_report.json"
            report.write_text(
                json.dumps(
                    {
                        "policy": "rl",
                        "task": "sort_cubes_by_color",
                        "task_complete": True,
                        "verified_pickups": 6,
                        "verified_placements": 6,
                        "picks": [],
                    }
                ),
                encoding="utf-8",
            )
            summary = MODULE.summarize([report])
        self.assertFalse(summary["runs"][0]["full_tower_complete"])

    def test_writes_csv_and_svg_without_plotting_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [{"seed": 1, "policy": "rl", "task_complete": True}]
            MODULE.write_metrics_csv(rows, root / "metrics.csv")
            MODULE.write_comparison_svg(
                {"rl": {"completion_rate": 1.0, "runs": 1, "completions": 1}},
                root / "chart.svg",
            )
            self.assertIn("seed,policy", (root / "metrics.csv").read_text())
            self.assertIn("100.0% (1/1)", (root / "chart.svg").read_text())


if __name__ == "__main__":
    unittest.main()
