import json
import tempfile
import unittest
from pathlib import Path

from stacking_recovery.bench.metrics import diagnose_run, summarize_run, summarize_suite


class TestMetrics(unittest.TestCase):
    def test_summarize_run_from_fixtures(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "execution_report.json").write_text(
                json.dumps(
                    {
                        "task_complete": True,
                        "verified_pickups": 1,
                        "events": [
                            {
                                "closed_loop_grasp": {"seated": True},
                                "pickup_verified": False,
                            },
                            {
                                "closed_loop_grasp": {"seated": False},
                                "pickup_verified": True,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (run / "manipulation.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "action": "grasp_check",
                                "seated": False,
                                "outcome": "retry",
                                "attempt_index": 0,
                                "reason": "EMPTY_GRASP",
                            }
                        ),
                        json.dumps(
                            {
                                "action": "regrasp",
                                "outcome": "retry",
                                "attempt_index": 1,
                                "reason": "EMPTY_GRASP",
                            }
                        ),
                        json.dumps(
                            {
                                "action": "grasp_check",
                                "seated": True,
                                "outcome": "success",
                                "attempt_index": 1,
                                "reason": "OK",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            row = summarize_run(run)
            self.assertTrue(row["report_found"])
            self.assertEqual(row["observed_picks"], 2)
            self.assertFalse(row["first_grasp_success"])
            self.assertEqual(row["regrasp_count"], 1)
            self.assertEqual(row["grasp_confusion"]["false_positive"], 1)
            self.assertEqual(row["grasp_confusion"]["false_negative"], 1)
            self.assertEqual(row["false_positive_rate"], 1.0)
            self.assertEqual(row["false_negative_rate"], 1.0)

    def test_suite_rates(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = []
            for complete in (True, False):
                d = Path(tmp) / ("ok" if complete else "bad")
                d.mkdir()
                (d / "execution_report.json").write_text(
                    json.dumps({"task_complete": complete}),
                    encoding="utf-8",
                )
                runs.append(d)
            suite = summarize_suite(runs)
            self.assertEqual(suite["n_runs"], 2)
            self.assertAlmostEqual(suite["tower_completion_rate"], 0.5)

    def test_current_report_picks_are_compared_and_diagnosed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "execution_report.json").write_text(
                json.dumps(
                    {
                        "task_complete": False,
                        "verified_pickups": 1,
                        "verified_placements": 0,
                        "abort_reason": "Closed-loop grasp stop at level 0: EMPTY_GRASP",
                        "picks": [
                            {
                                "closed_loop_grasp": {"seated": False},
                                "pickup_verified": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            summary = summarize_run(run)
            self.assertEqual(summary["grasp_confusion"]["false_negative"], 1)
            self.assertIsNone(summary["first_failure"])
            self.assertIn(
                "rejected at least one physically successful grasp",
                diagnose_run(summary)[0],
            )

    def test_false_positive_is_treated_as_unsafe(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "execution_report.json").write_text(
                json.dumps(
                    {
                        "task_complete": False,
                        "verified_pickups": 0,
                        "verified_placements": 0,
                        "picks": [
                            {
                                "detection_id": 7,
                                "pickup_verified": False,
                                "placement_verified": False,
                                "closed_loop_grasp": {
                                    "seated": True,
                                    "reason": "OK",
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            summary = summarize_run(run)
            self.assertEqual(summary["grasp_confusion"]["false_positive"], 1)
            self.assertEqual(summary["first_failure"]["detection_id"], 7)
            self.assertIn("accepted at least one grasp", diagnose_run(summary)[0])

    def test_missing_report_has_startup_diagnosis(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = summarize_run(Path(tmp))
            self.assertFalse(summary["report_found"])
            self.assertIn("No execution_report.json", diagnose_run(summary)[0])


if __name__ == "__main__":
    unittest.main()
