import json
import tempfile
import unittest
from pathlib import Path

from stacking_recovery.bench.metrics import summarize_run, summarize_suite


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


if __name__ == "__main__":
    unittest.main()
