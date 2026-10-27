import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_lab_readiness_report.py"
SPEC = importlib.util.spec_from_file_location("generate_lab_readiness_report", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestLabReadinessReport(unittest.TestCase):
    def test_pending_report_counts_binary_test_quotas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {"split": "test", "label": "GOOD", "path": "a.png", "seed": 1},
                {"split": "test", "label": "SHIFTED", "path": "b.png", "seed": 2},
                {"split": "train", "label": "GOOD", "path": "c.png", "seed": 3},
            ]
            (root / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
            report = MODULE.build_report(root, None, None, 2)
        self.assertEqual(report["overall"], "pending")
        self.assertEqual(report["dataset"]["target_test_samples"], 4)
        self.assertEqual(report["dataset"]["missing"], {"GOOD": 1, "FAILURE": 1})

    def test_go_requires_all_five_documented_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {"split": "test", "label": label, "path": f"{label}-{i}.png", "seed": i + offset}
                for label, offset in (("GOOD", 0), ("SHIFTED", 10))
                for i in range(2)
            ]
            (root / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
            classifier = root / "classifier.json"
            classifier.write_text(
                json.dumps({"test": {"n": 4, "accuracy": 1.0, "unsafe_good_rate": 0.0}}),
                encoding="utf-8",
            )
            rl = root / "rl.json"
            rl.write_text(
                json.dumps({"acceptance": {"status": "pass"}, "aggregate": {}}),
                encoding="utf-8",
            )
            failures = root / "failures.json"
            failures.write_text(
                json.dumps(
                    {
                        "total_trials": 20,
                        "scenario_counts": {
                            "missed_grasp": 5,
                            "shifted_cube": 5,
                            "occlusion": 5,
                            "placement_offset": 5,
                        },
                        "safe_outcomes": 20,
                        "unsafe_carries": 0,
                        "ignored_verification_failures": 0,
                    }
                ),
                encoding="utf-8",
            )
            repeatability = root / "repeatability.json"
            repeatability.write_text(
                json.dumps(
                    {
                        "clean_restarts": 3,
                        "successful_restarts": 3,
                        "identical_plan": True,
                    }
                ),
                encoding="utf-8",
            )
            report = MODULE.build_report(
                root, classifier, rl, 2, failures, repeatability
            )
        self.assertEqual(report["overall"], "go")

    def test_three_model_gates_cannot_hide_missing_safety_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {"split": "test", "label": label, "path": f"{label}-{i}.png"}
                for label in ("GOOD", "SHIFTED")
                for i in range(2)
            ]
            (root / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
            classifier = root / "classifier.json"
            classifier.write_text(
                json.dumps({"test": {"n": 4, "accuracy": 1.0, "unsafe_good_rate": 0.0}}),
                encoding="utf-8",
            )
            rl = root / "rl.json"
            rl.write_text(
                json.dumps({"acceptance": {"status": "pass"}}), encoding="utf-8"
            )
            report = MODULE.build_report(root, classifier, rl, 2)
        self.assertEqual(report["overall"], "pending")
        self.assertEqual(report["failure_handling"]["status"], "pending")
        self.assertEqual(report["repeatability"]["status"], "pending")


if __name__ == "__main__":
    unittest.main()
