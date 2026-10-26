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

    def test_go_requires_dataset_classifier_and_isaac_rl(self) -> None:
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
            report = MODULE.build_report(root, classifier, rl, 2)
        self.assertEqual(report["overall"], "go")


if __name__ == "__main__":
    unittest.main()
