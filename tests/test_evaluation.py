"""Classification metrics used by offline Isaac evaluation."""

import unittest

from stacking_recovery.data.evaluation import classification_report


class TestClassificationReport(unittest.TestCase):
    def test_report_counts_unsafe_good_predictions(self) -> None:
        report = classification_report(
            [("GOOD", "GOOD"), ("SHIFTED", "GOOD"), ("SHIFTED", "SHIFTED")],
            ["GOOD", "SHIFTED"],
        )
        self.assertEqual(report["accuracy"], 2 / 3)
        self.assertEqual(report["unsafe_good_count"], 1)
        self.assertEqual(report["unsafe_good_rate"], 0.5)
        self.assertEqual(report["per_class"]["SHIFTED"]["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
