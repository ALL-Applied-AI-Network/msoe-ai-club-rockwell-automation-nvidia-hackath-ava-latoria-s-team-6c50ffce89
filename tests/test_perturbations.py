import unittest
from pathlib import Path

from stacking_recovery.bench.perturbations import load_trials, trial_to_overrides

ROOT = Path(__file__).resolve().parents[1]


class TestPerturbations(unittest.TestCase):
    def test_load_default_trials(self):
        trials = load_trials(ROOT / "configs" / "trials")
        self.assertGreaterEqual(len(trials), 1)
        ov = trial_to_overrides(trials[0])
        self.assertIn("condition_id", ov)
        self.assertIn("policy", ov)


if __name__ == "__main__":
    unittest.main()
