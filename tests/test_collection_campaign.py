from __future__ import annotations

import unittest

from stacking_recovery.data.collection_campaign import (
    CampaignTargets,
    binary_label,
    campaign_complete,
    campaign_progress,
    next_collection_attempt,
)
from stacking_recovery.data.dataset_split import split_for_seed


class TestCollectionCampaign(unittest.TestCase):
    def test_detailed_failures_normalize_to_failure(self) -> None:
        self.assertEqual(binary_label("GOOD"), "GOOD")
        for label in ("SHIFTED", "ROTATED", "EMPTY_GRASP", "MISALIGNED"):
            self.assertEqual(binary_label(label), "FAILURE")

    def test_next_attempt_fills_actual_split_deficit(self) -> None:
        rows = [
            {"label": "GOOD", "split": "train", "seed": 0},
            {"label": "SHIFTED", "split": "train", "seed": 2},
        ]
        targets = CampaignTargets(train=1, val=1, test=1)
        attempt = next_collection_attempt(rows, targets)
        self.assertIsNotNone(attempt)
        assert attempt is not None
        self.assertIn(attempt.target_label, ("GOOD", "FAILURE"))
        self.assertIn(attempt.split, ("val", "test"))
        self.assertEqual(split_for_seed(attempt.seed), attempt.split)
        self.assertNotIn(attempt.seed, {0, 2})

    def test_failed_outcome_does_not_get_counted_as_requested_label(self) -> None:
        rows = [{"label": "GOOD", "split": "train", "seed": 0}]
        targets = CampaignTargets(train=1, val=1, test=1)
        progress = campaign_progress(rows, targets)
        self.assertEqual(progress["train"]["GOOD"]["count"], 1)
        self.assertEqual(progress["train"]["FAILURE"]["count"], 0)

    def test_attempt_without_saved_image_moves_to_a_fresh_seed(self) -> None:
        targets = CampaignTargets(train=1, val=1, test=1)
        first = next_collection_attempt([], targets)
        self.assertIsNotNone(first)
        assert first is not None
        second = next_collection_attempt(
            [], targets, attempted_seeds={first.seed}
        )
        self.assertIsNotNone(second)
        assert second is not None
        self.assertNotEqual(first.seed, second.seed)
        self.assertEqual(first.split, second.split)

    def test_campaign_completion_checks_both_labels_in_every_split(self) -> None:
        rows = [
            {"label": label, "split": split, "seed": i}
            for i, split in enumerate(("train", "val", "test"))
            for label in ("GOOD", "ROTATED")
        ]
        self.assertTrue(campaign_complete(rows, CampaignTargets(1, 1, 1)))

    def test_failure_trials_vary_and_are_auditable(self) -> None:
        rows = [
            {"label": label, "split": split, "seed": seed}
            for seed, (split, label) in enumerate(
                (
                    ("train", "GOOD"),
                    ("val", "GOOD"),
                    ("val", "SHIFTED"),
                    ("test", "GOOD"),
                    ("test", "ROTATED"),
                )
            )
        ]
        attempt = next_collection_attempt(rows, CampaignTargets(1, 1, 1))
        self.assertIsNotNone(attempt)
        assert attempt is not None
        self.assertEqual(attempt.target_label, "FAILURE")
        self.assertEqual(attempt.trial["campaign_target"], "FAILURE")
        self.assertEqual(attempt.trial["campaign_scenario"], attempt.scenario)


if __name__ == "__main__":
    unittest.main()
