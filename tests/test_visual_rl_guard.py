import unittest
from dataclasses import dataclass

from stacking_recovery.integration.visual_rl_guard import (
    VisualGateAction,
    visual_pickup_gate,
)


@dataclass(frozen=True)
class Prediction:
    class_name: str
    confidence: float
    probabilities: dict[str, float]


def prediction(label: str, probability_good: float) -> Prediction:
    return Prediction(
        label,
        max(probability_good, 1.0 - probability_good),
        {"GOOD": probability_good, "SHIFTED": 1.0 - probability_good},
    )


class TestVisualRLGate(unittest.TestCase):
    def test_high_confidence_good_proceeds(self) -> None:
        decision = visual_pickup_gate(prediction("GOOD", 0.93))
        self.assertEqual(decision.action, VisualGateAction.PROCEED)

    def test_borderline_good_requests_second_look(self) -> None:
        decision = visual_pickup_gate(prediction("GOOD", 0.72))
        self.assertEqual(decision.action, VisualGateAction.SECOND_LOOK)

    def test_two_good_looks_can_proceed(self) -> None:
        decision = visual_pickup_gate(
            prediction("GOOD", 0.72), prediction("GOOD", 0.74)
        )
        self.assertEqual(decision.action, VisualGateAction.PROCEED)
        self.assertEqual(decision.looks, 2)

    def test_failure_prediction_stops_before_placement(self) -> None:
        decision = visual_pickup_gate(prediction("SHIFTED", 0.12))
        self.assertEqual(decision.action, VisualGateAction.SAFE_STOP)

    def test_disagreement_fails_closed(self) -> None:
        decision = visual_pickup_gate(
            prediction("GOOD", 0.72), prediction("SHIFTED", 0.20)
        )
        self.assertEqual(decision.action, VisualGateAction.SAFE_STOP)


if __name__ == "__main__":
    unittest.main()
