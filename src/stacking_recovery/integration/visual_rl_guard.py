"""Fail-closed visual verification before an RL-directed placement."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Protocol


class VisualPrediction(Protocol):
    class_name: str
    probabilities: dict[str, float]


class VisualGateAction(str, Enum):
    PROCEED = "proceed"
    SECOND_LOOK = "second_look"
    SAFE_STOP = "safe_stop"


@dataclass(frozen=True)
class VisualGateDecision:
    action: VisualGateAction
    good_probability: float
    reason: str
    looks: int

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["action"] = self.action.value
        return result


def good_probability(prediction: VisualPrediction) -> float:
    """Return P(GOOD), even when another class wins the argmax."""
    return float(prediction.probabilities.get("GOOD", 0.0))


def visual_pickup_gate(
    first: VisualPrediction,
    second: VisualPrediction | None = None,
    *,
    direct_threshold: float = 0.80,
    minimum_threshold: float = 0.525,
    consensus_threshold: float = 0.70,
) -> VisualGateDecision:
    """Require a confident GOOD prediction before RL may place a cube.

    One very confident observation may proceed.  A borderline GOOD result must
    be confirmed by a second observation.  Any predicted failure or
    disagreement fails closed before the carry-to-stack segment.
    """
    p1 = good_probability(first)
    if first.class_name != "GOOD" or p1 < minimum_threshold:
        return VisualGateDecision(
            VisualGateAction.SAFE_STOP, p1, "first_look_rejected_pickup", 1
        )
    if p1 >= direct_threshold:
        return VisualGateDecision(
            VisualGateAction.PROCEED, p1, "high_confidence_pickup", 1
        )
    if second is None:
        return VisualGateDecision(
            VisualGateAction.SECOND_LOOK, p1, "borderline_pickup_needs_confirmation", 1
        )

    p2 = good_probability(second)
    mean_probability = (p1 + p2) / 2.0
    if (
        second.class_name == "GOOD"
        and p2 >= minimum_threshold
        and mean_probability >= consensus_threshold
    ):
        return VisualGateDecision(
            VisualGateAction.PROCEED,
            mean_probability,
            "two_look_consensus",
            2,
        )
    return VisualGateDecision(
        VisualGateAction.SAFE_STOP,
        mean_probability,
        "two_look_disagreement_or_low_confidence",
        2,
    )
