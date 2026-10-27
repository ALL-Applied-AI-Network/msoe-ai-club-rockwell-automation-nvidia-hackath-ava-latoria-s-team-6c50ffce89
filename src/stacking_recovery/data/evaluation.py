"""Dependency-free classification metrics for grasp-quality evaluation."""

from __future__ import annotations

from typing import Iterable


def classification_report(
    pairs: Iterable[tuple[str, str]], classes: list[str]
) -> dict:
    pairs = list(pairs)
    confusion = {truth: {pred: 0 for pred in classes} for truth in classes}
    for truth, pred in pairs:
        if truth in confusion and pred in confusion[truth]:
            confusion[truth][pred] += 1

    per_class = {}
    for label in classes:
        tp = confusion[label][label]
        actual = sum(confusion[label].values())
        predicted = sum(confusion[truth][label] for truth in classes)
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": actual,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    total = len(pairs)
    correct = sum(confusion[label][label] for label in classes)
    non_good = sum(1 for truth, _ in pairs if truth != "GOOD")
    unsafe_good = sum(1 for truth, pred in pairs if truth != "GOOD" and pred == "GOOD")
    return {
        "n": total,
        "accuracy": correct / total if total else 0.0,
        "unsafe_good_rate": unsafe_good / non_good if non_good else 0.0,
        "unsafe_good_count": unsafe_good,
        "non_good_count": non_good,
        "per_class": per_class,
        "confusion": confusion,
    }
