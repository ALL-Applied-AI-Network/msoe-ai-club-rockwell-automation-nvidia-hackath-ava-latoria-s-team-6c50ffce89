#!/usr/bin/env python3
"""Create presentation-ready plots from grasp model benchmark JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    raise SystemExit("Install plotting dependencies: pip install matplotlib numpy") from None


COLORS = {"accuracy": "#2563eb", "macro_f1": "#16a34a", "unsafe": "#dc2626"}


def save_figure(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_training(result: dict, output: Path) -> None:
    history = result["history"]
    epochs = [row["epoch"] for row in history]
    fig, (loss_ax, score_ax) = plt.subplots(1, 2, figsize=(11, 4.2))
    loss_ax.plot(epochs, [row["loss"] for row in history], marker="o", color="#7c3aed")
    loss_ax.set(title="Training loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    loss_ax.grid(alpha=0.25)
    score_ax.plot(epochs, [row["val"]["accuracy"] for row in history], marker="o", label="Accuracy", color=COLORS["accuracy"])
    score_ax.plot(epochs, [row["val"]["macro_f1"] for row in history], marker="o", label="Macro-F1", color=COLORS["macro_f1"])
    score_ax.plot(epochs, [row["val"]["unsafe_good_rate"] for row in history], marker="o", label="Unsafe GOOD rate", color=COLORS["unsafe"])
    score_ax.set(title="Validation metrics", xlabel="Epoch", ylabel="Score", ylim=(-0.03, 1.03))
    score_ax.grid(alpha=0.25)
    score_ax.legend(loc="lower right")
    fig.suptitle(f"{result['name']} training history")
    save_figure(fig, output / "training_curves.png")


def plot_confusion(result: dict, output: Path) -> None:
    matrix = np.asarray(result["test"]["confusion"])
    classes = result["classes"]
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=14, color="white" if matrix[row, col] > matrix.max() / 2 else "black")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set(xlabel="Predicted label", ylabel="True label", title="Held-out test confusion matrix")
    fig.colorbar(image, ax=ax, shrink=0.8)
    save_figure(fig, output / "confusion_matrix.png")


def plot_per_class(result: dict, output: Path) -> None:
    classes = result["classes"]
    metrics = ("precision", "recall", "f1")
    x = np.arange(len(classes))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for offset, metric in enumerate(metrics):
        values = [result["test"]["per_class"][name][metric] for name in classes]
        ax.bar(x + (offset - 1) * width, values, width, label=metric.replace("_", " ").title())
    ax.set_xticks(x, classes)
    ax.set(title="Held-out test metrics by class", ylabel="Score", ylim=(0, 1.05))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, output / "precision_recall_f1.png")


def plot_comparison(results: list[dict], output: Path) -> None:
    names = [row["name"].replace("_", "\n") for row in results]
    x = np.arange(len(results))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, [r["test"]["accuracy"] for r in results], width, label="Accuracy", color=COLORS["accuracy"])
    ax.bar(x, [r["test"]["macro_f1"] for r in results], width, label="Macro-F1", color=COLORS["macro_f1"])
    ax.bar(x + width, [r["test"]["unsafe_good_rate"] for r in results], width, label="Unsafe GOOD rate", color=COLORS["unsafe"])
    ax.set_xticks(x, names)
    ax.set(title="Model comparison on the held-out test split", ylabel="Score", ylim=(0, 1.05))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, output / "model_comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True, help="Directory containing benchmark result JSON files")
    parser.add_argument("--winner", type=Path, required=True, help="Winning model result JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    winner = json.loads(args.winner.read_text(encoding="utf-8"))
    comparison = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.results.glob("*.json")) if path.name != "summary.json"]
    if not comparison:
        comparison = [winner]
    plot_training(winner, args.output)
    plot_confusion(winner, args.output)
    plot_per_class(winner, args.output)
    plot_comparison(comparison, args.output)
    (args.output / "metrics.json").write_text(json.dumps({"winner": winner["name"], "validation": winner["validation"], "test": winner["test"], "test_latency_ms": winner["test_latency_ms"]}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote benchmark report to {args.output}")


if __name__ == "__main__":
    main()
