#!/usr/bin/env python3
"""Compare pretrained and Siamese models on the fixed table-pair split."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import cv2
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    raise SystemExit("Install ML deps: pip install -e '.[ml]'") from None

from stacking_recovery.data.wrist_augment import augment_table_pair_bgr
from stacking_recovery.vision.grasp_quality_model import preprocess
from stacking_recovery.vision.pair_models import (
    ModelSpec,
    build_pair_model,
    classifier_parameters,
)

CLASSES = ["GOOD", "SHIFTED"]


def binary_label(label: str) -> int:
    return 0 if label == "GOOD" else 1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TablePairDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        root: Path,
        pair_mode: str,
        input_size: int,
        *,
        augment: bool,
        seed: int,
    ) -> None:
        self.rows = rows
        self.root = root
        self.pair_mode = pair_mode
        self.input_size = input_size
        self.augment = augment
        self.rng = random.Random(seed)
        self.augmentation = {
            "brightness_delta": 0.12,
            "contrast_delta": 0.10,
            "gamma_min": 0.88,
            "gamma_max": 1.12,
            "saturation_delta": 0.10,
            "warmth_delta": 0.06,
            "noise_std": 5,
            "blur_prob": 0.12,
            "blur_ksize": 3,
            "jpeg_prob": 0.12,
            "jpeg_quality_min": 75,
            "jpeg_quality_max": 95,
            "shadow_prob": 0.15,
            "shadow_strength_max": 0.12,
            "affine_prob": 0.45,
            "rotation_deg": 2.0,
            "translate_frac": 0.02,
            "scale_delta": 0.03,
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        pair = cv2.imread(str(self.root / row["path"]), cv2.IMREAD_COLOR)
        if pair is None:
            raise FileNotFoundError(row["path"])
        if self.augment:
            pair = augment_table_pair_bgr(pair, self.augmentation, rng=self.rng)
        height, width = pair.shape[:2]
        if width < 2 or width % 2:
            raise ValueError(f"not a side-by-side pair: {row['path']} {pair.shape}")
        before = pair[:, : width // 2]
        after = pair[:, width // 2 :]
        if self.pair_mode == "siamese":
            x = torch.cat(
                [
                    preprocess(before, self.input_size),
                    preprocess(after, self.input_size),
                ],
                dim=0,
            )
        elif self.pair_mode == "delta":
            before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
            after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
            delta = cv2.absdiff(before_gray, after_gray)
            stacked = np.stack([delta, after_gray, before_gray], axis=2)
            x = preprocess(stacked, self.input_size)
        else:
            raise ValueError(self.pair_mode)
        return x, binary_label(str(row["label"])), str(row["sample_id"])


def metrics(truth: list[int], prob_good: list[float], threshold: float) -> dict:
    pred = [0 if probability >= threshold else 1 for probability in prob_good]
    confusion = [[0, 0], [0, 0]]
    for expected, actual in zip(truth, pred):
        confusion[expected][actual] += 1
    per_class = []
    for cls in range(2):
        tp = confusion[cls][cls]
        fp = sum(confusion[row][cls] for row in range(2) if row != cls)
        fn = sum(confusion[cls][col] for col in range(2) if col != cls)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"precision": precision, "recall": recall, "f1": f1})
    correct = sum(confusion[i][i] for i in range(2))
    non_good = sum(1 for value in truth if value != 0)
    unsafe = sum(1 for expected, actual in zip(truth, pred) if expected != 0 and actual == 0)
    return {
        "accuracy": correct / len(truth) if truth else 0.0,
        "macro_f1": sum(row["f1"] for row in per_class) / 2,
        "unsafe_good_rate": unsafe / non_good if non_good else 0.0,
        "confusion": confusion,
        "per_class": {CLASSES[i]: per_class[i] for i in range(2)},
        "threshold_good": threshold,
        "n": len(truth),
    }


def choose_threshold(truth: list[int], prob_good: list[float]) -> tuple[float, dict]:
    candidates = sorted({0.25, 0.5, 0.75, *[round(x, 3) for x in prob_good]})
    scored = [(threshold, metrics(truth, prob_good, threshold)) for threshold in candidates]
    safe = [item for item in scored if item[1]["unsafe_good_rate"] <= 0.05]
    pool = safe or scored
    return max(
        pool,
        key=lambda item: (
            item[1]["macro_f1"],
            item[1]["accuracy"],
            -item[1]["unsafe_good_rate"],
        ),
    )


@torch.no_grad()
def predict(model, loader, device) -> tuple[list[int], list[float], float]:
    model.eval()
    truth: list[int] = []
    probabilities: list[float] = []
    elapsed = 0.0
    seen = 0
    for xb, yb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        logits = model(xb)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - started
        seen += len(yb)
        probabilities.extend(torch.softmax(logits, dim=1)[:, 0].cpu().tolist())
        truth.extend(yb.tolist())
    return truth, probabilities, 1000 * elapsed / max(seen, 1)


def run_experiment(
    spec: ModelSpec,
    rows: list[dict],
    data_root: Path,
    output: Path,
    *,
    epochs: int,
    warmup_epochs: int,
    batch_size: int,
    input_size: int,
    seed: int,
    workers: int,
) -> dict:
    seed_everything(seed)
    split_rows = {
        split: [row for row in rows if row.get("split") == split]
        for split in ("train", "val", "test")
    }
    loaders = {}
    for split, selected in split_rows.items():
        dataset = TablePairDataset(
            selected,
            data_root,
            spec.pair_mode,
            input_size,
            augment=split == "train",
            seed=seed,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0 and split == "train",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_pair_model(spec, len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    history: list[dict] = []
    best_score = -1.0
    best_state = None
    stale = 0

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in classifier_parameters(model, spec.pair_mode):
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=8e-4, weight_decay=1e-4
    )

    for epoch in range(epochs):
        if epoch == warmup_epochs:
            for parameter in model.parameters():
                parameter.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=2e-4)
        model.train()
        loss_total = 0.0
        for xb, yb, _ in loaders["train"]:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.item())
        truth, probability, latency = predict(model, loaders["val"], device)
        threshold, report = choose_threshold(truth, probability)
        score = report["macro_f1"] - 0.25 * report["unsafe_good_rate"]
        row = {
            "epoch": epoch + 1,
            "loss": loss_total / max(len(loaders["train"]), 1),
            "val": report,
            "latency_ms": latency,
        }
        history.append(row)
        improved = score > best_score + 1e-6
        if improved:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        print(
            f"{spec.architecture}/{spec.pair_mode} epoch={epoch + 1:02d} "
            f"loss={row['loss']:.4f} val_acc={report['accuracy']:.3f} "
            f"f1={report['macro_f1']:.3f} unsafe={report['unsafe_good_rate']:.3f} "
            f"threshold={threshold:.3f}{' *' if improved else ''}",
            flush=True,
        )
        if epoch >= warmup_epochs and stale >= 5:
            break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    val_truth, val_probability, _ = predict(model, loaders["val"], device)
    threshold, val_report = choose_threshold(val_truth, val_probability)
    test_truth, test_probability, latency = predict(model, loaders["test"], device)
    test_report = metrics(test_truth, test_probability, threshold)
    name = f"{spec.architecture}_{spec.pair_mode}"
    checkpoint = output / f"{name}.pt"
    torch.save(model.state_dict(), checkpoint)
    result = {
        "name": name,
        "architecture": spec.architecture,
        "pair_mode": spec.pair_mode,
        "input_size": input_size,
        "seed": seed,
        "classes": CLASSES,
        "splits": {key: len(value) for key, value in split_rows.items()},
        "validation": val_report,
        "test": test_report,
        "test_latency_ms": latency,
        "checkpoint": str(checkpoint),
        "history": history,
    }
    (output / f"{name}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "model_search")
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help="architecture:pair_mode; repeatable",
    )
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    experiments = args.experiment or [
        "mobilenet_v3_small:delta",
        "mobilenet_v3_small:siamese",
        "resnet18:siamese",
        "efficientnet_b0:siamese",
    ]
    results = []
    for value in experiments:
        architecture, pair_mode = value.split(":", 1)
        results.append(
            run_experiment(
                ModelSpec(architecture, pair_mode),
                rows,
                args.data,
                args.output,
                epochs=args.epochs,
                warmup_epochs=args.warmup_epochs,
                batch_size=args.batch_size,
                input_size=args.input_size,
                seed=args.seed,
                workers=args.workers,
            )
        )
    results.sort(
        key=lambda row: (
            row["test"]["macro_f1"],
            row["test"]["accuracy"],
            -row["test"]["unsafe_good_rate"],
        ),
        reverse=True,
    )
    summary = {
        "dataset": str(args.data),
        "ranking": [
            {
                "name": row["name"],
                "accuracy": row["test"]["accuracy"],
                "macro_f1": row["test"]["macro_f1"],
                "unsafe_good_rate": row["test"]["unsafe_good_rate"],
                "latency_ms": row["test_latency_ms"],
                "threshold_good": row["test"]["threshold_good"],
                "checkpoint": row["checkpoint"],
            }
            for row in results
        ],
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
