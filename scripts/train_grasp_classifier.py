#!/usr/bin/env python3
"""Train MobileNetV3 grasp-quality classifier from data/grasp_quality/manifest.json."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import cv2
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    raise SystemExit("Install ML deps: pip install -e '.[ml]'") from None

from stacking_recovery.data.wrist_augment import augment_wrist_bgr
from stacking_recovery.data.grasp_dataset_store import resolve_data_root
from stacking_recovery.data.dataset_split import split_manifest_by_seed
from stacking_recovery.vision.grasp_quality_model import (
    build_mobilenet_v3,
    configured_classes,
    crop_for_observation,
    load_grasp_config,
    normalize_grasp_label,
    preprocess,
    save_model_checkpoint,
)


class GraspDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        data_root: Path,
        roi: dict,
        size: int,
        *,
        augment: bool = False,
        aug_cfg: dict | None = None,
        seed: int = 0,
        config: dict | None = None,
    ) -> None:
        self.samples: list[tuple[torch.Tensor, int]] = []
        rng = random.Random(seed)
        config = config or {}
        class_to_index = {
            label: index for index, label in enumerate(configured_classes(config))
        }
        for row in rows:
            img = cv2.imread(str(data_root / row["path"]))
            if img is None:
                raise FileNotFoundError(row["path"])
            crop = crop_for_observation(img, roi, row.get("observation_source"))
            if augment:
                crop = augment_wrist_bgr(crop, aug_cfg, rng=rng)
            label = normalize_grasp_label(row["label"], config)
            if label not in class_to_index:
                raise ValueError(f"label {label!r} is not in configured classes")
            self.samples.append((preprocess(crop, size), class_to_index[label]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Dataset root (default: GRASP_DATA_ROOT or repo/data/grasp_quality)",
    )
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "grasp_model.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument(
        "--fast",
        action="store_true",
        help="train classifier head only (much faster on CPU)",
    )
    ap.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="stop if val_acc does not improve for N epochs (default from config or off)",
    )
    ap.add_argument(
        "--observation-source",
        default=None,
        help="Train only one observation format, e.g. table_before_after",
    )
    args = ap.parse_args()
    args.data = resolve_data_root(ROOT, args.data)
    cfg = load_grasp_config(args.config)
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    if args.observation_source:
        manifest = [
            row
            for row in manifest
            if row.get("observation_source") == args.observation_source
        ]
    train_rows, val_rows, test_rows = split_manifest_by_seed(manifest)
    if not train_rows or not val_rows:
        raise SystemExit(
            "Dataset needs non-empty train and val seed splits. Collect multiple seeds "
            "or add explicit split fields to manifest.json."
        )
    print(
        f"dataset split: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}",
        flush=True,
    )
    roi = cfg["roi"]
    size = int(cfg["input_size"])
    aug_cfg = cfg.get("augmentation") or {}
    train_ds = GraspDataset(
        train_rows,
        args.data,
        roi,
        size,
        augment=True,
        aug_cfg=aug_cfg,
        seed=42,
        config=cfg,
    )
    val_ds = GraspDataset(val_rows, args.data, roi, size, augment=False, config=cfg)
    batch = int(cfg.get("training", {}).get("batch_size", 32))
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch)

    classes = configured_classes(cfg)
    model = build_mobilenet_v3(len(classes))
    if args.fast:
        for p in model.features.parameters():
            p.requires_grad = False
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(cfg.get("training", {}).get("lr", 1e-3)),
        weight_decay=float(cfg.get("training", {}).get("weight_decay", 1e-4)),
    )
    criterion = nn.CrossEntropyLoss()
    train_cfg = cfg.get("training", {})
    epochs = args.epochs or int(train_cfg.get("epochs", 15))
    if args.early_stop_patience is not None:
        patience = args.early_stop_patience
    elif train_cfg.get("early_stop_patience") is not None:
        patience = int(train_cfg["early_stop_patience"])
    else:
        patience = 0

    best_acc = -1.0
    best_state: dict | None = None
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            train_loss += float(loss.item())
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb).argmax(dim=1)
                correct += int((pred == yb).sum())
                total += len(yb)
        acc = correct / total if total else 0.0
        improved = acc > best_acc
        if improved:
            best_acc = acc
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        elif patience > 0:
            stale_epochs += 1
        mark = " *" if improved else ""
        print(
            f"epoch {epoch + 1}/{epochs} loss={train_loss / max(len(train_loader), 1):.4f} val_acc={acc:.3f}{mark}",
            flush=True,
        )
        if patience > 0 and stale_epochs >= patience:
            print(
                f"early stop: no val_acc improvement for {patience} epochs (best={best_acc:.3f} @ epoch {best_epoch})",
                flush=True,
            )
            break

    out_path = ROOT / cfg["model_path"]
    if best_state is not None:
        model.load_state_dict(best_state)
    save_model_checkpoint(
        model,
        out_path,
        meta={
            "trained_at": time.time(),
            "n_train": len(train_rows),
            "source": "mixed",
            "best_val_acc": best_acc,
            "best_epoch": best_epoch,
            "n_val": len(val_rows),
            "n_test_held_out": len(test_rows),
            "classes": classes,
            "label_mode": cfg.get("label_mode", "detailed"),
            "config": str(args.config),
        },
    )
    print(f"Saved {out_path} (best val_acc={best_acc:.3f} at epoch {best_epoch})")


if __name__ == "__main__":
    main()
