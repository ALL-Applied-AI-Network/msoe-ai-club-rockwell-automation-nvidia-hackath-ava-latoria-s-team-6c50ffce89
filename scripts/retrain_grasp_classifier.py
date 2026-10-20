#!/usr/bin/env python3
"""
Fine-tune grasp classifier after feedback / Isaac samples land in manifest.json.

Use after closed-loop or lab runs append rows via GraspFeedbackRecorder, or after
merging a new Isaac export into data/grasp_quality/.
"""

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

from stacking_recovery.data.grasp_dataset_store import load_manifest, resolve_data_root
from stacking_recovery.data.wrist_augment import augment_wrist_bgr
from stacking_recovery.vision.grasp_labels import CLASS_TO_INDEX, GRASP_CLASSES
from stacking_recovery.vision.grasp_quality_model import (
    build_mobilenet_v3,
    crop_roi,
    load_grasp_config,
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
    ) -> None:
        self.samples: list[tuple[torch.Tensor, int]] = []
        self.augment = augment
        self.aug_cfg = aug_cfg
        self.rng = random.Random(seed)
        for row in rows:
            img = cv2.imread(str(data_root / row["path"]))
            if img is None:
                raise FileNotFoundError(row["path"])
            crop = crop_roi(img, roi)
            if augment:
                crop = augment_wrist_bgr(crop, self.aug_cfg, rng=self.rng)
            self.samples.append((preprocess(crop, size), CLASS_TO_INDEX[row["label"]]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def split_manifest(manifest: list[dict], val_frac: float, seed: int) -> tuple[list, list]:
    rng = random.Random(seed)
    by_label: dict[str, list] = {c: [] for c in GRASP_CLASSES}
    for row in manifest:
        by_label[row["label"]].append(row)
    train, val = [], []
    for rows in by_label.values():
        rng.shuffle(rows)
        n_val = max(1, int(len(rows) * val_frac))
        val.extend(rows[:n_val])
        train.extend(rows[n_val:])
    return train, val


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
        "--finetune",
        action="store_true",
        help="load existing checkpoint and train with lower LR",
    )
    ap.add_argument("--fast", action="store_true", help="classifier head only")
    args = ap.parse_args()
    args.data = resolve_data_root(ROOT, args.data)
    cfg = load_grasp_config(args.config)
    manifest = load_manifest(args.data / "manifest.json")
    if not manifest:
        raise SystemExit(f"No manifest at {args.data / 'manifest.json'}")

    by_source: dict[str, int] = {}
    for row in manifest:
        src = row.get("source", "unknown")
        by_source[src] = by_source.get(src, 0) + 1
    print(f"manifest n={len(manifest)} by_source={json.dumps(by_source)}", flush=True)

    train_rows, val_rows = split_manifest(manifest, 0.15, 42)
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
    )
    val_ds = GraspDataset(val_rows, args.data, roi, size, augment=False)

    batch = int(cfg.get("training", {}).get("batch_size", 32))
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch)

    model = build_mobilenet_v3(len(GRASP_CLASSES))
    train_cfg = cfg.get("training", {})
    lr = float(train_cfg.get("lr", 1e-3))
    if args.finetune:
        ckpt = ROOT / cfg["model_path"]
        if not ckpt.is_file():
            raise SystemExit(f"--finetune requires {ckpt}")
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        lr = float(train_cfg.get("finetune_lr", lr * 0.1))

    if args.fast:
        for p in model.features.parameters():
            p.requires_grad = False

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    criterion = nn.CrossEntropyLoss()
    epochs = args.epochs or int(train_cfg.get("finetune_epochs", train_cfg.get("epochs", 8)))

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
        print(
            f"epoch {epoch + 1}/{epochs} loss={train_loss / max(len(train_loader), 1):.4f} val_acc={acc:.3f}",
            flush=True,
        )

    out_path = ROOT / cfg["model_path"]
    save_model_checkpoint(
        model,
        out_path,
        meta={
            "trained_at": time.time(),
            "n_train": len(train_rows),
            "finetune": args.finetune,
            "sources": by_source,
        },
    )
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
