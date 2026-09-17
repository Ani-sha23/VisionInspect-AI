"""Training entry point for the multi-label U-Net.

Run this where a GPU is available (Kaggle/Colab) and copy the resulting
checkpoint back into models/. torch is imported inside the functions so the rest
of the project stays usable without it.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

from .config_loader import Config
from .dataset import build_image_index, load_annotations
from .exceptions import DataError, ModelError
from .logger import get_logger
from .model import build_unet, resolve_device
from .rle import build_multilabel_mask

logger = get_logger(__name__)


def _dataset_class():
    try:
        from torch.utils.data import Dataset
    except Exception as exc:  # pragma: no cover
        raise ModelError("PyTorch is required for training. Install it with: pip install torch") from exc

    class SteelDataset(Dataset):
        """Yields (image CHW float32, mask CxHxW float32) pairs."""

        def __init__(self, image_ids, index, images_dir, height, width, mean, std, augment=False):
            self.image_ids = list(image_ids)
            self.index = index
            self.images_dir = images_dir
            self.height, self.width = int(height), int(width)
            self.mean = np.asarray(mean, dtype=np.float32)
            self.std = np.asarray(std, dtype=np.float32)
            self.augment = augment

        def __len__(self):
            return len(self.image_ids)

        def __getitem__(self, idx):
            image_id = self.image_ids[idx]
            path = os.path.join(self.images_dir, image_id)
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                raise DataError(f"Training image unreadable: {path}")
            original_h, original_w = image.shape[:2]
            mask = build_multilabel_mask(self.index.get(image_id, {}), (original_h, original_w))

            if (original_h, original_w) != (self.height, self.width):
                image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
                mask = np.stack([
                    cv2.resize(m, (self.width, self.height), interpolation=cv2.INTER_NEAREST) for m in mask
                ])

            if self.augment and np.random.rand() < 0.5:
                image = np.ascontiguousarray(image[:, ::-1])
                mask = np.ascontiguousarray(mask[:, :, ::-1])
            if self.augment and np.random.rand() < 0.5:
                image = np.ascontiguousarray(image[::-1])
                mask = np.ascontiguousarray(mask[:, ::-1])

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb = (rgb - self.mean) / self.std
            return np.transpose(rgb, (2, 0, 1)).copy(), mask.astype(np.float32)

    return SteelDataset


def dice_loss(logits, targets, eps: float = 1e-6):
    import torch
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    intersection = (probs * targets).sum(dims)
    cardinality = probs.sum(dims) + targets.sum(dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


def validate(model, loader, device) -> Tuple[float, float]:
    import torch
    import torch.nn as nn

    model.eval()
    bce = nn.BCEWithLogitsLoss()
    losses: List[float] = []
    dices: List[float] = []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = bce(logits, masks) + dice_loss(logits, masks)
            losses.append(float(loss.item()))
            preds = (torch.sigmoid(logits) >= 0.5).float()
            intersection = (preds * masks).sum((1, 2, 3))
            union = preds.sum((1, 2, 3)) + masks.sum((1, 2, 3))
            dice = torch.where(union > 0, (2 * intersection) / union, torch.ones_like(union))
            dices.append(float(dice.mean().item()))
    return float(np.mean(losses)), float(np.mean(dices))


def train(cfg: Config, splits_dir: str, epochs: int | None = None, limit: int | None = None) -> Dict:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ModelError(
            "PyTorch is required for training but is not installed. Install it with "
            "`pip install torch`, or train on Kaggle/Colab where it is preinstalled "
            "(see docs/TRAINING.md)."
        ) from exc

    annotations = load_annotations(cfg.get("data.train_csv"))
    index = build_image_index(annotations)
    images_dir = cfg.get("data.images_dir")

    def read_split(name: str) -> List[str]:
        path = os.path.join(splits_dir, f"{name}.csv")
        if not os.path.isfile(path):
            raise DataError(f"Split file missing: {path}. Run `prepare-data` first.")
        ids = pd.read_csv(path)["ImageId"].astype(str).tolist()
        return ids[:limit] if limit else ids

    SteelDataset = _dataset_class()
    height = cfg.get("preprocessing.target_height")
    width = cfg.get("preprocessing.target_width")
    mean = cfg.get("preprocessing.normalize.mean")
    std = cfg.get("preprocessing.normalize.std")

    train_ds = SteelDataset(read_split("train"), index, images_dir, height, width, mean, std, augment=True)
    val_ds = SteelDataset(read_split("val"), index, images_dir, height, width, mean, std, augment=False)

    batch_size = int(cfg.get("train.batch_size"))
    workers = int(cfg.get("train.num_workers"))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers)

    device = resolve_device(cfg.get("model.device"))
    model = build_unet(
        int(cfg.get("model.in_channels")),
        int(cfg.get("model.num_classes")),
        int(cfg.get("model.base_filters")),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.get("train.lr")), weight_decay=float(cfg.get("train.weight_decay"))
    )
    bce = nn.BCEWithLogitsLoss()
    total_epochs = int(epochs or cfg.get("train.epochs"))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs))

    checkpoint_dir = cfg.get("train.checkpoint_dir")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = cfg.get("model.weights")
    best_dice = -1.0
    patience = int(cfg.get("train.early_stopping_patience"))
    stale = 0
    history: List[Dict] = []

    logger.info("Training on %d images, validating on %d, device=%s", len(train_ds), len(val_ds), device)

    for epoch in range(1, total_epochs + 1):
        model.train()
        started = time.perf_counter()
        running: List[float] = []
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = bce(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            running.append(float(loss.item()))

        scheduler.step()
        val_loss, val_dice = validate(model, val_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": round(float(np.mean(running)), 4),
            "val_loss": round(val_loss, 4),
            "val_dice": round(val_dice, 4),
            "seconds": round(time.perf_counter() - started, 1),
        }
        history.append(record)
        logger.info("epoch %d | train_loss %.4f | val_loss %.4f | val_dice %.4f | %.1fs",
                    epoch, record["train_loss"], record["val_loss"], record["val_dice"], record["seconds"])

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({"model_state": model.state_dict(), "val_dice": best_dice, "epoch": epoch}, best_path)
            logger.info("New best checkpoint saved to %s (val_dice=%.4f)", best_path, best_dice)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                logger.info("Early stopping after %d epochs without improvement.", patience)
                break

    return {"best_val_dice": round(best_dice, 4), "epochs_run": len(history), "history": history,
            "checkpoint": best_path}
