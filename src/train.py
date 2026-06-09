from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import KvasirSegDataset
from src.metrics import BCEDiceLoss, average_metric_sums, merge_metric_sums, segmentation_metrics
from src.model import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a segmentation_models_pytorch U-Net on Kvasir-SEG.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/kvasir-seg"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder-name", type=str, default="efficientnet-b0")
    parser.add_argument("--encoder-weights", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--metrics-path", type=Path, default=Path("outputs/metrics/train_history.csv"))
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(
    data_root: Path,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    augment: bool,
    max_samples: int | None,
    seed: int,
) -> DataLoader:
    dataset = KvasirSegDataset(
        root=data_root,
        split=split,
        image_size=image_size,
        augment=augment,
        max_samples=max_samples,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def train_one_epoch(
    model: UNet,
    loader: DataLoader,
    criterion: BCEDiceLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    metric_sums: dict[str, float] = {}
    total = 0

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        total_loss += float(loss.item()) * batch_size
        merge_metric_sums(metric_sums, segmentation_metrics(logits.detach(), masks), batch_size)
        total += batch_size

    metrics = average_metric_sums(metric_sums, total)
    metrics["loss"] = total_loss / total
    return metrics


@torch.no_grad()
def evaluate(
    model: UNet,
    loader: DataLoader,
    criterion: BCEDiceLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    metric_sums: dict[str, float] = {}
    total = 0

    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = criterion(logits, masks)

        batch_size = images.shape[0]
        total_loss += float(loss.item()) * batch_size
        merge_metric_sums(metric_sums, segmentation_metrics(logits, masks), batch_size)
        total += batch_size

    metrics = average_metric_sums(metric_sums, total)
    metrics["loss"] = total_loss / total
    return metrics


def write_history_row(path: Path, row: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path: Path, model: UNet, args: argparse.Namespace, epoch: int, val_metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "val_metrics": val_metrics,
            "model_args": {
                "encoder_name": args.encoder_name,
                "encoder_weights": args.encoder_weights,
            },
            "training_args": {
                "image_size": args.image_size,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "seed": args.seed,
                "max_train_samples": args.max_train_samples,
                "max_val_samples": args.max_val_samples,
            },
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = build_loader(
        args.data_root,
        "train",
        args.image_size,
        args.batch_size,
        args.num_workers,
        augment=True,
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    val_loader = build_loader(
        args.data_root,
        "validation",
        args.image_size,
        args.batch_size,
        args.num_workers,
        augment=False,
        max_samples=args.max_val_samples,
        seed=args.seed + 1,
    )

    model = UNet(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
    ).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_dice = -1.0
    best_path = args.checkpoint_dir / "unet_best.pt"
    args.metrics_path.unlink(missing_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        write_history_row(args.metrics_path, row)

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            save_checkpoint(best_path, model, args, epoch, val_metrics)

        print(json.dumps(row, indent=2))

    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
