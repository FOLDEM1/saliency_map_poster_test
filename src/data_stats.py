from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import KvasirSegDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute RGB channel stats for Kvasir-SEG.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/kvasir-seg"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics/train_channel_stats.json"))
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = KvasirSegDataset(
        root=args.data_root,
        split=args.split,
        image_size=args.image_size,
        augment=False,
        max_samples=args.max_samples,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sum_sq = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0

    for batch in tqdm(loader, desc=f"stats {args.split}"):
        images = batch["image"].double()
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_sum_sq += (images * images).sum(dim=(0, 2, 3))
        pixel_count += images.shape[0] * images.shape[2] * images.shape[3]

    mean = channel_sum / pixel_count
    variance = channel_sum_sq / pixel_count - mean * mean
    std = torch.sqrt(torch.clamp(variance, min=0.0))

    stats = {
        "split": args.split,
        "image_size": args.image_size,
        "num_images": len(dataset),
        "mean_rgb": [float(value) for value in mean],
        "std_rgb": [float(value) for value in std],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
