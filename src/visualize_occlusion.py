from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.dataset import KvasirSegDataset
from src.evaluate import load_model
from src.occlusion import apply_square_occlusion, square_bounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a small occlusion visualization panel.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/kvasir-seg"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/unet_best.pt"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--mask-size", type=int, default=48)
    parser.add_argument("--cx", type=int, default=None)
    parser.add_argument("--cy", type=int, default=None)
    parser.add_argument("--image-id", type=str, default=None)
    parser.add_argument("--csv", type=Path, default=Path("outputs/occlusion_runs/bo_budget25_seed0.csv"))
    parser.add_argument("--stats-path", type=Path, default=Path("outputs/metrics/train_channel_stats.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/figures/occlusion_example.png"))
    return parser.parse_args()


def load_fill_rgb(stats_path: Path) -> torch.Tensor:
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    return torch.tensor(stats["mean_rgb"], dtype=torch.float32)


def best_location_from_csv(path: Path, image_id: str | None, default_size: int) -> tuple[str, int, int, int]:
    if not path.exists():
        if image_id is None:
            image_id = "0000"
        return image_id, 128, 128, default_size

    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    if image_id is not None:
        rows = [row for row in rows if row["image_id"] == image_id]

    if not rows:
        raise ValueError(f"No rows found in {path} for image_id={image_id}")

    best = max(rows, key=lambda row: float(row["best_score"]))
    size = int(float(best.get("best_mask_size") or best.get("mask_size") or default_size))
    return best["image_id"], int(float(best["best_cx"])), int(float(best["best_cy"])), size


def find_dataset_index(dataset: KvasirSegDataset, image_id: str) -> int:
    for index, path in enumerate(dataset.image_paths):
        if path.stem == image_id:
            return index
    raise ValueError(f"Image id {image_id} not found in split {dataset.split}")


def tensor_to_rgb(image: torch.Tensor) -> Image.Image:
    array = (image.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def mask_to_rgb(mask: torch.Tensor, color: tuple[int, int, int]) -> Image.Image:
    array = (mask.squeeze().cpu().numpy() >= 0.5).astype(np.uint8)
    rgb = np.zeros((array.shape[0], array.shape[1], 3), dtype=np.uint8)
    rgb[array > 0] = color
    return Image.fromarray(rgb, mode="RGB")


def probability_to_rgb(probability: torch.Tensor) -> Image.Image:
    array = (probability.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="L").convert("RGB")


def difference_to_rgb(a: torch.Tensor, b: torch.Tensor) -> Image.Image:
    diff = torch.abs(a - b).squeeze().cpu().numpy()
    diff = diff / max(float(diff.max()), 1e-7)
    red = (diff * 255).clip(0, 255).astype(np.uint8)
    rgb = np.zeros((diff.shape[0], diff.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = red
    rgb[:, :, 2] = 255 - red
    return Image.fromarray(rgb, mode="RGB")


def draw_square(image: Image.Image, cx: int, cy: int, size: int) -> Image.Image:
    output = image.copy()
    bounds = square_bounds(cx, cy, size, image.size[0])
    draw = ImageDraw.Draw(output)
    draw.rectangle([bounds.x0, bounds.y0, bounds.x1 - 1, bounds.y1 - 1], outline=(255, 0, 0), width=4)
    return output


def label_panel(image: Image.Image, label: str) -> Image.Image:
    label_height = 28
    output = Image.new("RGB", (image.width, image.height + label_height), color=(255, 255, 255))
    output.paste(image, (0, label_height))
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    draw.text((8, 8), label, fill=(0, 0, 0), font=font)
    return output


@torch.no_grad()
def main() -> None:
    args = parse_args()
    image_id, csv_cx, csv_cy, csv_size = best_location_from_csv(args.csv, args.image_id, args.mask_size)
    cx = args.cx if args.cx is not None else csv_cx
    cy = args.cy if args.cy is not None else csv_cy
    mask_size = args.mask_size if args.cx is not None or args.cy is not None else csv_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = KvasirSegDataset(args.data_root, args.split, image_size=args.image_size)
    sample = dataset[find_dataset_index(dataset, image_id)]

    image = sample["image"].to(device)
    gt = sample["mask"].to(device)
    fill_rgb = load_fill_rgb(args.stats_path).to(device)
    model = load_model(args.checkpoint, device)

    occluded = apply_square_occlusion(image, cx, cy, mask_size, fill_rgb)
    base_probability = torch.sigmoid(model(image.unsqueeze(0))).squeeze(0)
    occluded_probability = torch.sigmoid(model(occluded.unsqueeze(0))).squeeze(0)

    panels = [
        label_panel(draw_square(tensor_to_rgb(image), cx, cy, mask_size), "image + occlusion region"),
        label_panel(tensor_to_rgb(occluded), "occluded image"),
        label_panel(mask_to_rgb(gt, (255, 255, 255)), "ground truth"),
        label_panel(probability_to_rgb(base_probability), "prediction P0"),
        label_panel(probability_to_rgb(occluded_probability), "prediction Ptheta"),
        label_panel(difference_to_rgb(base_probability, occluded_probability), "|P0 - Ptheta|"),
    ]

    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    figure = Image.new("RGB", (width, height), color=(255, 255, 255))
    x = 0
    for panel in panels:
        figure.paste(panel, (x, 0))
        x += panel.width

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.save(args.output)
    print(f"Saved occlusion visualization to {args.output}")
    print(f"image_id={image_id}, cx={cx}, cy={cy}, mask_size={mask_size}")


if __name__ == "__main__":
    main()
