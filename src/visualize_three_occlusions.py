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
    parser = argparse.ArgumentParser(description="Compare random, BO and sliding occlusions on one image.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/kvasir-seg"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/unet_best.pt"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--mask-size", type=int, default=48)
    parser.add_argument("--image-id", type=str, default=None)
    parser.add_argument("--random-csv", type=Path, default=Path("outputs/occlusion_runs/random_budget25_seed0.csv"))
    parser.add_argument("--bo-csv", type=Path, default=Path("outputs/occlusion_runs/bo_budget25_seed0.csv"))
    parser.add_argument("--sliding-csv", type=Path, default=Path("outputs/occlusion_runs/sliding_stride32_10.csv"))
    parser.add_argument("--random-budget", type=int, default=25)
    parser.add_argument("--bo-budget", type=int, default=25)
    parser.add_argument("--sliding-budget", type=int, default=None)
    parser.add_argument("--stats-path", type=Path, default=Path("outputs/metrics/train_channel_stats.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/figures/three_occlusions.png"))
    return parser.parse_args()


def load_fill_rgb(stats_path: Path) -> torch.Tensor:
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    return torch.tensor(stats["mean_rgb"], dtype=torch.float32)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def select_image_id(bo_csv: Path, image_id: str | None) -> str:
    if image_id is not None:
        return image_id

    rows = read_rows(bo_csv)
    if not rows:
        raise ValueError(f"No rows found in {bo_csv}")

    best = max(rows, key=lambda row: float(row["best_score"]))
    return best["image_id"]


def best_row_for_image(path: Path, image_id: str, budget: int | None) -> dict[str, str]:
    rows = [row for row in read_rows(path) if row["image_id"] == image_id]
    if budget is not None:
        rows = [row for row in rows if int(float(row["step"])) <= budget]

    if not rows:
        raise ValueError(f"No rows found in {path} for image_id={image_id}, budget={budget}")

    final_step = max(int(float(row["step"])) for row in rows)
    final_rows = [row for row in rows if int(float(row["step"])) == final_step]
    return final_rows[-1]


def find_dataset_index(dataset: KvasirSegDataset, image_id: str) -> int:
    for index, path in enumerate(dataset.image_paths):
        if path.stem == image_id:
            return index
    raise ValueError(f"Image id {image_id} not found in split {dataset.split}")


def tensor_to_rgb(image: torch.Tensor) -> Image.Image:
    array = (image.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def mask_to_rgb(mask: torch.Tensor, color: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
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


def draw_square(image: Image.Image, cx: int, cy: int, size: int, color: tuple[int, int, int]) -> Image.Image:
    output = image.copy()
    bounds = square_bounds(cx, cy, size, image.size[0])
    draw = ImageDraw.Draw(output)
    draw.rectangle([bounds.x0, bounds.y0, bounds.x1 - 1, bounds.y1 - 1], outline=color, width=4)
    return output


def label_panel(image: Image.Image, label: str, fill: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    label_height = 34
    output = Image.new("RGB", (image.width, image.height + label_height), color=fill)
    output.paste(image, (0, label_height))
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    draw.text((8, 10), label, fill=(0, 0, 0), font=font)
    return output


def make_spacer(width: int, height: int) -> Image.Image:
    return Image.new("RGB", (width, height), color=(255, 255, 255))


@torch.no_grad()
def main() -> None:
    args = parse_args()
    image_id = select_image_id(args.bo_csv, args.image_id)

    method_rows = [
        ("Random", (230, 60, 60), best_row_for_image(args.random_csv, image_id, args.random_budget)),
        ("BO-SegOcc", (60, 140, 255), best_row_for_image(args.bo_csv, image_id, args.bo_budget)),
        ("Sliding", (30, 180, 90), best_row_for_image(args.sliding_csv, image_id, args.sliding_budget)),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = KvasirSegDataset(args.data_root, args.split, image_size=args.image_size)
    sample = dataset[find_dataset_index(dataset, image_id)]
    image = sample["image"].to(device)
    gt = sample["mask"].to(device)
    fill_rgb = load_fill_rgb(args.stats_path).to(device)
    model = load_model(args.checkpoint, device)

    base_probability = torch.sigmoid(model(image.unsqueeze(0))).squeeze(0)
    original_rgb = tensor_to_rgb(image)

    rows: list[list[Image.Image]] = [
        [
            label_panel(original_rgb, f"image {image_id}"),
            label_panel(mask_to_rgb(gt), "ground truth"),
            label_panel(probability_to_rgb(base_probability), "prediction P0"),
            label_panel(make_spacer(args.image_size, args.image_size), ""),
        ]
    ]

    for method_name, color, row in method_rows:
        cx = int(float(row["best_cx"]))
        cy = int(float(row["best_cy"]))
        mask_size = int(float(row.get("best_mask_size") or row.get("mask_size") or args.mask_size))
        best_score = float(row["best_score"])

        occluded = apply_square_occlusion(image, cx, cy, mask_size, fill_rgb)
        occluded_probability = torch.sigmoid(model(occluded.unsqueeze(0))).squeeze(0)

        rows.append(
            [
                label_panel(
                    draw_square(original_rgb, cx, cy, mask_size, color),
                    f"{method_name}: best mask",
                ),
                label_panel(tensor_to_rgb(occluded), f"occluded ({cx}, {cy}, {mask_size})"),
                label_panel(probability_to_rgb(occluded_probability), "prediction Ptheta"),
                label_panel(difference_to_rgb(base_probability, occluded_probability), f"|P0-Ptheta| J={best_score:.3f}"),
            ]
        )

    panel_width = args.image_size
    panel_height = args.image_size + 34
    figure = Image.new("RGB", (panel_width * 4, panel_height * len(rows)), color=(255, 255, 255))

    for row_idx, row in enumerate(rows):
        for col_idx, panel in enumerate(row):
            figure.paste(panel, (col_idx * panel_width, row_idx * panel_height))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.save(args.output)
    print(f"Saved three-occlusion comparison to {args.output}")
    print(f"image_id={image_id}")


if __name__ == "__main__":
    main()
