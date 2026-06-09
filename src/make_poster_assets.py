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
from src.experiment_io import load_fill_rgb
from src.occlusion import apply_square_occlusion, square_bounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create poster-ready qualitative and quantitative BO-SegOcc figures.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/kvasir-seg"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/unet_best.pt"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--bo-csv", type=Path, default=Path("outputs/occlusion_runs/bo_variable_size_pilot.csv"))
    parser.add_argument("--stats-path", type=Path, default=Path("outputs/metrics/train_channel_stats.json"))
    parser.add_argument("--fixed-summary", type=Path, default=Path("outputs/metrics/occlusion_comparison_fixed_size_pilot.json"))
    parser.add_argument("--variable-summary", type=Path, default=Path("outputs/metrics/occlusion_comparison_variable_size_pilot.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures/poster"))
    parser.add_argument("--image-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def normalize_image_id(image_id: str | None) -> str | None:
    if image_id is None:
        return None
    stripped = image_id.strip()
    if stripped.isdigit():
        return f"{int(stripped):04d}"
    return stripped


def select_best_row(path: Path, image_id: str | None, seed: int | None) -> dict[str, str]:
    rows = read_rows(path)
    image_id = normalize_image_id(image_id)
    if image_id is not None:
        rows = [row for row in rows if row["image_id"] == image_id]
    if seed is not None:
        rows = [row for row in rows if int(float(row["seed"])) == seed]
    if not rows:
        available = sorted({row["image_id"] for row in read_rows(path)})
        preview = ", ".join(available[:20])
        raise ValueError(
            f"No matching rows in {path} for image_id={image_id!r}, seed={seed}. "
            f"Available image IDs: {preview}"
        )
    return max(rows, key=lambda row: float(row["best_score"]))


def find_dataset_index(dataset: KvasirSegDataset, image_id: str) -> int:
    for index, path in enumerate(dataset.image_paths):
        if path.stem == image_id:
            return index
    raise ValueError(f"Image id {image_id} not found in split {dataset.split}")


def tensor_to_rgb(image: torch.Tensor) -> Image.Image:
    array = (image.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def probability_to_heat(probability: torch.Tensor, color: tuple[int, int, int]) -> Image.Image:
    values = probability.detach().cpu().squeeze().numpy()
    values = (values * 255).clip(0, 255).astype(np.uint8)
    rgb = np.zeros((values.shape[0], values.shape[1], 3), dtype=np.uint8)
    for channel, component in enumerate(color):
        rgb[:, :, channel] = (values.astype(np.float32) * component / 255.0).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def difference_to_rgb(base: torch.Tensor, degraded: torch.Tensor) -> Image.Image:
    diff = torch.abs(base - degraded).detach().cpu().squeeze().numpy()
    diff = diff / max(float(diff.max()), 1e-7)
    red = (diff * 255).clip(0, 255).astype(np.uint8)
    rgb = np.zeros((diff.shape[0], diff.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = red
    rgb[:, :, 1] = (red * 0.25).astype(np.uint8)
    rgb[:, :, 2] = 255 - red
    return Image.fromarray(rgb, mode="RGB")


def draw_occlusion_box(image: Image.Image, cx: int, cy: int, size: int) -> Image.Image:
    output = image.copy()
    bounds = square_bounds(cx, cy, size, image.width)
    draw = ImageDraw.Draw(output)
    draw.rectangle([bounds.x0, bounds.y0, bounds.x1 - 1, bounds.y1 - 1], outline=(255, 210, 0), width=5)
    return output


def labeled_panel(image: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    label_height = 56
    output = Image.new("RGB", (image.width, image.height + label_height), color=(255, 255, 255))
    output.paste(image, (0, label_height))
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    draw.text((10, 8), title, fill=(0, 0, 0), font=font)
    if subtitle:
        draw.text((10, 30), subtitle, fill=(70, 70, 70), font=font)
    return output


def save_individual_poster_images(
    output_dir: Path,
    image_id: str,
    seed: int,
    images: dict[str, Image.Image],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, image in images.items():
        path = output_dir / f"poster_{name}_{image_id}_seed{seed}.png"
        image.save(path)
        paths[name] = path
    return paths


@torch.no_grad()
def make_qualitative_figure(args: argparse.Namespace) -> tuple[Path, dict[str, object], dict[str, Path]]:
    row = select_best_row(args.bo_csv, args.image_id, args.seed)
    image_id = row["image_id"]
    seed = int(float(row["seed"]))
    cx = int(float(row["best_cx"]))
    cy = int(float(row["best_cy"]))
    size = int(float(row.get("best_mask_size") or row.get("mask_size") or 48))
    score = float(row["best_score"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = KvasirSegDataset(args.data_root, args.split, image_size=args.image_size)
    sample = dataset[find_dataset_index(dataset, image_id)]
    image = sample["image"].to(device)
    fill_rgb = load_fill_rgb(args.stats_path).to(device)
    model = load_model(args.checkpoint, device)

    baseline = torch.sigmoid(model(image.unsqueeze(0))).squeeze(0)
    occluded = apply_square_occlusion(image, cx, cy, size, fill_rgb)
    degraded = torch.sigmoid(model(occluded.unsqueeze(0))).squeeze(0)

    original_rgb = tensor_to_rgb(image)
    individual_images = {
        "original_image": original_rgb,
        "occluded_image": tensor_to_rgb(occluded),
        "baseline_prediction": probability_to_heat(baseline, (0, 180, 255)),
        "degraded_prediction": probability_to_heat(degraded, (255, 60, 60)),
        "prediction_change": difference_to_rgb(baseline, degraded),
    }
    individual_paths = save_individual_poster_images(args.output_dir, image_id, seed, individual_images)

    panels = [
        labeled_panel(individual_images["original_image"], "Original Image"),
        labeled_panel(individual_images["occluded_image"], "Occluded Image", f"BO mask: ({cx}, {cy}), size={size}"),
        labeled_panel(individual_images["baseline_prediction"], "Baseline Prediction", "P0"),
        labeled_panel(individual_images["degraded_prediction"], "Degraded Prediction", f"Ptheta, J={score:.3f}"),
        labeled_panel(individual_images["prediction_change"], "Prediction Change", "|P0 - Ptheta|"),
    ]

    gap = 14
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    height = max(panel.height for panel in panels)
    figure = Image.new("RGB", (width, height), color=(255, 255, 255))
    x = 0
    for panel in panels:
        figure.paste(panel, (x, 0))
        x += panel.width + gap

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"poster_qualitative_degradation_{image_id}_seed{seed}.png"
    figure.save(output)
    return output, {"image_id": image_id, "seed": seed, "cx": cx, "cy": cy, "size": size, "score": score}, individual_paths


def load_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_bar_chart(data: list[dict[str, object]], output: Path) -> None:
    width = 1800
    height = 1200
    margin_left = 170
    margin_right = 80
    margin_top = 190
    margin_bottom = 250
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    y_max = max(float(row["value"]) + float(row["std"]) for row in data) * 1.18

    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    title_font = load_font(58, bold=True)
    subtitle_font = load_font(31)
    tick_font = load_font(26)
    label_font = load_font(32, bold=True)
    meta_font = load_font(25)
    value_font = load_font(34, bold=True)

    draw.text((margin_left, 54), "Occlusion Search Effectiveness", fill=(0, 0, 0), font=title_font)
    draw.text(
        (margin_left, 120),
        "Mean best degradation score, J = 1 - SoftDice(P0, Ptheta). Higher is stronger.",
        fill=(75, 75, 75),
        font=subtitle_font,
    )

    axis_x = margin_left
    axis_y = margin_top + plot_height
    draw.line((axis_x, margin_top, axis_x, axis_y), fill=(20, 20, 20), width=3)
    draw.line((axis_x, axis_y, width - margin_right, axis_y), fill=(20, 20, 20), width=3)

    for tick in np.linspace(0, y_max, 5):
        y = axis_y - int((tick / y_max) * plot_height)
        draw.line((axis_x, y, width - margin_right, y), fill=(228, 228, 228), width=2)
        draw.text((72, y - 16), f"{tick:.2f}", fill=(70, 70, 70), font=tick_font)

    colors = {
        "random": (214, 87, 80),
        "sliding": (86, 154, 104),
        "fixed": (77, 132, 196),
        "variable": (122, 88, 176),
    }
    slot = plot_width / len(data)
    bar_width = int(slot * 0.48)

    for idx, row in enumerate(data):
        label = str(row["label"])
        value = float(row["value"])
        std = float(row["std"])
        kind = str(row["kind"])
        queries = int(row["queries"])
        elapsed = float(row["elapsed"])

        center = axis_x + int(slot * (idx + 0.5))
        bar_height = int((value / y_max) * plot_height)
        x0 = center - bar_width // 2
        x1 = center + bar_width // 2
        y0 = axis_y - bar_height

        draw.rounded_rectangle((x0, y0, x1, axis_y), radius=16, fill=colors[kind])

        err_top = axis_y - int((min(value + std, y_max) / y_max) * plot_height)
        err_bottom = axis_y - int((max(value - std, 0.0) / y_max) * plot_height)
        draw.line((center, err_top, center, err_bottom), fill=(45, 45, 45), width=3)
        draw.line((center - 18, err_top, center + 18, err_top), fill=(45, 45, 45), width=3)
        draw.line((center - 18, err_bottom, center + 18, err_bottom), fill=(45, 45, 45), width=3)

        draw.text((center - 66, y0 - 48), f"{value:.3f}", fill=(0, 0, 0), font=value_font)
        draw.text((center - 70, axis_y + 24), label, fill=(0, 0, 0), font=label_font)
        draw.text((center - 84, axis_y + 70), f"{queries} asks", fill=(80, 80, 80), font=meta_font)
        draw.text((center - 78, axis_y + 104), f"{elapsed:.1f}s mean", fill=(80, 80, 80), font=meta_font)

    draw.text(
        (margin_left, height - 62),
        "Ask count excludes the baseline P0 prediction. Error bars show standard deviation over 30 image-seed pairs.",
        fill=(80, 80, 80),
        font=meta_font,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def method_plot_data(args: argparse.Namespace) -> list[dict[str, object]]:
    fixed = load_summary(args.fixed_summary)
    variable = load_summary(args.variable_summary)
    random_queries = int(variable["inputs"]["random_budget"])
    bo_queries = int(variable["inputs"]["bo_budget"])
    sliding_queries = max(int(float(row["step"])) for row in read_rows(Path(variable["inputs"]["sliding_csv"])))
    return [
        {
            "label": "Random",
            "value": float(variable["random_budget"]["best_score_mean"]),
            "std": float(variable["random_budget"]["best_score_std"]),
            "kind": "random",
            "queries": random_queries,
            "elapsed": float(variable["random_budget"]["elapsed_sec_mean"]),
        },
        {
            "label": "Sliding full",
            "value": float(variable["sliding_full"]["best_score_mean"]),
            "std": float(variable["sliding_full"]["best_score_std"]),
            "kind": "sliding",
            "queries": sliding_queries,
            "elapsed": float(variable["sliding_full"]["elapsed_sec_mean"]),
        },
        {
            "label": "BO fixed",
            "value": float(fixed["bo_budget"]["best_score_mean"]),
            "std": float(fixed["bo_budget"]["best_score_std"]),
            "kind": "fixed",
            "queries": bo_queries,
            "elapsed": float(fixed["bo_budget"]["elapsed_sec_mean"]),
        },
        {
            "label": "BO variable",
            "value": float(variable["bo_budget"]["best_score_mean"]),
            "std": float(variable["bo_budget"]["best_score_std"]),
            "kind": "variable",
            "queries": bo_queries,
            "elapsed": float(variable["bo_budget"]["elapsed_sec_mean"]),
        },
    ]


def make_comparison_chart(args: argparse.Namespace) -> tuple[Path, list[dict[str, object]]]:
    data = method_plot_data(args)
    output = args.output_dir / "poster_method_comparison.png"
    draw_bar_chart(data, output)
    return output, data


def write_results_snippet(
    args: argparse.Namespace,
    example: dict[str, object],
    qualitative: Path,
    chart: Path,
    individual_paths: dict[str, Path],
    plot_data: list[dict[str, object]],
) -> Path:
    fixed = load_summary(args.fixed_summary)
    variable = load_summary(args.variable_summary)
    test_metrics = json.loads(Path("outputs/metrics/test_metrics.json").read_text(encoding="utf-8"))

    text = f"""# Poster Results Snippet

## Recommended Figures

- Qualitative degradation panel: `{qualitative}`
- Method comparison chart: `{chart}`
- BO saliency overlays: `outputs/saliency_maps/bo_variable_size/variable_size/`

Separate qualitative images:

```text
Original Image        {individual_paths['original_image']}
Occluded Image        {individual_paths['occluded_image']}
Baseline Prediction   {individual_paths['baseline_prediction']}
Degraded Prediction   {individual_paths['degraded_prediction']}
Prediction Change     {individual_paths['prediction_change']}
```

## Qualitative Example

Selected image `{example['image_id']}`, seed `{example['seed']}`.

Best BO variable-size mask:

```text
cx={example['cx']}
cy={example['cy']}
size={example['size']}
J={example['score']:.3f}
```

## Segmentation Quality

The frozen U-Net reached:

```text
Dice      {test_metrics['dice']:.3f}
IoU       {test_metrics['iou']:.3f}
Precision {test_metrics['precision']:.3f}
Recall    {test_metrics['recall']:.3f}
```

## Occlusion Results

Pilot setup: 10 test images, 3 seeds, 25-query budget for random and BO.

Mean best degradation score `J = 1 - SoftDice(P0, Ptheta)`:

```text
Random occlusion   {variable['random_budget']['best_score_mean']:.3f} ± {variable['random_budget']['best_score_std']:.3f}
Sliding full       {variable['sliding_full']['best_score_mean']:.3f} ± {variable['sliding_full']['best_score_std']:.3f}
BO fixed-size      {fixed['bo_budget']['best_score_mean']:.3f} ± {fixed['bo_budget']['best_score_std']:.3f}
BO variable-size   {variable['bo_budget']['best_score_mean']:.3f} ± {variable['bo_budget']['best_score_std']:.3f}
```

Model asks and runtime:

```text
{"Method":<16} {"Asks":>6} {"Mean time":>12}
{str(plot_data[0]['label']):<16} {int(plot_data[0]['queries']):>6} {float(plot_data[0]['elapsed']):>9.1f}s
{str(plot_data[1]['label']):<16} {int(plot_data[1]['queries']):>6} {float(plot_data[1]['elapsed']):>9.1f}s
{str(plot_data[2]['label']):<16} {int(plot_data[2]['queries']):>6} {float(plot_data[2]['elapsed']):>9.1f}s
{str(plot_data[3]['label']):<16} {int(plot_data[3]['queries']):>6} {float(plot_data[3]['elapsed']):>9.1f}s
```

Pairwise result:

```text
BO variable-size vs random:
mean delta {variable['pairwise_delta_best_score']['bo_budget_minus_random_budget']['mean']:.3f}
wins       {variable['pairwise_delta_best_score']['bo_budget_minus_random_budget']['wins']}/{variable['pairwise_delta_best_score']['bo_budget_minus_random_budget']['n']}

BO variable-size vs sliding full:
mean delta {-variable['pairwise_delta_best_score']['sliding_full_minus_bo_budget']['mean']:.3f}
wins       {variable['pairwise_delta_best_score']['sliding_full_minus_bo_budget']['losses']}/{variable['pairwise_delta_best_score']['sliding_full_minus_bo_budget']['n']}
```

## Poster Text

Bayesian optimization identified occlusions that produced stronger degradation of the frozen segmentation model than random search under the same query budget. The variable-size BO variant achieved the highest mean degradation score and won against random occlusion in all image-seed pairs in the pilot experiment. Ground-truth masks were not used during optimization; they were used only for post-hoc spatial evaluation.
"""
    output = args.output_dir / "poster_results_snippet.md"
    output.write_text(text, encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    qualitative, example, individual_paths = make_qualitative_figure(args)
    chart, plot_data = make_comparison_chart(args)
    snippet = write_results_snippet(args, example, qualitative, chart, individual_paths, plot_data)
    print(f"Saved qualitative figure to {qualitative}")
    for name, path in individual_paths.items():
        print(f"Saved {name} to {path}")
    print(f"Saved comparison chart to {chart}")
    print(f"Saved results snippet to {snippet}")


if __name__ == "__main__":
    main()
