from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import KvasirSegDataset
from src.metrics import average_metric_sums, merge_metric_sums, segmentation_metrics
from src.model import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained U-Net checkpoint.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/kvasir-seg"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/unet_best.pt"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--metrics-path", type=Path, default=Path("outputs/metrics/test_metrics.json"))
    parser.add_argument("--predictions-dir", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--save-examples", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> UNet:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint.get("model_args", {})
    model = UNet(
        encoder_name=model_args.get("encoder_name", "resnet34"),
        encoder_weights=None,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def save_prediction_examples(
    images: torch.Tensor,
    masks: torch.Tensor,
    probabilities: torch.Tensor,
    image_ids: list[str],
    output_dir: Path,
    max_to_save: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for image, mask, probability, image_id in zip(images, masks, probabilities, image_ids):
        if saved >= max_to_save:
            break

        image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        mask_np = (mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        pred_np = ((probability.squeeze(0).cpu().numpy() >= 0.5) * 255).astype(np.uint8)

        overlay = image_np.copy()
        overlay[pred_np > 0] = (0.6 * overlay[pred_np > 0] + np.array([255, 0, 0]) * 0.4).astype(np.uint8)

        panel = np.concatenate(
            [
                image_np,
                np.repeat(mask_np[:, :, None], 3, axis=2),
                np.repeat(pred_np[:, :, None], 3, axis=2),
                overlay,
            ],
            axis=1,
        )
        Image.fromarray(panel).save(output_dir / f"{image_id}_panel.png")
        saved += 1

    return saved


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = KvasirSegDataset(
        root=args.data_root,
        split=args.split,
        image_size=args.image_size,
        augment=False,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model = load_model(args.checkpoint, device)

    metric_sums: dict[str, float] = {}
    total = 0
    saved_examples = 0

    for batch in tqdm(loader, desc=f"eval {args.split}"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        probabilities = torch.sigmoid(logits)

        batch_size = images.shape[0]
        merge_metric_sums(metric_sums, segmentation_metrics(logits, masks), batch_size)
        total += batch_size

        if saved_examples < args.save_examples:
            remaining = args.save_examples - saved_examples
            saved_examples += save_prediction_examples(
                images.cpu(),
                masks.cpu(),
                probabilities.cpu(),
                list(batch["image_id"]),
                args.predictions_dir / args.split,
                remaining,
            )

    metrics = average_metric_sums(metric_sums, total)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
