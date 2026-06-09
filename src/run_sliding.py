from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from src.dataset import KvasirSegDataset
from src.evaluate import load_model
from src.experiment_io import load_fill_rgb, prune_incomplete_groups, write_rows, write_run_metadata
from src.occlusion import mask_to_binary, occlusion_objective, overlap_with_gt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sliding window occlusion on a trained segmenter.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/kvasir-seg"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/unet_best.pt"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--mask-size", type=int, default=48)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--stats-path", type=Path, default=Path("outputs/metrics/train_channel_stats.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/occlusion_runs/sliding.csv"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip images already completed in the output CSV.")
    return parser.parse_args()


def grid_centers(image_size: int, mask_size: int, stride: int) -> list[tuple[int, int]]:
    half = mask_size // 2
    min_center = half
    max_center = image_size - half

    coords = list(range(min_center, max_center + 1, stride))
    if coords[-1] != max_center:
        coords.append(max_center)

    return [(cx, cy) for cy in coords for cx in coords]


@torch.no_grad()
def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    dataset = KvasirSegDataset(
        root=args.data_root,
        split=args.split,
        image_size=args.image_size,
        augment=False,
        max_samples=args.max_samples,
    )
    fill_rgb = load_fill_rgb(args.stats_path).to(device)
    centers = grid_centers(args.image_size, args.mask_size, args.stride)
    done_raw = prune_incomplete_groups(args.output, ["image_id"], len(centers)) if args.resume else set()
    done = {image_id for (image_id,) in done_raw}
    if not args.resume:
        args.output.unlink(missing_ok=True)

    for sample in tqdm(dataset, desc="sliding"):
        image = sample["image"].to(device)
        gt = sample["mask"].to(device)
        image_id = str(sample["image_id"])
        if image_id in done:
            continue

        baseline_prediction = torch.sigmoid(model(image.unsqueeze(0)))
        best_score = float("-inf")
        best_cx = -1
        best_cy = -1
        start_time = time.perf_counter()
        rows: list[dict[str, int | float | str]] = []

        for step, (cx, cy) in enumerate(centers, start=1):
            score, _ = occlusion_objective(
                model=model,
                image=image,
                baseline_prediction=baseline_prediction,
                cx=cx,
                cy=cy,
                size=args.mask_size,
                fill_rgb=fill_rgb,
            )

            if score > best_score:
                best_score = score
                best_cx = cx
                best_cy = cy

            best_mask = mask_to_binary(best_cx, best_cy, args.mask_size, args.image_size, device=device)
            best_overlap = overlap_with_gt(best_mask, gt)
            elapsed = time.perf_counter() - start_time

            rows.append(
                {
                    "image_id": image_id,
                    "method": "sliding",
                    "seed": "",
                    "stride": args.stride,
                    "grid_size": len(centers),
                    "step": step,
                    "cx": cx,
                    "cy": cy,
                    "score": score,
                    "best_score": best_score,
                    "best_cx": best_cx,
                    "best_cy": best_cy,
                    "best_overlap_mask": best_overlap["overlap_mask"],
                    "best_polyp_coverage": best_overlap["polyp_coverage"],
                    "best_mask_gt_iou": best_overlap["mask_gt_iou"],
                    "elapsed_sec": elapsed,
                }
            )

        write_rows(args.output, rows)

    write_run_metadata(
        args.output,
        {
            "method": "sliding",
            "data_root": str(args.data_root),
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "image_size": args.image_size,
            "mask_size": args.mask_size,
            "stride": args.stride,
            "grid_size": len(centers),
            "stats_path": str(args.stats_path),
            "max_samples": args.max_samples,
            "num_dataset_samples": len(dataset),
            "expected_rows_if_complete": len(dataset) * len(centers),
        },
    )
    print(f"Saved sliding window results to {args.output}")


if __name__ == "__main__":
    main()
