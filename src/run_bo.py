from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from skopt import Optimizer
from skopt.space import Categorical, Integer
from tqdm import tqdm

from src.dataset import KvasirSegDataset
from src.evaluate import load_model
from src.experiment_io import load_fill_rgb, parse_int_list, prune_incomplete_groups, stable_seed, write_rows, write_run_metadata
from src.occlusion import mask_to_binary, occlusion_objective, overlap_with_gt
from src.saliency import Observation, build_and_save_gp_saliency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BO-SegOcc on a trained segmenter.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/kvasir-seg"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/unet_best.pt"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--mask-size", type=int, default=48)
    parser.add_argument("--optimize-size", action="store_true", help="Optimize theta=(cx, cy, size) instead of fixed-size theta=(cx, cy).")
    parser.add_argument("--size-candidates", type=parse_int_list, default=[32, 48, 64])
    parser.add_argument("--budget", type=int, default=25)
    parser.add_argument("--seeds", type=parse_int_list, default=[0])
    parser.add_argument("--n-initial-points", type=int, default=5)
    parser.add_argument("--acq-func", type=str, default="EI")
    parser.add_argument("--stats-path", type=Path, default=Path("outputs/metrics/train_channel_stats.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/occlusion_runs/bo.csv"))
    parser.add_argument("--saliency-dir", type=Path, default=Path("outputs/saliency_maps/bo"))
    parser.add_argument("--saliency-grid-stride", type=int, default=8)
    parser.add_argument("--no-save-saliency-maps", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip image/seed pairs already completed in the output CSV.")
    return parser.parse_args()


def build_optimizer(
    image_size: int,
    mask_size: int,
    optimize_size: bool,
    size_candidates: list[int],
    seed: int,
    n_initial_points: int,
    acq_func: str,
) -> Optimizer:
    max_size = max(size_candidates) if optimize_size else mask_size
    half = max_size // 2
    dimensions = [
        Integer(half, image_size - half, name="cx"),
        Integer(half, image_size - half, name="cy"),
    ]
    if optimize_size:
        dimensions.append(Categorical(size_candidates, name="size"))
    return Optimizer(
        dimensions=dimensions,
        base_estimator="GP",
        acq_func=acq_func,
        n_initial_points=n_initial_points,
        random_state=seed,
    )


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
    size_candidates = sorted(set(args.size_candidates if args.optimize_size else [args.mask_size]))
    done_raw = prune_incomplete_groups(args.output, ["image_id", "seed"], args.budget) if args.resume else set()
    done = {(image_id, int(seed)) for image_id, seed in done_raw}
    if not args.resume:
        args.output.unlink(missing_ok=True)

    for seed in args.seeds:
        for sample in tqdm(dataset, desc=f"bo seed {seed}"):
            image = sample["image"].to(device)
            gt = sample["mask"].to(device)
            image_id = str(sample["image_id"])
            if (image_id, seed) in done:
                continue

            optimizer = build_optimizer(
                image_size=args.image_size,
                mask_size=args.mask_size,
                optimize_size=args.optimize_size,
                size_candidates=size_candidates,
                seed=stable_seed("bo", seed, image_id, args.image_size, args.mask_size, args.budget),
                n_initial_points=args.n_initial_points,
                acq_func=args.acq_func,
            )

            baseline_prediction = torch.sigmoid(model(image.unsqueeze(0)))
            best_score = float("-inf")
            best_cx = -1
            best_cy = -1
            best_size = -1
            start_time = time.perf_counter()
            rows: list[dict[str, int | float | str]] = []
            observations: list[Observation] = []

            for step in range(1, args.budget + 1):
                point = optimizer.ask()
                if args.optimize_size:
                    cx, cy, proposed_size = point
                    current_size = int(proposed_size)
                else:
                    cx, cy = point
                    current_size = args.mask_size
                score, _ = occlusion_objective(
                    model=model,
                    image=image,
                    baseline_prediction=baseline_prediction,
                    cx=cx,
                    cy=cy,
                    size=current_size,
                    fill_rgb=fill_rgb,
                )

                # skopt minimizes, while BO-SegOcc maximizes J(theta).
                optimizer.tell(point, -score)
                observations.append(Observation(cx=float(cx), cy=float(cy), size=current_size, score=score))

                if score > best_score:
                    best_score = score
                    best_cx = int(cx)
                    best_cy = int(cy)
                    best_size = current_size

                best_mask = mask_to_binary(best_cx, best_cy, best_size, args.image_size, device=device)
                best_overlap = overlap_with_gt(best_mask, gt)
                elapsed = time.perf_counter() - start_time

                rows.append(
                    {
                        "image_id": image_id,
                        "method": "bo",
                        "seed": seed,
                        "max_budget": args.budget,
                        "step": step,
                        "cx": int(cx),
                        "cy": int(cy),
                        "mask_size": current_size,
                        "score": score,
                        "best_score": best_score,
                        "best_cx": best_cx,
                        "best_cy": best_cy,
                        "best_mask_size": best_size,
                        "best_overlap_mask": best_overlap["overlap_mask"],
                        "best_polyp_coverage": best_overlap["polyp_coverage"],
                        "best_mask_gt_iou": best_overlap["mask_gt_iou"],
                        "elapsed_sec": elapsed,
                    }
                )

            write_rows(args.output, rows)
            if not args.no_save_saliency_maps and len(observations) >= 2:
                mode = "variable_size" if args.optimize_size else "fixed_size"
                output_prefix = args.saliency_dir / mode / f"{image_id}_seed{seed}"
                build_and_save_gp_saliency(
                    observations=observations,
                    image=image,
                    image_size=args.image_size,
                    size_candidates=size_candidates,
                    grid_stride=args.saliency_grid_stride,
                    output_prefix=output_prefix,
                    random_state=stable_seed("saliency", seed, image_id, args.image_size, tuple(size_candidates)),
                )

    write_run_metadata(
        args.output,
        {
            "method": "bo",
            "data_root": str(args.data_root),
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "image_size": args.image_size,
            "mask_size": args.mask_size,
            "optimize_size": args.optimize_size,
            "size_candidates": size_candidates,
            "budget": args.budget,
            "seeds": args.seeds,
            "n_initial_points": args.n_initial_points,
            "acq_func": args.acq_func,
            "saliency_maps_saved": not args.no_save_saliency_maps,
            "saliency_dir": str(args.saliency_dir),
            "saliency_grid_stride": args.saliency_grid_stride,
            "stats_path": str(args.stats_path),
            "max_samples": args.max_samples,
            "num_dataset_samples": len(dataset),
            "expected_rows_if_complete": len(dataset) * len(args.seeds) * args.budget,
        },
    )
    print(f"Saved BO-SegOcc results to {args.output}")


if __name__ == "__main__":
    main()
