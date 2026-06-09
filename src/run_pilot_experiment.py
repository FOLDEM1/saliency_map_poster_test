from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import fmean, stdev
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a smaller BO-SegOcc pilot experiment.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/unet_best.pt"))
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--budget", type=int, default=25)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--sliding-stride", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/occlusion_runs/pilot"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("outputs/metrics/pilot"))
    parser.add_argument("--no-resume", action="store_true", help="Start output files from scratch.")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze existing pilot CSV files.")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("\n==>", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "random": output_dir / "random_pilot.csv",
        "bo": output_dir / "bo_pilot.csv",
        "sliding": output_dir / "sliding_pilot.csv",
    }


def run_experiment(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resume_flag = [] if args.no_resume else ["--resume"]

    run_command(
        [
            sys.executable,
            "-m",
            "src.run_random",
            "--checkpoint",
            str(args.checkpoint),
            "--budgets",
            str(args.budget),
            "--seeds",
            args.seeds,
            "--max-samples",
            str(args.max_samples),
            "--output",
            str(paths["random"]),
            *resume_flag,
        ]
    )
    run_command(
        [
            sys.executable,
            "-m",
            "src.run_bo",
            "--checkpoint",
            str(args.checkpoint),
            "--budget",
            str(args.budget),
            "--seeds",
            args.seeds,
            "--max-samples",
            str(args.max_samples),
            "--output",
            str(paths["bo"]),
            *resume_flag,
        ]
    )
    run_command(
        [
            sys.executable,
            "-m",
            "src.run_sliding",
            "--checkpoint",
            str(args.checkpoint),
            "--stride",
            str(args.sliding_stride),
            "--max-samples",
            str(args.max_samples),
            "--output",
            str(paths["sliding"]),
            *resume_flag,
        ]
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def to_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def to_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def final_by_image_seed(rows: list[dict[str, str]], budget: int) -> dict[tuple[str, int], dict[str, str]]:
    final: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        step = to_int(row, "step")
        if step > budget:
            continue

        key = (row["image_id"], to_int(row, "seed"))
        current = final.get(key)
        if current is None or step > to_int(current, "step"):
            final[key] = row
    return final


def final_by_image(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    final: dict[str, dict[str, str]] = {}
    for row in rows:
        image_id = row["image_id"]
        current = final.get(image_id)
        if current is None or to_int(row, "step") > to_int(current, "step"):
            final[image_id] = row
    return final


def mean(values: list[float]) -> float:
    return fmean(values) if values else float("nan")


def std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def win_summary(values: list[float]) -> dict[str, int | float]:
    return {
        "n": len(values),
        "mean": mean(values),
        "std": std(values),
        "wins": sum(value > 0 for value in values),
        "ties": sum(value == 0 for value in values),
        "losses": sum(value < 0 for value in values),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    random_final = final_by_image_seed(read_rows(paths["random"]), args.budget)
    bo_final = final_by_image_seed(read_rows(paths["bo"]), args.budget)
    sliding_final = final_by_image(read_rows(paths["sliding"]))

    common_pairs = sorted(set(random_final) & set(bo_final))
    comparison_rows: list[dict[str, Any]] = []

    for image_id, seed in common_pairs:
        if image_id not in sliding_final:
            continue

        random_row = random_final[(image_id, seed)]
        bo_row = bo_final[(image_id, seed)]
        sliding_row = sliding_final[image_id]

        random_score = to_float(random_row, "best_score")
        bo_score = to_float(bo_row, "best_score")
        sliding_score = to_float(sliding_row, "best_score")

        comparison_rows.append(
            {
                "image_id": image_id,
                "seed": seed,
                "random_best_score": random_score,
                "bo_best_score": bo_score,
                "sliding_best_score": sliding_score,
                "bo_minus_random": bo_score - random_score,
                "bo_minus_sliding": bo_score - sliding_score,
                "random_minus_sliding": random_score - sliding_score,
                "random_overlap_mask": to_float(random_row, "best_overlap_mask"),
                "bo_overlap_mask": to_float(bo_row, "best_overlap_mask"),
                "sliding_overlap_mask": to_float(sliding_row, "best_overlap_mask"),
                "random_polyp_coverage": to_float(random_row, "best_polyp_coverage"),
                "bo_polyp_coverage": to_float(bo_row, "best_polyp_coverage"),
                "sliding_polyp_coverage": to_float(sliding_row, "best_polyp_coverage"),
                "random_elapsed_sec": to_float(random_row, "elapsed_sec"),
                "bo_elapsed_sec": to_float(bo_row, "elapsed_sec"),
                "sliding_elapsed_sec": to_float(sliding_row, "elapsed_sec"),
            }
        )

    image_level_sliding = list(sliding_final.values())
    summary = {
        "inputs": {
            "max_samples": args.max_samples,
            "budget": args.budget,
            "seeds": args.seeds,
            "sliding_stride": args.sliding_stride,
            "random_csv": str(paths["random"]),
            "bo_csv": str(paths["bo"]),
            "sliding_csv": str(paths["sliding"]),
        },
        "counts": {
            "image_seed_pairs": len(comparison_rows),
            "sliding_images": len(image_level_sliding),
        },
        "method_summary": {
            "random": {
                "best_score_mean": mean([row["random_best_score"] for row in comparison_rows]),
                "best_score_std": std([row["random_best_score"] for row in comparison_rows]),
                "overlap_mask_mean": mean([row["random_overlap_mask"] for row in comparison_rows]),
                "polyp_coverage_mean": mean([row["random_polyp_coverage"] for row in comparison_rows]),
                "elapsed_sec_mean": mean([row["random_elapsed_sec"] for row in comparison_rows]),
            },
            "bo": {
                "best_score_mean": mean([row["bo_best_score"] for row in comparison_rows]),
                "best_score_std": std([row["bo_best_score"] for row in comparison_rows]),
                "overlap_mask_mean": mean([row["bo_overlap_mask"] for row in comparison_rows]),
                "polyp_coverage_mean": mean([row["bo_polyp_coverage"] for row in comparison_rows]),
                "elapsed_sec_mean": mean([row["bo_elapsed_sec"] for row in comparison_rows]),
            },
            "sliding": {
                "best_score_mean": mean([to_float(row, "best_score") for row in image_level_sliding]),
                "best_score_std": std([to_float(row, "best_score") for row in image_level_sliding]),
                "overlap_mask_mean": mean([to_float(row, "best_overlap_mask") for row in image_level_sliding]),
                "polyp_coverage_mean": mean([to_float(row, "best_polyp_coverage") for row in image_level_sliding]),
                "elapsed_sec_mean": mean([to_float(row, "elapsed_sec") for row in image_level_sliding]),
            },
        },
        "pairwise": {
            "bo_minus_random": win_summary([row["bo_minus_random"] for row in comparison_rows]),
            "bo_minus_sliding": win_summary([row["bo_minus_sliding"] for row in comparison_rows]),
            "random_minus_sliding": win_summary([row["random_minus_sliding"] for row in comparison_rows]),
        },
    }

    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = args.metrics_dir / "pilot_comparison.csv"
    summary_path = args.metrics_dir / "pilot_summary.json"
    write_csv(comparison_path, comparison_rows)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nPilot summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved comparison to {comparison_path}")
    print(f"Saved summary to {summary_path}")
    return summary


def main() -> None:
    args = parse_args()
    paths = output_paths(args.output_dir)

    if not args.analyze_only:
        run_experiment(args, paths)

    analyze(args, paths)


if __name__ == "__main__":
    main()
