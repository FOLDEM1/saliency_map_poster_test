from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


MetricRow = dict[str, str | int | float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare occlusion runs saved as CSV files.")
    parser.add_argument("--random-csv", type=Path, default=Path("outputs/occlusion_runs/random_budget25_seed0.csv"))
    parser.add_argument("--sliding-csv", type=Path, default=Path("outputs/occlusion_runs/sliding_stride32_10.csv"))
    parser.add_argument("--bo-csv", type=Path, default=None)
    parser.add_argument("--random-budget", type=int, default=25)
    parser.add_argument("--sliding-budget", type=int, default=25)
    parser.add_argument("--bo-budget", type=int, default=25)
    parser.add_argument("--default-mask-size", type=int, default=48)
    parser.add_argument(
        "--include-sliding-prefix",
        action="store_true",
        help="Also report the first N sliding-window steps. This is order-dependent and not a fair budgeted baseline.",
    )
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/metrics/occlusion_comparison.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("outputs/metrics/occlusion_comparison_summary.json"))
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def to_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else float("nan")


def to_int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    return int(float(value)) if value != "" else 0


def final_rows_by_image(rows: list[dict[str, str]], max_step: int | None = None) -> dict[str, dict[str, str]]:
    final: dict[str, dict[str, str]] = {}

    for row in rows:
        step = to_int(row, "step")
        if max_step is not None and step > max_step:
            continue

        image_id = row["image_id"]
        current = final.get(image_id)
        if current is None or step > to_int(current, "step"):
            final[image_id] = row

    return final


def final_rows_by_image_seed(rows: list[dict[str, str]], max_step: int) -> dict[tuple[str, int], dict[str, str]]:
    final: dict[tuple[str, int], dict[str, str]] = {}

    for row in rows:
        step = to_int(row, "step")
        if step > max_step:
            continue

        key = (row["image_id"], to_int(row, "seed"))
        current = final.get(key)
        if current is None or step > to_int(current, "step"):
            final[key] = row

    return final


def result_row(source: dict[str, str], method_label: str, seed: int | str = "", default_mask_size: int = 0) -> MetricRow:
    best_mask_size = to_int(source, "best_mask_size") or to_int(source, "mask_size") or default_mask_size
    return {
        "image_id": source["image_id"],
        "seed": seed,
        "method": method_label,
        "step": to_int(source, "step"),
        "best_score": to_float(source, "best_score"),
        "best_cx": to_int(source, "best_cx"),
        "best_cy": to_int(source, "best_cy"),
        "best_mask_size": best_mask_size,
        "best_overlap_mask": to_float(source, "best_overlap_mask"),
        "best_polyp_coverage": to_float(source, "best_polyp_coverage"),
        "best_mask_gt_iou": to_float(source, "best_mask_gt_iou"),
        "elapsed_sec": to_float(source, "elapsed_sec"),
    }


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(rows: list[MetricRow]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    methods = sorted({str(row["method"]) for row in rows})

    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        summary[method] = {
            "n": len(method_rows),
            "best_score_mean": mean([float(row["best_score"]) for row in method_rows]),
            "best_score_std": stdev([float(row["best_score"]) for row in method_rows]),
            "best_overlap_mask_mean": mean([float(row["best_overlap_mask"]) for row in method_rows]),
            "best_polyp_coverage_mean": mean([float(row["best_polyp_coverage"]) for row in method_rows]),
            "elapsed_sec_mean": mean([float(row["elapsed_sec"]) for row in method_rows]),
        }

    return summary


def add_pairwise_deltas(summary: dict[str, Any], rows: list[MetricRow]) -> None:
    by_method_pair = {(str(row["method"]), str(row["image_id"]), str(row["seed"])): row for row in rows}
    pairs = sorted({(str(row["image_id"]), str(row["seed"])) for row in rows})
    present_methods = {str(row["method"]) for row in rows}

    def deltas(left: str, right: str, key: str) -> list[float]:
        values = []
        for image_id, seed in pairs:
            left_row = by_method_pair.get((left, image_id, seed))
            right_row = by_method_pair.get((right, image_id, seed))
            if left_row is not None and right_row is not None:
                values.append(float(left_row[key]) - float(right_row[key]))
        return values

    comparisons = [
        ("bo_budget", "random_budget"),
        ("sliding_full", "bo_budget"),
        ("sliding_full", "random_budget"),
        ("sliding_prefix", "random_budget"),
        ("sliding_full", "sliding_prefix"),
    ]
    summary["pairwise_delta_best_score"] = {}
    for left, right in comparisons:
        if left not in present_methods or right not in present_methods:
            continue
        values = deltas(left, right, "best_score")
        if not values:
            continue
        summary["pairwise_delta_best_score"][f"{left}_minus_{right}"] = {
            "n": len(values),
            "mean": mean(values),
            "std": stdev(values),
            "wins": sum(value > 0 for value in values),
            "ties": sum(value == 0 for value in values),
            "losses": sum(value < 0 for value in values),
        }


def write_csv(path: Path, rows: list[MetricRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    random_rows = read_rows(args.random_csv)
    sliding_rows = read_rows(args.sliding_csv)
    bo_rows = read_rows(args.bo_csv) if args.bo_csv is not None else []

    random_final = final_rows_by_image_seed(random_rows, max_step=args.random_budget)
    sliding_prefix_final = final_rows_by_image(sliding_rows, max_step=args.sliding_budget)
    sliding_full_final = final_rows_by_image(sliding_rows, max_step=None)
    bo_final = final_rows_by_image_seed(bo_rows, max_step=args.bo_budget) if bo_rows else {}

    common_sets = [set(random_final)]
    if bo_final:
        common_sets.append(set(bo_final))
    common_pairs = sorted(set.intersection(*common_sets))
    comparison_rows: list[MetricRow] = []

    for image_id, seed in common_pairs:
        if image_id not in sliding_full_final:
            continue

        comparison_rows.append(result_row(random_final[(image_id, seed)], "random_budget", seed, args.default_mask_size))
        if bo_final:
            comparison_rows.append(result_row(bo_final[(image_id, seed)], "bo_budget", seed, args.default_mask_size))
        if args.include_sliding_prefix and image_id in sliding_prefix_final:
            comparison_rows.append(result_row(sliding_prefix_final[image_id], "sliding_prefix", seed, args.default_mask_size))
        comparison_rows.append(result_row(sliding_full_final[image_id], "sliding_full", seed, args.default_mask_size))

    summary = summarize(comparison_rows)
    summary["inputs"] = {
        "random_csv": str(args.random_csv),
        "sliding_csv": str(args.sliding_csv),
        "bo_csv": str(args.bo_csv) if args.bo_csv is not None else None,
        "random_budget": args.random_budget,
        "sliding_budget": args.sliding_budget,
        "bo_budget": args.bo_budget,
        "default_mask_size": args.default_mask_size,
        "include_sliding_prefix": args.include_sliding_prefix,
        "common_image_seed_pairs": len({(row["image_id"], row["seed"]) for row in comparison_rows}),
        "common_images": len({row["image_id"] for row in comparison_rows}),
    }
    add_pairwise_deltas(summary, comparison_rows)

    write_csv(args.output_csv, comparison_rows)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved comparison rows to {args.output_csv}")
    print(f"Saved summary to {args.summary_json}")


if __name__ == "__main__":
    main()
