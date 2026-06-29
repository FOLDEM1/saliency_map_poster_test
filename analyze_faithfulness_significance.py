from __future__ import annotations
import argparse
import csv
import json
import statistics
import numpy as np
from collections import defaultdict
from pathlib import Path
from scipy.stats import wilcoxon

METRIC_DIRECTION = {
    #+1 zysk, -1 koszt
    "deletion_auc": -1,
    "insertion_auc": +1,
    "deletion_aligned": +1,
    "diff": +1,
    "geomean": +1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Wilocxona dla AUC wiarygodnosci")
    parser.add_argument("--csv", type=Path, required=True,help="Plik faithfulness_*.csv z Wynikami wiarygodnosci metod (image/method/seed)")
    parser.add_argument("--metrics", type=str, default="geomean,deletion_auc,insertion_auc",help="Metryki do testu : (wypisac po przecinku )geomean,deletion_auc,insertion_auc.")
    parser.add_argument("--pairs", type=str, default="bo:random,bo:sliding,sliding:random",help="Pary porównan metod a:b ,c:d, ... ")
    parser.add_argument("--output-json", type=Path, default=None, help="mozliwy zapis wyników do json'a ")
    return parser.parse_args()

def load_per_image_metric(csv_path: Path, metric: str) -> dict[str, dict[str, float]]:
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with csv_path.open(newline="") as file:
        for row in csv.DictReader(file):
            buckets[row["method"]][row["image_id"]].append(float(row[metric]))
    return {method: { image_id: statistics.fmean(vals) for image_id, vals in images.items()} for method, images in buckets.items()}

def paired_test(values_a: dict[str, float], values_b: dict[str, float], direction: int) -> dict[str, object]:
    """
    Wilcoxon signed-rank na wspólnych obrazach dla obu metod
    """
    common = sorted(set(values_a) & set(values_b))
    if len(common) < 1:
        return {"n_pairs": 0, "note": "brak wspólnych obrazów dla metod"}
    a = [values_a[i] for i in common]
    b = [values_b[i] for i in common]
    # Różnica z poprawką względem kierunku (zysk/koszt)
    diffs = [direction * (av - bv) for av, bv in zip(a, b)]

    wins_a = sum(1 for d in diffs if d > 0)
    median_diff = statistics.median(diffs)

    # Przy teście Wilocxona w scipy wartosci 0 wyrzucały by błędy 
    nonzero = [d for d in diffs if d != 0.0]
    if not nonzero:
        # jezeli test nie ma z czego liczyc, mamy brak dowodu na róznice
        stat, p_value, effect = float("nan"), 1.0, 0.0
    else:
        result = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        stat, p_value = float(result.statistic), float(result.pvalue)

        ranks = np.argsort(np.argsort([abs(d) for d in nonzero])) + 1
        sum_pos = float(sum(r for r, d in zip(ranks, nonzero) if d > 0))
        sum_neg = float(sum(r for r, d in zip(ranks, nonzero) if d < 0))

        total = sum_pos + sum_neg
        effect = (sum_pos - sum_neg) / total if total > 0 else 0.0

    return {"n_pairs": len(common),
        "median_a": statistics.median(a),
        "median_b": statistics.median(b),
        "median_diff_a_minus_b_dir": median_diff,
        "wins_a": wins_a,
        "wins_b": len(common) - wins_a,
        "wilcoxon_stat": stat,
        "p_value": p_value,
        "effect_size_rank_biserial": effect,
        "significant_0_05": bool(p_value < 0.05)}

def main() -> None:
    args = parse_args()
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    pairs = [tuple(p.split(":")) for p in args.pairs.split(",") if ":" in p]

    summary: dict[str, dict[str, object]] = {}
    print(f"\n źródło danych -> : {args.csv}")
    for metric in metrics:
        if metric not in METRIC_DIRECTION:
            print(f"Metryka nie znana '{metric}' -> skip")
            continue
        per_image = load_per_image_metric(args.csv, metric)
        direction = METRIC_DIRECTION[metric]
        arrow = "wyżej oznacza lepiej" if direction > 0 else "niżej oznacza lepiej"
        print(f"\n \t \t Metryka: {metric}  ({arrow})")
        print("para (A vs B) | n | med_A | med_B | Δ(dir) | wins_A | p | r | sig")
        for a, b in pairs:
            if a not in per_image or b not in per_image:
                print(f"{a} vs {b}  -- brak danych dla którejś metody")
                continue
            res = paired_test(per_image[a], per_image[b], direction)
            summary[f"{metric}|{a}_vs_{b}"] = res
            if res["n_pairs"] == 0:
                print(f"{a} vs {b}  -- brak wspólnych obrazów")
                continue
            sig = "**" if res["p_value"] < 0.01 else ("*" if res["p_value"] < 0.05 else "ns")
            print(f"{a} vs {b} | n={res['n_pairs']} | med_A={res['median_a']:.3f} | med_B={res['median_b']:.4f} | Δ={res['median_diff_a_minus_b_dir']:.3f} | wins_A={res['wins_a']} | p={res['p_value']:.4f} | r={res['effect_size_rank_biserial']:.3f} | {sig}")

    print("\n Oznaczenia Δ(dir)>0 -> metoda A wierniejsza | r = rank-biserial (|0.1| mały, |0.3| średni, |0.5| duży)")
    print(" \t \t \t sig: * p<0.05(roznica istotna), ** p<0.01(roznica bardzo istotna), ns (nieistotna róznica)\n")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Zapisano: {args.output_json}")

if __name__ == "__main__":
    main()
