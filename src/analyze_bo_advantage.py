"""
================================================================================
 MODUŁ: analyze_bo_advantage.py
 GDZIE BO BŁYSZCZY: best_score przy USTALONYM, MAŁYM BUDŻECIE (BO vs random)
================================================================================

TEZA, KTÓREJ TEN TEST BRONI
--------------------------------------------------------------------------------
Faithfulness całej mapy premiuje pokrycie (wygrywa sliding). Efektywność "do
progu" bywa za łatwa (wszyscy go biją). NAJOSTRZEJSZA przewaga BO jest tu:
przy MAŁEJ liczbie zapytań sprytny dobór punktów (BO) powinien znaleźć
istotnie bardziej destrukcyjny region niż ślepe losowanie (random).

CO LICZY (czyta tylko CSV — bez modelu, bez GPU)
--------------------------------------------------------------------------------
Dla każdego budżetu k z --budgets:
  * best_score@k per (obraz, seed)  -> uśrednione po seedach do jednej liczby
    na (obraz, metoda)  -> sparowane po obrazie BO vs random,
  * parowany Wilcoxon signed-rank na różnicy (bo - random) per obraz,
  * median_uplift = mediana (bo - random)  (>0 => BO lepsze),
  * wins_bo, p_value, efekt rank-biserial,
  * "regret" = (CEL - best_score@k)/CEL, gdzie CEL = max best_score na obrazie:
    o ile procent metoda jest jeszcze od najlepszego osiągalnego (mniej = lepiej).

JAK CZYTAĆ
--------------------------------------------------------------------------------
Szukasz budżetu, przy którym median_uplift jest dodatni, p<0.05 i efekt duży —
to jest zdanie do pracy: "przy k zapytaniach BO znajduje region o Δ wyższym
best_score niż random (Wilcoxon p=..., r=...)". Zwykle przewaga jest największa
przy małym k i topnieje, gdy budżet rośnie (obie metody i tak dojdą do maksimum).
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Przewaga BO przy małym budżecie (best_score@k vs random).")
    parser.add_argument("--bo-csv", type=Path, required=True)
    parser.add_argument("--random-csv", type=Path, required=True)
    parser.add_argument("--sliding-csv", type=Path, default=None,
                        help="Opcjonalnie: linia odniesienia (best_score@k wzdłuż siatki).")
    parser.add_argument("--budgets", type=str, default="3,5,8,10,15,20,25",
                        help="Budżety k do porównania (po przecinku).")
    parser.add_argument("--output-json", type=Path, default=Path("outputs/metrics/bo_advantage.json"))
    parser.add_argument("--figure", type=Path, default=Path("outputs/figures/bo_advantage.png"))
    return parser.parse_args()


def best_score_at_step(path: Path) -> dict[tuple[str, str], dict[int, float]]:
    """{(image_id, seed): {step: best_score}} — best_score jest monotoniczny."""
    table: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            key = (row["image_id"], row.get("seed", ""))
            table[key][int(row["step"])] = float(row["best_score"])
    return table


def value_at_budget(step_map: dict[int, float], k: int) -> float:
    """best_score po k zapytaniach (lub po ostatnim dostępnym, jeśli trajektoria krótsza)."""
    steps = [s for s in step_map if s <= k]
    if steps:
        return step_map[max(steps)]
    return step_map[min(step_map)]


def per_image_mean(table: dict[tuple[str, str], dict[int, float]], k: int) -> dict[str, float]:
    """Uśrednia best_score@k po seedach w obrębie obrazu -> {image_id: wartość}."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for (image_id, _seed), step_map in table.items():
        buckets[image_id].append(value_at_budget(step_map, k))
    return {image_id: statistics.fmean(vals) for image_id, vals in buckets.items()}


def paired_wilcoxon(a: dict[str, float], b: dict[str, float]) -> dict[str, object]:
    """Wilcoxon na (a - b) po wspólnych obrazach. >0 => a lepsze."""
    common = sorted(set(a) & set(b))
    diffs = [a[i] - b[i] for i in common]
    wins_a = sum(1 for d in diffs if d > 0)
    median_diff = statistics.median(diffs) if diffs else float("nan")
    nonzero = [d for d in diffs if d != 0.0]
    if not nonzero:
        stat, p_value, effect = float("nan"), 1.0, 0.0
    else:
        result = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        stat, p_value = float(result.statistic), float(result.pvalue)
        ranks = np.argsort(np.argsort([abs(d) for d in nonzero])) + 1
        sum_pos = float(sum(r for r, d in zip(ranks, nonzero) if d > 0))
        sum_neg = float(sum(r for r, d in zip(ranks, nonzero) if d < 0))
        total = sum_pos + sum_neg
        effect = (sum_pos - sum_neg) / total if total > 0 else 0.0
    return {
        "n_pairs": len(common),
        "median_bo": statistics.median([a[i] for i in common]) if common else float("nan"),
        "median_random": statistics.median([b[i] for i in common]) if common else float("nan"),
        "median_uplift_bo_minus_random": median_diff,
        "wins_bo": wins_a,
        "wins_random": len(common) - wins_a,
        "wilcoxon_stat": stat,
        "p_value": p_value,
        "effect_size_rank_biserial": effect,
        "significant_0_05": bool(p_value < 0.05),
    }


def main() -> None:
    args = parse_args()
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    bo = best_score_at_step(args.bo_csv)
    rnd = best_score_at_step(args.random_csv)
    sliding = best_score_at_step(args.sliding_csv) if args.sliding_csv else None

    # CEL per obraz = max best_score osiągnięty przez bo lub random (lub sliding).
    target: dict[str, float] = defaultdict(float)
    for table in filter(None, [bo, rnd, sliding]):
        for (image_id, _seed), step_map in table.items():
            target[image_id] = max(target[image_id], max(step_map.values()))

    results: dict[str, object] = {"budgets": {}}
    print(f"\nBO vs RANDOM — best_score przy ustalonym budżecie (uplift>0 => BO lepsze)\n")
    header = f"{'k':>4s} {'n':>3s} {'med_BO':>8s} {'med_rnd':>8s} {'uplift':>9s} {'wins_BO':>8s} {'p':>9s} {'r':>7s} sig  regret_BO"
    print(header)
    for k in budgets:
        a = per_image_mean(bo, k)
        b = per_image_mean(rnd, k)
        res = paired_wilcoxon(a, b)
        # regret BO: ile jeszcze brakuje do maksimum (mniej = lepiej)
        regret = statistics.median([(target[i] - a[i]) / target[i] for i in a if target[i] > 0])
        res["regret_bo_median"] = regret
        results["budgets"][k] = res
        sig = "***" if res["p_value"] < 0.01 else ("*" if res["p_value"] < 0.05 else "ns")
        print(f"{k:4d} {res['n_pairs']:3d} {res['median_bo']:8.4f} {res['median_random']:8.4f} "
              f"{res['median_uplift_bo_minus_random']:+9.4f} {res['wins_bo']:8d} {res['p_value']:9.4f} "
              f"{res['effect_size_rank_biserial']:+7.2f} {sig:>3s}  {regret:8.1%}")

    # budżet maksymalnej przewagi BO (największy istotny efekt)
    sig_budgets = {k: v for k, v in results["budgets"].items() if v["significant_0_05"]}
    if sig_budgets:
        best_k = max(sig_budgets, key=lambda k: abs(sig_budgets[k]["effect_size_rank_biserial"]))
        results["best_budget_for_bo"] = best_k
        print(f"\n>>> BO błyszczy najmocniej przy k={best_k}: "
              f"uplift={sig_budgets[best_k]['median_uplift_bo_minus_random']:+.4f}, "
              f"p={sig_budgets[best_k]['p_value']:.4f}, r={sig_budgets[best_k]['effect_size_rank_biserial']:+.2f}")
    else:
        print("\n>>> Brak budżetu z istotną przewagą BO nad random (p<0.05).")

    print("\nLegenda: uplift=med(best_score_BO - best_score_random); regret_BO=ile %% do maksimum brakuje BO")
    print("         sig: * p<0.05, *** p<0.01, ns nieistotne\n")

    # --- wykres: best_score@k dla obu metod (+ sliding jako linia) -------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        med_bo = [results["budgets"][k]["median_bo"] for k in budgets]
        med_rnd = [results["budgets"][k]["median_random"] for k in budgets]
        plt.figure(figsize=(7, 5))
        plt.plot(budgets, med_bo, marker="o", label="bo")
        plt.plot(budgets, med_rnd, marker="o", label="random")
        if sliding is not None:
            med_sl = [statistics.median(per_image_mean(sliding, k).values()) for k in budgets]
            plt.plot(budgets, med_sl, marker="o", linestyle="--", label="sliding (odniesienie)")
        plt.xlabel("budżet (liczba zapytań do modelu)")
        plt.ylabel("mediana best_score")
        plt.title("Przewaga BO przy małym budżecie")
        plt.legend()
        plt.tight_layout()
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.figure, dpi=150)
        plt.close()
        print(f"Zapisano wykres: {args.figure}")
    except Exception as exc:
        print(f"UWAGA: pomijam wykres ({exc}).")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Zapisano: {args.output_json}\n")


if __name__ == "__main__":
    main()
