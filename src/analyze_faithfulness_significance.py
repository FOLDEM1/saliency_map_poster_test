"""
================================================================================
 MODUŁ: analyze_faithfulness_significance.py
 PAROWANY TEST ISTOTNOŚCI DLA WYNIKÓW FAITHFULNESS (Wilcoxon signed-rank)
================================================================================

PO CO TO ISTNIEJE
--------------------------------------------------------------------------------
evaluate_faithfulness.py daje AUC per (image_id, method, seed) i ładny wykres.
Ale wykres NIE odpowiada na pytanie recenzenta: "czy różnica BO vs random jest
PRAWDZIWA, czy mieści się w szumie?". Pasma ±std na wykresie się nakładają —
to za mało. Ten moduł zamienia surowe AUC w TWARDY wynik statystyczny.

DLACZEGO TEST PAROWANY (a nie t-test/Manna-Whitneya)
--------------------------------------------------------------------------------
Każdą metodę liczymy na TYCH SAMYCH obrazach. AUC są więc SPAROWANE po image_id:
trudny obraz podbija/obniża AUC wszystkim metodom jednocześnie. Test parowany
(Wilcoxon signed-rank na RÓŻNICACH per obraz) usuwa tę wspólną zmienność obrazu
i pyta wprost: "czy mapa metody A jest wierniejsza niż metody B NA TYM SAMYM
obrazie?". Wilcoxon (nieparametryczny) — bo AUC nie muszą być normalne, a
obrazów bywa mało.

JAK ZAPEWNIAMY UCZCIWOŚĆ (FAIR)
--------------------------------------------------------------------------------
  1. BUDŻET: porównuj CSV policzony z --match-budget (np. 25), żeby różnica nie
     brała się z liczby zapytań (sliding ma ich z natury więcej). Ten skrypt
     tylko czyta wyniki — o zrównanie budżetu dba evaluate_faithfulness.
  2. PAROWANIE: najpierw uśredniamy AUC po seedach W OBRĘBIE obrazu (jedna
     liczba na (obraz, metoda)), potem parujemy po przecięciu image_id obu
     metod. Dzięki temu seedy nie zawyżają sztucznie n.
  3. KIERUNEK: deletion "niżej=lepiej", insertion/geomean "wyżej=lepiej" —
     skrypt sam ustawia znak, żeby dodatni "median_diff" zawsze znaczył
     "pierwsza metoda lepsza".

CO ZWRACA
--------------------------------------------------------------------------------
Dla każdej pary metod i każdej metryki:
  * n_pairs           — ile obrazów weszło do testu (po przecięciu),
  * median_a, median_b— mediany metryki dla obu metod,
  * median_diff       — mediana różnicy (a - b) z poprawką kierunku (>0 = a lepsza),
  * wins_a            — na ilu obrazach a było lepsze od b (sign test pomocniczo),
  * wilcoxon_stat, p_value,
  * effect_size       — rank-biserial r dla par (|r|: 0.1 mały, 0.3 średni, 0.5 duży).
Wynik na ekran (czytelna tabela) i opcjonalnie do JSON.
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from scipy.stats import wilcoxon

# Kierunek "lepiej": +1 => wyższa wartość lepsza, -1 => niższa lepsza.
METRIC_DIRECTION = {
    "deletion_auc": -1,    # niżej = lepiej
    "insertion_auc": +1,   # wyżej = lepiej
    "deletion_aligned": +1,
    "diff": +1,
    "geomean": +1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parowany test istotności (Wilcoxon) dla AUC faithfulness.")
    parser.add_argument("--csv", type=Path, required=True,
                        help="Plik faithfulness_*.csv z evaluate_faithfulness (per image/method/seed).")
    parser.add_argument("--metrics", type=str, default="geomean,deletion_auc,insertion_auc",
                        help="Metryki do testu (po przecinku). Domyślnie: geomean,deletion_auc,insertion_auc.")
    parser.add_argument("--pairs", type=str, default="bo:random,bo:sliding,sliding:random",
                        help="Pary metod A:B (po przecinku). Dodatni median_diff = A lepsza.")
    parser.add_argument("--output-json", type=Path, default=None, help="Opcjonalny zapis wyniku do JSON.")
    return parser.parse_args()


def load_per_image_metric(csv_path: Path, metric: str) -> dict[str, dict[str, float]]:
    """
    Zwraca {method: {image_id: wartość}} — wartość = ŚREDNIA metryki po seedach
    w obrębie obrazu. Uśrednienie po seedach najpierw chroni przed sztucznym
    zawyżaniem liczby par (BO/random mają po kilka seedów, sliding jeden).
    """
    # buckets[method][image_id] = lista wartości (po seedach)
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with csv_path.open(newline="") as file:
        for row in csv.DictReader(file):
            buckets[row["method"]][row["image_id"]].append(float(row[metric]))
    return {
        method: {image_id: statistics.fmean(vals) for image_id, vals in images.items()}
        for method, images in buckets.items()
    }


def paired_test(values_a: dict[str, float], values_b: dict[str, float], direction: int) -> dict[str, object]:
    """
    Wilcoxon signed-rank na obrazach wspólnych dla obu metod.
    direction wyrównuje znak, żeby dodatni median_diff zawsze = "A lepsza".
    """
    common = sorted(set(values_a) & set(values_b))
    if len(common) < 1:
        return {"n_pairs": 0, "note": "brak wspólnych obrazów"}

    a = [values_a[i] for i in common]
    b = [values_b[i] for i in common]
    # Różnica z poprawką kierunku: po przemnożeniu dodatnia = A wierniejsza.
    diffs = [direction * (av - bv) for av, bv in zip(a, b)]

    wins_a = sum(1 for d in diffs if d > 0)
    median_diff = statistics.median(diffs)

    # Wilcoxon wymaga >=1 niezerowej różnicy; przy samych zerach jest nieokreślony.
    nonzero = [d for d in diffs if d != 0.0]
    if not nonzero:
        stat, p_value, effect = float("nan"), 1.0, 0.0
    else:
        # zero_method="wilcox" odrzuca różnice zerowe (klasyczny wariant).
        result = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        stat, p_value = float(result.statistic), float(result.pvalue)
        # Rank-biserial dla par: r = W+/sumrank - W-/sumrank = (T+ - T-) / T.
        # Praktyczny estymator przez sumy rang znaku różnicy:
        import numpy as np
        ranks = np.argsort(np.argsort([abs(d) for d in nonzero])) + 1
        sum_pos = float(sum(r for r, d in zip(ranks, nonzero) if d > 0))
        sum_neg = float(sum(r for r, d in zip(ranks, nonzero) if d < 0))
        total = sum_pos + sum_neg
        effect = (sum_pos - sum_neg) / total if total > 0 else 0.0

    return {
        "n_pairs": len(common),
        "median_a": statistics.median(a),
        "median_b": statistics.median(b),
        "median_diff_a_minus_b_dir": median_diff,  # >0 => A wierniejsza
        "wins_a": wins_a,
        "wins_b": len(common) - wins_a,
        "wilcoxon_stat": stat,
        "p_value": p_value,
        "effect_size_rank_biserial": effect,
        "significant_0_05": bool(p_value < 0.05),
    }


def main() -> None:
    args = parse_args()
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    pairs = [tuple(p.split(":")) for p in args.pairs.split(",") if ":" in p]

    summary: dict[str, dict[str, object]] = {}
    print(f"\nŹródło: {args.csv}")
    for metric in metrics:
        if metric not in METRIC_DIRECTION:
            print(f"  POMIJAM nieznaną metrykę: {metric}")
            continue
        per_image = load_per_image_metric(args.csv, metric)
        direction = METRIC_DIRECTION[metric]
        arrow = "wyżej=lepiej" if direction > 0 else "niżej=lepiej"
        print(f"\n=== Metryka: {metric}  ({arrow}) ===")
        print(f"{'para (A vs B)':22s} {'n':>3s} {'med_A':>8s} {'med_B':>8s} {'Δ(dir)':>9s} {'wins_A':>7s} {'p':>9s} {'r':>7s} sig")
        for a, b in pairs:
            if a not in per_image or b not in per_image:
                print(f"{a+' vs '+b:22s}  -- brak danych dla którejś metody")
                continue
            res = paired_test(per_image[a], per_image[b], direction)
            summary[f"{metric}|{a}_vs_{b}"] = res
            if res["n_pairs"] == 0:
                print(f"{a+' vs '+b:22s}  -- brak wspólnych obrazów")
                continue
            sig = "***" if res["p_value"] < 0.01 else ("*" if res["p_value"] < 0.05 else "ns")
            print(f"{a+' vs '+b:22s} {res['n_pairs']:3d} {res['median_a']:8.4f} {res['median_b']:8.4f} "
                  f"{res['median_diff_a_minus_b_dir']:+9.4f} {res['wins_a']:7d} {res['p_value']:9.4f} "
                  f"{res['effect_size_rank_biserial']:+7.2f} {sig}")

    print("\nLegenda: Δ(dir)>0 => metoda A wierniejsza | r = rank-biserial (|0.1| mały, |0.3| średni, |0.5| duży)")
    print("         sig: * p<0.05, *** p<0.01, ns nieistotne\n")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Zapisano: {args.output_json}")


if __name__ == "__main__":
    main()
