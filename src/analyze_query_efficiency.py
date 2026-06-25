"""
================================================================================
 MODUŁ: analyze_query_efficiency.py
 EFEKTYWNOŚĆ PRÓBKOWANIA: best_score vs LICZBA ZAPYTAŃ (gdzie BO ma wygrywać)
================================================================================

PO CO TO ISTNIEJE
--------------------------------------------------------------------------------
Metryka faithfulness ocenia mapę CAŁEGO obrazu i premiuje pokrycie — tam wygrywa
wyczerpujący sliding. Ale to NIE jest to, w czym dobry jest BO. BO to oszczędny
OPTYMALIZATOR: ma znaleźć najbardziej destrukcyjny region w JAK NAJMNIEJSZEJ
liczbie wywołań modelu. Ten moduł mierzy dokładnie to.

CO LICZY (czyta tylko istniejące CSV — bez modelu, bez GPU)
--------------------------------------------------------------------------------
  1. KRZYWA ZBIEŻNOŚCI: best_score (najlepszy dotychczas) jako funkcja kroku
     budżetu, uśredniona po (obraz, seed). Pokazuje, jak szybko metoda dochodzi
     do dobrego rozwiązania.
  2. ZAPYTANIA-DO-PROGU: dla każdego obrazu bierzemy CEL = najlepszy best_score
     osiągnięty przez KTÓRĄKOLWIEK metodę na tym obrazie (czyli realne maksimum,
     zwykle wyznaczone przez wyczerpujący sliding). Liczymy, ile zapytań każda
     metoda potrzebuje, by sięgnąć --threshold * CEL. Mediana po obrazach.
     To jest TWARDA liczba do pracy: "BO osiąga 95% maksimum w N zapytań,
     random w M, sliding dopiero przy pełnej siatce".

KLUCZOWA UCZCIWOŚĆ
--------------------------------------------------------------------------------
Wszystkie metody mierzymy tą samą wielkością (best_score = 1 - dice do baseline)
i tym samym CELEM per obraz. Sliding dostaje swój pełny budżet (to JEGO koszt) —
i właśnie o to chodzi: pokazujemy, że BO dochodzi do tego samego TANIEJ.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Efektywność próbkowania: best_score vs liczba zapytań.")
    parser.add_argument("--bo-csv", type=Path, required=True)
    parser.add_argument("--random-csv", type=Path, required=True)
    parser.add_argument("--sliding-csv", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Ułamek maksimum per obraz, który metoda ma osiągnąć (domyślnie 0.95).")
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/metrics/query_efficiency_curve.csv"),
                        help="Uśredniona krzywa best_score vs krok (do wykresu).")
    parser.add_argument("--output-json", type=Path, default=Path("outputs/metrics/query_efficiency.json"))
    parser.add_argument("--figure", type=Path, default=Path("outputs/figures/query_efficiency.png"))
    return parser.parse_args()


def read_trajectories(path: Path) -> dict[tuple[str, str], list[tuple[int, float]]]:
    """
    Zwraca {(image_id, seed): [(step, best_score), ...]} posortowane po step.
    best_score to "najlepszy dotychczas" — monotoniczny z definicji run_*.
    """
    traj: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            key = (row["image_id"], row.get("seed", ""))
            traj[key].append((int(row["step"]), float(row["best_score"])))
    for key in traj:
        traj[key].sort()
    return traj


def mean_curve(traj: dict[tuple[str, str], list[tuple[int, float]]], max_step: int) -> np.ndarray:
    """
    Uśredniona krzywa best_score po (obraz, seed). Krótsze trajektorie (np. BO 25
    vs sliding 64) przedłużamy ostatnią wartością (best_score już nie spada).
    """
    series = []
    for steps_scores in traj.values():
        scores = [s for _, s in steps_scores]
        if len(scores) < max_step:                       # przedłuż ostatnią wartością
            scores = scores + [scores[-1]] * (max_step - len(scores))
        series.append(scores[:max_step])
    return np.mean(np.asarray(series, dtype=float), axis=0)


def queries_to_threshold(
    method_traj: dict[tuple[str, str], list[tuple[int, float]]],
    target_per_image: dict[str, float],
    threshold: float,
) -> dict[str, object]:
    """
    Dla każdego (obraz, seed): pierwszy krok, w którym best_score >= threshold*CEL.
    Jeśli nie osiągnięto — liczymy jako długość trajektorii + brak (cap). Zwraca
    medianę/średnią liczby zapytań i odsetek osiągnięć.
    """
    needed: list[int] = []
    reached = 0
    total = 0
    for (image_id, _seed), steps_scores in method_traj.items():
        if image_id not in target_per_image:
            continue
        total += 1
        goal = threshold * target_per_image[image_id]
        hit = next((step for step, score in steps_scores if score >= goal), None)
        if hit is not None:
            needed.append(hit)
            reached += 1
        else:
            needed.append(steps_scores[-1][0])           # cap = pełny budżet (nie osiągnął)
    return {
        "median_queries": statistics.median(needed) if needed else float("nan"),
        "mean_queries": statistics.fmean(needed) if needed else float("nan"),
        "reached_fraction": reached / total if total else float("nan"),
        "n": total,
    }


def main() -> None:
    args = parse_args()
    methods = {
        "bo": read_trajectories(args.bo_csv),
        "random": read_trajectories(args.random_csv),
        "sliding": read_trajectories(args.sliding_csv),
    }

    # CEL per obraz = najlepszy best_score osiągnięty przez KTÓRĄKOLWIEK metodę.
    target_per_image: dict[str, float] = defaultdict(float)
    for traj in methods.values():
        for (image_id, _seed), steps_scores in traj.items():
            best = max(s for _, s in steps_scores)
            target_per_image[image_id] = max(target_per_image[image_id], best)

    # --- zapytania-do-progu ---------------------------------------------------
    summary = {"threshold": args.threshold, "per_method": {}}
    print(f"\nCEL = {args.threshold:.0%} maksimum best_score per obraz "
          f"({len(target_per_image)} obrazów)\n")
    print(f"{'metoda':10s} {'mediana zapytań':>16s} {'średnia':>9s} {'% osiągnięć':>12s}")
    for method, traj in methods.items():
        res = queries_to_threshold(traj, target_per_image, args.threshold)
        summary["per_method"][method] = res
        print(f"{method:10s} {res['median_queries']:16.1f} {res['mean_queries']:9.1f} "
              f"{res['reached_fraction']:11.0%}")

    # --- uśrednione krzywe zbieżności (do wykresu i CSV) ----------------------
    max_step = max(max(s for s, _ in ss) for traj in methods.values() for ss in traj.values())
    curves = {m: mean_curve(t, max_step) for m, t in methods.items()}

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step"] + list(curves.keys()))
        for i in range(max_step):
            writer.writerow([i + 1] + [f"{curves[m][i]:.6f}" for m in curves])

    # --- wykres (opcjonalnie) -------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps = np.arange(1, max_step + 1)
        plt.figure(figsize=(7, 5))
        for method, curve in curves.items():
            plt.plot(steps, curve, marker=".", label=method)
        plt.xlabel("liczba zapytań do modelu (budżet)")
        plt.ylabel("best_score (1 - dice do baseline)")
        plt.title("Efektywność próbkowania: im szybciej w górę, tym lepiej")
        plt.legend()
        plt.tight_layout()
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.figure, dpi=150)
        plt.close()
        print(f"\nZapisano wykres: {args.figure}")
    except Exception as exc:
        print(f"\nUWAGA: pomijam wykres ({exc}).")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Zapisano: {args.output_csv}")
    print(f"Zapisano: {args.output_json}\n")


if __name__ == "__main__":
    main()
