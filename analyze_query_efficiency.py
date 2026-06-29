from __future__ import annotations
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
import numpy as np

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Efektywnosc probkowania: best_score vs liczba zapytan")
    parser.add_argument("--bo-csv", type=Path, required=True)
    parser.add_argument("--random-csv", type=Path, required=True)
    parser.add_argument("--sliding-csv", type=Path, required=True)
    # ile procent maksimum z analizowanego zbioru metoda ma osiagnac
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/metrics/query_efficiency_curve.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/metrics/query_efficiency.json"))
    parser.add_argument("--figure", type=Path, default=Path("outputs/figures/query_efficiency.png"))
    return parser.parse_args()


def read_trajectories(path: Path) -> dict[tuple[str, str], list[tuple[int, float]]]:
    """
    Czyta CSV i zwraca {(image_id, seed): [(step, best_score), ...]}
    """
    traj: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            key = (row["image_id"], row.get("seed", ""))
            traj[key].append((int(row["step"]), float(row["best_score"])))
    # sortujemy kazdą trajektorie po numerze kroku
    for key in traj:
        traj[key].sort()
    return traj


def mean_curve(traj: dict[tuple[str, str], list[tuple[int, float]]], max_step: int) -> np.ndarray:
    """
    Uśredniona krzywą best_score po grupach (obraz, seed)
    """
    series = []
    for steps_scores in traj.values():
        scores = [s for _, s in steps_scores]
        if len(scores) < max_step:
            scores = scores + [scores[-1]] * (max_step - len(scores))
        series.append(scores[:max_step])
    return np.mean(np.asarray(series, dtype=float), axis=0)


def queries_to_threshold(method_traj: dict[tuple[str, str], list[tuple[int, float]]],target_per_image: dict[str, float],threshold: float) -> dict[str, object]:
    """
    Ile zapytan metoda potrzebuje, żeby sięgnac threshold * cel dla obrazu (np. 95% maksimum)
    """
    needed: list[int] = []
    reached = 0
    total = 0
    for (image_id, _seed), steps_scores in method_traj.items():
        if image_id not in target_per_image:
            continue
        total += 1
        goal = threshold * target_per_image[image_id]
        # pierwszy krok ktory przebija prog
        hit = next((step for step, score in steps_scores if score >= goal), None)
        if hit is not None:
            needed.append(hit)
            reached += 1
        else:
            # nie osiagnal progu, liczymy caly budzet
            needed.append(steps_scores[-1][0])
    return {
        "median_queries": statistics.median(needed) if needed else float("nan"),
        "mean_queries": statistics.fmean(needed) if needed else float("nan"),
        "reached_fraction": reached / total if total else float("nan"),
        "n": total}

def main() -> None:
    args = parse_args()
    methods = {"bo": read_trajectories(args.bo_csv),"random": read_trajectories(args.random_csv),"sliding": read_trajectories(args.sliding_csv)}

    # cel per obraz -> najlepszy best_score osiagniety przez ktorąkolwiek metode
    target_per_image: dict[str, float] = defaultdict(float)
    for traj in methods.values():
        for (image_id, _seed), steps_scores in traj.items():
            best = max(s for _, s in steps_scores)
            target_per_image[image_id] = max(target_per_image[image_id], best)

    # zapytania do progu
    summary = {"threshold": args.threshold, "per_method": {}}
    print(f"\ncel = {args.threshold:.0%} maksimum best_score per obraz ({len(target_per_image)} obrazow)\n")
    print("metoda | mediana zapytan | srednia | % osiagniec")
    for method, traj in methods.items():
        res = queries_to_threshold(traj, target_per_image, args.threshold)
        summary["per_method"][method] = res
        print(f"{method} | {res['median_queries']} | {res['mean_queries']:.1f} | {res['reached_fraction']:.0%}")

    # uśrednione krzywe zbieżności, do wykresu i csv
    max_step = max(max(s for s, _ in ss) for traj in methods.values() for ss in traj.values())
    curves = {m: mean_curve(t, max_step) for m, t in methods.items()}

    # zapis krzywej do csv
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step"] + list(curves.keys()))
        for i in range(max_step):
            writer.writerow([i + 1] + [f"{curves[m][i]:.6f}" for m in curves])

    # wykresy
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps = np.arange(1, max_step + 1)
        plt.figure(figsize=(7, 5))
        # co ile krokow podpisac wartosc, zeby nie zlepic 200 etykiet
        every = max(1, max_step // 8)
        for method, curve in curves.items():
            line, = plt.plot(steps, curve, marker="*", label=method)
            # podpis wartosci co kilka krokow, zaokraglone do 3 miejsc
            for i in range(0, max_step, every):
                plt.annotate(f"{curve[i]:.3f}", (steps[i], curve[i]),
                             textcoords="offset points", xytext=(0, 6),
                             ha="center", fontsize=7, color=line.get_color())
        plt.xlabel("liczba zapytan do modelu")
        plt.ylabel("best_score (1 - dice do baseline)")
        plt.title("Efektywność próbkowania")
        plt.legend()
        plt.tight_layout()
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.figure, dpi=150)
        plt.close()
        print(f"\n Zapisano wykres ->  {args.figure}")
    except Exception as exc:
        print(f"\nbrak matplotlib -> ({exc})")

    # zapis do json'a
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"zapisano: {args.output_csv}")
    print(f"zapisano: {args.output_json}")

if __name__ == "__main__":
    main()
