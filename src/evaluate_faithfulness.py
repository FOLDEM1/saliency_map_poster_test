"""
================================================================================
 MODUŁ: evaluate_faithfulness.py
 OCENA WIERNOŚCI MAP SALIENCY METODĄ INSERTION / DELETION
================================================================================

ABSTRAKT (co ten moduł robi i po co istnieje)
--------------------------------------------------------------------------------
Reszta projektu (random / sliding / BO) szuka destrukcyjnych okluzji i buduje
z nich mapy saliency. Ale dotąd NIE było żadnej metryki, która mówi:
"czy ta mapa NAPRAWDĘ wskazuje piksele, od których zależy model?".

Ten moduł to dokłada. Implementuje standardową metrykę z literatury
(Insertion / Deletion, RISE — Petsiuk i in.), która działa tak:

  * DELETION: bierzemy mapę saliency, sortujemy piksele od NAJWAŻNIEJSZYCH,
    i kolejno je "usuwamy" (zastępujemy rozmytym tłem). Po każdej porcji pytamy
    model, ile polipa jeszcze widzi. DOBRA mapa -> predykcja wali się od razu.

  * INSERTION: odwrotnie. Startujemy od obrazu w pełni rozmytego i dokładamy
    piksele od najważniejszych. DOBRA mapa -> predykcja szybko wraca.

Pole pod każdą krzywą (AUC) daje JEDNĄ porównywalną liczbę. Niskie Deletion AUC
oraz wysokie Insertion AUC = wierna mapa. To pozwala UCZCIWIE uszeregować
metody: BO vs random vs sliding.

DLACZEGO TO JEST MOCNE
--------------------------------------------------------------------------------
  1. Mierzy WIERNOŚĆ (czy mapa oddaje, na czym polega model), a nie ładność.
  2. NIE potrzebuje ground truth — sam model jest sędzią.
  3. Łapie "dobrze, ale z złych powodów" (overlap z GT tego nie wykryje).
  4. Daje jeden skalar (AUC) do rankingu metod.

ZAŁOŻENIA PROJEKTOWE (ustalone świadomie — patrz praca)
--------------------------------------------------------------------------------
  * SKALAR siły predykcji = suma sigmoidów, znormalizowana do startu 1.0:
        siła(x) = Σ sigmoid(model(x_zasłonięty)) / Σ sigmoid(model(oryginał))
    Dzięki normalizacji każdy obraz startuje z 1.0 i można uśredniać krzywe.

  * BASELINE (czym zastępujemy/od czego startujemy) = płaski obraz o średniej
    RGB z datasetu treningowego ("mean image"). To podejście z oryginalnej pracy
    RISE (Petsiuk et al., 2018, sekcja 3.1), które spełnia dwa warunki:
      (a) in-distribution — model widział te wartości kolorów podczas treningu,
      (b) brak struktury przestrzennej — predykcja na płaskim obrazie ≈ 0,
          więc krzywa deletion ma faktycznie "dokąd spaść".
    Alternatywny wariant (--baseline=blur, Gauss) jest dostępny do porównań,
    ale nie jest domyślny — rozmycie zachowuje niskie częstotliwości przestrzenne,
    przez co predykcja baseline bywa zbliżona do oryginału i AUC deletion ≈ 1.

  * HARMONOGRAM porcji = gęsto na początku (co 1% aż do 20%), rzadko potem
    (co 5%). Bo najważniejszy jest POCZĄTEK krzywej — tam dobra mapa zrzuca
    predykcję z klifu.

  * BUDOWA MAP = jednolita dla wszystkich trzech metod (random/sliding/bo), żeby
    jedyną różnicą między mapami był SPOSÓB PRÓBKOWANIA, a nie metoda budowy.
    Składa się z DWÓCH kroków:
      (1) rasteryzacja obserwacji z CSV (rasterize_window_scores) — uśrednia score
          na nakładających się oknach;
      (2) GĘSTNIENIE (densify): piksele, których ŻADNE okno nie dotknęło,
          dostają wartość NAJBLIŻSZEGO odwiedzonego piksela (wypełnienie Voronoi).
    Krok (2) jest KLUCZOWY i został dodany po wykryciu wady architektonicznej —
    patrz sekcja "WADA ARCHITEKTONICZNA…" niżej. Bez niego porównanie jest
    nieuczciwe i metryka mierzy POKRYCIE mapy, a nie jakość próbkowania.

  * AUC = reguła trapezów po PRAWDZIWEJ osi X (rzeczywisty % usuniętych pikseli).

WADA ARCHITEKTONICZNA WYKRYTA W PIERWOTNEJ WERSJI (A1) I JEJ NAPRAWA
--------------------------------------------------------------------------------
OBJAW (pilot, 10 obrazów): krzywe `bo` i `random` były prawie nieodróżnialne
i "poszarpane" (niemonotoniczne garby w okolicy 0.6–0.9 usuniętych pikseli),
a `sliding` deklasował obie. Sugerowało to, że metoda główna (BO) jest bez-
wartościowa — co było ARTEFAKTEM METRYKI, nie właściwością BO.

PRZYCZYNA: pierwotna budowa map ("A1") rasteryzowała tylko surowe obserwacje
i NIE wypełniała pikseli nieodwiedzonych. A że metody mają różne pokrycie:
      bo      ~25 okien  -> ~65% pokrycia  (~35% pikseli = 0)
      random  ~25 okien  -> ~54% pokrycia  (~46% pikseli = 0)
      sliding ~64 okna   -> 100% pokrycia  (0% pikseli = 0)
…to u bo/random OGROMNA masa pikseli miała saliency = 0 (remis). Stabilny
argsort szeregował te remisy w KOLEJNOŚCI RASTROWEJ (skan obrazu), czyli
przypadkowo — i to ta przypadkowa kolejność, nie saliency, sterowała drugą
połową krzywej. Stąd (a) garby/niemonotoniczność, (b) bo≈random (oba dzielą
ten sam tie-break), (c) pozorna dominacja sliding (jako jedyny miał kompletny,
sensowny ranking całego obrazu). Metryka mierzyła POKRYCIE, nie próbkowanie.

DODATKOWY ZARZUT: A1 wyrzucało właściwy produkt BO — gęstą mapę z GP — i oceniało
BO na reprezentacji, której samo nie używa.

NAPRAWA (ta wersja):
  1. GĘSTNIENIE (densify_saliency): każda mapa jest domykana wypełnieniem
     Voronoi (najbliższy odwiedzony piksel) PRZED rankingiem. Po tym kroku
     ŻADNA metoda nie ma remisów zerowych — cały ranking jest sensowny, a
     niemonotoniczne garby z tie-breaku znikają. Operacja jest deterministyczna
     i identyczna dla wszystkich metod (uczciwość zachowana).
  2. BUDŻET jako jawny konfounder: sliding z natury ma więcej okien niż bo/random.
     Flaga --match-budget N pozwala zrównać liczbę obserwacji per (obraz, seed)
     przez deterministyczne losowe podpróbkowanie (RNG o stałym ziarnie), żeby
     różnica nie brała się z samej liczby zapytań. Domyślnie wyłączone, ale
     fakt nierównego budżetu jest zapisywany w summary JSON.
  3. Flaga --no-densify odtwarza stare (wadliwe) zachowanie A1 — wyłącznie do
     porównań/diagnostyki, NIE do raportowania wyników.

IZOLACJA / ŁATWE USUNIĘCIE
--------------------------------------------------------------------------------
Cały kod siedzi w TYM jednym pliku. Z istniejących modułów tylko CZYTAMY
(load_model, KvasirSegDataset, Observation, rasterize_window_scores) — niczego
w nich nie zmieniamy. Żeby wypiąć całą funkcję: skasuj ten plik + usuń
faithfulness_command() i jedną linijkę run_command z src/main.py.

EFEKT KOŃCOWY (pliki, które ten moduł tworzy)
--------------------------------------------------------------------------------
  outputs/metrics/faithfulness_<profile>.csv
        -> wiersz na (image_id, method, seed): deletion_auc, insertion_auc,
           deletion_aligned (=1-deletion), diff (=insertion-deletion),
           geomean (=sqrt(insertion*(1-deletion)))
  outputs/metrics/faithfulness_<profile>.json
        -> agregaty per metoda (średnia ± odchylenie po obrazach i seedach)
           + zapis ustawień eksperymentu
  outputs/figures/faithfulness_curves_<profile>.png   (opcjonalnie)
        -> uśrednione krzywe Deletion i Insertion: BO vs random vs sliding
================================================================================
"""

from __future__ import annotations

# --- importy biblioteki standardowej -------------------------------------------
import argparse                      # parsowanie argumentów wiersza poleceń
import csv                           # czytanie plików CSV z wynikami okluzji
import json                          # zapis podsumowania w formacie JSON
import math                          # sqrt do średniej geometrycznej
import statistics                    # średnia i odchylenie standardowe agregatów
from pathlib import Path             # bezpieczne ścieżki niezależne od systemu

# --- importy zewnętrzne --------------------------------------------------------
import numpy as np                   # operacje na mapach saliency i sortowanie
# NumPy >=2.0 ma trapezoid; <2.0 tylko trapz. getattr z gołym np.trapz jako
# domyślnym wysadziłby się na NumPy>=2.0 (trapz usunięty) — stąd jawny fallback.
_np_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
import torch                         # inferencja modelu (czarna skrzynka)
import torchvision.transforms.functional as TF  # rozmycie Gaussa (baseline)

# --- importy z istniejących modułów projektu (TYLKO ODCZYT) --------------------
from src.dataset import KvasirSegDataset          # loader obrazów Kvasir-SEG
from src.evaluate import load_model               # ładowanie checkpointu U-Net
from src.experiment_io import stable_seed, write_run_metadata  # ziarno + zapis metadanych runu
from src.occlusion import square_bounds           # granice okna -> maska pokrycia (TYLKO ODCZYT)
from src.saliency import Observation, rasterize_window_scores  # budowa mapy

# Mała stała chroniąca przed dzieleniem przez zero przy normalizacji.
EPS = 1e-7


# ==============================================================================
# 1. ARGUMENTY WIERSZA POLECEŃ
# ==============================================================================
def parse_args() -> argparse.Namespace:
    """Definiuje wszystkie wejścia modułu. Domyślne ścieżki pasują do pipeline'u."""
    parser = argparse.ArgumentParser(description="Insertion/Deletion faithfulness of saliency maps.")
    # Skąd brać obrazy i model:
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/kvasir-seg"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/unet_best.pt"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--image-size", type=int, default=256)
    # Domyślny rozmiar maski — używany, gdy CSV nie ma kolumny mask_size
    # (random i sliding zapisują stały rozmiar, więc nie trzymają go w wierszu).
    parser.add_argument("--default-mask-size", type=int, default=48)
    # Trzy pliki CSV z historią zapytań — po jednym na metodę:
    parser.add_argument("--random-csv", type=Path, required=True)
    parser.add_argument("--sliding-csv", type=Path, required=True)
    parser.add_argument("--bo-csv", type=Path, required=True)
    # Statystyki kanałów — używane gdy --baseline=mean do budowy płaskiego obrazu bazowego.
    parser.add_argument("--stats-path", type=Path, default=Path("outputs/metrics/train_channel_stats.json"))
    # Typ baseline'u (RISE: mean image; alternatywnie Gauss):
    parser.add_argument("--baseline", type=str, default="mean", choices=["mean", "blur"],
                        help="mean = płaski obraz o średniej treningowej (RISE); blur = Gauss.")
    # Parametry baseline'u rozmytego (używane tylko gdy --baseline=blur):
    parser.add_argument("--blur-kernel", type=int, default=31, help="Rozmiar jądra Gaussa (nieparzysty).")
    parser.add_argument("--blur-sigma", type=float, default=7.0, help="Sigma rozmycia Gaussa.")
    # Parametry harmonogramu porcji:
    parser.add_argument("--dense-until", type=float, default=0.20, help="Do jakiego ułamka próbkujemy gęsto.")
    parser.add_argument("--dense-step", type=float, default=0.01, help="Krok gęsty (początek krzywej).")
    parser.add_argument("--sparse-step", type=float, default=0.05, help="Krok rzadki (dalsza część krzywej).")
    # GĘSTNIENIE mapy (domyślnie ON) — domyka piksele nieodwiedzone wartością
    # najbliższego odwiedzonego (Voronoi). Usuwa masę remisów zerowych, która
    # w wersji A1 fałszowała ranking. --no-densify odtwarza stare zachowanie.
    parser.add_argument("--no-densify", dest="densify", action="store_false",
                        help="Wyłącz gęstnienie map (odtwarza wadliwe A1 — tylko do diagnostyki).")
    parser.set_defaults(densify=True)
    # Zrównanie budżetu obserwacji per (obraz, seed). None = użyj wszystkich.
    # Wartość N = deterministyczne losowe podpróbkowanie do N obserwacji, żeby
    # różnica między metodami nie brała się z samej liczby zapytań (sliding > bo/random).
    parser.add_argument("--match-budget", type=int, default=None,
                        help="Podpróbkuj każdą metodę do N obserwacji per (obraz, seed).")
    # Ograniczenie liczby obrazów (do szybkich testów / zgodnie z profilem):
    parser.add_argument("--max-samples", type=int, default=None)
    # Gdzie zapisać wyniki:
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/metrics/faithfulness.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("outputs/metrics/faithfulness.json"))
    parser.add_argument("--figure", type=Path, default=None, help="Opcjonalny PNG z krzywymi.")
    return parser.parse_args()


# ==============================================================================
# 2. HARMONOGRAM PORCJI (gęsto na początku, rzadko potem)
# ==============================================================================
def build_schedule(dense_until: float, dense_step: float, sparse_step: float) -> np.ndarray:
    """
    Zwraca rosnącą tablicę ułamków usuniętych/dodanych pikseli, np.:
        [0.00, 0.01, 0.02, ..., 0.20, 0.25, 0.30, ..., 1.00]
    Gwarantuje obecność 0.0 (start) i 1.0 (koniec).
    """
    dense = np.arange(0.0, dense_until + 1e-9, dense_step)          # gęsta część
    sparse = np.arange(dense_until, 1.0 + 1e-9, sparse_step)        # rzadka część
    fractions = np.concatenate([dense, sparse, [0.0, 1.0]])         # sklejamy + skrajne
    fractions = np.clip(fractions, 0.0, 1.0)                        # przytnij do [0,1]
    return np.unique(fractions)                                     # posortuj i odduplikuj


# ==============================================================================
# 3. POMOCNICZE: SKALAR SIŁY PREDYKCJI
# ==============================================================================
@torch.no_grad()  # ocena — gradienty modelu niepotrzebne (czarna skrzynka)
def sigmoid_mass(model: torch.nn.Module, image_chw: torch.Tensor, device: torch.device) -> float:
    """
    Liczy "ile polipa model widzi" = sumę prawdopodobieństw sigmoid po pikselach.
    image_chw: tensor CxHxW (jeden obraz). Zwraca pojedynczą liczbę (float).
    """
    logits = model(image_chw.unsqueeze(0).to(device))   # dodaj wymiar batcha -> 1xCxHxW
    return float(torch.sigmoid(logits).sum().item())    # sigmoid -> suma -> liczba


# ==============================================================================
# 4. BUDOWA MAPY SALIENCY Z WIERSZY CSV (A1: prosta rasteryzacja)
# ==============================================================================
def saliency_from_rows(rows: list[dict[str, str]], image_size: int, default_mask_size: int) -> np.ndarray:
    """
    Zamienia listę zapytań danej metody (cx, cy, mask_size, score) na mapę 2D.
    Każde zapytanie to jeden kwadrat z wynikiem 'score'; rasterize_window_scores
    sumuje score na obszarze kwadratu i uśrednia po nakładających się oknach.
    To DOKŁADNIE ta sama funkcja, której używa BO — dlatego porównanie jest uczciwe.
    """
    observations: list[Observation] = []                          # lista okien
    for row in rows:                                              # każdy wiersz = jedno zapytanie
        size_raw = row.get("mask_size", "")                      # BO ma tę kolumnę, random/sliding nie
        size = int(float(size_raw)) if size_raw not in ("", None) else default_mask_size
        observations.append(
            Observation(
                cx=float(row["cx"]),                             # środek X kwadratu
                cy=float(row["cy"]),                             # środek Y kwadratu
                size=size,                                       # bok kwadratu
                score=float(row["score"]),                       # destrukcyjność tej okluzji
            )
        )
    return rasterize_window_scores(observations, image_size=image_size)  # -> mapa HxW (float)


def coverage_mask_from_rows(rows: list[dict[str, str]], image_size: int, default_mask_size: int) -> np.ndarray:
    """
    Zwraca maskę bool HxW: True tam, gdzie JAKIEKOLWIEK okno tej metody dotknęło
    piksela. Potrzebna do gęstnienia — odróżnia piksel "odwiedzony o wyniku 0" od
    piksela "nigdy nie odwiedzony". (saliency==0 NIE wystarcza: okno o score 0 też
    daje 0, a jest legalną obserwacją.)
    """
    mask = np.zeros((image_size, image_size), dtype=bool)
    for row in rows:
        size_raw = row.get("mask_size", "")
        size = int(float(size_raw)) if size_raw not in ("", None) else default_mask_size
        bounds = square_bounds(float(row["cx"]), float(row["cy"]), size, image_size)
        mask[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1] = True
    return mask


def densify_saliency(saliency: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """
    Domyka mapę: każdy piksel NIEODWIEDZONY dostaje wartość NAJBLIŻSZEGO
    odwiedzonego piksela (wypełnienie Voronoi przez transformatę odległości).
    Po tym kroku nie ma masy remisów zerowych, więc cały ranking pikseli jest
    sensowny i identycznie konstruowany dla każdej metody.

    Brzegowe przypadki:
      * całość odwiedzona  -> zwracamy bez zmian,
      * nic nie odwiedzone -> nie ma czym wypełnić, zwracamy bez zmian,
      * brak scipy         -> degradujemy do A1 (ostrzeżenie), zwracamy bez zmian.
    """
    if covered.all() or not covered.any():
        return saliency
    try:
        from scipy import ndimage  # import lokalny — zależność miękka
    except Exception as exc:
        print(f"UWAGA: scipy niedostępny ({exc}) — pomijam gęstnienie (zachowanie A1).", flush=True)
        return saliency
    # Dla każdego piksela indeks najbliższego piksela z covered==True.
    indices = ndimage.distance_transform_edt(~covered, return_distances=False, return_indices=True)
    return saliency[tuple(indices)]


def subsample_rows(rows: list[dict[str, str]], budget: int, seed_key: tuple[str, str]) -> list[dict[str, str]]:
    """
    Deterministycznie podpróbkowuje obserwacje do 'budget' sztuk, żeby zrównać
    liczbę zapytań między metodami (sliding ma ich z natury więcej niż bo/random).
    Losowanie (a nie 'pierwsze N') chroni przed biasem przestrzennym — np. siatka
    sliding zapisana wierszami zaczyna od rogu, więc 'pierwsze N' to lewy-górny pas.
    Ziarno wyprowadzone z (image_id, seed), więc wynik jest powtarzalny.
    """
    if budget is None or len(rows) <= budget:
        return rows
    # stable_seed (blake2b) jest powtarzalne MIĘDZY procesami, w odróżnieniu od
    # wbudowanego hash() (losowanego przez PYTHONHASHSEED).
    rng = np.random.default_rng(stable_seed("subsample", *seed_key, budget))
    chosen = rng.choice(len(rows), size=budget, replace=False)
    return [rows[i] for i in sorted(chosen.tolist())]


# ==============================================================================
# 5. JĄDRO: KRZYWA DELETION ALBO INSERTION
# ==============================================================================
@torch.no_grad()
def evaluate_curve(
    model: torch.nn.Module,
    image: torch.Tensor,        # oryginał, CxHxW, na device
    blurred: torch.Tensor,      # rozmyta kopia, CxHxW, na device
    ranking: torch.Tensor,      # indeksy pikseli posortowane od najważniejszych (long, device)
    fractions: np.ndarray,      # harmonogram ułamków
    baseline_mass: float,       # Σ sigmoid na czystym obrazie (do normalizacji)
    device: torch.device,
    mode: str,                  # "deletion" albo "insertion"
) -> np.ndarray:
    """
    Zwraca tablicę 'y' (znormalizowana siła predykcji) o długości == len(fractions).
      * deletion:  start = oryginał;  zastępujemy top-piksele rozmyciem.
      * insertion: start = rozmycie;  przywracamy top-piksele z oryginału.
    """
    n_pixels = ranking.shape[0]                          # ile pikseli ma obraz (H*W)
    ys: list[float] = []                                 # tu zbieramy kolejne wartości siły

    # Spłaszczone widoki C x (H*W) — żeby adresować piksele jednym indeksem.
    image_flat = image.view(image.shape[0], -1)          # oryginał, spłaszczony
    blurred_flat = blurred.view(blurred.shape[0], -1)    # rozmycie, spłaszczone

    for fraction in fractions:                           # idziemy po harmonogramie
        k = int(round(float(fraction) * n_pixels))       # ilu pikseli dotyczy ten krok

        if mode == "deletion":                           # --- DELETION ---
            work = image.clone()                         # zaczynamy od pełnego obrazu
            if k > 0:                                     # przy k=0 nic nie ruszamy (y=1.0)
                idx = ranking[:k]                        # k NAJWAŻNIEJSZYCH pikseli
                work_flat = work.view(work.shape[0], -1) # widok spłaszczony (współdzieli pamięć)
                work_flat[:, idx] = blurred_flat[:, idx] # "usuń" je -> wartość rozmyta
        else:                                            # --- INSERTION ---
            work = blurred.clone()                       # zaczynamy od obrazu rozmytego
            if k > 0:
                idx = ranking[:k]                        # k najważniejszych pikseli
                work_flat = work.view(work.shape[0], -1)
                work_flat[:, idx] = image_flat[:, idx]   # "przywróć" je z oryginału

        mass = sigmoid_mass(model, work, device)         # ile polipa model teraz widzi
        ys.append(mass / (baseline_mass + EPS))          # normalizacja do startu 1.0

    return np.asarray(ys, dtype=np.float64)


# ==============================================================================
# 6. AUC (pole pod krzywą) PO PRAWDZIWEJ OSI X
# ==============================================================================
def area_under_curve(fractions: np.ndarray, ys: np.ndarray) -> float:
    """
    Reguła trapezów. KLUCZOWE: całkujemy po faktycznych 'fractions' (nierówne
    odstępy z harmonogramu), a nie zakładając równomierne kroki — inaczej AUC
    byłoby przekłamane.
    """
    return float(_np_trapezoid(ys, fractions))


# ==============================================================================
# 7. ŁĄCZENIE DWÓCH LICZB W JEDNĄ (różnica i średnia geometryczna)
# ==============================================================================
def combine_scores(insertion_auc: float, deletion_auc: float) -> dict[str, float]:
    """
    Deletion i Insertion idą w PRZECIWNE strony, więc najpierw wyrównujemy:
        deletion_aligned = 1 - deletion_auc   (teraz większe = lepsze).
    Potem:
        diff    = insertion - deletion         (klasyczny "jeden wynik")
        geomean = sqrt(insertion * (1-deletion))  (karze nierównowagę)
    max(...,0) chroni sqrt, gdyby któraś składowa wyszła ujemnie (rzadki przypadek,
    gdy okluzja zwiększa przewidzianą masę).
    """
    deletion_aligned = 1.0 - deletion_auc
    diff = insertion_auc - deletion_auc
    geomean = math.sqrt(max(insertion_auc, 0.0) * max(deletion_aligned, 0.0))
    return {
        "deletion_aligned": deletion_aligned,
        "diff": diff,
        "geomean": geomean,
    }


# ==============================================================================
# 8. POMOCNICZE: czytanie CSV i grupowanie po (image_id, seed)
# ==============================================================================
def read_rows(path: Path) -> list[dict[str, str]]:
    """Wczytuje CSV jako listę słowników. Pusta lista, gdy pliku brak."""
    if not path.exists():
        print(f"UWAGA: brak pliku {path} — pomijam tę metodę.", flush=True)
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def group_by_image_seed(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    """Grupuje wiersze w słownik {(image_id, seed): [wiersze]}. Seed bywa pusty (sliding)."""
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["image_id"], row.get("seed", ""))     # sliding ma seed == ""
        groups.setdefault(key, []).append(row)
    return groups


# ==============================================================================
# 9. OPCJONALNY WYKRES (matplotlib jest opcjonalny — brak nie psuje modułu)
# ==============================================================================
def save_figure(
    fractions: np.ndarray,
    curves: dict[str, dict[str, np.ndarray]],   # {metoda: {"deletion": y, "insertion": y}}
    output_path: Path,
) -> None:
    """Rysuje uśrednione krzywe. Jeśli nie ma matplotlib — po prostu pomija wykres."""
    try:
        import matplotlib                       # import wewnątrz funkcji = zależność opcjonalna
        matplotlib.use("Agg")                   # backend bez okienka (zapis do pliku)
        import matplotlib.pyplot as plt
    except Exception as exc:                    # brak biblioteki lub błąd importu
        print(f"UWAGA: pomijam wykres (matplotlib niedostępny: {exc}).", flush=True)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_del, ax_ins) = plt.subplots(1, 2, figsize=(12, 5))  # dwa panele obok siebie
    for method, data in sorted(curves.items()):                  # po jednej linii na metodę
        line_del, = ax_del.plot(fractions, data["deletion"], marker=".", label=method)
        line_ins, = ax_ins.plot(fractions, data["insertion"], marker=".", label=method)
        # Pasmo ±1 std (gdy dostępne) — uwidacznia wariancję między obrazami,
        # szczególnie istotną przy małej próbie (pilot).
        if "deletion_std" in data:
            ax_del.fill_between(fractions, data["deletion"] - data["deletion_std"],
                                data["deletion"] + data["deletion_std"], alpha=0.15, color=line_del.get_color())
            ax_ins.fill_between(fractions, data["insertion"] - data["insertion_std"],
                                data["insertion"] + data["insertion_std"], alpha=0.15, color=line_ins.get_color())
    ax_del.set_title("Deletion (niżej = lepiej)")
    ax_del.set_xlabel("ułamek usuniętych pikseli"); ax_del.set_ylabel("siła predykcji")
    ax_ins.set_title("Insertion (wyżej = lepiej)")
    ax_ins.set_xlabel("ułamek dodanych pikseli"); ax_ins.set_ylabel("siła predykcji")
    ax_del.legend(); ax_ins.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Zapisano wykres: {output_path}", flush=True)


# ==============================================================================
# 10. GŁÓWNA PROCEDURA
# ==============================================================================
@torch.no_grad()
def main() -> None:
    args = parse_args()

    # --- 10a. urządzenie + model + dane ---------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)                  # zamrożony segmentator
    dataset = KvasirSegDataset(                                  # loader obrazów testowych
        root=args.data_root,
        split=args.split,
        image_size=args.image_size,
        augment=False,
        max_samples=args.max_samples,
    )

    # Mapujemy image_id -> obraz (CxHxW na device). Tylko obrazy z datasetu
    # (czyli z uwzględnieniem max_samples) będą oceniane.
    image_map: dict[str, torch.Tensor] = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        image_map[str(sample["image_id"])] = sample["image"].contiguous().to(device)

    # --- 10b. harmonogram porcji (wspólny dla wszystkich) ----------------------
    fractions = build_schedule(args.dense_until, args.dense_step, args.sparse_step)

    # Budujemy baseline raz — albo płaski mean image (RISE), albo Gauss per obraz.
    #
    # Mean image (domyślny, --baseline=mean):
    #   Petsiuk et al. "RISE: Randomized Input Sampling for Explanation of
    #   Black-box Models", BMVC 2018, sekcja 3.1 — autorzy używają średniego
    #   obrazu z datasetu jako punktu startowego insertion i końcowego deletion.
    #   Uzasadnienie: płaski obraz bez struktury przestrzennej daje predykcję ≈ 0,
    #   więc krzywa deletion ma realne "dno" do osiągnięcia. Jednocześnie wartości
    #   pikseli są in-distribution (model widział je podczas treningu).
    #
    # Gauss (alternatywny, --baseline=blur):
    #   Zachowuje lokalne średnie otoczenia, przez co predykcja baseline może być
    #   zbliżona do oryginału — deletion AUC "utyka" przy 1.0 i traci czułość.
    if args.baseline == "mean":
        stats = json.loads(args.stats_path.read_text(encoding="utf-8"))
        mean_rgb = torch.tensor(stats["mean_rgb"], dtype=torch.float32, device=device)
        _mean_baseline: torch.Tensor | None = mean_rgb.view(3, 1, 1).expand(3, args.image_size, args.image_size).contiguous()
    else:
        _mean_baseline = None

    # Cache baseline'ów i mas bazowych — liczone raz na obraz, nie raz na metodę.
    blurred_cache: dict[str, torch.Tensor] = {}
    baseline_mass_cache: dict[str, float] = {}

    def get_blurred(image_id: str, image: torch.Tensor) -> torch.Tensor:
        """Zwraca (i zapamiętuje) baseline dla obrazu — mean image (RISE) lub Gauss."""
        if image_id not in blurred_cache:
            if _mean_baseline is not None:
                blurred_cache[image_id] = _mean_baseline  # ten sam tensor dla każdego obrazu
            else:
                blurred = TF.gaussian_blur(image, kernel_size=[args.blur_kernel, args.blur_kernel],
                                           sigma=[args.blur_sigma, args.blur_sigma])
                blurred_cache[image_id] = blurred.contiguous()
        return blurred_cache[image_id]

    def get_baseline_mass(image_id: str, image: torch.Tensor) -> float:
        """Zwraca (i zapamiętuje) Σ sigmoid na czystym obrazie — mianownik normalizacji."""
        if image_id not in baseline_mass_cache:
            baseline_mass_cache[image_id] = sigmoid_mass(model, image, device)
        return baseline_mass_cache[image_id]

    # --- 10c. przejście po trzech metodach -------------------------------------
    method_files = {                                             # etykieta -> plik CSV
        "random": args.random_csv,
        "sliding": args.sliding_csv,
        "bo": args.bo_csv,
    }

    result_rows: list[dict[str, object]] = []                   # wiersze do CSV wynikowego
    # Akumulator krzywych do uśrednienia per metoda:
    curve_acc: dict[str, dict[str, list[np.ndarray]]] = {
        m: {"deletion": [], "insertion": []} for m in method_files
    }

    for method, csv_path in method_files.items():                # po jednej metodzie naraz
        groups = group_by_image_seed(read_rows(csv_path))        # {(image_id, seed): wiersze}

        for (image_id, seed), rows in groups.items():            # po każdej parze obraz+seed
            if image_id not in image_map:                        # obraz spoza wybranego podzbioru
                continue

            image = image_map[image_id]                          # oryginał na device
            blurred = get_blurred(image_id, image)               # baseline (rozmycie)
            baseline_mass = get_baseline_mass(image_id, image)   # mianownik normalizacji
            if baseline_mass < EPS:                              # model nic nie wykrył ->
                continue                                         # nie ma czego mierzyć, pomijamy

            # 0) (opcjonalnie) zrównaj budżet obserwacji między metodami.
            rows = subsample_rows(rows, args.match_budget, (image_id, seed))

            # 1) Zbuduj mapę saliency z zapytań tej metody (rasteryzacja okien).
            saliency = saliency_from_rows(rows, args.image_size, args.default_mask_size)

            # 1b) GĘSTNIENIE: domknij piksele nieodwiedzone (Voronoi). Bez tego
            #     masa remisów zerowych (różna per metoda) fałszuje ranking — patrz
            #     sekcja "WADA ARCHITEKTONICZNA…" w nagłówku modułu.
            if args.densify:
                covered = coverage_mask_from_rows(rows, args.image_size, args.default_mask_size)
                saliency = densify_saliency(saliency, covered)

            # 2) Ranking pikseli: od najważniejszych. argsort(-x) + stabilność,
            #    by ewentualne pozostałe remisy miały tę samą kolejność dla każdej metody.
            order = np.argsort(-saliency.reshape(-1), kind="stable")
            ranking = torch.from_numpy(order.copy()).long().to(device)

            # 3) Policz obie krzywe.
            deletion_y = evaluate_curve(model, image, blurred, ranking, fractions,
                                        baseline_mass, device, mode="deletion")
            insertion_y = evaluate_curve(model, image, blurred, ranking, fractions,
                                         baseline_mass, device, mode="insertion")

            # 4) Pola pod krzywymi.
            deletion_auc = area_under_curve(fractions, deletion_y)
            insertion_auc = area_under_curve(fractions, insertion_y)
            combined = combine_scores(insertion_auc, deletion_auc)

            # 5) Zapamiętaj wiersz wynikowy.
            result_rows.append({
                "image_id": image_id,
                "method": method,
                "seed": seed,
                "deletion_auc": deletion_auc,
                "insertion_auc": insertion_auc,
                "deletion_aligned": combined["deletion_aligned"],
                "diff": combined["diff"],
                "geomean": combined["geomean"],
            })

            # 6) Dorzuć krzywe do akumulatora (do uśrednienia na wykresie).
            curve_acc[method]["deletion"].append(deletion_y)
            curve_acc[method]["insertion"].append(insertion_y)

        print(f"Metoda '{method}': oceniono {sum(1 for r in result_rows if r['method'] == method)} par obraz+seed.",
              flush=True)

    # --- 10d. zapis wyników per wiersz (CSV, nadpisanie) -----------------------
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if result_rows:
        with args.output_csv.open("w", newline="") as file:      # "w" = świeży plik
            writer = csv.DictWriter(file, fieldnames=list(result_rows[0].keys()))
            writer.writeheader()
            writer.writerows(result_rows)

    # --- 10e. agregaty per metoda (średnia ± odchylenie) -----------------------
    def mean(values: list[float]) -> float:
        return statistics.fmean(values) if values else float("nan")

    def stdev(values: list[float]) -> float:
        return statistics.stdev(values) if len(values) > 1 else 0.0

    summary: dict[str, object] = {}
    for method in method_files:                                  # podsumuj każdą metodę
        rows_m = [r for r in result_rows if r["method"] == method]
        if not rows_m:
            continue
        summary[method] = {
            "n": len(rows_m),
            "deletion_auc_mean": mean([float(r["deletion_auc"]) for r in rows_m]),
            "deletion_auc_std": stdev([float(r["deletion_auc"]) for r in rows_m]),
            "insertion_auc_mean": mean([float(r["insertion_auc"]) for r in rows_m]),
            "insertion_auc_std": stdev([float(r["insertion_auc"]) for r in rows_m]),
            "deletion_aligned_mean": mean([float(r["deletion_aligned"]) for r in rows_m]),
            "diff_mean": mean([float(r["diff"]) for r in rows_m]),
            "geomean_mean": mean([float(r["geomean"]) for r in rows_m]),
        }

    # Dorzucamy do podsumowania ustawienia eksperymentu (powtarzalność).
    summary["inputs"] = {
        "random_csv": str(args.random_csv),
        "sliding_csv": str(args.sliding_csv),
        "bo_csv": str(args.bo_csv),
        "image_size": args.image_size,
        "default_mask_size": args.default_mask_size,
        "blur_kernel": args.blur_kernel,
        "blur_sigma": args.blur_sigma,
        "schedule_points": int(len(fractions)),
        "dense_until": args.dense_until,
        "dense_step": args.dense_step,
        "sparse_step": args.sparse_step,
        "max_samples": args.max_samples,
        "baseline": args.baseline,
        "densify": args.densify,
        "match_budget": args.match_budget,
        "observations_per_group": {
            method: sorted({len(rows) for rows in group_by_image_seed(read_rows(path)).values()})
            for method, path in method_files.items()
        },
        "note": (
            "Mapy budowane jednolicie: rasteryzacja obserwacji + gęstnienie Voronoi "
            "(densify) dla random/sliding/bo. Gdy observations_per_group różni się "
            "między metodami, budżet jest konfounderem — rozważ --match-budget."
        ),
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Metadane runu obok CSV (spójnie z resztą pipeline'u).
    write_run_metadata(args.output_csv, {"method": "faithfulness", **summary["inputs"]})

    # --- 10f. opcjonalny wykres (uśrednione krzywe) ----------------------------
    if args.figure is not None:
        averaged: dict[str, dict[str, np.ndarray]] = {}
        for method, data in curve_acc.items():
            if not data["deletion"]:                             # metoda bez wyników -> pomiń
                continue
            del_stack = np.stack(data["deletion"])
            ins_stack = np.stack(data["insertion"])
            averaged[method] = {
                "deletion": np.mean(del_stack, axis=0),
                "insertion": np.mean(ins_stack, axis=0),
                "deletion_std": np.std(del_stack, axis=0),
                "insertion_std": np.std(ins_stack, axis=0),
            }
        if averaged:
            save_figure(fractions, averaged, args.figure)

    # --- 10g. wypisz podsumowanie na ekran -------------------------------------
    print(json.dumps(summary, indent=2))
    print(f"Zapisano wyniki per wiersz: {args.output_csv}")
    print(f"Zapisano podsumowanie:      {args.summary_json}")


if __name__ == "__main__":
    main()

# ==============================================================================
# EFEKT KOŃCOWY TEGO PLIKU
# ------------------------------------------------------------------------------
# Po uruchomieniu (samodzielnie lub z pipeline'u) powstają:
#   * outputs/metrics/faithfulness_<profile>.csv   — surowe AUC per (obraz, metoda, seed)
#   * outputs/metrics/faithfulness_<profile>.json  — agregaty per metoda + ustawienia
#   * outputs/figures/faithfulness_curves_<profile>.png (jeśli podano --figure i jest matplotlib)
#
# INTERPRETACJA:
#   * deletion_auc  — NIŻEJ = lepiej (dobra mapa szybko niszczy predykcję)
#   * insertion_auc — WYŻEJ = lepiej (dobra mapa szybko ją odbudowuje)
#   * diff / geomean — jeden skalar do rankingu BO vs random vs sliding (WYŻEJ = lepiej)
#
# Jeśli BO ma wyższe diff/geomean niż random/sliding, jego mapy są WIERNIEJSZE —
# czyli faktycznie wskazują piksele, na których polega model.
#
# UWAGA METODOLOGICZNA: wyniki mają sens TYLKO przy włączonym gęstnieniu
# (domyślnie). Przy --no-densify (stare A1) druga połowa krzywej bo/random jest
# sterowana kolejnością rastrową remisów zerowych, nie saliency — patrz sekcja
# "WADA ARCHITEKTONICZNA…" w nagłówku. Gdy liczba obserwacji różni się między
# metodami (np. sliding >> bo/random), zgłaszaj to lub użyj --match-budget.
# ==============================================================================
