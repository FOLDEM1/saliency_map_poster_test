from __future__ import annotations
import argparse                      
import csv                           
import json                          
import math                         
import statistics                    
from pathlib import Path            
import numpy as np
import torch                        
import torchvision.transforms.functional as TF 


from src.dataset import KvasirSegDataset          
from src.evaluate import load_model               
from src.experiment_io import stable_seed, write_run_metadata  
from src.occlusion import square_bounds   
from src.saliency import Observation, rasterize_window_scores 



"""
Funkcja czytająca paramtetry testu z wiersza poleceń
"""
def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(description="Insertion/Deletion faithfulness of saliency maps.")
    # --data-root -> Ścieżka do głownego katalogu z danymi i maskami
    # --checkpoint -> Ścieżka do pliku punktu kontrolnego/wag modelu
    # --split -> wybór który zbiór ma być ewaluowany train/validation/test
    # --image-size -> Rozmiar (kwadratu) na jakim model był trenowany
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/kvasir-seg"),help='Ścieżka do katalogu danych')
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/unet_best.pt"),help="Ścieżka do pliku punktu kontrolnego modelu")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"],help='Zbiór danych do oceny, jeden z train/validation/test')
    parser.add_argument("--image-size", type=int, default=256)
    # --default-mask-size -> domyślny rozmiar maski, jeżeli nie występuje w pliku csv . BO zapisuje mask_size oraz jej rozmiar musi zgadzać się z rozmiarem użytym w fazie RUN 
    parser.add_argument("--default-mask-size", type=int, default=48,help='Rozmiar okna okluzji jezeli nie ma go w pliku CSV, musi zgadzać się z rozmiarem z fazy RUN ')
    
    # --*-csv -> Pliki *.csv z historią zapytań , jeden na metode
    parser.add_argument("--random-csv", type=Path, required=True)
    parser.add_argument("--sliding-csv", type=Path, required=True)
    parser.add_argument("--bo-csv", type=Path, required=True)

    # --stats-path -> Ścieżka z której będą wczytywane dane dla każdego kanału RGB z fazy treningowej do obliczenia "neutralnego" koloru otoczenia, potrzebny gdy --baseline mean
    parser.add_argument("--stats-path", type=Path, default=Path("outputs/metrics/train_channel_stats.json"))
    # --baseline -> wartość wypełnienia pikselu które zostaną usunięte w fazie DELETION oraz wartość wypełnienia całego obrazu w fazie INSERTION
    # RISE: Randomized Input Sampling for Explanation of Black-box Models https://arxiv.org/abs/1806.07421
    parser.add_argument("--baseline", type=str, default="mean", choices=["mean", "blur"],help="mean płaski obraz wyliczany na podstawie statystyk z fazy treningowej/ blur(Gaussian blur)")
    
    # --blur-kernel -> Szerokość rozmycia kernela, musi być to liczba nieparzysta, żeby piksel miał konkretny środek
    # --blur-sigma -> Moc rozmycia wewnątrz kernela
    parser.add_argument("--blur-kernel", type=int, default=31, help="Rozmiar kernela (liczba nieparzysta).")
    parser.add_argument("--blur-sigma", type=float, default=7.0, help="Sigma - siła rozmycia ")

    # Parametry harmonogramu
    # --dense-until -> do jakiego progu usuwamy gęsto najważniejsze fragmenty np do 20%
    # --dense-step -> z jakim krokiem próbkujemy gęsto np co 1%
    # --sparse-step -> kiedy przekroczymy próg np 20% gęstego próbkowania, zwiększamy go do np 5%

    parser.add_argument("--dense-until", type=float, default=0.20, help="Do jakiego ułamka obrazu próbkujemy gęsto")
    parser.add_argument("--dense-step", type=float, default=0.01, help="Krok gęstego próbkowania ")
    parser.add_argument("--sparse-step", type=float, default=0.05, help="Krok próbkowania po przekroczeniu progu")

    
    # --no-densify -> W przypadku gdy metody random oraz BO mogą nie mieć kontaktu z wszystkimi pikselami
    #                 więc w mapie będą miały wartość 0, lecz w przypadku tworzenia rankingu pojawia się mnóstwo pikseli z wartością 0. Gdy są sortowane nie dzieje się to głównie względem mapy lecz też kolejności rastrowej .
    #                 Wypełnienie wartości za pomocą Voronoi na podstawie transformaty odległości pozwala zaadresować ten problem.
    parser.add_argument("--no-densify", dest="densify", action="store_false",help="Ustawienie tej flagi spowoduje brak użycia wypełnienia Voronoi fragmentów obrazów nie odwiedzonych przez metody, ranking pikseli nieodwiedzonych będzie wtedy oparty na przypadkowej kolejności rastrowej - bias przestrzenny")
    parser.set_defaults(densify=True)


    # --match-budget -> flaga pozwalająca wymusić na każdej metodzie budżet punktów zapytań na jeden obraz
    parser.add_argument("--match-budget", type=int, default=None,
                        help="Podpróbkuj każdą metodę do N obserwacji per (obraz, seed).")
    # --max-samples -> liczy metryki dla pierwszych N obrazów:
    parser.add_argument("--max-samples", type=int, default=None)

    # Ścieżki gdzie mają być zapisane wyniki
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/metrics/mask_eval_scores_output.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("outputs/metrics/mask_eval_scores_summary.json"))
    parser.add_argument("--figure", type=Path, default=None, help="Flaga czy generować wykres z wynikami czy tez nie")
    return parser.parse_args()


"""
Przygotowania planu próbkowania

"""

def build_schedule(dense_until: float, dense_step: float, sparse_step: float) -> np.ndarray:
    """
    Zwraca np.array z wartosciami kroków dla INSERTION/DELETION 
    """
    # część gdzie próbkujemy gęsto
    dense = np.arange(0.0, dense_until + 1e-9, dense_step)
    # część gdzie próbkujemy rzadziej, od dense_until do 1
    sparse = np.arange(dense_until, 1.0 + 1e-9, sparse_step)
    # połączenie tablic w jedną + gwarancja zawierania wartości 0,1
    fractions = np.concatenate([dense, sparse, [0.0, 1.0]])
    # usunięcie wartości związanych z + epsilon
    fractions = np.clip(fractions, 0.0, 1.0)
    # usuwamy duplikaty i sortujemy rosnąco
    return np.unique(fractions)


""" 
Siła predykcji wystąpeinia polipa , która potem będzie przeskalowana w zależności od masy obrazu
"""
@torch.no_grad() 
def sigmoid_mass(model: torch.nn.Module, image_chw: torch.Tensor, device: torch.device) -> float:
    """
    Zlicza ile model aktualnie widzi polipa poprzez sume prawdobieństwa, że dany piksel należy do polipa
    """
    logits = model(image_chw.unsqueeze(0).to(device)) # przepuszczamy obraz przez model
    return float(torch.sigmoid(logits).sum().item()) # bierzemy siłę pewności że po zasłonięciu polipa nadal występuje na obrazie ze wszystkich pikseli

"""
Łączenie obserwacji do budowy saliency map z danych z CSV
"""

def saliency_from_rows(rows: list[dict[str, str]], image_size: int, default_mask_size: int) -> np.ndarray:
    """
    Nakładamy wartości z okien ,żeny potem je uśrednić  
    """
    #lista okien
    observations: list[Observation] = []
    # dla kazdego zapytania  
    for row in rows:  
        # Jeżeli to BO , ma on parametr mask_size                                           
        size_raw = row.get("mask_size", "")
        # zamieniamy wartość z CSV na wartość liczbową 
        size = int(float(size_raw)) if size_raw not in ("", None) else default_mask_size
        observations.append(
            Observation(
                # środeki x i y kwadratu
                cx=float(row["cx"]),
                cy=float(row["cy"]),

                # bok kwadratu
                size=size,

                # moc szkody okluzji
                score=float(row["score"]),))
    return rasterize_window_scores(observations, image_size=image_size)  #zwracamy mape


def coverage_mask_from_rows(rows: list[dict[str, str]], image_size: int, default_mask_size: int) -> np.ndarray:
    """
    Zwraca informacje czy dany piksel zawiera się w jakiej kolwiek masce,
    Jeżeli nie oraz flaga --no-densify nie była ustawiona, będzie to podstawą do wypełnienia ich odpowiednimi wartościami
    """
    # incjalizujemy zmienną wynikową, o wymiarach odpowiadającym obrazowi wypełnionymi wartościami False/0
    mask = np.zeros((image_size, image_size), dtype=bool)
    #Dla każdego okna 
    for row in rows:
        # wyciagamy wartość mask_size jeżeli nie w jej miejsce wstawiamy domyślną wartość
        size_raw = row.get("mask_size", "")
        #podwójna konwersja w przypadku str:"123.0"->float:123.0 -> int:123
        size = int(float(size_raw)) if size_raw not in ("", None) else default_mask_size
        #wyznaczamy za pomocą funckji square_bounds granice okna/kwadratu
        bounds = square_bounds(float(row["cx"]), float(row["cy"]), size, image_size)
        # kazdy kwadrat zamalowuje swój obszar
        mask[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1] = True
    return mask # zwracamy maske


def densify_saliency(saliency: np.ndarray, covered: np.ndarray) -> np.ndarray:
    
    """
    Każdy piksel z mapy który był nie odwiedzony oraz nie występuje flaga -no-denisfy, otrzyma wartość najbliższego sąsiada
    z pikseli które zostały otagowane przez okna za pomcą wypełnienia Voronoi na podstawie transformaty odległości
    """
    # jezeli kazdy piksel został pokryty zwracamy mape bez modyfikacji
    if covered.all() or not covered.any():
        return saliency
    try:
        # lokalnie importujemy scipy , jezeli go nie wyswietlamy komunikat oraz zwracamy surową mape
        from scipy import ndimage
    except Exception as exc:
        print(f"Brak scipy :({exc}) fallback do zachowania bez wypełniania ", flush=True)
        return saliency
    # Dla każdego piksela tam gdzie False wstaw indeks najbliższego piksela/sąsiada
    indices = ndimage.distance_transform_edt(~covered, return_distances=False, return_indices=True)
    # wypełniamy mape i zwracamy ją 
    return saliency[tuple(indices)]


def subsample_rows(rows: list[dict[str, str]], budget: int, seed_key: tuple[str, str]) -> list[dict[str, str]]:

    """ 
    funkcja ścinająca liczbę okien, do wyzanczonego budżetu, (żeby sliding nie miał niesprawiedliwej przewagi związną z liczbą zapytań)
    """
    # gdy budżetu nie ma lub jest więskzy od liczby obserwacji, nie ma czego scinac
    if budget is None or len(rows) <= budget:
        return rows
    # tworzymy generator losowy podanym ziarnem , lecz taki zeby był deterministyczny co uruchomenia
    rng = np.random.default_rng(stable_seed("subsample", *seed_key, budget))
    # losujemy indeksy okien o liczności odpowiadjącemu budżetowi (bez powtórzeń)
    chosen = rng.choice(len(rows), size=budget, replace=False)
    # zwracamy wybrane kwadaraty okluzji
    return [rows[i] for i in sorted(chosen.tolist())]


"""
Krzywe deletion/insertion
"""
@torch.no_grad()
def evaluate_curve(
    #model 
    model: torch.nn.Module,
    #oryginalny obraz
    image: torch.Tensor, 
    #rozmyta/płaska kopia
    blurred: torch.Tensor,
    # posortowane indeksy pikseli
    ranking: torch.Tensor,
    # harmonogram kroków z "schedulera"
    fractions: np.ndarray,
    # Suma prawodpobienst pikseli należności do polipa
    baseline_mass: float,
    device: torch.device,
    # tryp deletion/insertion
    mode: str) -> np.ndarray:
  
    """ 
    Zwraca liczbe "ile model widzi polipa" przy usuwaniu/dodawaniu fragemntów polipa
    deletion -> zaczynamy od oryginalanego obrazu i usuwamy te piksele które miały największą szanse przynależności do polipa
    insertion -> do rozmytego zaczynamy dodawac najlepsze fragmenty i patrzymy na poprawe jakości modelu
    """
    
    ys: list[float] = []
    # Spłaszczone widoki poniewaz indeksy to lista 1D
    image_flat = image.view(image.shape[0], -1) # spłasczony orginał
    blurred_flat = blurred.view(blurred.shape[0], -1) # spłaszczone rozmycie

    # po każdej frakcji z schedulera
    for fraction in fractions:
        #ile pikseli z obrazka bierzemy pod uwage
        k = int(round(float(fraction) *ranking.shape[0]))
        # jezeli tryb to deletion
        if mode == "deletion":
            #kopiujemy obraz 
            work = image.clone()
            #jezeli liczba pikseli to conajmniej 1 (moze sie zdarzyc przy zaoorkglaniu i malych frakcjach)
            if k > 0: 
                idx = ranking[:k] # bierzemy k najlepszych pikseli
                # zamazujemy je , uzwajac widoku
                work_flat = work.view(work.shape[0], -1) 
                work_flat[:, idx] = blurred_flat[:, idx]
        else: # insertion
            # robimy kopie obrazu rozmytego
            work = blurred.clone()
            if k > 0:
                idx = ranking[:k] # bierzemy k najlepszych pikseli
                # tworzymy widok na obraz i wstawiamy do zamazanego obrazu najlepsze piksele
                work_flat = work.view(work.shape[0], -1)
                work_flat[:, idx] = image_flat[:, idx]
        # liczmy ile poliba model widzi aktualnie
        mass = sigmoid_mass(model, work, device)
        # normalizujemy wartosci oraz dodanie wyniku do listy
        ys.append(mass / (baseline_mass + 1e-7))
    # zwracamy liste wynikow 
    return np.asarray(ys, dtype=np.float64)

def area_under_curve(fractions: np.ndarray, ys: np.ndarray) -> float:
    """
    Liczymy pole pod krzywą jakosci modelu do frakcji usunietych/dodanych elementow
    """
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapz(ys, fractions))

"""
łączenie wyników metryk w jedną liczbe 
"""
def combine_scores(insertion_auc: float, deletion_auc: float) -> dict[str, float]:
    """
    Metryki łączące dwa wartosci AUC z insertion/deletion w jedną liczbe
    wybrano G-mean za karanie odstępstw od wartości , jednakże można by użyć dowolnej "lubianej" metryki
    """
    deletion_aligned = max(1.0 - deletion_auc,0)
    
    diff = insertion_auc - deletion_auc
    geomean = math.sqrt(deletion_aligned*min(insertion_auc,1))
    # zwracamy dict wartosci poniewaz samo diff moze tez być elmentem wartym uwagi
    return {
        "deletion_aligned": deletion_aligned,
        "diff": diff,
        "geomean": geomean}

"""
funkcje pomocnicze do czytania i grupowania po (image_id,seed)

"""
def read_rows(path: Path) -> list[dict[str, str]]:
    """
    Funckja czyta dane z CSV jako liste słowników. Gdy pliku nie ma zwracana jest pusta lista
    """
    if not path.exists():
        print(f"brak pliku {path} -> zwrocono pusta liste", flush=True)
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def group_by_image_seed(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    """ 
    Funkcja grupuje wiersze w słownik {(image_id, seed): [wiersze]}. Seed moze być pusty (sliding)
    """
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["image_id"], row.get("seed", "")) # sliding ma seed = ""/pusty
        groups.setdefault(key, []).append(row)
    return groups

"""
Wykresy 
"""
def save_figure(
    # np.array frakcji z "schedulera"
    fractions: np.ndarray,
    # {metoda: {"deletion": y, "insertion": y}}
    curves: dict[str, dict[str, np.ndarray]], 
    # ścieżka docelowa na zapis
    output_path: Path) -> None:
    """
    Funkcja rysuje uśrednione krzywe Deletion/Insertion. Jeśli brakuje bilbioteki matplotlib , pomija wykres.
    """
    try:
        #import wewntętrzny
        import matplotlib
        #bez okienek  
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Brak biblioteki matplotlib {exc}).", flush=True)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # dwa wykresy obok siebie
    fig, (ax_del, ax_ins) = plt.subplots(1, 2, figsize=(12, 5))
    # po jednej linii na metode 
    for method, data in sorted(curves.items()):
        line_del,= ax_del.plot(fractions, data["deletion"], marker=".", label=method)
        line_ins,= ax_ins.plot(fractions, data["insertion"], marker=".", label=method)
        # dodatkowo odchylenia standardowe przdatne przy oglądaniu wyników
        if "deletion_std" in data:
            ax_del.fill_between(fractions, data["deletion"] - data["deletion_std"],data["deletion"] + data["deletion_std"], alpha=0.15, color=line_del.get_color())
            ax_ins.fill_between(fractions, data["insertion"] - data["insertion_std"],data["insertion"] + data["insertion_std"], alpha=0.15, color=line_ins.get_color())
    ax_del.set_title("Deletion (niżej -> lepiej)")
    ax_del.set_xlabel("ułamek usuniętych pikseli"); ax_del.set_ylabel("siła predykcji")
    ax_ins.set_title("Insertion (wyżej -> lepiej)")
    ax_ins.set_xlabel("ułamek dodanych pikseli"); ax_ins.set_ylabel("siła predykcji")
    ax_del.legend(); ax_ins.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Zapisano wykres: {output_path}", flush=True)

""" 
main modulu 
"""
@torch.no_grad()
def main() -> None:
    
    args = parse_args()

    # jezeli mozemy uzywamy gpu jezeli nie cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #wczytujemy segmentator
    model = load_model(args.checkpoint, device)
    #loader obrazów
    dataset = KvasirSegDataset(root=args.data_root,split=args.split,image_size=args.image_size, augment=False, max_samples=args.max_samples)

    # Mapowanie image_id na obraz 
    image_map: dict[str, torch.Tensor] = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        # view wymaga ciągłego układu pamięci ianczej mogą być błędy
        image_map[str(sample["image_id"])] = sample["image"].contiguous().to(device)

    # Tworzymy scheduler frakcji gęsto/rzadko
    fractions = build_schedule(args.dense_until, args.dense_step, args.sparse_step)

    # Czytamy dane w zależności od parametrów potrzbnych do obliczenia obrazu bazowego do insertion/deletion
    if args.baseline == "mean":
        stats = json.loads(args.stats_path.read_text(encoding="utf-8"))
        mean_rgb = torch.tensor(stats["mean_rgb"], dtype=torch.float32, device=device)
        mean_baseline: torch.Tensor | None = mean_rgb.view(3, 1, 1).expand(3, args.image_size, args.image_size).contiguous()
    else:
        mean_baseline = None

    # Cache mas i obrazów bazowych
    blurred_cache: dict[str, torch.Tensor] = {}
    baseline_mass_cache: dict[str, float] = {}

    def get_blurred(image_id: str, image: torch.Tensor) -> torch.Tensor:
        """
        Funckja tworzy i zwraca obraz bazowy
        """
        if image_id not in blurred_cache:
            if mean_baseline is not None:
                # Jeden tensor dla wszystkich obrazów
                blurred_cache[image_id] = mean_baseline
            else:
                #rozmycie gausowskie zgodnie z podanymi parametrami
                blurred = TF.gaussian_blur(image, kernel_size=[args.blur_kernel, args.blur_kernel],sigma=[args.blur_sigma, args.blur_sigma])
                blurred_cache[image_id] = blurred.contiguous()
        return blurred_cache[image_id]

    def get_baseline_mass(image_id: str, image: torch.Tensor) -> float:
        """
        Zwraca sumę wartości prawdopodobieństw dla oryginalnego obrazu. Przyda się do normalizacji
        """
        if image_id not in baseline_mass_cache:
            baseline_mass_cache[image_id] = sigmoid_mass(model, image, device)
        return baseline_mass_cache[image_id]

    # Ścieżki do rezultatów eksperymentów
    method_files = {
        "random": args.random_csv,
        "sliding": args.sliding_csv,
        "bo": args.bo_csv,
    }
    #miejsce na wyniki
    result_rows: list[dict[str, object]] = []
    # Agregat krzywych do uśrednienia dla każdej z  metod
    curve_agg: dict[str, dict[str, list[np.ndarray]]] = {m: {"deletion": [], "insertion": []} for m in method_files}
    
    #dla każdej z metod
    for method, csv_path in method_files.items(): 
        # {(image_id, seed): rows}
        groups = group_by_image_seed(read_rows(csv_path))        
        #dla każdej pary (obraz,seed)
        for (image_id, seed), rows in groups.items():
            #jezeli nie mamy mapy dla danego obrazy
            if image_id not in image_map:
                continue

            image = image_map[image_id]
            #obraz bazowy -> rozmyty w wybrany sposób
            blurred = get_blurred(image_id, image)
            # minownik do normalizacji
            baseline_mass = get_baseline_mass(image_id, image)
            # model niczego nie wykrył więc go pomijamy
            if baseline_mass < 1e-7:
                continue

            # Równoważymy budżet
            rows = subsample_rows(rows, args.match_budget, (image_id, seed))

            # budujemy mapę saliency z zapytań danej metody/rasteryzacja okien
            saliency = saliency_from_rows(rows, args.image_size, args.default_mask_size)


            # jeżeli wybranie domknięcie za pomocą Voronoi, tak też robimy
            if args.densify:
                covered = coverage_mask_from_rows(rows, args.image_size, args.default_mask_size)
                saliency = densify_saliency(saliency, covered)

            # posortowanie najważniejszych pikseli oraz zachowanie stabilności kolejności pikseli na obrazie
            order = np.argsort(-saliency.reshape(-1), kind="stable")
            ranking = torch.from_numpy(order.copy()).long().to(device)

            # Obliczamy krzywe insertion/deletion
            deletion_y = evaluate_curve(model, image, blurred, ranking, fractions,
                                        baseline_mass, device, mode="deletion")
            insertion_y = evaluate_curve(model, image, blurred, ranking, fractions,
                                         baseline_mass, device, mode="insertion")

            # AUC.
            deletion_auc = area_under_curve(fractions, deletion_y)
            insertion_auc = area_under_curve(fractions, insertion_y)
            combined = combine_scores(insertion_auc, deletion_auc)

            # Dodajemy wiersz wynikowy
            result_rows.append({
                "image_id": image_id,"method": method,"seed": seed,"deletion_auc": deletion_auc, "insertion_auc": insertion_auc,"deletion_aligned": combined["deletion_aligned"],"diff": combined["diff"],"geomean": combined["geomean"]})

            # Dodajemy krzywe do aggregata, do wykresów.
            curve_agg[method]["deletion"].append(deletion_y)
            curve_agg[method]["insertion"].append(insertion_y)

        print(f"Metoda: '{method}' ->  oceniono {sum(1 for r in result_rows if r['method'] == method)} par (obraz,seed)",
              flush=True)

    #Zapisanie/nadpisanie wyników do CSV
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if result_rows:
        with args.output_csv.open("w", newline="") as file:      # "w" = świeży plik
            writer = csv.DictWriter(file, fieldnames=list(result_rows[0].keys()))
            writer.writeheader()
            writer.writerows(result_rows)

    #aggreagty dla metod -> średnia +- std
    def mean(values: list[float]) -> float:
        return statistics.fmean(values) if values else float("nan")

    def stdev(values: list[float]) -> float:
        return statistics.stdev(values) if len(values) > 1 else 0.0

    
    #incjalizacja slownika na wyniki
    summary: dict[str, object] = {}
    # wpisanie danych do slownika wynikowego
    for method in method_files:
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

    # Dopsainei do summary paramterów wprowadzonych eskperymentu
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
        }
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Metadane przebiegu obok CSV .
    write_run_metadata(args.output_csv, {"method": "faithfulness", **summary["inputs"]})

    # tworzenie i zapisywanie wykresów
    if args.figure is not None:
        averaged: dict[str, dict[str, np.ndarray]] = {}
        for method, data in curve_agg.items():
            if not data["deletion"]:
                continue
            del_stack = np.stack(data["deletion"])
            ins_stack = np.stack(data["insertion"])
            #średnie +- odchylenia dla obu metod
            averaged[method] = {
                "deletion": np.mean(del_stack, axis=0),
                "insertion": np.mean(ins_stack, axis=0),
                "deletion_std": np.std(del_stack, axis=0),
                "insertion_std": np.std(ins_stack, axis=0),
            }
        if averaged:
            save_figure(fractions, averaged, args.figure)

    # wyświetlenie wyników eksperymentu
    print(json.dumps(summary, indent=2))
    print(f"Zapisano wyniki per wiersz: {args.output_csv}")
    print(f"Zapisano podsumowanie:      {args.summary_json}")


if __name__ == "__main__":
    main()
