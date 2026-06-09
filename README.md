# BO-SegOcc Experiment README

## Cel

Projekt implementuje black-boxową analizę wrażliwości okluzyjnej dla binarnej segmentacji polipów. Model segmentacyjny jest traktowany jako zamrożona funkcja:

```text
image -> segmentation prediction
```

Eksperyment nie wymaga gradientów, wag ani aktywacji modelu. Analiza polega wyłącznie na zapytaniach inferencyjnych do gotowego segmentatora.

## Idea Metody

Dla obrazu wejściowego `x` model generuje predykcję bazową:

```text
P0 = sigmoid(model(x))
```

Następnie wybrany region obrazu jest maskowany średnim kolorem kanałów RGB ze zbioru treningowego. Dla obrazu po okluzji model generuje:

```text
Ptheta = sigmoid(model(mask(x, theta)))
```

Funkcja celu mierzy zmianę predykcji segmentacyjnej:

```text
J(theta) = 1 - SoftDice(P0, Ptheta)
```

Duża wartość `J(theta)` oznacza, że zamaskowany region silnie zmienił predykcję modelu.

Ground truth nie jest używany do wyboru maski. Maska referencyjna służy tylko do późniejszej oceny, czy znaleziony region pokrywa się z polipem.

## Porównywane Metody

Pipeline uruchamia cztery warianty:

```text
random occlusion
sliding window occlusion
BO fixed-size:      theta = (cx, cy)
BO variable-size:   theta = (cx, cy, size)
```

`BO fixed-size` służy do czystego porównania lokalizacji maski przy stałej skali perturbacji.

`BO variable-size` jest bliższy paperowi `Black-Box Saliency Map Generation Using Bayesian Optimisation`, bo optymalizuje jednocześnie pozycję i rozmiar okna.

## Główne Pliki

```text
main.py                         root entrypoint
src/main.py                     orchestrator pipeline'u
src/dataset.py                  loader Kvasir-SEG
src/model.py                    U-Net z segmentation_models_pytorch
src/evaluate.py                 ewaluacja checkpointu
src/occlusion.py                wspólna logika okluzji i funkcji celu
src/run_random.py               random occlusion
src/run_sliding.py              sliding window
src/run_bo.py                   BO fixed-size i variable-size
src/saliency.py                 mapy saliency z Gaussian Process
src/analyze_occlusion.py        analiza wyników metod
src/validate_occlusion_runs.py  walidacja kompletności CSV
```

## Najważniejsze Polecenie

Jeżeli masz już wytrenowany model w:

```text
checkpoints/unet_best.pt
```

i przygotowane dane w:

```text
data/raw/kvasir-seg/
```

uruchom świeży eksperyment black-box bez trenowania:

```bash
uv run python main.py \
  --profile pilot \
  --skip-train \
  --skip-download \
  --keep-checkpoint \
  --no-fresh-data
```

To czyści `outputs/`, ale zachowuje dane i checkpoint.

## Sprawdzenie Bez Uruchamiania

Przed startem można wypisać plan komend:

```bash
uv run python main.py \
  --profile pilot \
  --skip-train \
  --skip-download \
  --keep-checkpoint \
  --no-fresh-data \
  --dry-run
```

## Profile Pipeline'u

```text
smoke  szybki test techniczny
pilot  sensowny eksperyment roboczy
full   pełny eksperyment
```

Rekomendowany tryb do uzyskania świeżych wyników bez dużego kosztu:

```bash
uv run python main.py --profile pilot --skip-train --skip-download --keep-checkpoint --no-fresh-data
```

Pełny run od zera, razem z pobraniem danych i treningiem:

```bash
uv run python main.py --profile full
```

Ten wariant usuwa stare dane, checkpointy i wyniki, dlatego nie jest rekomendowany, jeśli celem jest tylko black-box test gotowego modelu.

## Co Tworzy Pipeline

Po uruchomieniu powstają:

```text
outputs/metrics/
outputs/occlusion_runs/
outputs/saliency_maps/
outputs/figures/
outputs/predictions/
```

Najważniejsze pliki metryk:

```text
outputs/metrics/test_metrics.json
outputs/metrics/validation_metrics.json
outputs/metrics/occlusion_comparison_fixed_size_pilot.json
outputs/metrics/occlusion_comparison_variable_size_pilot.json
```

Najważniejsze CSV:

```text
outputs/occlusion_runs/random_pilot.csv
outputs/occlusion_runs/sliding_pilot.csv
outputs/occlusion_runs/bo_fixed_size_pilot.csv
outputs/occlusion_runs/bo_variable_size_pilot.csv
```

Każdy CSV zapisuje pełną historię zapytań:

```text
image_id
method
seed
step
cx
cy
mask_size
score
best_score
best_cx
best_cy
best_mask_size
overlap metrics
elapsed_sec
```

## Mapy Saliency

Dla BO generowane są mapy saliency z posterior mean Gaussian Process:

```text
outputs/saliency_maps/bo_fixed_size/
outputs/saliency_maps/bo_variable_size/
```

Dla każdego obrazu i seeda powstają:

```text
<image_id>_seed<seed>.npy
<image_id>_seed<seed>_saliency.png
<image_id>_seed<seed>_overlay.png
```

Overlay z `bo_variable_size` jest najlepszym materiałem wizualnym do pokazania finalnej mapy saliency.

## Interpretacja Wyników

W pliku:

```text
outputs/metrics/occlusion_comparison_variable_size_pilot.json
```

sprawdź:

```text
bo_budget.best_score_mean
random_budget.best_score_mean
sliding_full.best_score_mean
pairwise_delta_best_score
```

Jeżeli `bo_budget.best_score_mean` jest wyższe niż `random_budget.best_score_mean`, BO znajduje bardziej destrukcyjne maski przy tym samym budżecie.

Jeżeli overlap BO z ground truth jest wysoki, znalezione regiony są nie tylko destrukcyjne, ale też przestrzennie związane z polipem.

## Walidacja Runów

Pipeline automatycznie uruchamia walidację kompletności CSV. Można też sprawdzić ręcznie:

```bash
uv run python -m src.validate_occlusion_runs \
  outputs/occlusion_runs/bo_variable_size_pilot.csv \
  --expected-images 10 \
  --expected-seeds 3 \
  --expected-max-step 25
```


## Minimalna Procedura Robocza

1. Sprawdź, czy masz checkpoint:

```bash
ls checkpoints/unet_best.pt
```

2. Sprawdź, czy masz dane:

```bash
ls data/raw/kvasir-seg/test/images
```

3. Zrób dry-run:

```bash
uv run python main.py --profile pilot --skip-train --skip-download --keep-checkpoint --no-fresh-data --dry-run
```

4. Odpal eksperyment:

```bash
uv run python main.py --profile pilot --skip-train --skip-download --keep-checkpoint --no-fresh-data
```

5. Obejrzyj:

```text
outputs/metrics/occlusion_comparison_variable_size_pilot.json
outputs/figures/three_occlusions_pilot.png
outputs/saliency_maps/bo_variable_size/variable_size/
```
