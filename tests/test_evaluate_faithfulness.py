from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.evaluate_faithfulness import (
    EPS,
    area_under_curve,
    build_schedule,
    combine_scores,
    coverage_mask_from_rows,
    densify_saliency,
    evaluate_curve,
    group_by_image_seed,
    read_rows,
    saliency_from_rows,
    sigmoid_mass,
    subsample_rows,
)


# ---------------------------------------------------------------------------
# Fake model używany we wszystkich testach wymagających modelu.
# Zwraca stały tensor logitów — nie potrzebujemy checkpointu.
# ---------------------------------------------------------------------------
class ConstantModel(torch.nn.Module):
    """Zwraca jedną stałą wartość logitu na każdy piksel."""

    def __init__(self, logit_value: float = 0.0, h: int = 4, w: int = 4) -> None:
        super().__init__()
        self.logit_value = logit_value
        self.h = h
        self.w = w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        return torch.full((batch, 1, self.h, self.w), self.logit_value)


# ---------------------------------------------------------------------------
# 1. build_schedule
# ---------------------------------------------------------------------------
class BuildScheduleTests(unittest.TestCase):

    def test_starts_at_zero_and_ends_at_one(self) -> None:
        fracs = build_schedule(dense_until=0.20, dense_step=0.01, sparse_step=0.05)
        self.assertAlmostEqual(float(fracs[0]), 0.0)
        self.assertAlmostEqual(float(fracs[-1]), 1.0)

    def test_strictly_increasing(self) -> None:
        fracs = build_schedule(dense_until=0.20, dense_step=0.01, sparse_step=0.05)
        self.assertTrue(bool(np.all(np.diff(fracs) > 0)))

    def test_all_values_in_unit_interval(self) -> None:
        fracs = build_schedule(dense_until=0.20, dense_step=0.01, sparse_step=0.05)
        self.assertTrue(bool(np.all(fracs >= 0.0)))
        self.assertTrue(bool(np.all(fracs <= 1.0)))

    def test_dense_region_has_fine_steps(self) -> None:
        fracs = build_schedule(dense_until=0.20, dense_step=0.01, sparse_step=0.05)
        dense = fracs[fracs <= 0.20 + 1e-9]
        # w gęstej części co 0.01, więc powinno być ~21 punktów (0.00 .. 0.20)
        self.assertGreaterEqual(len(dense), 20)

    def test_no_duplicate_values(self) -> None:
        fracs = build_schedule(dense_until=0.20, dense_step=0.01, sparse_step=0.05)
        self.assertEqual(len(fracs), len(np.unique(fracs)))


# ---------------------------------------------------------------------------
# 2. sigmoid_mass
# ---------------------------------------------------------------------------
class SigmoidMassTests(unittest.TestCase):

    def setUp(self) -> None:
        self.device = torch.device("cpu")

    def test_returns_float(self) -> None:
        model = ConstantModel(logit_value=0.0, h=4, w=4)
        image = torch.zeros(3, 4, 4)
        result = sigmoid_mass(model, image, self.device)
        self.assertIsInstance(result, float)

    def test_logit_zero_gives_half_per_pixel(self) -> None:
        # sigmoid(0) = 0.5; 4x4 = 16 pikseli -> suma = 8.0
        model = ConstantModel(logit_value=0.0, h=4, w=4)
        image = torch.zeros(3, 4, 4)
        result = sigmoid_mass(model, image, self.device)
        self.assertAlmostEqual(result, 8.0, places=4)

    def test_large_positive_logit_approaches_n_pixels(self) -> None:
        # sigmoid(+100) ≈ 1.0; 4x4 = 16 pikseli -> suma ≈ 16.0
        model = ConstantModel(logit_value=100.0, h=4, w=4)
        image = torch.zeros(3, 4, 4)
        result = sigmoid_mass(model, image, self.device)
        self.assertAlmostEqual(result, 16.0, places=2)

    def test_large_negative_logit_approaches_zero(self) -> None:
        model = ConstantModel(logit_value=-100.0, h=4, w=4)
        image = torch.zeros(3, 4, 4)
        result = sigmoid_mass(model, image, self.device)
        self.assertAlmostEqual(result, 0.0, places=2)


# ---------------------------------------------------------------------------
# 3. saliency_from_rows
# ---------------------------------------------------------------------------
class SaliencyFromRowsTests(unittest.TestCase):

    def test_output_shape_matches_image_size(self) -> None:
        rows = [{"cx": "4", "cy": "4", "score": "1.0"}]
        sal = saliency_from_rows(rows, image_size=8, default_mask_size=2)
        self.assertEqual(sal.shape, (8, 8))

    def test_uses_default_mask_size_when_column_missing(self) -> None:
        # Wiersz bez 'mask_size' (jak random/sliding) — powinien użyć default_mask_size.
        rows = [{"cx": "4", "cy": "4", "score": "1.0"}]
        sal = saliency_from_rows(rows, image_size=8, default_mask_size=2)
        self.assertGreater(float(sal.max()), 0.0)

    def test_uses_mask_size_column_when_present(self) -> None:
        # BO zapisuje mask_size; duży kwadrat powinien dać więcej niezerowych pikseli.
        rows_small = [{"cx": "4", "cy": "4", "score": "1.0", "mask_size": "2"}]
        rows_large = [{"cx": "4", "cy": "4", "score": "1.0", "mask_size": "6"}]
        sal_small = saliency_from_rows(rows_small, image_size=8, default_mask_size=2)
        sal_large = saliency_from_rows(rows_large, image_size=8, default_mask_size=2)
        nonzero_small = int(np.count_nonzero(sal_small))
        nonzero_large = int(np.count_nonzero(sal_large))
        self.assertLess(nonzero_small, nonzero_large)

    def test_empty_mask_size_string_falls_back_to_default(self) -> None:
        rows = [{"cx": "4", "cy": "4", "score": "1.0", "mask_size": ""}]
        sal = saliency_from_rows(rows, image_size=8, default_mask_size=2)
        self.assertEqual(sal.shape, (8, 8))

    def test_higher_score_yields_higher_map_value(self) -> None:
        rows_low  = [{"cx": "4", "cy": "4", "score": "0.1"}]
        rows_high = [{"cx": "4", "cy": "4", "score": "0.9"}]
        sal_low  = saliency_from_rows(rows_low,  image_size=8, default_mask_size=2)
        sal_high = saliency_from_rows(rows_high, image_size=8, default_mask_size=2)
        self.assertGreater(float(sal_high.max()), float(sal_low.max()))


# ---------------------------------------------------------------------------
# 4. evaluate_curve
# ---------------------------------------------------------------------------
class EvaluateCurveTests(unittest.TestCase):

    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.h, self.w = 4, 4
        self.image   = torch.rand(3, self.h, self.w)
        self.blurred = torch.zeros(3, self.h, self.w)  # baseline = czerń
        self.ranking = torch.arange(self.h * self.w)   # piksele po kolei
        self.fracs   = build_schedule(0.20, 0.10, 0.25)

    def test_output_length_matches_fractions(self) -> None:
        model = ConstantModel(logit_value=0.0, h=self.h, w=self.w)
        baseline = sigmoid_mass(model, self.image, self.device)
        ys = evaluate_curve(model, self.image, self.blurred, self.ranking,
                            self.fracs, baseline, self.device, mode="deletion")
        self.assertEqual(len(ys), len(self.fracs))

    def test_deletion_at_fraction_zero_is_one(self) -> None:
        # Przy ułamku 0 nic nie usunęliśmy — predykcja = baseline -> y = 1.0
        model = ConstantModel(logit_value=2.0, h=self.h, w=self.w)
        baseline = sigmoid_mass(model, self.image, self.device)
        ys = evaluate_curve(model, self.image, self.blurred, self.ranking,
                            self.fracs, baseline, self.device, mode="deletion")
        self.assertAlmostEqual(float(ys[0]), 1.0, places=4)

    def test_insertion_at_fraction_zero_is_lower_than_at_fraction_one(self) -> None:
        # Przy f=0 zaczynamy od rozmazanego (czarnego) obrazu.
        # Model widzi mniej niż przy f=1.0 (pełny obraz).
        model = ConstantModel(logit_value=2.0, h=self.h, w=self.w)
        baseline = sigmoid_mass(model, self.image, self.device)
        ys = evaluate_curve(model, self.image, self.blurred, self.ranking,
                            self.fracs, baseline, self.device, mode="insertion")
        self.assertLessEqual(float(ys[0]), float(ys[-1]) + 1e-6)

    def test_deletion_returns_float64_array(self) -> None:
        model = ConstantModel(logit_value=0.0, h=self.h, w=self.w)
        baseline = sigmoid_mass(model, self.image, self.device)
        ys = evaluate_curve(model, self.image, self.blurred, self.ranking,
                            self.fracs, baseline, self.device, mode="deletion")
        self.assertEqual(ys.dtype, np.float64)

    def test_constant_model_deletion_stays_constant(self) -> None:
        # Model ignoruje zawartość obrazu -> zamiana pikseli nic nie zmienia.
        model = ConstantModel(logit_value=1.0, h=self.h, w=self.w)
        baseline = sigmoid_mass(model, self.image, self.device)
        ys = evaluate_curve(model, self.image, self.blurred, self.ranking,
                            self.fracs, baseline, self.device, mode="deletion")
        # Każdy krok daje tę samą masę -> y ≈ const przez cały przebieg.
        self.assertTrue(bool(np.allclose(ys, ys[0], atol=1e-4)))


# ---------------------------------------------------------------------------
# 5. area_under_curve
# ---------------------------------------------------------------------------
class AreaUnderCurveTests(unittest.TestCase):

    def test_constant_one_gives_auc_one(self) -> None:
        fracs = np.linspace(0.0, 1.0, 11)
        ys    = np.ones(11)
        self.assertAlmostEqual(area_under_curve(fracs, ys), 1.0, places=6)

    def test_constant_zero_gives_auc_zero(self) -> None:
        fracs = np.linspace(0.0, 1.0, 11)
        ys    = np.zeros(11)
        self.assertAlmostEqual(area_under_curve(fracs, ys), 0.0, places=6)

    def test_triangle_gives_half(self) -> None:
        # Trójkąt od (0,0) do (1,1): pole = 0.5
        fracs = np.array([0.0, 1.0])
        ys    = np.array([0.0, 1.0])
        self.assertAlmostEqual(area_under_curve(fracs, ys), 0.5, places=6)

    def test_auc_with_uneven_fractions(self) -> None:
        # Potwierdzamy, że trapz używa prawdziwej osi X (nierównomierne kroki).
        fracs = np.array([0.0, 0.1, 0.5, 1.0])
        ys    = np.ones(4)  # stała 1 -> AUC = 1 bez względu na kroki
        self.assertAlmostEqual(area_under_curve(fracs, ys), 1.0, places=6)


# ---------------------------------------------------------------------------
# 6. combine_scores
# ---------------------------------------------------------------------------
class CombineScoresTests(unittest.TestCase):

    def test_perfect_scores_give_max_combined(self) -> None:
        # insertion=1.0, deletion=0.0 -> deletion_aligned=1.0, diff=1.0, geomean=1.0
        result = combine_scores(insertion_auc=1.0, deletion_auc=0.0)
        self.assertAlmostEqual(result["deletion_aligned"], 1.0)
        self.assertAlmostEqual(result["diff"], 1.0)
        self.assertAlmostEqual(result["geomean"], 1.0)

    def test_worst_scores_give_zero_combined(self) -> None:
        # insertion=0.0, deletion=1.0 -> deletion_aligned=0.0, diff=-1.0, geomean=0.0
        result = combine_scores(insertion_auc=0.0, deletion_auc=1.0)
        self.assertAlmostEqual(result["deletion_aligned"], 0.0)
        self.assertAlmostEqual(result["diff"], -1.0)
        self.assertAlmostEqual(result["geomean"], 0.0)

    def test_symmetric_scores_give_expected_geomean(self) -> None:
        # insertion=0.8, deletion=0.2 -> deletion_aligned=0.8
        # geomean = sqrt(0.8 * 0.8) = 0.8
        result = combine_scores(insertion_auc=0.8, deletion_auc=0.2)
        self.assertAlmostEqual(result["geomean"], 0.8, places=5)

    def test_all_keys_present(self) -> None:
        result = combine_scores(insertion_auc=0.5, deletion_auc=0.5)
        self.assertIn("deletion_aligned", result)
        self.assertIn("diff", result)
        self.assertIn("geomean", result)

    def test_geomean_nonnegative_when_insertion_negative(self) -> None:
        # Rzadki przypadek: okluzja zwiększa masę -> insertion_auc ujemne.
        result = combine_scores(insertion_auc=-0.1, deletion_auc=0.5)
        self.assertGreaterEqual(result["geomean"], 0.0)


# ---------------------------------------------------------------------------
# 7. group_by_image_seed
# ---------------------------------------------------------------------------
class GroupByImageSeedTests(unittest.TestCase):

    def test_groups_by_image_and_seed(self) -> None:
        rows = [
            {"image_id": "img1", "seed": "0", "score": "0.5"},
            {"image_id": "img1", "seed": "1", "score": "0.7"},
            {"image_id": "img2", "seed": "0", "score": "0.3"},
        ]
        groups = group_by_image_seed(rows)
        self.assertIn(("img1", "0"), groups)
        self.assertIn(("img1", "1"), groups)
        self.assertIn(("img2", "0"), groups)
        self.assertEqual(len(groups[("img1", "0")]), 1)

    def test_missing_seed_defaults_to_empty_string(self) -> None:
        # sliding nie zapisuje seed — group_by_image_seed ma fallback na "".
        rows = [{"image_id": "img1", "score": "0.5"}]
        groups = group_by_image_seed(rows)
        self.assertIn(("img1", ""), groups)

    def test_multiple_rows_same_group_are_collected(self) -> None:
        rows = [
            {"image_id": "img1", "seed": "0", "step": "1"},
            {"image_id": "img1", "seed": "0", "step": "2"},
        ]
        groups = group_by_image_seed(rows)
        self.assertEqual(len(groups[("img1", "0")]), 2)


# ---------------------------------------------------------------------------
# 8. read_rows (z tymczasowym plikiem CSV)
# ---------------------------------------------------------------------------
class ReadRowsTests(unittest.TestCase):

    def test_returns_empty_list_for_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "does_not_exist.csv"
            rows = read_rows(missing)
        self.assertEqual(rows, [])

    def test_reads_csv_correctly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["image_id", "cx", "cy", "score"])
                writer.writeheader()
                writer.writerow({"image_id": "abc", "cx": "10", "cy": "20", "score": "0.5"})

            rows = read_rows(path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["image_id"], "abc")
        self.assertEqual(rows[0]["score"], "0.5")


# ---------------------------------------------------------------------------
# 9. coverage_mask_from_rows  (maska odwiedzonych pikseli)
# ---------------------------------------------------------------------------
class CoverageMaskTests(unittest.TestCase):

    def test_single_window_marks_its_square(self) -> None:
        rows = [{"cx": "4", "cy": "4", "score": "0.0"}]
        mask = coverage_mask_from_rows(rows, image_size=8, default_mask_size=2)
        # Okno o score 0 NADAL liczy się jako odwiedzone (maska != saliency).
        self.assertTrue(bool(mask.any()))
        self.assertEqual(mask.dtype, bool)

    def test_uncovered_pixels_stay_false(self) -> None:
        rows = [{"cx": "1", "cy": "1", "score": "1.0"}]
        mask = coverage_mask_from_rows(rows, image_size=8, default_mask_size=2)
        # Przeciwległy róg nie jest dotknięty przez małe okno w (1,1).
        self.assertFalse(bool(mask[7, 7]))

    def test_full_grid_covers_everything(self) -> None:
        rows = [{"cx": str(x), "cy": str(y), "score": "1.0"}
                for x in range(0, 8) for y in range(0, 8)]
        mask = coverage_mask_from_rows(rows, image_size=8, default_mask_size=4)
        self.assertTrue(bool(mask.all()))


# ---------------------------------------------------------------------------
# 10. densify_saliency  (wypełnienie Voronoi pikseli nieodwiedzonych)
# ---------------------------------------------------------------------------
class DensifySaliencyTests(unittest.TestCase):

    def test_full_coverage_is_unchanged(self) -> None:
        sal = np.arange(16, dtype=np.float32).reshape(4, 4)
        covered = np.ones((4, 4), dtype=bool)
        out = densify_saliency(sal, covered)
        self.assertTrue(bool(np.array_equal(out, sal)))

    def test_no_coverage_returns_input(self) -> None:
        sal = np.zeros((4, 4), dtype=np.float32)
        covered = np.zeros((4, 4), dtype=bool)
        out = densify_saliency(sal, covered)
        self.assertTrue(bool(np.array_equal(out, sal)))

    def test_uncovered_pixels_get_nearest_value(self) -> None:
        # Jedyny odwiedzony piksel ma wartość 5 -> po gęstnieniu CAŁA mapa = 5.
        sal = np.zeros((4, 4), dtype=np.float32)
        sal[0, 0] = 5.0
        covered = np.zeros((4, 4), dtype=bool)
        covered[0, 0] = True
        out = densify_saliency(sal, covered)
        self.assertTrue(bool(np.allclose(out, 5.0)))

    def test_no_zero_ties_after_densify_on_partial_coverage(self) -> None:
        # Po gęstnieniu nie powinno zostać "dziur" zerowych z braku pokrycia,
        # gdy odwiedzone wartości są niezerowe — to sedno naprawy artefaktu rankingu.
        sal = np.zeros((6, 6), dtype=np.float32)
        sal[0, 0] = 1.0
        sal[5, 5] = 2.0
        covered = np.zeros((6, 6), dtype=bool)
        covered[0, 0] = True
        covered[5, 5] = True
        out = densify_saliency(sal, covered)
        self.assertEqual(int(np.count_nonzero(out == 0.0)), 0)


# ---------------------------------------------------------------------------
# 11. subsample_rows  (zrównanie budżetu obserwacji)
# ---------------------------------------------------------------------------
class SubsampleRowsTests(unittest.TestCase):

    def _rows(self, n: int) -> list[dict[str, str]]:
        return [{"image_id": "img", "seed": "0", "cx": str(i), "cy": "0", "score": "0.0"} for i in range(n)]

    def test_none_budget_keeps_all(self) -> None:
        rows = self._rows(40)
        self.assertEqual(len(subsample_rows(rows, None, ("img", "0"))), 40)

    def test_budget_larger_than_rows_keeps_all(self) -> None:
        rows = self._rows(10)
        self.assertEqual(len(subsample_rows(rows, 25, ("img", "0"))), 10)

    def test_budget_trims_to_n(self) -> None:
        rows = self._rows(64)
        self.assertEqual(len(subsample_rows(rows, 25, ("img", "0"))), 25)

    def test_deterministic_for_same_key(self) -> None:
        rows = self._rows(64)
        a = subsample_rows(rows, 25, ("img", "0"))
        b = subsample_rows(rows, 25, ("img", "0"))
        self.assertEqual([r["cx"] for r in a], [r["cx"] for r in b])


if __name__ == "__main__":
    unittest.main()
