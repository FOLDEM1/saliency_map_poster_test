from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import torch

from src.analyze_occlusion import final_rows_by_image_seed
from src.experiment_io import prune_incomplete_groups, stable_seed
from src.occlusion import apply_square_occlusion, mask_to_binary, overlap_with_gt, square_bounds
from src.saliency import Observation, rasterize_window_scores


class OcclusionTests(unittest.TestCase):
    def test_square_bounds_clamps_to_image(self) -> None:
        bounds = square_bounds(cx=0, cy=0, size=48, image_size=256)
        self.assertEqual((bounds.x0, bounds.y0, bounds.x1, bounds.y1), (0, 0, 48, 48))

        bounds = square_bounds(cx=255, cy=255, size=48, image_size=256)
        self.assertEqual((bounds.x0, bounds.y0, bounds.x1, bounds.y1), (208, 208, 256, 256))

    def test_apply_square_occlusion_uses_fill_rgb(self) -> None:
        image = torch.zeros((3, 8, 8))
        fill = torch.tensor([0.1, 0.2, 0.3])
        occluded = apply_square_occlusion(image, cx=4, cy=4, size=2, fill_rgb=fill)

        self.assertTrue(torch.allclose(occluded[:, 3:5, 3:5], fill.view(3, 1, 1)))
        self.assertEqual(float(occluded[:, 0, 0].sum()), 0.0)

    def test_overlap_with_gt(self) -> None:
        mask = mask_to_binary(cx=2, cy=2, size=2, image_size=4)
        gt = torch.zeros((1, 4, 4))
        gt[:, 1:3, 1:2] = 1.0

        overlap = overlap_with_gt(mask, gt)
        self.assertAlmostEqual(overlap["overlap_mask"], 0.5)
        self.assertAlmostEqual(overlap["polyp_coverage"], 1.0)


class AnalysisTests(unittest.TestCase):
    def test_final_rows_by_image_seed_keeps_seeds_separate(self) -> None:
        rows = [
            {"image_id": "a", "seed": "0", "step": "1"},
            {"image_id": "a", "seed": "1", "step": "1"},
            {"image_id": "a", "seed": "0", "step": "2"},
        ]
        final = final_rows_by_image_seed(rows, max_step=2)

        self.assertEqual(set(final), {("a", 0), ("a", 1)})
        self.assertEqual(final[("a", 0)]["step"], "2")
        self.assertEqual(final[("a", 1)]["step"], "1")

    def test_stable_seed_is_reproducible_and_keyed(self) -> None:
        self.assertEqual(stable_seed("random", 0, "0001"), stable_seed("random", 0, "0001"))
        self.assertNotEqual(stable_seed("random", 0, "0001"), stable_seed("random", 0, "0002"))

    def test_prune_incomplete_groups_keeps_only_complete_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.csv"
            rows = [
                {"image_id": "a", "seed": "0", "step": "1"},
                {"image_id": "a", "seed": "0", "step": "2"},
                {"image_id": "b", "seed": "0", "step": "1"},
            ]
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["image_id", "seed", "step"])
                writer.writeheader()
                writer.writerows(rows)

            complete = prune_incomplete_groups(path, ["image_id", "seed"], required_steps=2)

            self.assertEqual(complete, {("a", "0")})
            with path.open(newline="") as file:
                remaining = list(csv.DictReader(file))
            self.assertEqual(len(remaining), 2)
            self.assertEqual({row["image_id"] for row in remaining}, {"a"})


class SaliencyTests(unittest.TestCase):
    def test_rasterize_window_scores_averages_overlapping_windows(self) -> None:
        windows = [
            Observation(cx=2, cy=2, size=2, score=1.0),
            Observation(cx=3, cy=2, size=2, score=3.0),
        ]

        saliency = rasterize_window_scores(windows, image_size=5)

        self.assertAlmostEqual(float(saliency[1, 1]), 1.0)
        self.assertAlmostEqual(float(saliency[1, 2]), 2.0)
        self.assertAlmostEqual(float(saliency[1, 3]), 3.0)


if __name__ == "__main__":
    unittest.main()
