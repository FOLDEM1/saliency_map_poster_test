from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import torch
from PIL import Image
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from src.occlusion import square_bounds


@dataclass(frozen=True)
class Observation:
    cx: float
    cy: float
    size: int
    score: float


def normalize_observations(observations: list[Observation], image_size: int, max_mask_size: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[obs.cx / image_size, obs.cy / image_size, obs.size / max_mask_size] for obs in observations], dtype=float)
    y = np.array([obs.score for obs in observations], dtype=float)
    return x, y


def fit_gp_saliency_model(
    observations: list[Observation],
    image_size: int,
    max_mask_size: int,
    random_state: int,
) -> GaussianProcessRegressor:
    if len(observations) < 2:
        raise ValueError("At least two BO observations are required to fit a saliency GP.")

    x, y = normalize_observations(observations, image_size, max_mask_size)
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(3), nu=2.5) + WhiteKernel(
        noise_level=1e-6,
        noise_level_bounds=(1e-8, 1e-2),
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        random_state=random_state,
        n_restarts_optimizer=2,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(x, y)
    return gp


def gp_window_predictions(
    gp: GaussianProcessRegressor,
    image_size: int,
    size_candidates: list[int],
    grid_stride: int,
    max_mask_size: int,
) -> list[Observation]:
    max_half = max(size_candidates) // 2
    coords = list(range(max_half, image_size - max_half + 1, grid_stride))
    if coords[-1] != image_size - max_half:
        coords.append(image_size - max_half)

    windows: list[Observation] = []
    for size in size_candidates:
        query = np.array(
            [[cx / image_size, cy / image_size, size / max_mask_size] for cy in coords for cx in coords],
            dtype=float,
        )
        means = gp.predict(query)
        for (cy, cx), mean in zip(((cy, cx) for cy in coords for cx in coords), means):
            windows.append(Observation(cx=float(cx), cy=float(cy), size=int(size), score=float(mean)))
    return windows


def rasterize_window_scores(windows: list[Observation], image_size: int) -> np.ndarray:
    saliency_sum = np.zeros((image_size, image_size), dtype=np.float32)
    saliency_count = np.zeros((image_size, image_size), dtype=np.float32)

    for window in windows:
        bounds = square_bounds(window.cx, window.cy, window.size, image_size)
        saliency_sum[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1] += window.score
        saliency_count[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1] += 1.0

    saliency = np.divide(
        saliency_sum,
        np.maximum(saliency_count, 1.0),
        out=np.zeros_like(saliency_sum),
        where=saliency_count > 0,
    )
    return saliency


def normalize_map(values: np.ndarray) -> np.ndarray:
    finite = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(finite.min())
    max_value = float(finite.max())
    if max_value <= min_value:
        return np.zeros_like(finite)
    return (finite - min_value) / (max_value - min_value)


def save_saliency_outputs(
    image: torch.Tensor,
    saliency: np.ndarray,
    output_prefix: Path,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_prefix.with_suffix(".npy"), saliency)

    normalized = normalize_map(saliency)
    gray = (normalized * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(gray).save(output_prefix.with_name(output_prefix.name + "_saliency.png"))

    image_np = (image.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    heat = np.zeros_like(image_np)
    heat[..., 0] = gray
    overlay = (0.55 * image_np + 0.45 * heat).clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(output_prefix.with_name(output_prefix.name + "_overlay.png"))


def build_and_save_gp_saliency(
    observations: list[Observation],
    image: torch.Tensor,
    image_size: int,
    size_candidates: list[int],
    grid_stride: int,
    output_prefix: Path,
    random_state: int,
) -> np.ndarray:
    max_mask_size = max(size_candidates)
    gp = fit_gp_saliency_model(
        observations=observations,
        image_size=image_size,
        max_mask_size=max_mask_size,
        random_state=random_state,
    )
    windows = gp_window_predictions(
        gp=gp,
        image_size=image_size,
        size_candidates=size_candidates,
        grid_stride=grid_stride,
        max_mask_size=max_mask_size,
    )
    saliency = rasterize_window_scores(windows, image_size=image_size)
    save_saliency_outputs(image=image, saliency=saliency, output_prefix=output_prefix)
    return saliency
