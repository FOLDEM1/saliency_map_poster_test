from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SquareBounds:
    x0: int
    y0: int
    x1: int
    y1: int


def square_bounds(cx: float, cy: float, size: int, image_size: int) -> SquareBounds:
    half = size // 2
    center_x = int(round(cx))
    center_y = int(round(cy))

    x0 = max(0, center_x - half)
    y0 = max(0, center_y - half)
    x1 = min(image_size, x0 + size)
    y1 = min(image_size, y0 + size)

    x0 = max(0, x1 - size)
    y0 = max(0, y1 - size)
    return SquareBounds(x0=x0, y0=y0, x1=x1, y1=y1)


def apply_square_occlusion(
    image: torch.Tensor,
    cx: float,
    cy: float,
    size: int,
    fill_rgb: torch.Tensor,
) -> torch.Tensor:
    if image.ndim not in {3, 4}:
        raise ValueError(f"Expected image with shape CxHxW or BxCxHxW, got {tuple(image.shape)}")

    batched = image.ndim == 4
    output = image.clone()
    height = output.shape[-2]
    width = output.shape[-1]
    if height != width:
        raise ValueError("Only square images are supported for now.")

    bounds = square_bounds(cx, cy, size, height)
    fill = fill_rgb.to(device=output.device, dtype=output.dtype).view(1, 3, 1, 1)

    if batched:
        output[:, :, bounds.y0 : bounds.y1, bounds.x0 : bounds.x1] = fill
    else:
        output[:, bounds.y0 : bounds.y1, bounds.x0 : bounds.x1] = fill.squeeze(0)

    return output


def soft_dice_between_predictions(
    prediction_a: torch.Tensor,
    prediction_b: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    a = prediction_a.reshape(prediction_a.shape[0], -1)
    b = prediction_b.reshape(prediction_b.shape[0], -1)
    intersection = (a * b).sum(dim=1)
    denominator = a.sum(dim=1) + b.sum(dim=1)
    return ((2.0 * intersection + eps) / (denominator + eps)).mean()


@torch.no_grad()
def occlusion_objective(
    model: torch.nn.Module,
    image: torch.Tensor,
    baseline_prediction: torch.Tensor,
    cx: float,
    cy: float,
    size: int,
    fill_rgb: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    occluded = apply_square_occlusion(image, cx, cy, size, fill_rgb)
    logits = model(occluded.unsqueeze(0))
    prediction = torch.sigmoid(logits)
    score = 1.0 - soft_dice_between_predictions(baseline_prediction, prediction)
    return float(score.item()), prediction


def mask_to_binary(cx: float, cy: float, size: int, image_size: int, device: torch.device | None = None) -> torch.Tensor:
    mask = torch.zeros((1, image_size, image_size), dtype=torch.float32, device=device)
    bounds = square_bounds(cx, cy, size, image_size)
    mask[:, bounds.y0 : bounds.y1, bounds.x0 : bounds.x1] = 1.0
    return mask


def overlap_with_gt(mask: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7) -> dict[str, float]:
    mask_bool = mask > 0.5
    gt_bool = gt > 0.5
    intersection = (mask_bool & gt_bool).sum().float()
    mask_area = mask_bool.sum().float()
    gt_area = gt_bool.sum().float()
    union = (mask_bool | gt_bool).sum().float()

    return {
        "overlap_mask": float((intersection / (mask_area + eps)).item()),
        "polyp_coverage": float((intersection / (gt_area + eps)).item()),
        "mask_gt_iou": float((intersection / (union + eps)).item()),
    }
