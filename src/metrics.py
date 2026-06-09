from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def soft_dice_score(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    probabilities = probabilities.reshape(probabilities.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)
    intersection = (probabilities * targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
    return ((2.0 * intersection + eps) / (denominator + eps)).mean()


class DiceLoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        return 1.0 - soft_dice_score(probabilities, targets)


class BCEDiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, targets) + self.dice(logits, targets)


@torch.no_grad()
def segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    probabilities = torch.sigmoid(logits)
    predictions = probabilities >= threshold
    targets_bool = targets >= 0.5

    predictions_flat = predictions.reshape(predictions.shape[0], -1)
    targets_flat = targets_bool.reshape(targets_bool.shape[0], -1)

    tp = (predictions_flat & targets_flat).sum(dim=1).float()
    fp = (predictions_flat & ~targets_flat).sum(dim=1).float()
    fn = (~predictions_flat & targets_flat).sum(dim=1).float()

    eps = 1e-7
    dice = ((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)).mean()
    iou = ((tp + eps) / (tp + fp + fn + eps)).mean()
    precision = ((tp + eps) / (tp + fp + eps)).mean()
    recall = ((tp + eps) / (tp + fn + eps)).mean()

    return {
        "dice": float(dice.item()),
        "iou": float(iou.item()),
        "precision": float(precision.item()),
        "recall": float(recall.item()),
    }


def merge_metric_sums(metric_sums: dict[str, float], batch_metrics: dict[str, float], batch_size: int) -> None:
    for key, value in batch_metrics.items():
        metric_sums[key] = metric_sums.get(key, 0.0) + value * batch_size


def average_metric_sums(metric_sums: dict[str, float], total: int) -> dict[str, float]:
    return {key: value / total for key, value in metric_sums.items()}
