from __future__ import annotations

import torch
from torch import nn
import segmentation_models_pytorch as smp


class UNet(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = None,
        in_channels: int = 3,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        if encoder_weights in {"", "none", "None", "null"}:
            encoder_weights = None

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
