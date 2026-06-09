from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class KvasirSegDataset(Dataset):
    def __init__(
        self,
        root: str | Path = "data/raw/kvasir-seg",
        split: str = "train",
        image_size: int = 256,
        augment: bool = False,
        max_samples: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.augment = augment

        images_dir = self.root / split / "images"
        masks_dir = self.root / split / "masks"
        self.image_paths = sorted(images_dir.glob("*.png"))
        self.mask_paths = [masks_dir / path.name for path in self.image_paths]

        if max_samples is not None:
            self.image_paths = self.image_paths[:max_samples]
            self.mask_paths = self.mask_paths[:max_samples]

        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {images_dir}")

        missing_masks = [path for path in self.mask_paths if not path.exists()]
        if missing_masks:
            raise FileNotFoundError(f"Missing mask for {missing_masks[0]}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)

        if self.augment:
            image, mask = self._augment(image, mask)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)

        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_id": image_path.stem,
        }

    @staticmethod
    def _augment(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        return image, mask
