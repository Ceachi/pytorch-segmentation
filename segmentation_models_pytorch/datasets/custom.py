from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """Generic dataset for semantic segmentation.

    Expects the following directory structure::

        root/
          train/
            images/*.png
            labels/*.png
          val/
            images/*.png
            labels/*.png
    """

    def __init__(self, root: str, split: str = "train", transform: Optional[Callable] = None) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        self.root = Path(root)
        self.split = split
        self.images_dir = self.root / split / "images"
        self.labels_dir = self.root / split / "labels"
        self.transform = transform

        self.images = sorted(self.images_dir.glob("*.png"))
        self.labels = sorted(self.labels_dir.glob("*.png"))
        if len(self.images) != len(self.labels):
            raise RuntimeError("Number of images and labels does not match")
        for img, lbl in zip(self.images, self.labels):
            if img.stem != lbl.stem:
                raise RuntimeError(f"Mismatched image and label: {img.name} vs {lbl.name}")

    def __len__(self) -> int:  # noqa: D401
        return len(self.images)

    def __getitem__(self, index: int):
        image = np.array(Image.open(self.images[index]).convert("RGB"), dtype=np.float32) / 255.0
        mask = np.array(Image.open(self.labels[index]), dtype=np.int64)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        return image, mask
