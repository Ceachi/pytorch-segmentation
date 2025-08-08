from typing import List, Optional

import numpy as np
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from segmentation_models_pytorch.datasets import CustomDataset


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int = 4,
        train_transform: Optional[object] = None,
        val_transform: Optional[object] = None,
        class_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.class_names = class_names or []
        self.num_classes = len(self.class_names)

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: D401
        self.train_dataset = CustomDataset(self.data_dir, split="train", transform=self.train_transform)
        self.val_dataset = CustomDataset(self.data_dir, split="val", transform=self.val_transform)
        if self.num_classes:
            self._validate_dataset(self.train_dataset)
            self._validate_dataset(self.val_dataset)

    def _validate_dataset(self, dataset: CustomDataset) -> None:
        for label_path in dataset.labels:
            mask = np.array(Image.open(label_path), dtype=np.int64)
            if mask.min() < 0 or mask.max() >= self.num_classes:
                raise ValueError(
                    f"Mask {label_path.name} contains pixels outside class range 0-{self.num_classes-1}"
                )

    def train_dataloader(self) -> DataLoader:  # noqa: D401
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:  # noqa: D401
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
