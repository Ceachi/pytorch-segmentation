from typing import List, Optional

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch

try:  # torchmetrics>=0.11 provides task specific metrics
    from torchmetrics.classification import (
        MulticlassF1Score as F1Score,
        MulticlassJaccardIndex as JaccardIndex,
    )
    _metric_kwargs = {}
except ImportError:  # fallback for newer torchmetrics versions
    from torchmetrics import F1Score, JaccardIndex

    _metric_kwargs = {"task": "multiclass"}

from .losses import (
    BinaryDiceLoss,
    BinaryIoULoss,
    DiceLoss,
    GeneralizedDiceLoss,
    GeneralizedIoULoss,
    IoULoss,
)


class SegmentationModel(pl.LightningModule):
    def __init__(
        self,
        arch: str,
        encoder_name: str,
        encoder_weights: str,
        in_channels: int,
        classes: int,
        lr: float,
        scheduler: Optional[dict] = None,
        optimizer: Optional[dict] = None,
        class_names: Optional[List[str]] = None,
        loss: str = "dice",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = getattr(smp, arch)(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
        loss_map = {
            "dice": DiceLoss(),
            "binary_dice": BinaryDiceLoss(),
            "generalized_dice": GeneralizedDiceLoss(),
            "iou": IoULoss(),
            "binary_iou": BinaryIoULoss(),
            "generalized_iou": GeneralizedIoULoss(),
        }
        self.loss_fn = loss_map.get(loss)
        if self.loss_fn is None:
            raise ValueError(f"Unsupported loss: {loss}")
        self.scheduler_cfg = scheduler
        self.optimizer_cfg = optimizer
        self.class_names = class_names or [str(i) for i in range(classes)]
        self.train_iou = JaccardIndex(num_classes=classes, average="none", **_metric_kwargs)
        self.val_iou = JaccardIndex(num_classes=classes, average="none", **_metric_kwargs)
        self.train_dice = F1Score(num_classes=classes, average="none", **_metric_kwargs)
        self.val_dice = F1Score(num_classes=classes, average="none", **_metric_kwargs)

    def forward(self, x):  # noqa: D401
        return self.model(x)

    def training_step(self, batch, batch_idx):  # noqa: D401
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_iou.update(preds, y)
        self.train_dice.update(preds, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):  # noqa: D401
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_iou.update(preds, y)
        self.val_dice.update(preds, y)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):  # noqa: D401
        opt_cfg = self.optimizer_cfg or {"name": "AdamW", "params": {}}
        opt_cls = getattr(torch.optim, opt_cfg["name"])
        optimizer = opt_cls(
            self.parameters(), lr=self.hparams.lr, **opt_cfg.get("params", {})
        )

        if self.scheduler_cfg:
            sched_name = self.scheduler_cfg["name"]
            sched_params = self.scheduler_cfg.get("params", {})
            scheduler_cls = getattr(torch.optim.lr_scheduler, sched_name)
            scheduler = scheduler_cls(optimizer, **sched_params)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return optimizer

    def _log_metrics(self, prefix: str, iou_metric, dice_metric) -> None:
        iou = iou_metric.compute()
        dice = dice_metric.compute()
        self.log(f"{prefix}/mIoU", iou.mean(), prog_bar=True)
        self.log(f"{prefix}/mDCC", dice.mean(), prog_bar=True)
        for idx, name in enumerate(self.class_names):
            self.log(f"{prefix}/class-{name}-mIoU", iou[idx])
            self.log(f"{prefix}/class-{name}-mDCC", dice[idx])
        iou_metric.reset()
        dice_metric.reset()

    def on_train_epoch_end(self) -> None:  # noqa: D401
        self._log_metrics("train", self.train_iou, self.train_dice)

    def on_validation_epoch_end(self) -> None:  # noqa: D401
        self._log_metrics("val", self.val_iou, self.val_dice)
