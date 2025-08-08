from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.optim.swa_utils import AveragedModel


class CheckpointSaver(pl.Callback):
    """Save latest, best, and averaged model weights and sync to W&B."""

    def __init__(
        self,
        dirpath: str,
        monitor: str = "val/mIoU",
        mode: str = "max",
        average: bool = True,
    ) -> None:
        super().__init__()
        self.dirpath = Path(dirpath)
        self.monitor = monitor
        self.mode = mode
        self.average = average
        self.best_score: Optional[float] = None
        self.avg_model: Optional[AveragedModel] = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: D401
        self.dirpath.mkdir(parents=True, exist_ok=True)
        if self.average:
            self.avg_model = AveragedModel(pl_module.model)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:  # noqa: D401
        if self.avg_model is not None:
            self.avg_model.update_parameters(pl_module.model)

    def _save_and_log(self, trainer: pl.Trainer, path: Path, state_dict: dict) -> None:
        torch.save(state_dict, path)
        logger = getattr(trainer, "logger", None)
        if hasattr(logger, "experiment") and logger.experiment is not None:
            try:
                logger.experiment.save(str(path), policy="live")
            except Exception:
                pass

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: D401
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            return
        current = metrics[self.monitor].item()

        latest_path = self.dirpath / "ckpt_latest.pth"
        self._save_and_log(trainer, latest_path, pl_module.model.state_dict())

        if (
            self.best_score is None
            or (self.mode == "max" and current > self.best_score)
            or (self.mode == "min" and current < self.best_score)
        ):
            self.best_score = current
            best_path = self.dirpath / "ckpt_best.pth"
            self._save_and_log(trainer, best_path, pl_module.model.state_dict())

        if self.avg_model is not None:
            avg_path = self.dirpath / "average_model.pth"
            self._save_and_log(trainer, avg_path, self.avg_model.module.state_dict())
