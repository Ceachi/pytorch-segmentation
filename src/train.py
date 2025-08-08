import argparse
import os
from pathlib import Path
import random

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import segmentation_models_pytorch as smp
import wandb
import yaml
from dotenv import load_dotenv

from .datamodule import SegmentationDataModule
from .lit_module import SegmentationModel
from .callbacks import CheckpointSaver
from segmentation_models_pytorch.augmentations import build_augmentations


def load_experiment(config_path: str, experiment_name: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    defaults = cfg.get("defaults", {})
    experiments = cfg.get("experiments", {})
    exp_cfg = experiments.get(experiment_name, {})
    params = {**defaults, **exp_cfg}
    return model_cfg, params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to experiment yaml")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name")
    args = parser.parse_args()

    load_dotenv()
    data_dir = os.environ["DATASET_PATH"]
    wandb_project = os.environ.get("WANDB_PROJECT")
    class_names = [n.strip() for n in os.environ.get("CLASS_NAMES", "").split(",") if n.strip()]

    model_cfg, params = load_experiment(args.config, args.experiment)
    model_cfg.setdefault("classes", len(class_names))

    train_transforms_cfg = params.get("train_transforms", [])
    val_transforms_cfg = params.get("val_transforms", [])

    if model_cfg.get("encoder_weights"):
        prep = smp.encoders.get_preprocessing_params(model_cfg["encoder_name"], model_cfg["encoder_weights"])
        norm_cfg = {
            "name": "normalize",
            "params": {"mean": prep["mean"], "std": prep["std"], "max_pixel_value": 1.0},
        }
        train_transforms_cfg.append(norm_cfg)
        val_transforms_cfg.append(norm_cfg)

    params["train_transforms"] = train_transforms_cfg
    params["val_transforms"] = val_transforms_cfg

    train_transform = build_augmentations(train_transforms_cfg) if train_transforms_cfg else None
    val_transform = build_augmentations(val_transforms_cfg) if val_transforms_cfg else None
    datamodule = SegmentationDataModule(
        data_dir=data_dir,
        batch_size=params["batch_size"],
        train_transform=train_transform,
        val_transform=val_transform,
        class_names=class_names,
    )
    datamodule.setup()

    model = SegmentationModel(
        **model_cfg,
        lr=params["lr"],
        scheduler=params.get("scheduler"),
        optimizer=params.get("optimizer"),
        class_names=class_names,
    )

    run_name = f"{Path(args.config).stem}-{args.experiment}"
    logger = WandbLogger(project=wandb_project, name=run_name, config={**model_cfg, **params})

    rng = np.random.default_rng(0)
    colors = [rng.integers(0, 255, size=3).tolist() for _ in class_names]
    class_labels = {i: name for i, name in enumerate(class_names)}
    samples = []
    for idx in random.sample(range(len(datamodule.train_dataset)), min(2, len(datamodule.train_dataset))):
        img, mask = datamodule.train_dataset[idx]
        img_np = img.permute(1, 2, 0).numpy()
        mask_np = mask.numpy()
        samples.append(
            wandb.Image(
                img_np,
                masks={
                    "ground_truth": {
                        "mask_data": mask_np,
                        "class_labels": class_labels,
                        "class_colors": colors,
                    }
                },
            )
        )
    if samples:
        logger.experiment.log({"examples": samples})

    callbacks = []
    if params.get("early_stopping"):
        es_cfg = params["early_stopping"]
        callbacks.append(
            EarlyStopping(
                monitor=es_cfg.get("monitor", "val/mIoU"),
                patience=es_cfg.get("patience", 3),
                mode=es_cfg.get("mode", "max"),
            )
        )

    monitor_metric = params.get("monitor", "val/mIoU")
    ckpt_dir = Path(params.get("ckpt_dir", "checkpoints")) / run_name
    callbacks.append(CheckpointSaver(dirpath=ckpt_dir, monitor=monitor_metric))

    trainer_kwargs = {"max_epochs": params["max_epochs"], "logger": logger, "callbacks": callbacks}
    if params.get("gpu") is not None:
        trainer_kwargs.update({"accelerator": "gpu", "devices": [params["gpu"]]})
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
