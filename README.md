# pytorch-segmentation

Simple yet flexible framework for training semantic segmentation models with
[`segmentation_models_pytorch`](https://github.com/qubvel/segmentation_models.pytorch)
and [PyTorch Lightning](https://lightning.ai). The project focuses on keeping
experiments reproducible and easy to configure.

## Dataset structure

Datasets are expected to have the following layout::

```
<root>/
  train/
    images/*.png
    labels/*.png
  val/
    images/*.png
    labels/*.png
```

Label values start from `0` (background) and increase with every class. Class
names and ordering are provided through the environment file.

## Environment

Create a `.env` file based on `.env.example`:

```
WANDB_PROJECT=your_project
DATASET_PATH=/path/to/dataset
CLASS_NAMES=background,class1,class2
```

To reproduce the Python environment with all dependencies, create a conda
environment using `environment.yaml`:

```
conda env create -f environment.yaml
conda activate pytorch-segmentation
```

## Experiments

Experiment configurations live under the `experiments/` directory. Each YAML
file defines model settings and a collection of named experiments. For
example, `experiments/unetpp.yaml` defines a Unet++ architecture and two
experiments that modify training hyper-parameters.

Augmentations are described in the experiment config via `train_transforms`
and `val_transforms`. Each transform entry specifies the augmentation `name`
and optional `params` and is composed in order using
[Albumentations](https://albumentations.ai). Available transforms are
registered under `segmentation_models_pytorch/augmentations`.

Learning rate schedulers can be configured by adding a `scheduler` block to an
experiment, e.g.:

```yaml
scheduler:
  name: CosineAnnealingLR
  params:
    T_max: 10
```

Early stopping is optional and can be enabled via an `early_stopping` block:

```yaml
early_stopping:
  monitor: val/mIoU
  patience: 3
  mode: max
```

During training both mean Dice coefficient (mDCC) and mean Intersection over
Union (mIoU) are computed for train and validation splits. Metrics are logged
overall and per class to Weights & Biases.

Before training begins the dataset is validated to ensure that every image has
a matching mask and that all mask pixel values fall within the declared class
indices. Two random samples from the training set are also uploaded to W&B to
visualize the image alongside its mask and legend.

## Training

Run training by specifying a configuration file and the experiment to use:

```
python -m src.train --config experiments/unetpp.yaml --experiment experiment-batch_size
```

Runs are logged to Weights & Biases with names in the form
`<config-file>-<experiment>` (e.g. `unetpp-experiment-batch_size`). All hyper
parameters used for the run are also stored in the logger configuration.

The training script accepts a `gpu` parameter in experiment configs to choose
which GPU to run on (e.g. `gpu: 1`). Optimizers and schedulers are configured
via `optimizer` and `scheduler` blocks and are passed through to PyTorch
accordingly.

### Checkpoints

During training, model weights are periodically written under
`checkpoints/<run-name>`:

- `ckpt_latest.pth` – weights from the most recent epoch
- `ckpt_best.pth` – weights achieving the best monitored metric
- `average_model.pth` – running average of model weights

All three files are also uploaded to the corresponding W&B run and updated as
training progresses.
