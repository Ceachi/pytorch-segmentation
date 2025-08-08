import albumentations as A
from typing import List, Dict, Any

AUGMENTATION_REGISTRY = {
    "resize": A.Resize,
    "horizontal_flip": A.HorizontalFlip,
    "vertical_flip": A.VerticalFlip,
    "random_rotate_90": A.RandomRotate90,
    "rotate": A.Rotate,
    "normalize": A.Normalize,
}


def build_augmentations(config: List[Dict[str, Any]]) -> A.Compose:
    """Build an Albumentations augmentation pipeline.

    Args:
        config: List of augmentation configs with ``name`` and optional ``params``.

    Returns:
        Composed Albumentations transform.
    """
    transforms = []
    for aug in config:
        name = aug["name"]
        params = aug.get("params", {})
        if name not in AUGMENTATION_REGISTRY:
            raise KeyError(f"Unknown augmentation: {name}")
        transforms.append(AUGMENTATION_REGISTRY[name](**params))
    return A.Compose(transforms)

__all__ = ["build_augmentations", "AUGMENTATION_REGISTRY"]
