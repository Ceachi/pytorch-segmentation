"""Loss functions for semantic segmentation.

This module adapts the Dice and IoU based losses from the
`super-gradients` project so they can be used inside this repository.
The losses support both multi-class and binary segmentation and include
generalized variants that weight each class inversely proportional to
its volume in the target mask in order to mitigate class imbalance.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _StructureLoss(nn.Module):
    """Common functionality for Dice/IoU style segmentation losses."""

    def __init__(
        self,
        mode: str,
        apply_softmax: bool = True,
        apply_sigmoid: bool = False,
        smooth: float = 1.0,
        eps: float = 1e-5,
        ignore_index: Optional[int] = None,
        generalized: bool = False,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.apply_softmax = apply_softmax
        self.apply_sigmoid = apply_sigmoid
        self.smooth = smooth
        self.eps = eps
        self.ignore_index = ignore_index
        self.generalized = generalized

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.apply_sigmoid:
            preds = torch.sigmoid(logits)
        elif self.apply_softmax:
            preds = torch.softmax(logits, dim=1)
        else:
            preds = logits

        if target.shape == preds.shape:
            target_one_hot = target.float()
        else:
            if target.dim() != preds.dim() - 1:
                raise ValueError("Target tensor shape does not match predictions")
            target_one_hot = F.one_hot(target.long(), num_classes=preds.shape[1])
            target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        if self.ignore_index is not None:
            mask = target.ne(self.ignore_index).unsqueeze(1)
            preds = preds * mask
            target_one_hot = target_one_hot * mask

        dims = (0,) + tuple(range(2, preds.dim()))
        intersection = torch.sum(preds * target_one_hot, dim=dims)

        if self.mode == "dice":
            denominator = torch.sum(preds + target_one_hot, dim=dims)
        elif self.mode == "iou":
            denominator = torch.sum(preds + target_one_hot - preds * target_one_hot, dim=dims)
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if self.generalized:
            target_sum = torch.sum(target_one_hot, dim=dims)
            weights = 1.0 / (target_sum * target_sum + self.eps)
            weights = torch.where(torch.isinf(weights), torch.zeros_like(weights), weights)
            intersection = intersection * weights
            denominator = denominator * weights

        if self.mode == "dice":
            numer = 2.0 * intersection
            loss = 1.0 - (numer + self.smooth) / (denominator + self.smooth + self.eps)
        else:  # iou
            loss = 1.0 - (intersection + self.smooth) / (denominator + self.smooth + self.eps)

        return loss.mean()


class DiceLoss(_StructureLoss):
    """Multi-class Dice loss."""

    def __init__(
        self,
        apply_softmax: bool = True,
        smooth: float = 1.0,
        eps: float = 1e-5,
        ignore_index: Optional[int] = None,
    ) -> None:
        super().__init__(
            mode="dice",
            apply_softmax=apply_softmax,
            smooth=smooth,
            eps=eps,
            ignore_index=ignore_index,
        )


class BinaryDiceLoss(DiceLoss):
    """Dice loss for binary segmentation tasks."""

    def __init__(self, apply_sigmoid: bool = True, smooth: float = 1.0, eps: float = 1e-5) -> None:
        super().__init__(apply_softmax=False, smooth=smooth, eps=eps)
        self.apply_sigmoid = apply_sigmoid


class GeneralizedDiceLoss(DiceLoss):
    """Generalized Dice loss that compensates for class imbalance."""

    def __init__(
        self,
        apply_softmax: bool = True,
        smooth: float = 0.0,
        eps: float = 1e-17,
        ignore_index: Optional[int] = None,
    ) -> None:
        super().__init__(
            apply_softmax=apply_softmax,
            smooth=smooth,
            eps=eps,
            ignore_index=ignore_index,
        )
        self.generalized = True


class IoULoss(_StructureLoss):
    """Multi-class IoU loss."""

    def __init__(
        self,
        apply_softmax: bool = True,
        smooth: float = 1.0,
        eps: float = 1e-5,
        ignore_index: Optional[int] = None,
    ) -> None:
        super().__init__(
            mode="iou",
            apply_softmax=apply_softmax,
            smooth=smooth,
            eps=eps,
            ignore_index=ignore_index,
        )


class BinaryIoULoss(IoULoss):
    """IoU loss for binary segmentation tasks."""

    def __init__(self, apply_sigmoid: bool = True, smooth: float = 1.0, eps: float = 1e-5) -> None:
        super().__init__(apply_softmax=False, smooth=smooth, eps=eps)
        self.apply_sigmoid = apply_sigmoid


class GeneralizedIoULoss(IoULoss):
    """Generalized IoU loss that compensates for class imbalance."""

    def __init__(
        self,
        apply_softmax: bool = True,
        smooth: float = 0.0,
        eps: float = 1e-17,
        ignore_index: Optional[int] = None,
    ) -> None:
        super().__init__(
            apply_softmax=apply_softmax,
            smooth=smooth,
            eps=eps,
            ignore_index=ignore_index,
        )
        self.generalized = True


__all__ = [
    "DiceLoss",
    "BinaryDiceLoss",
    "GeneralizedDiceLoss",
    "IoULoss",
    "BinaryIoULoss",
    "GeneralizedIoULoss",
]

