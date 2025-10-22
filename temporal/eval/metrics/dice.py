from functools import partial

import torch

from .metric import Metric


def dice(y_pred, y_true, apply_sigmoid=False, epsilon=1e-7):
    """
    Computes the Dice coefficient metric.

    Args:
        y_true (Tensor): Ground truth labels.
        y_pred (Tensor): Predicted labels.
        apply_sigmoid (bool): Whether to apply sigmoid activation to predictions.

    Returns:
        Tensor: Dice coefficient score.
    """
    assert y_pred.ndim == y_true.ndim, f"Got {y_pred.ndim}D and {y_true.ndim}D tensors."
    assert y_pred.ndim in [2, 3], f"Dice metric only supports 2D or 3D tensors, but got {y_pred.ndim}D."

    if y_pred.ndim == y_true.ndim == 2:
        y_pred = y_pred.unsqueeze(1)
        y_true = y_true.unsqueeze(1)

    y_pred, y_true = y_pred.float(), y_true.float()
    if apply_sigmoid:
        y_pred = y_pred.sigmoid()

    intersection = (y_true * y_pred).sum(dim=(1, 2))
    total = y_true.sum(dim=(1, 2)) + y_pred.sum(dim=(1, 2))
    batch_dice = 2 * intersection / total
    # If both y_true and y_pred are all zeros, dice is undefined, do not count it in the mean.
    return torch.nanmean(batch_dice)


class DiceMetric(Metric):
    def __init__(
        self,
        reduction="none",
        ignore_empty=True,
        apply_sigmoid=False,
    ):
        self.reduction = reduction
        self.ignore_empty = ignore_empty
        super().__init__(reduction=reduction, ignore_empty=ignore_empty)
        self.metric_fn = partial(dice, apply_sigmoid=apply_sigmoid)

        self._values = []
