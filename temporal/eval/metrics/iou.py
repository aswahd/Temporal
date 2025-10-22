from functools import partial

import torch

from .metric import Metric


def iou(y_pred, y_true, apply_sigmoid=False):
    """
    Computes the Intersection over Union (IoU) metric.

    Args:
        y_true (Tensor): Ground truth labels.
        y_pred (Tensor): Predicted labels.
        apply_sigmoid (bool): Whether to apply sigmoid activation to predictions.

    Returns:
        Tensor: IoU score.
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
    union = y_true.sum(dim=(1, 2)) + y_pred.sum(dim=(1, 2)) - intersection
    batch_iou = intersection / union
    # If both y_true and y_pred are all zeros, iou is undefined, do not count it in the mean.
    return torch.nanmean(batch_iou)


class IoUMetric(Metric):
    def __init__(self, reduction="none", ignore_empty=True, apply_sigmoid=False):
        super().__init__(reduction=reduction, ignore_empty=ignore_empty)
        self.metric_fn = partial(iou, apply_sigmoid=apply_sigmoid)
        self._values = []
