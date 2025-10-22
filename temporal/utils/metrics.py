import torch

def iou_score(y_pred, y_true, epsilon=1e-5, reduction="mean"):
    """
    y_pred: (N, 1, H, W).
    y_true: (N, 1, H, W).
    """
    y_pred, y_true = y_pred.float(), y_true.float()
    intersection = torch.sum(y_pred * y_true, dim=(1, 2, 3))
    union = (
        torch.sum(y_pred, dim=(1, 2, 3))
        + torch.sum(y_true, dim=(1, 2, 3))
        - intersection
        + epsilon
    )
    iou = intersection / union
    # iou[y_true.sum(dim=(1,2,3)) == 0] = torch.nan # MONAI implementation
    fp = (y_true.sum(dim=(1, 2, 3)) == 0) & (y_pred.sum(dim=(1, 2, 3)) == 1)
    empty = (y_true.sum(dim=(1, 2, 3)) == 0) & (y_pred.sum(dim=(1, 2, 3)) == 0)
    iou[fp] = 0
    iou[empty] = torch.nan

    if reduction == "mean":
        mean = torch.nanmean(iou)
        return 1 if torch.isnan(mean) else mean
    else:
        return iou


def dice_coef(y_pred, y_true, epsilon=1e-5, reduction="mean"):

    y_pred, y_true = y_pred.float(), y_true.float()
    intersection = torch.sum(y_pred * y_true, dim=(1, 2, 3))
    dice = 2.0 * intersection / torch.sum(y_pred + y_true + epsilon, dim=(1, 2, 3))
    # dice[y_true.sum(dim=(1,2,3)) == 0] = torch.nan # MONAI implementation (skip false positives and empty masks)

    fp = (y_true.sum(dim=(1, 2, 3)) == 0) & (y_pred.sum(dim=(1, 2, 3)) == 1)
    empty = (y_true.sum(dim=(1, 2, 3)) == 0) & (y_pred.sum(dim=(1, 2, 3)) == 0)

    dice[fp] = 0 # Penalize false positives
    dice[empty] = torch.nan # skip empty masks

    if reduction == "mean":
        mean = torch.nanmean(dice)
        return 1 if torch.isnan(mean) else mean
    else:
        return dice
