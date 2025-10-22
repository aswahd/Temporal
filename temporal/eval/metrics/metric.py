import torch


class Metric:
    def __init__(
        self,
        reduction="none",
        ignore_empty=True,
    ):
        self.reduction = reduction
        self.ignore_empty = ignore_empty
        self.metric_fn = lambda x: x
        self._values = []

    def __call__(self, y_hat_dict, y_true_dict, apply_sigmoid=False):
        """
        Computes the IoU metric.

        Args:
            y_true_dict (dict): Dictionary mapping from category IDs to ground truth masks.
            y_hat_dict (dict): Dictionary mapping from category IDs to predicted masks.

        Returns: None
        """

        assert set(y_true_dict.keys()) == set(y_hat_dict.keys()), "Mismatch in category IDs"

        per_class_metrics = {}
        for category_id, y_true in y_true_dict.items():
            y_hat = y_hat_dict[category_id]
            assert y_hat.ndim == y_true.ndim, f"Got {y_hat.ndim}D and {y_true.ndim}D tensors."
            score = self.metric_fn(y_hat, y_true, apply_sigmoid=apply_sigmoid)

            # ignore_empty: whether to ignore empty ground truth cases during calculation.
            # If `True`, NaN value will be set for empty ground truth cases.
            # If `False`, 1 will be set if the predictions of empty ground truth cases are also empty.

            if self.ignore_empty:
                if y_true.sum() == 0:
                    score = float("nan")
            else:
                if y_true.sum() == y_hat.sum() == 0:
                    score = 1.0

            per_class_metrics[category_id] = score

        # Sort by category ID
        per_class_metrics = {k: per_class_metrics[k] for k in sorted(list(per_class_metrics.keys()))}
        scores = list(per_class_metrics.values())
        self._values.append(scores)

    def reset(self):
        self._values = []

    def aggregate(self, reduction="none"):
        if reduction == "none":
            return torch.tensor(self._values)

        if reduction == "mean_batch":
            return torch.nanmean(self._values, dim=0)

        if reduction == "mean":
            return torch.nanmean(self._values)

        raise ValueError(f"Unknown reduction method: {reduction}")
