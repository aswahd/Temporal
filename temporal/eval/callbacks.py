import logging
from pathlib import Path
from typing import Tuple

import torch


class EvalLoggerCallback:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def _format_results(self, evaluator, dice_scores: torch.Tensor, iou_scores: torch.Tensor) -> str:
        """Format evaluation results as string"""
        results = ["=== Evaluation Results ==="]
        results.append(f"Context Size: {evaluator.context_size}")
        results.append(f"Categories: {evaluator.train_set.get_category_names()}\n")

        # Per-class metrics
        results.append("Per-Class Metrics:")
        results.append("{:<10}{:>10}{:>10}".format("Class", "Dice", "IoU"))
        results.append("-" * 30)

        for i, (dice, iou) in enumerate(zip(dice_scores, iou_scores, strict=True)):
            results.append("{:<10}{:>10.4f}{:>10.4f}".format(f"Class {i + 1}", dice.item(), iou.item()))

        # Average metrics
        results.append("\nAverage Metrics:")
        results.append("-" * 30)
        results.append(
            "{:<10}{:>10.4f}{:>10.4f}".format("Mean", dice_scores.nanmean().item(), iou_scores.nanmean().item())
        )

        return "\n".join(results)

    def before_forward(self, evaluator, query_image_path: str, query_image_size: Tuple[int, int]):
        # Log detailed scores
        image_name = Path(query_image_path).name
        self.logger.info("\n" + "=" * 50)
        self.logger.info(f"Detailed Results for: {image_name}")
        self.logger.info("-" * 50)
        self.logger.info("Per-class Dice scores:")

    def after_eval_batch(self, evaluator, dice_per_class: torch.Tensor, iou_per_class: torch.Tensor):
        # Log the total number of batches
        for i, dice in enumerate(dice_per_class, 1):
            self.logger.info(f"Class {i}: {dice.item():.4f}")
        self.logger.info(f"Mean Dice: {dice_per_class.nanmean().item():.4f}")
        self.logger.info("\nPer-class IoU scores:")
        for i, iou in enumerate(iou_per_class, 1):
            self.logger.info(f"Class {i}: {iou.item():.4f}")
        self.logger.info(f"Mean IoU: {iou_per_class.nanmean().item():.4f}")
        self.logger.info("=" * 50)

    def before_eval(self, evaluator, dataloader: torch.utils.data.DataLoader):
        # Log the number of batches
        self.logger.info("\n" + "=" * 50)
        self.logger.info("Starting Evaluation")
        self.logger.info("Context Size: %d", evaluator.context_size)
        self.logger.info("Categories: %s", evaluator.train_set.get_category_names())
        self.logger.info("=" * 50 + "\n")

    def after_eval(self, evaluator, dice_scores, iou_scores):
        results_str = self._format_results(evaluator, dice_scores, iou_scores)
        self.logger.info("\n" + "=" * 50)
        self.logger.info("Final Evaluation Results")
        self.logger.info("=" * 50)
        self.logger.info("\n" + results_str)
        self.logger.info("\n" + "=" * 50)
        self.logger.info("Evaluation Complete")
        self.logger.info("=" * 50)
