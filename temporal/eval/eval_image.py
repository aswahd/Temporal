from typing import Callable, Dict, Optional, Tuple

import hydra
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from torchvision import transforms
from tqdm import tqdm

from temporal import setup_device
from temporal.eval.callbacks import EvalLoggerCallback
from temporal.eval.metrics import DiceMetric, IoUMetric
from temporal.logging import setup_logging
from temporal.models.temporal.temporal_predictor import TemporalImagePredictor
from temporal.utils.dataset import ImageListDataset, MultiMaskDataset, SingleMaskDataset
from temporal.utils.icl import select_context


class ImageEvaluator:
    """Handles evaluation of segmentation predictions"""

    def __init__(
        self,
        predictor: TemporalImagePredictor,
        train_set: SingleMaskDataset | MultiMaskDataset,
        training_image_embeddings: torch.Tensor,
        context_size: int = 10,
        similarity_fn: Optional[Callable] = None,
        sampling_strategy: str = "topk",
        pool_multiplier: int = 5,
        lambda_param: float = 0.5,
    ):
        self.predictor = predictor
        self.train_set = train_set
        self.training_image_embeddings = training_image_embeddings
        self.context_size = context_size
        self.similarity_fn = similarity_fn
        self.sampling_strategy = sampling_strategy
        self.pool_multiplier = pool_multiplier
        self.lambda_param = lambda_param

        self.dice_metric = DiceMetric(reduction="mean_batch", ignore_empty=True)
        self.iou_metric = IoUMetric(reduction="mean_batch", ignore_empty=True)

        self.callbacks = []

    def reset_metrics(self):
        """Reset evaluation metrics"""
        self.dice_metric.reset()
        self.iou_metric.reset()

    def __call__(self, query_image_path: str):
        context_indices = select_context(
            query_image_path=query_image_path,
            training_embeddings=self.training_image_embeddings,
            categories=self.train_set.get_category_ids(),
            train_set=self.train_set,
            context_size=self.context_size,
            similarity_fn=self.similarity_fn,
            sampling_strategy=self.sampling_strategy,
            pool_multiplier=self.pool_multiplier,
            lambda_param=self.lambda_param,
        )

        images_context = [self.train_set.get_image_paths(i) for i in context_indices]
        masks_context = [self.train_set.get_mask_array(i) for i in context_indices]

        video_preds = self.predictor.forward_in_context(
            image_query=query_image_path,
            images_context=images_context,
            masks_context=masks_context,
            category_ids=self.train_set.get_category_ids(),
        )

        # {frame_idx: {category_id: mask}}
        mask_query_dict = video_preds[self.context_size]  # {category_id: mask}
        mask_query_dict = {k: v.squeeze(0) for k, v in mask_query_dict.items()}  # {category_id: mask (H, W)}
        return mask_query_dict

    def eval_batch(self, batch: dict):
        """Evaluate a single batch"""
        # Batch size should be 1.
        # The ground truth and prediction masks don't include the background class.
        assert len(batch) == 1, "Batch size should be 1 for image evaluation"
        batch = batch[0]
        query_image_path = batch["img_path"]
        mask_gt_dict = batch["masks"]
        img_size = batch["img_size"]

        self.before_forward(query_image_path=query_image_path, query_image_size=img_size)
        mask_pred_dict = self(query_image_path=query_image_path)
        self.after_forward(mask_pred_dict)

        # Calculate metrics
        self.dice_metric(mask_pred_dict, mask_gt_dict)
        self.iou_metric(mask_pred_dict, mask_gt_dict)

        # Calculate per-example dice scores
        dice_per_class = self.dice_metric.aggregate(reduction="none")[-1]
        iou_per_class = self.iou_metric.aggregate(reduction="none")[-1]
        return dice_per_class, iou_per_class

    @torch.no_grad()
    def eval(self, dataloader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate on full dataset"""
        self.before_eval(dataloader)
        self.reset_metrics()

        with tqdm(dataloader, total=len(dataloader), position=0, leave=True) as pbar:
            for batch in pbar:
                self.before_eval_batch(batch)
                self.eval_batch(batch)
                self.after_eval_batch(
                    self.dice_metric.aggregate(reduction="none").nanmean(0),
                    self.iou_metric.aggregate(reduction="none").nanmean(0),
                )
                dice_score = self.dice_metric.aggregate(reduction="none").nanmean(0).nanmean(0).item()
                iou_score = self.iou_metric.aggregate(reduction="none").nanmean(0).nanmean(0).item()
                pbar.set_description(f"Dice: {dice_score:.4f}, IoU: {iou_score:.4f}")

        # Get final results
        dice_scores = self.dice_metric.aggregate(reduction="none").nanmean(0)
        iou_scores = self.iou_metric.aggregate(reduction="none").nanmean(0)

        self.after_eval(dice_scores, iou_scores)
        return dice_scores, iou_scores

    def add_callback(self, callback):
        """Add a callback to the evaluator"""
        self.callbacks.append(callback)

    def before_eval(self, dataloader: torch.utils.data.DataLoader):
        """Called before evaluation starts"""
        for callback in self.callbacks:
            if hasattr(callback, "before_eval"):
                callback.before_eval(self, dataloader)

    def before_eval_batch(self, batch: dict):
        """Called before evaluating a batch"""
        for callback in self.callbacks:
            if hasattr(callback, "before_eval_batch"):
                callback.before_eval_batch(self, batch)

    def before_forward(self, query_image_path: str, query_image_size: Tuple[int, int]):
        """Called before forward pass"""
        for callback in self.callbacks:
            if hasattr(callback, "before_forward"):
                callback.before_forward(self, query_image_path=query_image_path, query_image_size=query_image_size)

    def after_forward(self, mask_pred_dict: Dict[int, torch.Tensor]):
        """Called after forward pass"""
        for callback in self.callbacks:
            if hasattr(callback, "after_forward"):
                callback.after_forward(self, mask_pred_dict)

    def after_eval_batch(self, dice_per_class: torch.tensor, iou_per_class: torch.tensor):
        """Called after evaluating a batch"""
        for callback in self.callbacks:
            if hasattr(callback, "after_eval_batch"):
                callback.after_eval_batch(self, dice_per_class=dice_per_class, iou_per_class=iou_per_class)

    def after_eval(self, dice_scores: torch.Tensor, iou_scores: torch.Tensor):
        """Called after evaluation ends"""
        for callback in self.callbacks:
            if hasattr(callback, "after_eval"):
                callback.after_eval(self, dice_scores=dice_scores, iou_scores=iou_scores)


def get_embeddings_from_dataloader(model, dataloader) -> torch.Tensor:
    """Extract embeddings for training images."""
    device = next(model.parameters()).device
    model.eval()
    embeddings = []

    with torch.no_grad():
        for images in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)
            features = model(images).view(images.size(0), -1)
            embeddings.append(features)

    embeddings = torch.cat(embeddings)
    return F.normalize(embeddings, p=2, dim=1)


def main():
    with initialize_config_module(config_module="temporal/inference_configs/"):
        cfg = compose(config_name="eval_image")
        OmegaConf.resolve(cfg)

    print(OmegaConf.to_yaml(cfg, resolve=True))
    device = setup_device()
    logger = setup_logging(cfg.logs, level=cfg.get("log_level", "INFO"))
    vision_encoder, sim_fn = hydra.utils.instantiate(cfg.prompt_retriever.embedding_fn)
    if vision_encoder is not None:
        vision_encoder = vision_encoder.to(device)
    print(f"Loading dataset from {cfg.train_dataset.root_dir}...")
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    print(f"Loading context from {cfg.train_dataset.root_dir}...")
    train_set = hydra.utils.instantiate(cfg.train_dataset)(transform=test_transform)
    print(f"Loading test dataset from {cfg.train_dataset.root_dir}...")
    test_set = hydra.utils.instantiate(cfg.test_dataset)(transform=test_transform)
    print("Extracting training embeddings...")
    if vision_encoder is not None:
        training_image_embeddings = get_embeddings_from_dataloader(
            model=vision_encoder,
            dataloader=torch.utils.data.DataLoader(
                ImageListDataset(
                    image_paths=train_set.get_image_paths(),
                    transform=test_transform,
                ),
                batch_size=128,
                shuffle=False,
                num_workers=4,
            ),
        )
        print(f"Training embeddings shape: {training_image_embeddings.shape}")
    else:
        training_image_embeddings = None
        print("No vision encoder provided, using random selection")

    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    print("Building predictor...")
    predictor = hydra.utils.instantiate(cfg.predictor)

    evaluator = hydra.utils.instantiate(cfg.evaluator)(
        predictor=predictor,
        train_set=train_set,
        training_image_embeddings=training_image_embeddings,
        similarity_fn=sim_fn,
    )

    evaluator.add_callback(EvalLoggerCallback(logger))
    dice_scores, iou_scores = evaluator.eval(test_dataloader)

    return dice_scores, iou_scores


if __name__ == "__main__":
    main()
