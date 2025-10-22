from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import hydra
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from temporal import setup_device
from temporal.eval.callbacks import EvalLoggerCallback
from temporal.eval.metrics import DiceMetric, IoUMetric
from temporal.logging import setup_logging
from temporal.models.temporal.temporal_predictor import TemporalVideoPredictor
from temporal.utils.dataset import ImageListDataset, MultiMaskDataset, SingleMaskDataset
from temporal.utils.icl import select_context


class VideoEvaluator:
    """
    Evaluates video segmentation predictions using either semi-supervised or ICL inference.
    Computes per-class Dice and IoU metrics and optionally saves debug visualizations.
    """

    def __init__(
        self,
        predictor: TemporalVideoPredictor,
        train_set: SingleMaskDataset | MultiMaskDataset,
        training_image_embeddings: torch.Tensor | None,
        context_size: int = 4,
        similarity_fn: Callable | None = None,
        num_keyframes: int = 20,
        temporal_distance: int = 4,
        confidence_threshold: float = 0.7,
        sampling_strategy: str = "topk",
        pool_multiplier: int = 5,
        lambda_param: float = 0.5,
        inference_mode: str = "icl",
    ):
        self.predictor = predictor
        self.train_set = train_set
        self.training_image_embeddings = training_image_embeddings
        self.context_size = context_size
        self.similarity_fn = similarity_fn
        self.num_keyframes = num_keyframes
        self.temporal_distance = temporal_distance
        self.confidence_threshold = confidence_threshold
        self.sampling_strategy = sampling_strategy
        self.pool_multiplier = pool_multiplier
        self.lambda_param = lambda_param
        self.inference_mode = inference_mode

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.dice_metric = DiceMetric(reduction="mean_batch", ignore_empty=True)
        self.iou_metric = IoUMetric(reduction="mean_batch", ignore_empty=True)

        self.callbacks = []

    def reset_metrics(self):
        """Reset evaluation metrics."""
        self.dice_metric.reset()
        self.iou_metric.reset()

    def select_key_frames(self, similarities: torch.Tensor) -> List[int]:
        """Select diverse keyframes using similarity scores and temporal constraints."""
        assert isinstance(similarities, torch.Tensor), "Similarities must be a torch.Tensor"
        assert similarities.ndim == 1, "Expected 1D tensor, got shape {}".format(similarities.shape)
        assert similarities.numel() > 0, "Empty similarities tensor"

        selected = []
        sorted_indices = similarities.argsort(descending=True).cpu().tolist()

        for idx in sorted_indices:
            if len(selected) >= self.num_keyframes:
                break
            if not selected or all(abs(idx - s) > self.temporal_distance for s in selected):
                selected.append(idx)
        return sorted(selected)

    def process_video_semisupervised(
        self,
        masks_gt_dict: dict[int, dict[int, torch.Tensor]],
        frame_paths: list[str | Path],
    ) -> dict[int, dict[int, torch.Tensor]]:
        """
        For each category, find the first frame where it is present in the ground truth and prompt the predictor.
        Then propagate segmentation masks through the video.
        """
        inference_state = self.predictor.set_predictor_state(image_paths=frame_paths)

        for category_id in self.train_set.get_category_ids():
            # Find first frame with the category present
            for frame_idx, gt_mask_dict in masks_gt_dict.items():
                if gt_mask_dict[category_id].sum() > 0:
                    mask_prompt = gt_mask_dict[category_id]
                    break
            else:
                # This is a fallback if no frame with the category is found
                # Will be prompted with empty mask
                frame_idx = 0
                mask_prompt = torch.zeros(self.predictor.image_size, self.predictor.image_size, dtype=torch.uint8)

            self.predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=category_id,
                mask=mask_prompt,
                is_init_cond_frame=True,
            )
            # print(f"Added mask for category {category_id} in frame {frame_idx}")

        return self.predictor.propagate_video_segments(inference_state, start_frame_idx=0)

    def compute_frame_similarities(self, frame_paths: List[str]) -> torch.Tensor:
        """Compute similarity scores for video frames using training embeddings."""
        embeddings = []
        for path in tqdm(frame_paths, desc="computing similarity to training images"):
            with torch.no_grad():
                embeddings.append(self.similarity_fn([path]))

        embeddings = torch.cat(embeddings, dim=0)
        embeddings = F.normalize(embeddings, dim=1, p=2)  # (num_frames, D)
        return embeddings @ self.training_image_embeddings.T

    def process_video_icl(self, frame_paths: List[str]) -> Dict[int, torch.Tensor]:
        if self.similarity_fn is not None:
            similarities = self.compute_frame_similarities(frame_paths)  # (num_frames, num_train_images)
            topk = similarities.max(dim=1).values
        else:
            topk = torch.randint(0, len(frame_paths), (len(frame_paths),))

        keyframe_indices = self.select_key_frames(topk)
        print(f"Selected keyframes: {keyframe_indices}")

        keyframe_masks = []
        for idx in tqdm(keyframe_indices, desc="Processing keyframes"):
            mask_query_dict = self.inference_image(query_path=frame_paths[idx])  # {category_id: mask, ...}
            keyframe_masks.append(mask_query_dict)
            # TODO: Add confidence thresholding
            # h, w = mask_query_dict[next(iter(mask_query_dict))].squeeze(0).shape

        video_preds = self.predictor.forward_vos(
            image_paths=frame_paths, keyframe_indices=keyframe_indices, keyframe_masks=keyframe_masks
        )  # {frame_idx: {category_id: mask, ...}, ...

        w, h = Image.open(frame_paths[0]).size

        # If any frame is missing a category (should not happen), fill with empty masks
        for frame_idx in range(len(frame_paths)):
            for category_id in self.train_set.get_category_ids():
                if category_id in video_preds[frame_idx]:
                    video_preds[frame_idx][category_id] = (
                        video_preds[frame_idx][category_id].type(torch.uint8).squeeze(0)
                    )
                else:
                    video_preds[frame_idx][category_id] = torch.zeros((h, w), dtype=torch.uint8)

        return video_preds

    def inference_image(self, query_path: str) -> Dict[int, torch.Tensor]:
        """Process single image (original functionality)"""
        context_indices = select_context(
            query_image_path=query_path,
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
            image_query=query_path,
            images_context=images_context,
            masks_context=masks_context,
            category_ids=self.train_set.get_category_ids(),
        )

        mask_query_dict = video_preds[self.context_size]  # {category_id: mask}

        return mask_query_dict

    def process_video(
        self, masks_gt_dict: dict[int, dict[int, torch.Tensor]], frame_paths: list[str]
    ) -> dict[int, dict[int, torch.Tensor]]:
        """
        Process a video using the specified inference mode.
        """

        if self.inference_mode == "semi_supervised":
            # self.logger.info("Using semi-supervised video segmentation")
            return self.process_video_semisupervised(masks_gt_dict=masks_gt_dict, frame_paths=frame_paths)
        else:
            # self.logger.info("Using ICL for video segmentation")
            return self.process_video_icl(frame_paths=frame_paths)

    def eval_batch(self, batch: dict):
        """Evaluate segmentation for a single video batch."""
        frames = batch["frames"]
        masks_gt_dict = batch["masks"]
        video_dir_str = batch["video_dir"]

        video_dir = Path(video_dir_str)
        frame_paths = sorted(video_dir.glob("*.jpg"))
        if not frame_paths:
            frame_paths = sorted(video_dir.glob("*.png"))
        frame_paths = [str(p) for p in frame_paths]

        masks_pred_dict = self.process_video(masks_gt_dict=masks_gt_dict, frame_paths=frame_paths)
        # Sort predictions by frame index
        masks_pred_dict = {k: masks_pred_dict[k] for k in range(len(frames))}

        # batch video predictions by category
        mask_gt_dict = {cat: [] for cat in self.train_set.get_category_ids()}
        for frame_idx in masks_gt_dict:
            for cat_id in self.train_set.get_category_ids():
                mask_gt_dict[cat_id].append(masks_gt_dict[frame_idx][cat_id])

        mask_gt_dict = {k: torch.stack(v) for k, v in mask_gt_dict.items()}
        mask_pred_dict = {cat: [] for cat in self.train_set.get_category_ids()}
        for frame_idx in masks_pred_dict:
            for cat_id in self.train_set.get_category_ids():
                mask_pred_dict[cat_id].append(masks_pred_dict[frame_idx][cat_id].squeeze(0).squeeze(0))
        mask_pred_dict = {k: torch.stack(v) for k, v in mask_pred_dict.items()}

        self.dice_metric(mask_pred_dict, mask_gt_dict)
        self.iou_metric(mask_pred_dict, mask_gt_dict)

    @torch.no_grad()
    def eval(self, dataloader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate segmentation on all videos from the dataloader."""
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
                pbar.update(1)

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

    def _format_results(self, dice_scores: torch.Tensor, iou_scores: torch.Tensor) -> str:
        results_str = "=== Video Evaluation Results ===\n\n"
        results_str += f"Context Size: {self.context_size}\n"
        results_str += f"Number of Keyframes: {self.num_keyframes}\n"
        results_str += f"Temporal Distance: {self.temporal_distance}\n"
        results_str += f"Categories: {self.train_set.get_category_names()}\n\n"
        results_str += "Per-Class Metrics:\n"
        results_str += "{:<10}{:>10}{:>10}\n".format("Class", "Dice", "IoU")
        results_str += "-" * 30 + "\n"
        for i, (dice, iou) in enumerate(zip(dice_scores, iou_scores, strict=True)):
            class_name = self.train_set.get_category_names()[i]
            results_str += "{:<10}{:>10.4f}{:>10.4f}\n".format(class_name, dice.item(), iou.item())
        results_str += "\nAverage Metrics:\n"
        results_str += "-" * 30 + "\n"
        results_str += "{:<10}{:>10.4f}{:>10.4f}\n".format(
            "Mean", dice_scores.nanmean().item(), iou_scores.nanmean().item()
        )
        return results_str


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
        cfg = compose(config_name="eval_video")
        OmegaConf.resolve(cfg)

    print(OmegaConf.to_yaml(cfg, resolve=True))
    device = setup_device()
    logger = setup_logging(cfg.logs, level=cfg.get("log_level", "INFO"))
    vision_encoder, sim_fn = hydra.utils.instantiate(cfg.prompt_retriever.embedding_fn)
    if vision_encoder is not None:
        vision_encoder = vision_encoder.to(device)
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    print(f"Building context from {cfg.train_dataset.root_dir}...")
    train_set = hydra.utils.instantiate(cfg.train_dataset)(transform=test_transform)
    print(f"Loading test dataset from {cfg.train_dataset.root_dir}...")
    test_set = hydra.utils.instantiate(cfg.test_dataset)(
        category_ids=train_set.get_category_ids(),
        transform=None,
    )
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

    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        num_workers=1,
        pin_memory=True if device.type == "cuda" else False,
        collate_fn=lambda x: x[0],
    )

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
