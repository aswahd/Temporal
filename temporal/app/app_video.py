from __future__ import annotations

import tempfile
from typing import Callable, Dict, List, Tuple

import cv2
import gradio as gr
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from temporal import setup_device
from temporal.models.temporal.temporal_predictor import TemporalVideoPredictor
from temporal.utils.dataset import ImageListDataset, MultiMaskDataset, SingleMaskDataset
from temporal.utils.icl import select_context


class Segmenter:
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

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def compute_frame_similarities(self, frame_paths: List[str]) -> torch.Tensor:
        """Compute similarity scores for video frames using training embeddings."""
        embeddings = []
        for path in tqdm(frame_paths, desc="computing similarity to training images"):
            with torch.no_grad():
                embeddings.append(self.similarity_fn([path]))

        embeddings = torch.cat(embeddings, dim=0)
        embeddings = F.normalize(embeddings, dim=1, p=2)  # (num_frames, D)
        return embeddings @ self.training_image_embeddings.T

    def select_key_frames(self, similarities: torch.Tensor) -> List[int]:
        """Select diverse keyframes using similarity scores and temporal constraints."""
        selected = []
        sorted_indices = similarities.argsort(descending=True).cpu().tolist()

        for idx in sorted_indices:
            if len(selected) >= self.num_keyframes:
                break
            if not selected or all(abs(idx - s) > self.temporal_distance for s in selected):
                selected.append(idx)
        return sorted(selected)

    def inference_vos(self, frames: List[np.ndarray]) -> Dict[int, np.ndarray]:
        # TODO: This can be improved by avoiding writing to disk
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_paths = [f"{tmpdir}/frame_{i:04d}.png" for i in range(len(frames))]

            for path, frame in tqdm(zip(frame_paths, frames, strict=True)):
                Image.fromarray(frame).save(path)

            similarities = self.compute_frame_similarities(frame_paths)  # (num_frames, num_train_images)
            keyframe_indices = self.select_key_frames(similarities.max(dim=1).values)
            print(f"Selected keyframes: {keyframe_indices}")

            keyframe_masks = []
            for idx in tqdm(keyframe_indices, desc="Processing keyframes"):
                mask_query_dict = self.inference_image(query_path=frame_paths[idx])  # {category_id: mask, ...}
                keyframe_masks.append(mask_query_dict)
                # TODO: Add confidence thresholding

            video_segments = self.predictor.forward_vos(
                image_paths=frame_paths, keyframe_indices=keyframe_indices, keyframe_masks=keyframe_masks
            )

        return video_segments  # {frame_idx: {category_id: mask, ...}, ...}

    def inference_image(self, query_path: str) -> Tuple[Dict[int, torch.Tensor], List[str], List[str]]:
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


def create_app(model: Segmenter) -> gr.Blocks:
    """Create Gradio interface for video segmentation."""

    def process_video(
        video_path: str | None,
    ) -> tuple[list[np.ndarray], dict[int, dict[int, torch.Tensor]], str]:
        """
        Process a video file into frames and segmentation masks.

        Args:
            video_path: Path to the video file to process, or None if no video uploaded

        Returns:
            A tuple containing:
            - List of frames as numpy arrays
            - Dictionary mapping frame indices to segmentation masks
            - Status message as string
        """
        if not video_path:
            return [], {}, "Please upload a video first."

        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            masks = model.inference_vos(frames)

            return (
                frames,
                masks,
                f"Processing complete! {len(frames)} frames analyzed.",
            )
        except Exception as e:
            print(f"Error processing video: {str(e)}")
            import traceback

            traceback.print_exc()
            return [], {}, f"Error processing video: {str(e)}"

    def process_images(
        image_list: list[tuple[str, None]],
    ) -> tuple[list[np.ndarray], dict[int, dict[int, torch.Tensor]], str]:
        """
        Process multiple images as if they were frames in a video.
        """

        frames = [img for img, _ in image_list]
        frames = [np.array(Image.open(img).convert("RGB")) for img in frames]

        print(f"Processing {len(frames)} frames...")
        masks = model.inference_vos(frames)

        return (
            frames,
            masks,
            f"Processing complete! {len(frames)} frames analyzed.",
        )

    def on_frame_select(
        evt: gr.SelectData,
        frames: list[np.ndarray],
        masks: dict[int, dict[int, torch.Tensor]],
    ) -> tuple[tuple[np.ndarray, list[tuple[np.ndarray, str]]], str]:
        """
        Process the selected frame and display its segmentation mask
        """
        idx = evt.index
        frame = frames[idx][0]  # Get image path
        mask_dict = masks.get(idx, {})

        # Create gradio annotations from masks
        annotations = []
        for cat, mask in mask_dict.items():
            mask_np = mask[0].long().cpu().numpy()  # HxW
            binary_mask = mask_np > 0

            annotations.append((binary_mask, str(cat)))

        # Return annotated image data
        label_info = "No segmentation found." if not annotations else f"Segmentation detected for frame {idx + 1}"
        return (frame, annotations), label_info

    with gr.Blocks(
        title="Temporal Video Segmentation",
    ) as app:
        with gr.Row(elem_classes="header"):
            gr.Markdown("# Temporal: Video Segmentation")

        with gr.Row(elem_classes="container"):
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem("Upload Video"):
                        video_input = gr.Video(
                            label="Input Video (Processing will start automatically after upload)",
                            elem_id="video-input",
                            elem_classes="video-preview",
                            interactive=True,
                        )

                    with gr.TabItem("Upload Images"):
                        image_input = gr.Gallery(
                            label="Upload Multiple Images (Processing will start automatically after upload)",
                            elem_id="image-input",
                            type="filepath",
                            preview=True,
                            show_label=True,
                            interactive=True,
                        )

                status = gr.Markdown("", elem_classes="status-msg")

                print("Debug info:", model.similarity_fn.__module__)
                with gr.Accordion("Advanced Info", open=False):
                    gr.Markdown(f"""
                        ### Processing Parameters
                        - Context size: {model.context_size}
                        - Number of keyframes: {model.num_keyframes}
                        - Temporal distance: {model.temporal_distance}
                        - Confidence threshold: {model.confidence_threshold}
                        - Categories: {", ".join(str(c) for c in model.train_set.get_category_names())}
                        - Embedding model: {model.similarity_fn.__module__.split(".")[-1] if model.similarity_fn else "Random"}
                        - Sampling strategy: {model.sampling_strategy}
                    """)

            with gr.Column(scale=2, elem_classes="results-container"):
                gr.Markdown("### Video Frames and Segmentation")

                frame_gallery = gr.Gallery(
                    label="Select a frame to view its segmentation",
                    elem_classes="gallery",
                    object_fit="contain",
                    show_label=True,
                    preview=True,
                )

                with gr.Column(elem_classes="segmentation-area"):
                    gr.Markdown("##### Frame Segmentation")
                    segmented_output = gr.AnnotatedImage(
                        label="Segmentation Visualization",
                        show_legend=True,
                    )
                    frame_info = gr.Markdown(
                        "Select a frame above to view its segmentation",
                        elem_classes="status-msg",
                    )

        mask_state = gr.State()

        video_input.change(
            process_video,
            inputs=video_input,
            outputs=[frame_gallery, mask_state, status],
        )

        image_input.change(
            process_images,
            inputs=image_input,
            outputs=[frame_gallery, mask_state, status],
        )

        frame_gallery.select(
            on_frame_select,
            inputs=[frame_gallery, mask_state],
            outputs=[segmented_output, frame_info],
        )

        with gr.Row(elem_classes="footer"):
            gr.Markdown("""
                ### About Temporal
                Temporal is a time-aware medical image segmentation framework that uses in-context learning with SAM2.
                For more information, visit our project repository.
            """)

    return app


def main():
    with initialize_config_module(config_module="temporal/inference_configs/"):
        cfg = compose(config_name="app_video")
        OmegaConf.resolve(cfg)

    print(OmegaConf.to_yaml(cfg, resolve=True))
    device = setup_device()
    vision_encoder, sim_fn = hydra.utils.instantiate(cfg.prompt_retriever.embedding_fn)
    if vision_encoder is not None:
        vision_encoder = vision_encoder.to(device)
    print(f"Loading dataset from {cfg.dataset.root_dir}...")
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_set = hydra.utils.instantiate(cfg.dataset)(transform=test_transform)
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

    print("Building predictor...")
    predictor = hydra.utils.instantiate(cfg.predictor)

    segmenter = hydra.utils.instantiate(cfg.segmenter)(
        predictor=predictor,
        train_set=train_set,
        training_image_embeddings=training_image_embeddings,
        similarity_fn=sim_fn,
    )
    print("Creating Gradio interface...")
    app = create_app(segmenter)
    app.launch(share=cfg.runtime.share)


if __name__ == "__main__":
    main()
