import os
from typing import Any, Callable, Dict, List, Optional, Tuple

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

    def __call__(self, query_path: str):
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

        # If we need similarity scores (for visualization), we need to compute them separately
        similarities = []
        if self.similarity_fn is not None and self.training_image_embeddings is not None:
            with torch.no_grad():
                query_embedding = self.similarity_fn([query_path])
                context_embeddings = self.training_image_embeddings[context_indices]
                similarities = F.cosine_similarity(query_embedding, context_embeddings).tolist()
        else:
            # If no similarity function provided, just use placeholder values
            similarities = [1.0] * len(context_indices)

        images_context = [self.train_set.get_image_paths(i) for i in context_indices]
        masks_context = [self.train_set.get_mask_array(i) for i in context_indices]

        video_preds = self.predictor.forward_in_context(
            image_query=query_path,
            images_context=images_context,
            masks_context=masks_context,
            category_ids=self.train_set.get_category_ids(),
        )

        # {frame_idx: {category_id: mask}}
        mask_query_dict = video_preds[self.context_size]  # {category_id: mask}

        return mask_query_dict, images_context, masks_context, similarities


class SegmentationUI:
    """Class to handle UI components and interactions"""

    def __init__(self, model: Segmenter):
        self.model = model

    def _create_upload_section(self) -> Tuple[gr.Gallery, gr.Markdown, gr.Video]:
        """Create the image/video upload section"""
        # For uploading a list of images
        with gr.Tab("Upload Image"):
            image_input = gr.Gallery(
                label="Upload images",
                elem_id="image-input",
                type="filepath",
                preview=True,
                show_label=True,
                interactive=True,
            )
            with gr.Accordion("Uploaded Files", open=False):
                uploaded_files_info = gr.Markdown("No images uploaded yet.")

        # For uploding a video
        with gr.Tab("Upload Video"):
            video_input = gr.Video(
                label="Upload a video to extract frames",
                elem_id="video-input",
            )

        return image_input, uploaded_files_info, video_input

    def _create_segmentation_section(
        self,
    ) -> Tuple[gr.Gallery, gr.AnnotatedImage, gr.Markdown, gr.Markdown]:
        """Create the segmentation visualization section"""
        with gr.Column(scale=2):
            image_gallery = gr.Gallery(
                label="Select an image to segment",
                show_label=True,
                elem_classes="gallery",
            )

            with gr.Group():
                gr.Markdown("## Segmentation Result")
                segmented_output = gr.AnnotatedImage(
                    show_legend=True,
                    height=350,
                    label="Segmentation Visualization",
                )
                segmentation_info = gr.Markdown("Select an image to see segmentation results")

        return image_gallery, segmented_output, segmentation_info

    def _create_context_section(self) -> Tuple[gr.Markdown, List[gr.AnnotatedImage]]:
        """Create the context images section"""
        context_info_md = gr.Markdown()
        context_images = []

        for i in range(0, self.model.context_size, 2):
            with gr.Row(elem_classes="context-grid"):
                for j in range(2):
                    if i + j < self.model.context_size:
                        annotated_img = gr.AnnotatedImage(
                            show_legend=True,
                            height=250,
                            label=f"Context {i + j + 1}",
                            elem_classes="context-item",
                        )
                        context_images.append(annotated_img)

        return context_info_md, context_images

    def handle_image_upload(self, gallery_images: List[Tuple[str, None]]) -> Tuple[List[Tuple[str, None]], str]:
        """Process uploaded images"""
        file_info = [f"{i + 1}. **{os.path.basename(img[0])}**" for i, img in enumerate(gallery_images)]
        return gallery_images, f"### Uploaded Files\n\n{''.join(file_info)}"

    @staticmethod
    def read_video(video_path: str) -> List[np.ndarray]:
        """Read video file and return a list of frames as numpy arrays"""
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        return frames

    def handle_video_upload(self, video_path: str) -> List[np.ndarray]:
        """
        Convert video to a list of frames (numpy arrays)

        """
        return self.read_video(video_path)

    def _prepare_context_data(
        self,
        images_context: List[str],
        masks_ctx_dicts: List[Dict[int, torch.Tensor]],
        similarities: List[float],
    ) -> Tuple[List[Tuple[np.ndarray, List[Tuple[np.ndarray, str]]]], List[str]]:
        """Prepare context images and their annotations

        Args:
            images_context: List of context image paths
            masks_ctx_dicts: List of context mask arrays (each array is a mapping from category id to mask tensor)
            similarities: List of similarity scores

        """
        context_outputs = []
        context_info = []

        for i, (img_path, masks_dict, sim_score) in enumerate(zip(images_context, masks_ctx_dicts, similarities)):
            img = Image.open(img_path)

            img_arr = np.array(img)

            mask_annotations = []

            for cat_id, cat_mask in masks_dict.items():
                cat_name = self.model.train_set.get_category_id_to_name()[cat_id]
                if (cat_mask > 0).any():
                    mask_annotations.append((cat_mask.cpu().numpy(), cat_name))

            context_outputs.append((img_arr, mask_annotations))
            img_filename = os.path.basename(img_path)
            context_info.append(
                f"**Context {i + 1}** - Similarity: {sim_score:.4f}\n\nFile: `{img_filename}`\n\nPath: `{img_path}`"
            )

        return context_outputs, context_info

    def handle_frame_select(
        self,
        evt: gr.SelectData,
        gallery: List[Dict[str, Any]],
        gallery_state: Optional[Dict],
    ) -> Tuple[Any, ...]:
        """Process selected frame/image"""
        query_path = evt.value["path"] if "path" in evt.value else evt.value["image"]["path"]

        mask_query_dict, images_context, mask_ctx_arrs, similarities = self.model(query_path=query_path)

        context_outputs, context_info = self._prepare_context_data(images_context, mask_ctx_arrs, similarities)

        query_arr = np.array(Image.open(query_path))
        annotations = []

        for cat_name, cat_id in zip(
            self.model.train_set.get_category_names(),
            self.model.train_set.get_category_ids(),
        ):
            cat_mask = mask_query_dict[cat_id].squeeze(0).cpu().numpy().astype(np.uint8)
            annotations.append((cat_mask, cat_name))

        detected_cats = [cat for m, cat in annotations if m.sum() > 0]
        detected_info = (
            f"### Segmentation Results\n\n{len(detected_cats)} categories detected: {', '.join(detected_cats)}"
            if detected_cats
            else "### No objects detected"
        )

        return (query_arr, annotations), context_outputs, "\n\n".join(context_info), detected_info

    def create_interface(self) -> gr.Blocks:
        """Create and return the complete Gradio interface"""
        with gr.Blocks(
            title="Temporal: Time-contrastive pretraing for in-context image and video segmentation",
        ) as app:
            self._create_header()

            with gr.Tabs():
                with gr.TabItem("Segmentation Tool"):
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1):
                            (
                                image_input,
                                uploaded_files_info,
                                video_input,
                            ) = self._create_upload_section()
                            with gr.Accordion("Dataset Information", open=False):
                                gr.Markdown(self._get_dataset_info())

                        (
                            image_gallery,
                            segmented_output,
                            segmentation_info,
                        ) = self._create_segmentation_section()

                    with gr.Accordion("Context Images", open=True):
                        gr.Markdown("### Context Images Used for Segmentation")
                        gr.Markdown(
                            "These are the most similar images from the training dataset that were used to guide the segmentation. "
                            "Masks are overlaid on the images."
                        )
                        context_info_md, context_images = self._create_context_section()

            gallery_state = gr.State()
            context_data_state = gr.State()

            image_input.change(
                fn=self.handle_image_upload,
                inputs=[image_input],
                outputs=[image_gallery, uploaded_files_info],
            )

            video_input.change(
                fn=self.handle_video_upload,
                inputs=[video_input],
                outputs=image_gallery,
            )

            image_gallery.select(
                fn=self.handle_frame_select,
                inputs=[image_gallery, gallery_state],
                outputs=[
                    segmented_output,
                    context_data_state,
                    context_info_md,
                    segmentation_info,
                ],
            ).then(
                fn=lambda context_outputs: [(img, masks) for img, masks in context_outputs],
                inputs=[context_data_state],
                outputs=context_images,
            )

        return app

    def _create_header(self):
        """Create the application header"""
        with gr.Row(elem_classes="header"):
            gr.Markdown("# Temporal: Time-contrastive pretraining for in-context image and video segmentation")

    def _get_dataset_info(self) -> str:
        """Get formatted dataset information"""
        category_stats = self.model.train_set.get_category_stats()
        stats_text = "\n".join(f"- {cat}: {count} images" for cat, count in category_stats.items())

        return f"""
        ## Dataset Information
        
        **Total images in training set:** {len(self.model.train_set)}
        **Context size:** {self.model.context_size}
        **Categories:** {", ".join(str(c) for c in self.model.train_set.get_category_names())}
        
        ### Category Distribution

        {stats_text}
        """


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
        cfg = compose(config_name="app_image")
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
    app = SegmentationUI(segmenter).create_interface()
    app.launch(share=cfg.runtime.share)


if __name__ == "__main__":
    main()
