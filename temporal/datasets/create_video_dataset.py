"""
Create a video dataset by finding similar images for each query image.

Directory structure:

input_dir/
├── train/
│   ├── imgs/
│   │   ├── labeled_frame{frame_idx}.png
│   │   └── ...
│   └── gts/
    │  ├── $ClassName$
    │       ├── ${PatientID}_labeled_frame{frame_idx}.png
    │       └── ...
└── test/
   Similar to train/


output_dir/
├── JPEGImages/
|     ├── ${className}/
│          ├── query_image1_name/
│          │   ├── 00000.jpg  # Least similar image
│          │   ├── 00001.jpg  # Second least similar
│          │   └── 000XX.jpg  # Original query image

└── Annotations/
    ├── ${className}/
    │   ├── query_image1_name/
    │   │   ├── 00000.png  # Mask for least similar
    │   │   ├── 00001.png  # Mask for second least similar
    │   │   └── 000XX.png  # Mask for query image
    │   └── query_image2_name/
    │       └── 00000.png
    └── ${className}/  # Repeat for each class
        └── ...
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from temporal.models.utils.embedding_fns import get_embedding_model
from temporal.utils.dataset import BinarySegmentationDataset, ImageListDataset
from temporal.utils.icl import select_context

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoDataset:
    def __init__(
        self,
        similarity_fn: Callable,
        training_image_embeddings: torch.Tensor,
        train_set: BinarySegmentationDataset,
        class_name: str,
    ):
        self.similarity_fn = similarity_fn
        self.training_image_embeddings = training_image_embeddings
        self.train_set = train_set
        self.class_name = class_name


def extract_training_embeddings(model: torch.nn.Module, dataloader: DataLoader) -> torch.Tensor:
    """Extract embeddings for training images."""
    model.eval()
    embeddings = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for images in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)
            features = model(images).view(images.size(0), -1)
            embeddings.append(features)

    return torch.cat(embeddings)


def make_video_dataset(
    creator: VideoDataset,
    output_dir: str,
    num_similar: int = 20,
    img_format: str = "jpg",
    sampling_strategy: str = "topk",
    pool_multiplier: int = 5,
    lambda_param: float = 0.5,
) -> None:
    """
    Create a video dataset by finding similar images for each query image.

    Args:
        creator: Instance of CreateVideoDataset with necessary data
        output_dir: Base output directory
        num_similar: Number of similar frames to include in each video
        img_format: Format for output images (jpg or png)
        sampling_strategy: Strategy for selecting similar frames ('topk' or 'diverse')
        pool_multiplier: Size multiplier for initial pool when using diverse strategy
        lambda_param: Diversity weight parameter for diverse selection
    """
    output_dir = Path(output_dir)
    image_dir = output_dir / "JPEGImages"
    mask_dir = output_dir / "Annotations"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    train_image_paths = creator.train_set.get_image_paths()
    # Process each image in the dataset for the current class
    for idx, query_path in enumerate(
        tqdm(
            train_image_paths,
            total=len(creator.train_set),
            desc=f"Processing {creator.class_name} images",
        )
    ):
        query_name = Path(query_path).stem

        context_indices = select_context(
            query_image_path=query_path,
            train_set=creator.train_set,
            categories=creator.train_set.get_category_ids(),
            query_image_embedding=creator.training_image_embeddings[idx].unsqueeze(0),
            training_embeddings=creator.training_image_embeddings,
            context_size=num_similar,
            similarity_fn=creator.similarity_fn,
            sampling_strategy=sampling_strategy,
            pool_multiplier=pool_multiplier,
            lambda_param=lambda_param,
        )

        # Create directories for this query image
        img_seq_dir = image_dir / query_name
        mask_seq_dir = mask_dir / query_name
        img_seq_dir.mkdir(parents=True, exist_ok=True)
        mask_seq_dir.mkdir(parents=True, exist_ok=True)

        # Save similar frames and their masks
        for j, img_idx in enumerate(context_indices):
            img_path = creator.train_set.get_image_paths(img_idx)
            shutil.copy(img_path, img_seq_dir / f"{j:03d}.{img_format}")

            # Get mask for this context image
            shutil.copy(
                creator.train_set.masks_dir / Path(img_path).name,
                mask_seq_dir / f"{j:03d}.png",
            )

        # Save query frame as last frame
        shutil.copy(query_path, img_seq_dir / f"{num_similar:03d}.{img_format}")

        # Get mask for query image
        shutil.copy(
            creator.train_set.masks_dir / Path(query_path).name,
            mask_seq_dir / f"{num_similar:03d}.png",
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Create video dataset from similar images")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Input data directory containing imgs/ and gts/",
    )
    parser.add_argument("--output_dir", required=True, help="Output directory for video dataset")

    # Model selection and configuration
    parser.add_argument(
        "--model_type",
        default="dinov2",
        choices=["resnet50", "resnet18", "clip", "dinov2", "medclip"],
        help="Type of model to use for similarity computation",
    )
    parser.add_argument("--checkpoint", help="Model checkpoint path (required for ResNet models)")
    parser.add_argument(
        "--dinov2_model_key",
        default="dinov2_vitb14",
        choices=["dinov2_vitb14", "dinov2_vitl14", "dinov2_vits14", "dinov2_vitg14"],
        help="DinoV2 model variant to use",
    )

    # Dataset creation parameters
    parser.add_argument("--num-similar", type=int, default=20, help="Number of similar frames")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-format", default="jpg", choices=["png", "jpg"])

    # Selection strategy parameters
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="topk",
        choices=["topk", "diverse"],
        help="Strategy for selecting similar frames",
    )
    parser.add_argument(
        "--pool-multiplier",
        type=int,
        default=5,
        help="Pool size multiplier for diverse sampling",
    )
    parser.add_argument(
        "--lambda-param",
        type=float,
        default=0.5,
        help="Diversity weight for greedy diverse selection",
    )

    # Add seed for reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Find all classes in the dataset
    mask_dir = Path(args.input_dir) / "gts"
    class_names = [d.name for d in mask_dir.iterdir() if d.is_dir()]
    logger.info(f"Found classes: {class_names}")

    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize embedding model
    logger.info(f"Building {args.model_type} model...")
    # Initialize model using the function from embedding_fns
    model_kwargs = {
        "checkpoint_path": args.checkpoint,
        "model_key": args.dinov2_model_key,
    }
    vision_encoder, similarity_fn = get_embedding_model(args.model_type, **model_kwargs)

    if vision_encoder is not None:
        vision_encoder = vision_encoder.to(device)

    # Process each class separately
    for class_name in class_names:
        logger.info(f"Processing class: {class_name}")

        # Load dataset for this class
        train_dataset = BinarySegmentationDataset(
            images_dir=Path(args.input_dir) / "imgs",
            masks_dir=mask_dir / class_name,
            transform=transform,
        )

        # Extract embeddings for images
        logger.info("Extracting embeddings for images...")
        dataloader = DataLoader(
            ImageListDataset(
                image_paths=train_dataset.get_image_paths(),
                transform=transform,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        if vision_encoder is not None:
            training_embeddings = extract_training_embeddings(vision_encoder, dataloader)
            training_embeddings = F.normalize(training_embeddings, p=2, dim=1)
            logger.info(f"Extracted embeddings shape: {training_embeddings.shape}")
        else:
            training_embeddings = None
            logger.warning("No embedding model provided, using random selection")

        # Create dataset generator
        creator = VideoDataset(
            similarity_fn=similarity_fn,
            training_image_embeddings=training_embeddings,
            train_set=train_dataset,
            class_name=class_name,
        )

        # Create output directory structure
        logger.info(f"Creating video dataset for {class_name}...")
        class_output_dir = Path(args.output_dir) / class_name
        make_video_dataset(
            creator=creator,
            output_dir=class_output_dir,
            num_similar=args.num_similar,
            img_format=args.image_format,
            sampling_strategy=args.sampling_strategy,
            pool_multiplier=args.pool_multiplier,
            lambda_param=args.lambda_param,
        )

        logger.info(f"Video dataset for {class_name} complete.")

    logger.info("Video dataset creation complete.")
