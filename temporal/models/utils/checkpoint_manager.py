import logging
from pathlib import Path
from typing import Optional, Tuple

import requests
import torch


class CheckpointManager:
    """Manages downloading and loading of model checkpoints."""

    # Default paths and URLs
    DEFAULT_CHECKPOINT_DIR = "checkpoints"
    MODEL_URLS = {
        "tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "small": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "base": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "large": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    }

    MODEL_CONFIGS = {
        "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "base": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
    }

    def __init__(self, checkpoint_dir: Optional[str] = None):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints. Defaults to 'checkpoints'.
        """
        self.checkpoint_dir = Path(checkpoint_dir or self.DEFAULT_CHECKPOINT_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def get_checkpoint_path(self, model_size: str) -> Path:
        """Get path for a specific model checkpoint."""
        if model_size not in self.MODEL_URLS:
            raise ValueError(f"Invalid model size {model_size}. Must be one of {list(self.MODEL_URLS.keys())}")
        return self.checkpoint_dir / Path(self.MODEL_URLS[model_size]).name

    def download_checkpoint(self, url: str, dest_path: Path) -> None:
        """Download a checkpoint file."""
        self.logger.info(f"Downloading checkpoint from {url} to {dest_path}")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Get total file size
        total_size = int(response.headers.get("content-length", 0))

        # Download with progress tracking
        with open(dest_path, "wb") as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                for data in response.iter_content(chunk_size=8192):
                    downloaded += len(data)
                    f.write(data)
                    done = int(50 * downloaded / total_size)
                    print(f"\rDownloading: [{'=' * done}{' ' * (50 - done)}] {downloaded}/{total_size} bytes", end="")
        print()

        self.logger.info(f"Successfully downloaded checkpoint to {dest_path}")

    def ensure_checkpoint_exists(self, model_size: str) -> Tuple[str, Path]:
        """Ensure checkpoint exists, downloading if necessary.

        Args:
            model_size: Size of model to ensure exists ("tiny", "small", "base", "large")

        Returns:
            Tuple of (config_path, checkpoint_path)
        """
        checkpoint_path = self.get_checkpoint_path(model_size)

        if not checkpoint_path.exists():
            self.download_checkpoint(self.MODEL_URLS[model_size], checkpoint_path)

        return self.MODEL_CONFIGS[model_size], checkpoint_path

    def load_checkpoint(self, model, checkpoint_path: Path, map_location: str = "cpu") -> None:
        """Load a checkpoint into a model.

        Args:
            model: Model to load checkpoint into
            checkpoint_path: Path to checkpoint file
            map_location: Device to load checkpoint to
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=map_location)["model"]

        # Load state dict and check for errors
        missing_keys, unexpected_keys = model.load_state_dict(state_dict)

        if missing_keys:
            self.logger.error(f"Missing keys: {missing_keys}")
            raise RuntimeError("Missing keys when loading checkpoint")

        if unexpected_keys:
            self.logger.error(f"Unexpected keys: {unexpected_keys}")
            raise RuntimeError("Unexpected keys when loading checkpoint")

        self.logger.info("Successfully loaded checkpoint")
