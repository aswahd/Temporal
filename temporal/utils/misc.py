import os
from tempfile import TemporaryDirectory
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
from PIL import Image


def convert_to_semantic(mask: torch.Tensor):
    """
    Convert a multi-channel mask to a semantic segmentation mask.

    Args:
    mask: (B, C, H, W) - A PyTorch tensor where B is batch size, C is number of classes,
                        H is height, and W is width.

    Returns:
    sem_mask: (B, H, W) - A PyTorch tensor representing the semantic segmentation mask.
    """

    # Get the class with the highest probability for each pixel (adding 1 to account for ignored background)
    sem_mask = mask.argmax(dim=1) + 1

    # Create a foreground mask
    fg = (mask > 0).any(dim=1).float()

    # Apply the foreground mask to sem_mask
    sem_mask = sem_mask * fg
    # plt.imshow(sem_mask[0].cpu().numpy())
    # plt.savefig("debug_semantic_mask.png")
    # plt.close()

    return sem_mask.long()


class DotDict(dict):
    """A dictionary that supports dot notation."""

    def __getattr__(self, attr):
        if attr in self:
            value = self[attr]
            if isinstance(value, dict) and not isinstance(value, DotDict):
                value = DotDict(value)
                self[attr] = value  # Store back to ensure the same object is used
            return value
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{attr}'"
        )

    def __setattr__(self, attr, value):
        self[attr] = value

    def __delattr__(self, attr):
        del self[attr]

    @staticmethod
    def from_dict(data):
        """Recursively converts a dictionary to DotDict."""
        if not isinstance(data, dict):
            return data
        return DotDict({k: DotDict.from_dict(v) for k, v in data.items()})


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_avg(self):
        return self.avg


def copy_images_to_temp_dir(
    image_paths: List[str], target_size: Tuple[int, int]
) -> TemporaryDirectory:
    """
    Copy a list of image paths to a temporary directory.

    Args:
        image_paths (list): List of paths to the images.
        target_size (Tuple[int, int]): The size of the query image. All context images will be resized to this size.

    Returns:
        TemporaryDirectory: Temporary directory containing the images.
    """
    temp_dir = TemporaryDirectory()

    # Ensure all images have the height and width of the first image
    for i, image_path in enumerate(image_paths):
        img = Image.open(image_path)
        if img.size != target_size:
            img = img.resize(target_size)

        # Rename the images to {i}.png
        img.save(os.path.join(temp_dir.name, f"{i}.png"))

    return temp_dir
