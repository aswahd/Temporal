import os
from tqdm import tqdm
import torch
from sam2.utils.misc import AsyncVideoFrameLoader, _load_img_as_tensor
import mimetypes
from typing import List, Tuple


def image_exts():
    return [
        ext
        for ext in mimetypes.types_map
        if mimetypes.types_map[ext].startswith("image")
    ]


def load_video_frames_from_images(
    *,
    image_paths: List[str],
    image_size: Tuple[int, int],
    offload_video_to_cpu: bool,
    img_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    img_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    async_loading_frames: bool = False,
    compute_device: torch.device = torch.device("cuda"),
) -> Tuple[torch.Tensor, int, int]:
    """
    Load and preprocess video frames from image paths.

    Args:
        img_paths: List of paths to image files
        image_size: Target size (height, width) for resizing frames
        offload_video_to_cpu: If True, keeps frames in CPU memory
        img_mean: Normalization mean values for RGB channels
        img_std: Normalization standard deviation values for RGB channels
        async_loading_frames: If True, loads frames asynchronously
        compute_device: Target device for tensor operations

    Returns:
        Tuple containing:
        - Normalized frame tensor of shape (num_frames, 3, height, width)
        - Original video height
        - Original video width
    """
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

    if async_loading_frames:
        lazy_images = AsyncVideoFrameLoader(
            image_paths,
            image_size,
            offload_video_to_cpu,
            img_mean,
            img_std,
            compute_device,
        )
        return lazy_images, lazy_images.video_height, lazy_images.video_width

    images = torch.zeros(len(image_paths), 3, image_size, image_size, dtype=torch.float32)
    for n, img_path in enumerate(tqdm(image_paths, desc="Loading frames")):
        images[n], video_height, video_width = _load_img_as_tensor(img_path, image_size)

    if not offload_video_to_cpu:
        images = images.to(compute_device)
        img_mean = img_mean.to(compute_device)
        img_std = img_std.to(compute_device)

    images = (images - img_mean) / img_std
    return images, video_height, video_width
