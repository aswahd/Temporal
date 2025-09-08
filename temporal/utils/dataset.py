import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import io


class VideoDataset(torch.utils.data.Dataset):
    """
    Video dataset that reads sequences from a test directory.
    Expects a directory structure with:
      - JPEGImages/ : folder containing subdirectories for each video with image frames.
      - Annotations/ : folder containing corresponding segmentation masks.
    Implements an updated interface similar to ImageFolder.
    """

    def __init__(
        self,
        video_dir: str | Path,
        category_ids: Tuple[int, ...],
        transform: Callable | None = None,
    ):
        video_dir = Path(video_dir)
        if not video_dir.exists():
            raise ValueError(f"Video directory {video_dir} does not exist")

        self.video_dir = video_dir
        self.images_root = video_dir / "JPEGImages"
        self.annot_root = video_dir / "Annotations"

        if not self.images_root.exists() or not self.annot_root.exists():
            raise ValueError(f"JPEGImages or Annotations directory not found in {video_dir}")

        # Each subdirectory in JPEGImages represents one video
        self.video_dirs = sorted([d for d in self.images_root.iterdir() if d.is_dir()])
        if not self.video_dirs:
            raise ValueError(f"No video directories found in {self.images_root}")

        self.transform = transform
        self._category_ids = category_ids

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx: int) -> Dict:
        video_dir = self.video_dirs[idx]
        # Assume corresponding annotations are in a parallel structure
        annot_dir = Path(str(video_dir).replace("JPEGImages", "Annotations"))

        # Get list of frame file paths (jpg preferred, fallback to png)
        frame_paths = sorted(video_dir.glob("*.jpg"))
        if not frame_paths:
            frame_paths = sorted(video_dir.glob("*.png"))
        if not frame_paths:
            raise RuntimeError(f"No image files found in {video_dir}")
        frame_paths = [str(p) for p in frame_paths]

        # Load frames (as tensors)
        try:
            frames = []
            for f in frame_paths:
                # Use torchvision.io to read image as tensor (C x H x W)
                img = io.read_image(f)
                if img.size(0) == 1:  # If grayscale, convert to RGB by repeating channels
                    img = img.repeat(3, 1, 1)
                # Optionally, apply a transform (if provided, convert to PIL, then back to tensor)
                if self.transform:
                    pil_img = Image.fromarray(img.permute(1, 2, 0).cpu().numpy())
                    img = self.transform(pil_img)
                frames.append(img)
            # Ensure all frames have the same spatial dimensions
            target_size = frames[0].shape[-2:]
            frames = [
                F.interpolate(
                    f.unsqueeze(0),
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )[0]
                if f.shape[-2:] != target_size
                else f
                for f in frames
            ]
        except Exception as e:
            raise RuntimeError(f"Error loading frames from {video_dir}: {e}")

        # Load masks from the annotations directory
        mask_paths = sorted(annot_dir.glob("*.png"))
        if not mask_paths:
            raise RuntimeError(f"No mask files found in {annot_dir}")
        mask_paths = [str(p) for p in mask_paths]
        if len(mask_paths) != len(frame_paths):
            raise ValueError(
                f"Number of frames ({len(frame_paths)}) does not match number of masks ({len(mask_paths)}) for video {video_dir}"
            )

        masks = {}
        for frame_idx, mask_path in enumerate(mask_paths):
            mask = self.read_mask(mask_path)
            if mask.shape != frames[frame_idx].shape[1:]:
                raise ValueError(
                    f"Mask shape {mask.shape} does not match frame shape {frames[frame_idx].shape[1:]} for {mask_path}"
                )
            masks[frame_idx] = self.get_mask_array(mask_path)

        return {
            "frames": frames,
            "masks": masks,
            "video_dir": str(video_dir),
        }

    def read_mask(self, path: str) -> np.ndarray:
        """
        Read a mask image and convert it to a numpy array.
        Assumes the mask is in grayscale format.
        """
        mask = Image.open(path).convert("L")
        return np.array(mask)

    def get_mask_array(self, path) -> Dict[int, torch.tensor]:
        mask_arr = np.array(self.read_mask(path))
        mask_res = {}
        for cat_id in self._category_ids:
            mask_res[cat_id] = torch.from_numpy((mask_arr == cat_id)).type(torch.uint8)
        return mask_res

    def get_category_ids(self) -> Tuple[int, ...]:
        return self._category_ids


class ImageDataset(torch.utils.data.Dataset):
    pass


class ImageListDataset(ImageDataset):
    """Dataset for loading and transforming images"""

    def __init__(self, image_paths: List[str], transform: Callable):
        self.transform = transform
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image_path = self.image_paths[idx]
        return self.transform(Image.open(image_path).convert("RGB"))


class SingleMaskDataset(ImageDataset):
    """
    Image dataset for semantic segmentation with a palette of colors.
    root/
         ├── imgs/
            │   ├── img1.png
            │   └── ...
            └── gts/
            │   ├── img1.png
    """

    def __init__(
        self,
        root_dir: str,
        category_map: Optional[Dict[str, int]] = None,
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        masks_dir = self.root_dir / "gts"
        self.masks = list(masks_dir.iterdir())
        self.images = []

        for mask in self.masks:
            img_path = self.root_dir / "imgs" / mask.name
            if img_path.exists():
                self.images.append(img_path)

        if category_map is None:
            unique_ids = set()
            for mask_path in self.masks:
                unique_ids.update(self._get_category_ids(mask_path))
            unique_ids = sorted(unique_ids)

            self.category_map = {str(i): i for i in unique_ids}
        else:
            self.category_map = category_map

        self._category_ids = self.get_category_ids()
        self._category_names = self.get_category_names()
        self._category_id_to_name = self.get_category_id_to_name()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = self.read_image(img_path)
        w, h = img.size
        masks = self.get_mask_array(idx)

        if self.transform is not None:
            img = self.transform(img)

        return {
            "image": img,
            "img_size": (h, w),
            "masks": masks,
            "img_path": img_path,
        }

    def _get_category_ids(self, mask_path: Path) -> List[int]:
        mask = np.array(Image.open(mask_path).convert("L"))
        categories = set(np.unique(mask).tolist())
        return list(categories - {0})

    @staticmethod
    def read_image(path):
        return Image.open(path).convert("RGB")

    @staticmethod
    def read_mask(path):
        return Image.open(path).convert("L")

    @property
    def num_classes(self) -> int:
        return len(self.category_map)

    def get_category_names(self) -> List[str]:
        return [k for k in self.category_map]

    def get_category_ids(self) -> List[int]:
        return [i for _, i in self.category_map.items()]

    def get_category_id_to_name(self) -> Dict[int, str]:
        return {v: k for k, v in self.category_map.items()}

    def get_category_stats(self) -> Dict[int, int]:
        """Return a count of images for each category in the training set"""
        category_counts = defaultdict(int)
        for mask_path in self.masks:
            mask = np.array(Image.open(mask_path).convert("L"))
            for cat_id in np.unique(mask).tolist():
                if cat_id in self._category_ids:
                    category_counts[self._category_id_to_name[cat_id]] += 1

        return category_counts

    def get_image_paths(self, idx=None):
        if idx is None:
            return self.images

        return self.images[idx]

    def get_mask_array(self, idx: int) -> Dict[int, torch.tensor]:
        mask_arr = np.array(self.read_mask(self.masks[idx]))
        mask_res = {}
        for cat_id in self._category_ids:
            mask_res[cat_id] = torch.from_numpy((mask_arr == cat_id)).type(torch.uint8)
        return mask_res


class MultiMaskDataset(torch.utils.data.Dataset):
    """
    root/
       ├── imgs/
       │   ├── img1.png
       │   └── ...
       └── gts/
        │  ├── img1
        │       ├── class1
        │          └── mask.png
        │       └── class2
        │          └── mask.png
        |           ...
    """

    def __init__(
        self,
        root_dir: str,
        category_names: Optional[List[str]] = None,
        category_map: Optional[Dict[str, int]] = None,
        transform=None,
        ignore_missing_masks: bool = True,
        **kwargs,
    ):
        assert category_names is not None or category_map is not None, ""
        assert (category_names and not category_map) or (not category_names and category_map), (
            "You can only pass category_names or category_map."
        )
        assert os.path.exists(root_dir), f"Root directory does not exist: {root_dir}"

        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images_dir = self.root_dir / "imgs"
        self.gts_dir = self.root_dir / "gts"
        if category_names is not None:
            self.category_map = {c: i for i, c in enumerate(category_names)}

        if category_map is not None:
            self.category_map = category_map

        self.ignore_missing_masks = ignore_missing_masks

        # TODO: remove limit after testing
        self.images = [f for f in self.images_dir.iterdir() if f.is_file()][:100]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        masks = self.get_mask_array(idx, target_size=(h, w))

        if self.transform:
            img = self.transform(img)

        return {
            "image": img,
            "img_size": (h, w),
            "masks": masks,
            "img_path": img_path,
        }

    @property
    def num_classes(self):
        return len(self.category_map)

    def get_category_names(self):
        return [k for k in self.category_map]

    def get_category_ids(self) -> Tuple[int, ...]:
        return [i for _, i in self.category_map.items()]

    def get_category_id_to_name(self):
        return {v: k for k, v in self.category_map.items()}

    def get_category_stats(self) -> Dict:
        """Return a count of images for each category in the training set"""
        category_counts = defaultdict(int)
        for image_path in self.images:
            for cat_name, _ in self.category_map.items():
                mask_path = ((self.gts_dir / image_path.stem) / cat_name) / "mask.png"

                if not mask_path.exists() and not self.ignore_missing_masks:
                    raise FileNotFoundError(f"Mask {mask_path} doesn't exists.")

                mask_arry = np.array(Image.open(mask_path).convert("L"))

                if (mask_arry > 0).any():
                    category_counts[cat_name] += 1

        return category_counts

    def get_image_paths(self, idx=None):
        if idx is None:
            return self.images

        return self.images[idx]

    def get_mask_array(self, idx: int, target_size: Tuple[int, int] | None = None) -> Dict[int, torch.tensor]:
        if target_size is not None:
            h, w = target_size

        image_path = self.images[idx]
        masks_res = {}

        for cat_name, cat_id in self.category_map.items():
            mask_path = ((self.gts_dir / image_path.stem) / cat_name) / "mask.png"

            if mask_path.exists():
                mask_arry = np.array(Image.open(mask_path).convert("L")) > 0
                masks_res[cat_id] = torch.from_numpy(mask_arry).byte()
            else:
                if not self.ignore_missing_masks:
                    raise FileNotFoundError(f"Mask {mask_path} doesn't exists.")
                else:
                    if target_size is None:
                        raise ValueError(
                            "Target size must be provided if mask is missing and ignore_missing_masks is True."
                        )
                    masks_res[cat_id] = torch.zeros(h, w, dtype=torch.uint8)

        return masks_res


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    masks = torch.stack([b["masks"] for b in batch])
    image_paths = [b["img_path"] for b in batch]
    image_size = [b["img_size"] for b in batch]

    return {
        "images": images,
        "img_size": image_size,
        "masks": masks,
        "img_path": image_paths,
    }
