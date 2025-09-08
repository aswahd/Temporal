from typing import Callable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import transforms


def prepare_clip(model_key: str, device: str = "cuda") -> Tuple[nn.Module, Callable]:
    import clip

    clip_model, preprocess = clip.load(model_key, device=device)
    vision_encoder = clip_model.visual
    vision_encoder.eval()

    @torch.no_grad()
    def _forward(image_paths: List[str]):
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        images = torch.stack([preprocess(Image.open(img_path).convert("RGB")) for img_path in image_paths]).cuda()

        image_features = vision_encoder(images)
        return F.normalize(image_features, p=2, dim=-1)

    return vision_encoder, _forward


def prepare_dinov2(model_key: str = "dinov2_vitb14", device: str = "cuda") -> Tuple[nn.Module, Callable]:
    def build():
        dinov2 = torch.hub.load("facebookresearch/dinov2", model_key).to(device)
        dinov2.eval()
        return dinov2

    vision_encoder = build()
    vision_encoder.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    @torch.no_grad()
    def _forward(image_paths: List[str]):
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        img = torch.stack([transform(Image.open(img_path).convert("RGB")) for img_path in image_paths]).to(device)

        embeddings = vision_encoder(img)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    return vision_encoder, _forward


def prepare_temporal(checkpoint_path: str, arch: str = "resnet50", device: str = "cuda") -> Tuple[nn.Module, Callable]:
    """
    ResNet50 trained with multi-positive time-contrastive learning.
    """

    def build() -> torch.nn.Module:
        resnet = getattr(torchvision.models, arch)()
        model = torch.nn.Sequential(*list(resnet.children())[:-1])
        # Remove ReLU activation before average pooling
        model[-2][2].relu = torch.nn.Identity()
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint["model_state_dict"]
        # Remove prefix
        state_dict = {k[len("backbone.") :]: v for k, v in state_dict.items() if "backbone" in k}
        model.load_state_dict(state_dict)
        return model

    vision_encoder = build()

    vision_encoder.eval()
    vision_encoder.to(device)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    @torch.no_grad()
    def _forward(image_paths: List[str]):
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        img = torch.stack([transform(Image.open(img_path).convert("RGB")) for img_path in image_paths]).to(device)
        embeddings = vision_encoder(img).view(len(image_paths), -1)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    return vision_encoder, _forward
