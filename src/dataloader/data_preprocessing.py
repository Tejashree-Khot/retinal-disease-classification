from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torchvision import transforms


def get_efficient_net_data_transforms(img_size: int = 224) -> dict[str, Callable]:
    """Return data augmentation and normalization transforms."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
    }


def get_transforms(image_size: int = 224) -> transforms.Compose:
    """Get a torchvision transform pipeline for preprocessing images.

    Args:
        image_size (int): Size to which the image will be resized.

    Returns:
        transforms.Compose: Composed transform for preprocessing.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


def load_image(image_path: Path, image_size: int = 224) -> torch.Tensor:
    """Load and preprocess an image for model inference.

    Args:
        image_path (str): Path to the image file.
        image_size (int): Size to which the image will be resized.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    image = Image.open(image_path).convert("RGB")
    transform = get_transforms(image_size)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image
