from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


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
