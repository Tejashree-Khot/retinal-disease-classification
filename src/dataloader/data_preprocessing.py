"""Image and text preprocessing helpers used by the dataloader.

This module contains transforms and small helpers for loading images.
The functions provide a consistent preprocessing pipeline so
that training, evaluation and inference use the same steps.

brightness=0.1	Real scanners vary, but 10% is high
contrast=0.1	Alters lesion visibility
saturation=0.0	Exudates & hemorrhages distorted
hue=0.0	Hue shift breaks medical meaning
scale=(0.85, 1.0) 15 % crop
ratio=(0.9, 1.1) 10 % crop
rotation=10 degrees
affine=0.05 translation, 5 degrees shear
"""

import logging
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor
from transformers import AutoProcessor

from dataloader.data_utils import CLASSES, CLASSES_DICT, LABEL_COLUMN_NAME
from utils.logger import configure_logging

configure_logging()

LOGGER = logging.getLogger("data_preprocessing")


def get_image_transforms(size: tuple[int], data_type: str) -> Callable:
    """Return data augmentation and normalization transforms.
    This is efficientnet image transforms but we will use it for other models as well.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if data_type == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                # transforms.RandomApply([CLAHE()], p=0.3),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), shear=5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        return transforms.Compose([transforms.Resize(size), transforms.ToTensor(), transforms.Normalize(mean, std)])


def load_image(image_path: Path, transform: Callable) -> torch.Tensor:
    """Load and preprocess an image for model inference.

    The returned tensor has a leading batch dimension (shape [1, C, H, W]) so
    it can be passed directly to models that expect batches.
    """
    pil_image = Image.open(image_path).convert("RGB")
    tensor_image = transform(pil_image)
    # Ensure the transformed result is a Tensor (some static analyzers may think it's a PIL Image)
    if isinstance(tensor_image, Image.Image):
        tensor_image = transforms.ToTensor()(tensor_image)
    return tensor_image


def load_image_paths_and_labels(dataset_path: Path) -> tuple[list[Path], list[int]]:
    """Read image paths and labels from the dataset."""
    image_paths = []
    labels = []

    data = pd.read_csv(dataset_path / "annotations.csv")
    LOGGER.info(f"Loading {len(data)} image_paths from {dataset_path}...")

    for _, row in data.iterrows():
        label = row[LABEL_COLUMN_NAME]
        image_path = dataset_path / "images" / f"{row['Image name']}"

        if image_path.exists():
            image_paths.append(image_path)
            labels.append(int(CLASSES_DICT[str(label)]))
    LOGGER.info(f"Loaded {len(image_paths)} image_paths from {dataset_path}.")
    return image_paths, labels


def get_image_transforms_from_processor(processor: AutoProcessor) -> Callable:
    """Get image transforms from processor."""
    print(processor.image_processor)
    size = processor.image_processor.size["height"]
    mean = processor.image_processor.image_mean
    std = processor.image_processor.image_std
    return Compose(
        [Resize((size, size), interpolation=InterpolationMode.BILINEAR), ToTensor(), Normalize(mean=mean, std=std)]
    )


def preprocess(examples, transform: Callable, processor: AutoProcessor):
    pixel_values = [transform(image.convert("RGB")) for image in examples["image"]]
    captions = [CLASSES[label] for label in examples["label"]]
    inputs = processor.tokenizer(
        captions, max_length=64, padding="max_length", truncation=True, return_attention_mask=True
    )
    inputs["pixel_values"] = pixel_values
    return inputs


def load_image_paths_and_labels_and_captions(dataset_path: Path) -> tuple[list[Path], list[int], list[str]]:
    """Read image paths and labels from the dataset."""
    image_paths = []
    labels = []
    captions = []

    data = pd.read_csv(dataset_path / "annotations.csv")
    LOGGER.info(f"Loading {len(data)} image_paths from {dataset_path}...")

    for _, row in data.iterrows():
        label = row[LABEL_COLUMN_NAME]
        caption = row["caption"]
        image_path = dataset_path / "images" / f"{row['Image name']}"

        if image_path.exists():
            image_paths.append(image_path)
            labels.append(int(CLASSES_DICT[str(label)]))
            captions.append(caption)
    LOGGER.info(f"Loaded {len(image_paths)} image_paths from {dataset_path}.")
    return image_paths, labels, captions
