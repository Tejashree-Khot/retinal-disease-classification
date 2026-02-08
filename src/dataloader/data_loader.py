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

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from transformers import AutoProcessor

from dataloader.data_utils import CLASSES_DICT, LABEL_COLUMN_NAME
from utils.logger import configure_logging

configure_logging()

LOGGER = logging.getLogger("data_loader")


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


def get_sampler(labels: list[int]) -> WeightedRandomSampler:
    """Get weighted random sampler for class balancing."""
    class_sample_count = np.bincount(labels, minlength=len(CLASSES_DICT))
    class_weights = 1.0 / np.clip(class_sample_count, 1, None)
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler


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


class CustomDataset(Dataset):
    """PyTorch custom dataloader for images + labels."""

    def __init__(self, dataset_path: Path, size: tuple[int], data_type: str):
        self.image_paths, self.labels, _ = load_image_paths_and_labels_and_captions(dataset_path)
        self.image_transform = get_image_transforms(size, data_type)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        label = torch.tensor(self.labels[index], dtype=torch.long)
        image = self.image_transform(Image.open(self.image_paths[index]).convert("RGB"))
        return image, label


class MedSigLIPDataset(Dataset):
    """PyTorch dataset for MedSigLIP model with images, text, and labels."""

    def __init__(self, processor: AutoProcessor, dataset_path: Path):
        self.image_paths, self.labels, self.captions = load_image_paths_and_labels_and_captions(dataset_path)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict:
        image = Image.open(self.image_paths[index]).convert("RGB")
        inputs = self.processor(
            text=self.captions[index], images=image, padding="max_length", truncation=True, return_tensors="pt"
        )
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "labels": self.labels[index],
        }


def get_data_loader(
    dataset: CustomDataset | MedSigLIPDataset, batch_size: int, use_weighted_sampler: bool = False
) -> DataLoader:
    """Get data loader (image + label)."""
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=get_sampler(dataset.labels) if use_weighted_sampler else None,
        shuffle=not use_weighted_sampler,
        num_workers=4,
        pin_memory=False,
        prefetch_factor=1,
        persistent_workers=True,
    )
    return data_loader


if __name__ == "__main__":
    # Example usage
    dataset_path = Path("../data/IDRiD/Train")
    size = (448, 448)
    batch_size = 8

    dataset = CustomDataset(dataset_path=dataset_path, size=size, data_type="train")
    print(f"Dataset length: {len(dataset)}")
    data_loader = get_data_loader(dataset, batch_size)

    for images, labels in data_loader:
        print(images.shape, labels)
        break

    print("Custom dataset is ready.")

    processor = AutoProcessor.from_pretrained("google/medsiglip-448")
    dataset = MedSigLIPDataset(processor=processor, dataset_path=dataset_path)
    print(f"Dataset length: {len(dataset)}")
    data_loader = get_data_loader(dataset, batch_size)

    for batch in data_loader:
        print(batch["pixel_values"].shape, batch["labels"])
        break

    print("MedSigLIP dataset is ready.")
