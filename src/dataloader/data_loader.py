"""PyTorch custom data loader."""

from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoProcessor

from dataloader.data_preprocessing import (
    get_image_transforms,
    get_image_transforms_from_processor,
    load_image,
    load_image_paths_and_labels,
    load_image_paths_and_labels_and_captions,
)
from dataloader.data_utils import CLASSES_DICT


class CustomDataset(Dataset):
    """PyTorch custom dataloader for images + labels."""

    def __init__(self, dataset_path: Path, size: tuple[int], data_type: str):
        self.image_paths, self.labels = load_image_paths_and_labels(dataset_path)
        self.image_transform = get_image_transforms(size, data_type)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        label = torch.tensor(self.labels[index], dtype=torch.long)
        image = load_image(self.image_paths[index], self.image_transform)
        return image, label


class MedSigLIPDataset(Dataset):
    """PyTorch dataset for MedSigLIP model with images, text, and labels."""

    def __init__(self, dataset_path: Path, processor: AutoProcessor):
        self.image_paths, self.labels, self.captions = load_image_paths_and_labels_and_captions(dataset_path)
        self.processor = processor
        self.transform = get_image_transforms_from_processor(processor)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict:
        image = load_image(self.image_paths[index], self.transform)
        inputs = self.processor.tokenizer(
            self.captions[index], max_length=64, padding="max_length", truncation=True, return_attention_mask=True
        )
        return {
            "pixel_values": image,
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": self.labels[index],
        }


def get_sampler(labels: list[int]) -> WeightedRandomSampler:
    """Get weighted random sampler for class balancing."""
    class_sample_count = np.bincount(labels, minlength=len(CLASSES_DICT))
    class_weights = 1.0 / np.clip(class_sample_count, 1, None)
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler


def get_data_loader(dataset: CustomDataset, batch_size: int, use_weighted_sampler: bool = False) -> DataLoader:
    """Get data loader (image + label)."""
    sampler = None
    if use_weighted_sampler:
        sampler = get_sampler(dataset.labels)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=not use_weighted_sampler,
        num_workers=4,
        pin_memory=False,
        prefetch_factor=1,
        persistent_workers=True,
    )

    return data_loader


def collate_fn_text_image(examples: list[dict]) -> dict:
    """Collate function for text and image data."""
    pixel_values = torch.tensor([example["pixel_values"] for example in examples])
    input_ids = torch.tensor([example["input_ids"] for example in examples])
    attention_mask = torch.tensor([example["attention_mask"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask, "return_loss": True}


if __name__ == "__main__":
    # Example usage
    dataset_path = Path("../data/IDRiD/Train")
    size = (512, 512)
    batch_size = 32

    dataset = CustomDataset(dataset_path=dataset_path, size=size, data_type="train")
    print(f"Dataset length: {len(dataset)}")
    data_loader = get_data_loader(dataset, batch_size)

    for images, labels in data_loader:
        print(images.shape, labels)
        break
    print("Data loader is ready.")
