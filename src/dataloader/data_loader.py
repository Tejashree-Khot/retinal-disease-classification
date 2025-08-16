"""PyTorch custom data loader."""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from dataloader.data_utils import CLASSES_DICT


class CustomDataset(Dataset):
    """Pytorch custom dataloader."""

    def __init__(
        self,
        images: list | np.ndarray,
        labels: list | np.ndarray,
        image_transform: transforms.Compose,
    ):
        self.images = images  # Keep as list to handle different image sizes
        self.labels = np.array(labels)
        self.image_transform = image_transform

    def __len__(self) -> int:
        size = len(self.images)
        return size

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        label = self.labels[index]

        # Convert numpy array to PIL Image for transforms
        img_array = self.images[index]
        img_pil = Image.fromarray(img_array)
        transformed = self.image_transform(img_pil)
        if isinstance(transformed, Image.Image):
            transformed = transforms.ToTensor()(transformed)
        img_tensor = cast(Tensor, transformed)

        label = torch.tensor(label, dtype=torch.long)

        return img_tensor, label


def image_transform(
    size: tuple | list, augment: bool, arch: str = "efficientnet-b0"
) -> transforms.Compose:
    """Image transformation for training and validation."""
    if arch == "efficientnet-b0":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean, std = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

    data_transform = transforms.Compose(
        [
            transforms.Resize((size[0], size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return data_transform


def load_images(dataset_path: Path) -> tuple[list, list]:
    """Load images and labels from the dataset."""
    images = []
    labels = []
    image_paths = []

    data = pd.read_csv(dataset_path / "annotations.csv")[:30]

    dataset_path = Path(dataset_path)
    for row in tqdm(data.iterrows()):
        label = row[1]["class"]
        image_path = dataset_path / "images" / f"{row[1]['Image name']}.jpg"
        if image_path.exists():
            image_paths.append(image_path)
            image = Image.open(image_path).convert("RGB")
            image_array = np.array(image)
            images.append(image_array)
            labels.append(CLASSES_DICT[label])

    print(f"Loaded {len(images)} images from {dataset_path}.")
    return images, labels


def get_data_loader(
    dataset_path: Path, size: tuple | list, batch_size: int, augment: bool
) -> DataLoader:
    """Get data loader for the dataset."""
    # for faster training, we load data images first
    images, labels = load_images(dataset_path)

    data_transform = image_transform(size, augment)

    data = CustomDataset(images, labels, data_transform)

    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=5)
    return data_loader


def prepare_data_loaders(
    data_dir: Path, size: tuple | list, batch_size: int, augment: bool = True
) -> tuple[DataLoader, DataLoader]:
    """Prepare data loaders for training and validation."""
    train_dataset_path = data_dir / "train"
    val_dataset_path = data_dir / "val"
    train_loader = get_data_loader(train_dataset_path, size, batch_size, augment)
    val_loader = get_data_loader(val_dataset_path, size, batch_size, augment)
    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    dataset_path = "../../../data/IDRiD/train"
    size = (224, 224)
    batch_size = 8
    augment = True

    data_loader = get_data_loader(Path(dataset_path), size, batch_size, augment)

    for images, labels in data_loader:
        print(images.shape, labels)
        break  # Just to show the first batch
    print("Data loader is ready.")
    for images, labels in data_loader:
        print(images.shape, labels)
        break  # Just to show the first batch
    print("Data loader is ready.")
