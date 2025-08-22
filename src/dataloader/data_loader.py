"""PyTorch custom data loader."""

from pathlib import Path
from typing import cast

import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from dataloader.data_utils import CLASSES_DICT
from dataloader.data_preprocessing import get_efficient_net_data_transforms


class CustomDataset(Dataset):
    """Pytorch custom dataloader."""

    def __init__(self, dataset_path: Path, image_transform: transforms.Compose):
        self.image_paths, self.labels = load_images(dataset_path)
        self.image_transform = image_transform

    def __len__(self) -> int:
        size = len(self.image_paths)
        return size

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        label = self.labels[index]
        image_path = self.image_paths[index]
        image = self.image_transform(Image.open(image_path).convert("RGB"))
        img_tensor = cast(Tensor, image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return img_tensor, label_tensor


def image_transform(
    img_size: tuple | list, augment: bool, arch: str = "efficientnet-b0"
) -> transforms.Compose:
    """Image transformation for training and validation."""
    if arch == "efficientnet-b0":
        # Use the dedicated EfficientNet transforms
        transforms_dict = get_efficient_net_data_transforms(
            img_size if isinstance(img_size, int) else img_size[0]
        )
        return transforms_dict["train"] if augment else transforms_dict["val"]
    else:
        # Use the previous transforms for other architectures
        mean, std = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

        if augment:
            # Training transforms with augmentation
            return transforms.Compose(
                [
                    transforms.Resize((size[0], size[1])),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
        else:
            # Validation transforms without augmentation
            return transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )


def load_images(dataset_path: Path) -> tuple[list[Path], list[int]]:
    """Read image images and labels from the dataset."""
    image_paths = []
    labels = []

    data = pd.read_csv(dataset_path / "annotations.csv")[:20]
    print(f"Loading {len(data)} image_paths from {dataset_path}...")

    for row in tqdm(data.iterrows()):
        label = row[1]["class"]
        image_path = dataset_path / "images" / f"{row[1]['Image name']}"
        if image_path.exists():
            image_paths.append(image_path)
            labels.append(int(CLASSES_DICT[label]))
        else:
            print(f"Image {image_path} does not exist, skipping.")

    print(f"Loaded {len(image_paths)} image_paths from {dataset_path}.")
    return image_paths, labels


def get_data_loader(
    dataset_path: Path, size: tuple | list, batch_size: int, augment: bool
) -> DataLoader:
    """Get data loader for the dataset."""
    # for faster training, we load data images first,
    # if memory is not an issue, you can uncomment the next line
    # images, labels = load_images(dataset_path)

    data_transform = image_transform(size, augment)

    data = CustomDataset(dataset_path, data_transform)

    data_loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=2,
    )
    return data_loader


if __name__ == "__main__":
    # Example usage
    dataset_path = "../../../data/IDRiD/Train"
    size = (448, 448)
    batch_size = 32
    augment = True

    data_loader = get_data_loader(Path(dataset_path), size, batch_size, augment)

    for images, labels in data_loader:
        print(images.shape, labels)
        break  # Just to show the first batch
    print("Data loader is ready.")
