"""Embedding visualization utilities for CNN models."""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch import Tensor

from dataloader.data_loader import CustomDataset, get_data_loader
from dataloader.data_utils import CLASSES
from models.base_model import BaseModel
from models.model_factory import create_model


def _get_feature_layer(model: BaseModel) -> nn.Module:
    """Get the feature extraction layer (avgpool) based on model architecture."""
    return model.model.avgpool


class FeatureExtractor:
    """Extract features from a model using forward hooks."""

    def __init__(self, model: BaseModel):
        self.model = model
        self.features: list[Tensor] = []
        self._hook_handle = None

    def _hook_fn(self, module: nn.Module, input: tuple, output: Tensor) -> None:
        self.features.append(output.detach())

    def __enter__(self) -> "FeatureExtractor":
        target_layer = _get_feature_layer(self.model)
        self._hook_handle = target_layer.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._hook_handle:
            self._hook_handle.remove()

    def clear(self) -> None:
        self.features = []


def visualize_embeddings(
    data_dir: Path,
    model: BaseModel | None = None,
    model_name: str = "resnet",
    model_path: str | None = None,
    num_classes: int = 4,
    max_images: int = 100,
    batch_size: int = 32,
    image_size: tuple[int, int] = (224, 224),
    device: str = "cpu",
    perplexity: float = 30.0,
    random_state: int = 42,
    show: bool = True,
    save_path: str | None = None,
    **model_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract embeddings from any BaseModel and visualize with t-SNE or UMAP."""
    if model is None:
        model = create_model(model_name, num_classes=num_classes, **model_kwargs)
        if model_path:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()

    dataset = CustomDataset(data_dir, size=image_size, data_type="test")
    dataloader = get_data_loader(dataset, batch_size=batch_size)

    embeddings = []
    labels = []

    with torch.no_grad(), FeatureExtractor(model) as extractor:
        collected = 0
        for images, lbls in dataloader:
            images = images.to(device)
            _ = model(images)

            feats = extractor.features[-1]
            feats = feats.view(feats.size(0), -1)

            take = min(images.size(0), max_images - collected)
            embeddings.append(feats[:take].cpu())
            labels.append(lbls[:take])
            collected += take
            extractor.clear()

            if collected >= max_images:
                break

    embeddings_tensor = torch.cat(embeddings).numpy()
    labels_array = torch.cat(labels).numpy()

    effective_perplexity = min(perplexity, max(1.0, len(embeddings_tensor) - 1))
    reducer = TSNE(n_components=2, random_state=random_state, perplexity=effective_perplexity)
    embeddings_2d = reducer.fit_transform(embeddings_tensor)

    if show or save_path:
        plt.figure(figsize=(10, 8))
        for label in sorted(set(labels_array)):
            idx = labels_array == label
            label_name = CLASSES[label] if label < len(CLASSES) else f"Class {label}"
            plt.scatter(
                embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=label_name, alpha=0.7, s=50
            )
        plt.legend(loc="best")
        plt.title(f"Embedding Visualization for {model_name}")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        if show:
            plt.show()
        plt.close()

    return embeddings_2d, labels_array


if __name__ == "__main__":
    data_dir = Path("../data/IDRiD/Train")
    output_dir = Path("../output/embeddings")

    for model_name in ["resnet", "vgg", "efficientnet", "convnext"]:
        visualize_embeddings(
            data_dir=data_dir,
            show=False,
            save_path=output_dir / f"embeddings_{model_name}.png",
            model_name=model_name,
        )
