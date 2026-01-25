"""Embedding visualization utilities for CNN models."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch import Tensor

sys.path.append(str(Path(__file__).parent.parent / "src"))

from dataloader.data_loader import CustomDataset, DataLoader, get_data_loader
from dataloader.data_utils import CLASSES
from models.base_model import BaseModel
from models.checkpoint import CheckpointManager
from models.model_factory import create_model
from utils.helper import get_device


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


def generate_embeddings(
    model: BaseModel, dataloader: DataLoader, max_images: int = 100, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Extract embeddings from any BaseModel and visualize with t-SNE or UMAP."""
    embeddings = []
    labels = []

    with torch.no_grad(), FeatureExtractor(model) as extractor:
        collected = 0
        for images, lbls in dataloader:
            images = images.to(get_device("auto"))
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

    perplexity = 30.0
    effective_perplexity = min(perplexity, max(1.0, len(embeddings_tensor) - 1))
    reducer = TSNE(n_components=2, random_state=random_state, perplexity=effective_perplexity)
    embeddings_2d = reducer.fit_transform(embeddings_tensor)
    return embeddings_2d, labels_array


def plot_embeddings(embeddings_2d, labels_array, title: str, show: bool = False, save_path: str | None = None):
    """Plot embeddings."""
    plt.figure(figsize=(10, 8))
    for label in sorted(set(labels_array)):
        idx = labels_array == label
        label_name = CLASSES[label] if label < len(CLASSES) else f"Class {label}"
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=label_name, alpha=0.7, s=50)
    plt.legend(loc="best")
    plt.title(f"Embedding Visualization for {title}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_all_embeddings(results, show: bool = False, save_path: str | None = None):
    """Plot all embeddings in a grid."""
    num_models = len(results)
    cols = 2
    rows = (num_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 6))
    axes = np.array(axes).reshape(-1)

    for ax, (title, emb, labels) in zip(axes, results):
        for label in sorted(set(labels)):
            idx = labels == label
            label_name = CLASSES[label] if label < len(CLASSES) else f"Class {label}"
            ax.scatter(emb[idx, 0], emb[idx, 1], label=label_name, alpha=0.7, s=40)

        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(fontsize=8)

    # Hide unused subplots
    for ax in axes[len(results) :]:
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()


def make_argparser():
    parser = argparse.ArgumentParser(description="Visualize feature maps and layer evolution of a CNN model.")
    parser.add_argument("--model_name", required=False)
    parser.add_argument("--variant", required=False)
    parser.add_argument("--image_path", type=Path, default=None, help="Path to the image to visualize.")
    return parser.parse_args()


def main(args: argparse.Namespace):
    root_dir = Path(__file__).parent.parent
    output_dir = root_dir / "output" / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = root_dir / "data" / "IDRiD" / "Test"

    if args.model_name:
        models = [(args.model_name, args.variant)]
    else:
        models = [("resnet", "50"), ("vgg", "16"), ("efficientnet", "b7"), ("convnext", "large")]

    results = []

    for model_name, variant in models:
        checkpoint_path = root_dir / "output" / "checkpoints" / f"{model_name}_{variant}_best_model.pt"

        model = create_model(model_name, variant, num_classes=len(CLASSES), pretrained=False)
        model = CheckpointManager.load_for_inference(model, checkpoint_path, device=get_device("auto"))

        dataset = CustomDataset(data_dir, size=model.get_input_size(), data_type="test")
        dataloader = get_data_loader(dataset, batch_size=4)

        embeddings_2d, labels = generate_embeddings(
            dataloader=dataloader, model=model, max_images=1000, random_state=42
        )
        results.append((f"{model_name}_{variant}", embeddings_2d, labels))
        save_path = output_dir / f"embeddings_{model_name}_{variant}.png"
        plot_embeddings(embeddings_2d, labels, f"{model_name}_{variant}", show=True, save_path=save_path)

    plot_all_embeddings(results, show=True, save_path=output_dir / "all_embeddings.png")


if __name__ == "__main__":
    main(make_argparser())
