"""Feature map visualization and layer evolution animation for CNN models."""

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.animation import FuncAnimation

sys.path.append(str(Path(__file__).parent.parent / "src"))

from dataloader.data_preprocessing import get_image_transforms, load_image
from dataloader.data_utils import CLASSES
from models.base_model import BaseModel
from models.checkpoint import CheckpointManager
from models.model_factory import create_model
from utils.helper import get_device
from utils.logger import configure_logging

configure_logging()
LOGGER = logging.getLogger("feature_maps")


class FeatureMapVisualizer:
    """Visualize intermediate feature maps of CNN layers."""

    def __init__(self, model: BaseModel, device: str = "cpu"):
        self.base_model = model
        self.model = model.model.to(device).eval()
        self.device = device
        self.conv_layers = model.get_selected_conv_layers_in_order()

    def _register_hook(self, layer: nn.Module):
        activations = {}

        def hook_fn(_, __, output):
            activations["feat"] = output.detach().cpu()

        return activations, layer.register_forward_hook(hook_fn)

    def visualize(
        self,
        image_path: Path,
        layer: nn.Module,
        max_channels: int = 16,
        cols: int = 8,
        figsize: tuple[int, int] | None = None,
        save_path: Path | None = None,
        show: bool = True,
    ) -> plt.Figure:
        """Visualize feature maps from a given layer."""
        transform = get_image_transforms(self.base_model.get_input_size(), data_type="test")
        image_tensor = load_image(image_path, transform).unsqueeze(0).to(self.device)

        activations, handle = self._register_hook(layer)
        with torch.no_grad():
            _ = self.model(image_tensor)
        handle.remove()

        feature_maps = activations["feat"].squeeze(0)
        num_channels = min(feature_maps.shape[0], max_channels)
        rows = (num_channels + cols - 1) // cols
        figsize = figsize or (cols * 3, rows * 3)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = np.atleast_1d(axes).ravel()

        for i in range(num_channels):
            fmap = feature_maps[i]
            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-6)
            axes[i].imshow(fmap, cmap="viridis")
            axes[i].set_title(f"Ch {i}")
            axes[i].axis("off")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=80)
        if show:
            plt.show()
        plt.close(fig)
        return fig


class LayerEvolutionAnimator:
    """Animate feature evolution across CNN layers."""

    def __init__(self, model: BaseModel, device: str = "cpu"):
        self.base_model = model
        self.model = model.model.to(device).eval()
        self.device = device
        self.conv_layers = model.get_selected_conv_layers_in_order()

    def animate(self, image_path: Path, save_path: Path | None = None, interval: int = 800, show: bool = True):
        """Create layer-wise evolution animation."""
        transform = get_image_transforms(self.base_model.get_input_size(), data_type="test")
        image_tensor = load_image(image_path, transform).unsqueeze(0).to(self.device)

        activations = []
        hooks = []

        def hook_fn(_, __, output):
            feat = output.detach().cpu().mean(dim=1).squeeze(0)
            feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-6)
            activations.append(feat.numpy())

        for layer in self.conv_layers:
            hooks.append(layer.register_forward_hook(hook_fn))

        with torch.no_grad():
            _ = self.model(image_tensor)

        for h in hooks:
            h.remove()

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(activations[0], cmap="inferno")
        ax.axis("off")

        def update(frame):
            im.set_array(activations[frame])
            ax.set_title(f"Layer {frame + 1}/{len(activations)}")
            return [im]

        anim = FuncAnimation(fig, update, frames=len(activations), interval=interval, blit=True)

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            anim.save(save_path, writer="pillow")

        if show:
            plt.show()
        plt.close(fig)


def make_argparser():
    parser = argparse.ArgumentParser(description="Visualize feature maps and layer evolution of a CNN model.")
    parser.add_argument(
        "--model_name", type=str, default="resnet", choices=["resnet", "vgg", "efficientnet", "convnext"]
    )
    parser.add_argument("--variant", type=str, default="18")
    parser.add_argument("--image_path", type=Path, default=None, help="Path to the image to visualize.")
    return parser.parse_args()


def main(args: argparse.Namespace):
    root = Path(__file__).parent.parent
    output_dir = root / "output" / "feature_maps"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model_name
    variant = args.variant
    checkpoint_path = root / "output" / "checkpoints" / f"{model_name}_{variant}_best_model.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    if args.image_path:
        image_path = args.image_path
    else:
        random.seed(42)  # for reproducibility for all the models with same input image
        image_dir = root / "data" / "IDRiD" / "Train" / "images"
        image_path = Path(random.choice(list(image_dir.rglob("*.jpg"))))

    LOGGER.info(f"Visualizing feature maps for {image_path.name}")

    model = create_model(model_name, variant, num_classes=len(CLASSES), pretrained=True)
    model = CheckpointManager.load_for_inference(model, checkpoint_path, device=get_device("auto"))
    LOGGER.info(f"Loaded model for visualization from {checkpoint_path}")

    shutil.copy(image_path, output_dir / f"{model_name}_{variant}_input_image.jpg")
    visualizer = FeatureMapVisualizer(model, device="cpu")
    for idx, layer in enumerate(visualizer.conv_layers):
        visualizer.visualize(
            image_path, layer, save_path=output_dir / f"{model_name}_{variant}_feature_map_{idx}.png", show=False
        )

    animator = LayerEvolutionAnimator(model, device="cpu")
    LOGGER.info(f"Animating layer evolution for {model_name}_{variant}")
    animator.animate(image_path, save_path=output_dir / f"{model_name}_{variant}_layer_evolution.gif", show=False)


if __name__ == "__main__":
    main(make_argparser())
