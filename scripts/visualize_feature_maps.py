"""Feature map visualization for CNN models."""

import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).parent.parent / "src"))

from dataloader.data_preprocessing import get_image_transforms, load_image
from dataloader.data_utils import CLASSES
from models.base_model import BaseModel
from models.model_factory import create_model


class FeatureMapVisualizer:
    """Visualize intermediate feature maps of CNN layers."""

    def __init__(self, model: BaseModel, device: str = "cpu"):
        self.base_model = model
        self.model = model.model.to(device).eval()
        self.device = device

    def _register_hook(self, layer: nn.Module):
        activations = {}

        def hook_fn(_, __, output):
            activations["feat"] = output.detach().cpu()

        handle = layer.register_forward_hook(hook_fn)
        return activations, handle

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
        """
        Visualize feature maps from a given layer.

        Args:
            image_path: Path to input image
            layer: Layer to extract feature maps from
            max_channels: Max number of channels to display
            cols: Number of columns in grid
        """

        transform = get_image_transforms(self.base_model.get_input_size(), data_type="test")
        image_tensor = load_image(image_path, transform).unsqueeze(0).to(self.device)

        activations, handle = self._register_hook(layer)

        with torch.no_grad():
            _ = self.model(image_tensor)

        handle.remove()

        feature_maps = activations["feat"].squeeze(0)  # [C, H, W]
        num_channels = min(feature_maps.shape[0], max_channels)
        rows = (num_channels + cols - 1) // cols

        if figsize is None:
            figsize = (cols * 3, rows * 3)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = np.atleast_1d(axes).ravel()

        for i in range(num_channels):
            fmap = feature_maps[i]
            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-6)

            axes[i].imshow(fmap, cmap="viridis")
            axes[i].axis("off")
            axes[i].set_title(f"Ch {i}")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return fig


if __name__ == "__main__":
    model_name = "convnext"

    root = Path(__file__).parent.parent.parent
    image_dir = root / "data" / "IDRiD" / "Train" / "images"
    output_dir = root / "output" / "feature_maps"
    checkpoint_path = root / "output" / "checkpoints" / f"{model_name}_best_model.pt"

    output_dir.mkdir(parents=True, exist_ok=True)

    model = create_model(model_name, num_classes=len(CLASSES), pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)

    image_path = glob.glob(str(image_dir / "*.jpg"))[0]

    visualizer = FeatureMapVisualizer(model, device="cpu")
    for layer_id in range(len(model.model.features[:])):
        print(f"Visualizing layer {type(model.model.features[layer_id])}")
        print(model.model.features[layer_id])
        # layer = model.model.features[layer_id][-1].block[0]

        # print(f"Layer {layer}")

        # visualizer.visualize(
        #     image_path,
        #     layer,
        #     save_path=output_dir / f"{model_name}_feature_map_{layer_id}.png",
        #     show=False,
        # )
