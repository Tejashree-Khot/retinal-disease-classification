"""Feature map visualization using Grad-CAM for CNN models."""

import os
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.append(str(Path(__file__).parent.parent / "src"))

from dataloader.data_preprocessing import get_image_transforms, load_image
from dataloader.data_utils import CLASSES
from models.base_model import BaseModel
from models.model_factory import create_model


class GradCAMVisualizer:
    """Grad-CAM visualization utility."""

    def __init__(
        self, model: BaseModel, method: Literal["gradcam", "gradcam++", "scorecam"] = "gradcam", device: str = "cpu"
    ):
        self.device = device

        # BaseModel wraps the actual torch.nn.Module
        self.base_model = model
        self.model = model.model.to(device).eval()

        target_layer = self.base_model.get_feature_layer()
        self.cam = GradCAM(model=self.model, target_layers=[target_layer])

    def visualize(self, image_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """Generate Grad-CAM visualization for a single image tensor."""
        input_tensor = image_tensor.unsqueeze(0).to(self.device)

        targets = [ClassifierOutputTarget(class_idx)] if class_idx is not None else None

        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)[0]

        # Convert tensor → numpy image for overlay (expects [0,1])
        img = image_tensor.permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        cam_image = show_cam_on_image(img.astype(np.float32), grayscale_cam, use_rgb=True)
        return cam_image

    def visualize_batch(
        self,
        image_paths: list[Path],
        class_indices: list[int] | None = None,
        cols: int = 4,
        figsize: tuple[int, int] | None = None,
        show: bool = True,
        save_path: str | None = None,
    ) -> plt.Figure:
        """Generate Grad-CAM visualizations for multiple images in a grid."""

        if not image_paths:
            raise ValueError("image_paths is empty. Provide at least one image.")

        n_images = len(image_paths)
        rows = max(1, (n_images + cols - 1) // cols)

        if figsize is None:
            figsize = (cols * 4, rows * 4)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = np.atleast_1d(axes).ravel()

        transform = get_image_transforms(self.base_model.get_input_size(), data_type="test")

        for i, img_path in enumerate(image_paths):
            class_idx = class_indices[i] if class_indices else None

            image_tensor = load_image(img_path, transform)
            vis = self.visualize(image_tensor, class_idx)

            axes[i].imshow(vis)
            axes[i].axis("off")

            if class_idx is not None:
                axes[i].set_title(f"Class {class_idx}")

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=150)

        if show:
            plt.show()

        plt.close(fig)
        return fig


if __name__ == "__main__":
    model_name = "convnext"

    root = Path(__file__).parent.parent.parent
    image_dir = root / "data" / "IDRiD" / "Train" / "images"
    output_dir = root / "output" / "gradcam"
    checkpoint_path = root / "output" / "checkpoints" / f"{model_name}_best_model.pt"

    output_dir.mkdir(parents=True, exist_ok=True)

    model = create_model(model_name, num_classes=len(CLASSES), pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)

    image_paths = list(image_dir.glob("*.jpg"))[:8]

    visualizer = GradCAMVisualizer(model, device="cpu")
    visualizer.visualize_batch(
        image_paths=image_paths, save_path=str(output_dir / f"{model_name}_gradcam.png"), show=False
    )
