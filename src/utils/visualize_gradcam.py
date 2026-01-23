"""Feature map visualization using Grad-CAM for CNN models."""

import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from dataloader.data_utils import CLASSES
from models.base_model import BaseModel
from models.model_factory import create_model

CAM_METHODS = {"gradcam": GradCAM, "gradcam++": GradCAMPlusPlus, "scorecam": ScoreCAM}


class GradCAMVisualizer:
    """Grad-CAM visualization utility."""

    def __init__(
        self,
        model: BaseModel,
        method: Literal["gradcam", "gradcam++", "scorecam"] = "gradcam",
        device: str = "cpu",
        input_size: tuple[int, int] = (224, 224),
    ):
        self.device = device

        # unwrap BaseModel safely
        self.model = model
        self.model.to(device).eval()

        target_layer = self.model.get_feature_layer()
        cam_cls = CAM_METHODS[method]

        self.cam = cam_cls(model=self.model, target_layers=[target_layer])

        self.transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _prepare_image(self, image: Image.Image):
        image = image.convert("RGB")
        rgb = np.array(image.resize((224, 224))) / 255.0
        tensor = self.transform(image).unsqueeze(0)
        return rgb.astype(np.float32), tensor.to(self.device)

    def visualize(self, image: Image.Image, class_idx: int | None = None) -> np.ndarray:
        rgb_img, input_tensor = self._prepare_image(image)

        targets = [ClassifierOutputTarget(class_idx)] if class_idx is not None else None

        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)[0]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        return cam_image

    def visualize_batch(
        self,
        image_paths: list[Path | Image.Image],
        class_indices: list[int] | None = None,
        cols: int = 4,
        figsize: tuple[int, int] | None = None,
        show: bool = True,
        save_path: str | None = None,
    ) -> plt.Figure:
        """Generate Grad-CAM visualizations for multiple images in a grid."""

        if len(image_paths) == 0:
            raise ValueError("image_paths is empty. Provide at least one image.")

        n_images = len(image_paths)
        rows = max(1, (n_images + cols - 1) // cols)

        if figsize is None:
            figsize = (cols * 4, rows * 4)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = np.atleast_1d(axes).ravel()

        for i, item in enumerate(image_paths):
            class_idx = class_indices[i] if class_indices else None

            if isinstance(item, (str, Path)):
                image = Image.open(item).convert("RGB")
            else:
                image = item

            vis = self.visualize(image, class_idx)
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
    image_dir = Path(__file__).parent.parent.parent / "data" / "IDRiD" / "Train" / "images"
    output_dir = Path(__file__).parent.parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoints" / f"{model_name}_best_model.pt"
    output_path = output_dir / "feature_maps" / f"{model_name}_gradcam.png"

    model = create_model(model_name, num_classes=len(CLASSES), pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)

    image_paths = list(image_dir.glob("*.jpg"))[:8]

    visualizer = GradCAMVisualizer(model, method="gradcam", device="cpu")
    visualizer.visualize_batch(image_paths=image_paths, save_path=str(output_path), show=False)
