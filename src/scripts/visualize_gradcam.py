"""Feature map visualization using Grad-CAM for CNN models."""

import argparse
import logging
import os
import random
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataloader.data_loader import get_image_transforms
from dataloader.data_utils import CLASSES, CLASSES_DICT
from models.base_model import BaseModel
from models.checkpoint import CheckpointManager
from models.model_factory import create_model
from utils.helper import get_device
from utils.logger import configure_logging

configure_logging()
LOGGER = logging.getLogger("gradcam")


def load_image(image_path: Path, transform) -> torch.Tensor:
    """Load a single image, apply transform, and return as tensor."""
    return transform(Image.open(image_path).convert("RGB"))


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


def plot_classwise_gradcam(
    df: pd.DataFrame,
    image_dir: Path,
    visualizer: GradCAMVisualizer,
    transform,
    pred_col="predictions",
    save_path: Path | None = None,
    show: bool = True,
):
    """Plot classwise Grad-CAM for a given dataframe of predictions."""
    image_paths = sorted(image_dir.glob("*.jpg"))
    n = len(CLASSES)

    fig, ax = plt.subplots(2, n, figsize=(4 * n, 8))
    random.seed(42)

    splits = [
        df[df.labels == df[pred_col]],  # correct
        df[df.labels != df[pred_col]],  # wrong
    ]

    for i, cls in enumerate(CLASSES):
        cls_id = CLASSES_DICT[cls]

        for r, sub_df in enumerate(splits):
            row = sub_df[sub_df.labels == cls_id].sample(1).iloc[0]
            img_path = image_paths[row.name]

            img_tensor = load_image(img_path, transform)
            cam = visualizer.visualize(img_tensor, class_idx=row[pred_col])

            ax[r, i].imshow(cam)
            ax[r, i].set_title(f"GT: {CLASSES[row.labels]} | Pred: {CLASSES[row[pred_col]]}", fontsize=9)
            ax[r, i].axis("off")

    ax[0, 0].set_ylabel("Correct")
    ax[1, 0].set_ylabel("Wrong")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=80)

    if show:
        plt.show()


def make_argparser():
    parser = argparse.ArgumentParser(description="Visualize feature maps and layer evolution of a CNN model.")
    parser.add_argument(
        "--model_name", type=str, default="convnext", choices=["resnet", "vgg", "efficientnet", "convnext"]
    )
    parser.add_argument("--variant", type=str, default="large")
    parser.add_argument("--image_path", type=Path, nargs="+", default=None, help="Path(s) to image(s) to visualize.")
    parser.add_argument("--prediction_csv", type=Path, default=None, help="Path to the prediction csv file.")
    return parser.parse_args()


def main(args: argparse.Namespace):
    """Run Grad-CAM visualization for a single image or classwise predictions."""
    random.seed(42)
    root = Path(__file__).parent.parent.parent
    output_dir = root / "output" / "gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model_name
    variant = args.variant

    checkpoint_path = root / "output" / "checkpoints" / f"{model_name}_{variant}_best_model.pt"
    model = create_model(model_name, variant, num_classes=len(CLASSES), pretrained=False)
    model = CheckpointManager.load_for_inference(model, checkpoint_path, device=get_device("auto"))
    LOGGER.info(f"Loaded model for visualization from {checkpoint_path}")

    transform = get_image_transforms(model.get_input_size(), data_type="test")
    visualizer = GradCAMVisualizer(model, device="cpu")

    if args.image_path:
        if len(args.image_path) == 1:
            image_path = args.image_path[0]
            LOGGER.info(f"Visualizing Grad-CAM for {image_path.name}")
            image_tensor = load_image(image_path, transform)
            cam_image = visualizer.visualize(image_tensor)
            save_file = output_dir / f"{model_name}_{variant}_gradcam.png"
            plt.imsave(str(save_file), cam_image)
            LOGGER.info(f"Saved Grad-CAM to {save_file}")
        else:
            LOGGER.info(f"Visualizing Grad-CAM for {len(args.image_path)} images")
            save_file = output_dir / f"{model_name}_{variant}_batch_gradcam.png"
            visualizer.visualize_batch(args.image_path, save_path=str(save_file), show=False)
            LOGGER.info(f"Saved batch Grad-CAM to {save_file}")
    elif args.prediction_csv:
        LOGGER.info(f"Plotting classwise Grad-CAM for {args.prediction_csv.name}")
        image_dir = root / "data" / "IDRiD" / "Train" / "images"
        prediction_df = pd.read_csv(args.prediction_csv)
        plot_classwise_gradcam(
            prediction_df,
            image_dir,
            visualizer,
            transform,
            save_path=output_dir / f"{model_name}_{variant}_classwise_gradcam.png",
            show=False,
        )


if __name__ == "__main__":
    main(make_argparser())
