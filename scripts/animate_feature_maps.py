"""Layer-wise feature evolution animation for CNN models."""

import glob
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib.animation import FuncAnimation

sys.path.append(str(Path(__file__).parent.parent / "src"))

from dataloader.data_preprocessing import get_image_transforms, load_image
from dataloader.data_utils import CLASSES
from models.base_model import BaseModel
from models.model_factory import create_model


class LayerEvolutionAnimator:
    """Animate feature evolution across CNN layers."""

    def __init__(self, model: BaseModel, device: str = "cpu"):
        self.base_model = model
        self.model = model.model.to(device).eval()
        self.device = device

    def _collect_layers(self) -> List[nn.Module]:
        """
        Select meaningful layers for evolution visualization.
        Adjust if needed per architecture.
        """
        layers = []

        if hasattr(self.model, "features"):  # ConvNeXt / EfficientNet
            layers.extend(self.model.features)
        elif hasattr(self.model, "layer1"):  # ResNet
            layers.extend([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])
        else:
            raise ValueError("Unsupported model architecture")

        return layers

    def animate(self, image_path: Path, save_path: Path | None = None, interval: int = 800):
        """Create layer-wise evolution animation."""

        transform = get_image_transforms(self.base_model.get_input_size(), data_type="test")
        image_tensor = load_image(image_path, transform).unsqueeze(0).to(self.device)

        layers = self._collect_layers()
        activations = []

        hooks = []

        def hook_fn(_, __, output):
            feat = output.detach().cpu()
            feat = feat.mean(dim=1).squeeze(0)  # channel-avg → [H, W]
            feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-6)
            activations.append(feat.numpy())

        # Register hooks
        for layer in layers:
            hooks.append(layer.register_forward_hook(hook_fn))

        with torch.no_grad():
            _ = self.model(image_tensor)

        for h in hooks:
            h.remove()

        # --- Animation ---
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

        plt.show()
        plt.close(fig)


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

    visualizer = LayerEvolutionAnimator(model, device="cpu")

    visualizer.animate(image_path, save_path=output_dir / f"{model_name}_feature_map.gif")
