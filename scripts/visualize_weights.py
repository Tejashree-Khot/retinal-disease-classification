"""Weight distribution visualization for CNN models."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

sys.path.append(str(Path(__file__).parent.parent / "src"))
from dataloader.data_utils import CLASSES
from models.base_model import BaseModel
from models.model_factory import create_model


class WeightsVisualizer:
    """Visualize weight distributions across model layers."""

    def __init__(self, model: BaseModel):
        self.model = model

    def get_weight_statistics(self, layer_types: tuple[type, ...] = (nn.Conv2d, nn.Linear)) -> pd.DataFrame:
        """Get summary statistics for all layers of specified types."""
        stats = []
        for name, module in self.model.named_modules():
            if isinstance(module, layer_types) and hasattr(module, "weight"):
                weight = module.weight.data.cpu().numpy().flatten()
                stats.append(
                    {
                        "layer": name,
                        "type": module.__class__.__name__,
                        "shape": str(list(module.weight.shape)),
                        "params": module.weight.numel(),
                        "mean": float(np.mean(weight)),
                        "std": float(np.std(weight)),
                        "min": float(np.min(weight)),
                        "max": float(np.max(weight)),
                        "sparsity": float(np.sum(np.abs(weight) < 1e-6) / len(weight)),
                    }
                )
        return pd.DataFrame(stats)

    def plot_weight_distributions(
        self,
        layer_types: tuple[type, ...] = (nn.Conv2d, nn.Linear),
        max_layers: int = 12,
        bins: int = 50,
        figsize: tuple[int, int] | None = None,
        show: bool = True,
        save_path: str | None = None,
    ) -> plt.Figure:
        """Plot weight histograms for each layer."""
        layers_data = []
        for name, module in self.model.named_modules():
            if isinstance(module, layer_types) and hasattr(module, "weight"):
                weight = module.weight.data.cpu().numpy().flatten()
                layers_data.append((name, module.__class__.__name__, weight))

        if len(layers_data) > max_layers:
            step = len(layers_data) // max_layers
            layers_data = layers_data[::step][:max_layers]

        n_layers = len(layers_data)
        cols = min(4, n_layers)
        rows = (n_layers + cols - 1) // cols

        if figsize is None:
            figsize = (cols * 4, rows * 3)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = np.array(axes).flatten() if n_layers > 1 else [axes]

        for i, (name, layer_type, weights) in enumerate(layers_data):
            ax = axes[i]
            ax.hist(weights, bins=bins, alpha=0.7, edgecolor="black", linewidth=0.5)
            short_name = name.split(".")[-1] if "." in name else name
            ax.set_title(f"{short_name}\n({layer_type})", fontsize=9)
            ax.set_xlabel("Weight Value", fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            ax.tick_params(labelsize=7)

            mean_val = np.mean(weights)
            ax.axvline(mean_val, color="red", linestyle="--", linewidth=1, label=f"μ={mean_val:.3f}")
            ax.legend(fontsize=7)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.suptitle("Weight Distributions per Layer", fontsize=12, y=1.02)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        if show:
            plt.show()
        plt.close()

        return fig

    def plot_weight_summary(
        self,
        layer_types: tuple[type, ...] = (nn.Conv2d, nn.Linear),
        figsize: tuple[int, int] = (12, 5),
        show: bool = True,
        save_path: str | None = None,
    ) -> plt.Figure:
        """Plot summary statistics (mean, std) for all layers."""
        stats_df = self.get_weight_statistics(layer_types)
        if stats_df.empty:
            raise ValueError("No layers found matching the specified types")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        x = range(len(stats_df))
        labels = [name.split(".")[-1][:15] for name in stats_df["layer"]]

        ax1.bar(x, stats_df["mean"], yerr=stats_df["std"], capsize=3, alpha=0.7, color="steelblue")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Weight Value")
        ax1.set_title("Mean ± Std of Weights per Layer")
        ax1.axhline(0, color="gray", linestyle="--", linewidth=0.5)

        colors = ["steelblue" if t == "Conv2d" else "coral" for t in stats_df["type"]]
        ax2.bar(x, stats_df["params"], alpha=0.7, color=colors)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Parameter Count")
        ax2.set_title("Parameters per Layer")
        ax2.set_yscale("log")

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        if show:
            plt.show()
        plt.close()

        return fig

    def compare_checkpoints(
        self,
        checkpoint_paths: list[str],
        layer_name: str,
        device: str = "cpu",
        bins: int = 50,
        figsize: tuple[int, int] = (12, 4),
        show: bool = True,
        save_path: str | None = None,
    ) -> plt.Figure:
        """Compare weight distributions of a specific layer across checkpoints."""
        fig, ax = plt.subplots(figsize=figsize)

        for i, path in enumerate(checkpoint_paths):
            state_dict = torch.load(path, map_location=device, weights_only=True)

            matching_keys = [k for k in state_dict.keys() if layer_name in k and "weight" in k]
            if not matching_keys:
                print(f"Warning: Layer '{layer_name}' not found in {path}")
                continue

            weights = state_dict[matching_keys[0]].cpu().numpy().flatten()
            label = os.path.basename(path).replace(".pth", "").replace(".pt", "")
            ax.hist(weights, bins=bins, alpha=0.5, label=label, edgecolor="black", linewidth=0.3)

        ax.set_xlabel("Weight Value")
        ax.set_ylabel("Count")
        ax.set_title(f"Weight Distribution Comparison: {layer_name}")
        ax.legend()

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        if show:
            plt.show()
        plt.close()

        return fig


def visualize_weights(
    model: BaseModel,
    layer_types: tuple[type, ...] = (nn.Conv2d, nn.Linear),
    max_layers: int = 12,
    show: bool = True,
    save_path: str | None = None,
) -> plt.Figure:
    """Convenience function to visualize weight distributions."""
    visualizer = WeightsVisualizer(model)
    return visualizer.plot_weight_distributions(
        layer_types=layer_types, max_layers=max_layers, show=show, save_path=save_path
    )


def get_weight_statistics(model: BaseModel, layer_types: tuple[type, ...] = (nn.Conv2d, nn.Linear)) -> pd.DataFrame:
    """Convenience function to get weight statistics."""
    visualizer = WeightsVisualizer(model)
    return visualizer.get_weight_statistics(layer_types=layer_types)


if __name__ == "__main__":
    model_name = "convnext"
    variant = "large"

    checkpoint_path = Path(__file__).parent.parent.parent / "output" / "checkpoints" / f"{model_name}_best_model.pt"
    output_dir = Path(__file__).parent.parent.parent / "output" / "weights"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = create_model(name=model_name, variant=variant, num_classes=len(CLASSES), pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=False)
    visualize_weights(model, save_path=str(output_dir / f"{model_name}_weights.png"), show=False)
