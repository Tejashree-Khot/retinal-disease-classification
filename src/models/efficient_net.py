"""EfficientNet model definition and visualization utilities."""

import torch
from torch import nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
from torchviz import make_dot


def get_efficientnet_model(
    num_classes: int = 2, pretrained: bool = True, fine_tune_all: bool = False
) -> nn.Module:
    """Return EfficientNet-B0 with final layer adjusted for num_classes."""
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    for param in model.parameters():
        param.requires_grad = fine_tune_all
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model


def visualize_model_architecture(
    model: nn.Module, filename: str = "efficientnet_model.png"
) -> None:
    """Visualize the model architecture and save as PNG."""
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    y = model(x)
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.render(filename.replace(".png", ""), format="png")
    print(f"Model visualization saved as '{filename}'")


if __name__ == "__main__":
    # Example usage
    model = get_efficientnet_model(num_classes=10, pretrained=True, fine_tune_all=True)
    visualize_model_architecture(model, "efficientnet_model.png")
    print("EfficientNet model and visualization ready.")
