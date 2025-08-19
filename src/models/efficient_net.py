"""EfficientNet model definition and visualization utilities."""

from torch import nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


def get_efficientnet_model(
    num_classes: int = 2, pretrained: bool = True, fine_tune_all: bool = False
) -> nn.Module:
    """Return EfficientNet-B0 with final layer adjusted for num_classes."""
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    # EfficientNet-B0's classifier is Sequential, last layer is Linear
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)  # type: ignore

    for param in model.parameters():
        param.requires_grad = fine_tune_all
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model


if __name__ == "__main__":
    # Example usage
    model = get_efficientnet_model(num_classes=5, pretrained=True, fine_tune_all=True)
    print("EfficientNet model and visualization ready.")
