"""ResNet model implementation for retinal disease classification."""

import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
)

from models.base_model import BaseModel

# Mapping of variant names to model functions and weights
RESNET_VARIANTS = {
    "18": (models.resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "34": (models.resnet34, ResNet34_Weights.IMAGENET1K_V1),
    "50": (models.resnet50, ResNet50_Weights.IMAGENET1K_V2),
    "101": (models.resnet101, ResNet101_Weights.IMAGENET1K_V2),
    "152": (models.resnet152, ResNet152_Weights.IMAGENET1K_V2),
}


class ResNetModel(BaseModel):
    """ResNet model wrapper for classification.

    Supports ResNet-18, 34, 50, 101, and 152 with ImageNet pretrained weights.
    """

    def __init__(self, num_classes: int, variant: str, pretrained: bool = True):
        """Initialize ResNet model."""
        if variant.lower() not in RESNET_VARIANTS:
            raise ValueError(f"Unsupported variant: {variant}. Choose from: {list(RESNET_VARIANTS.keys())}")
        super().__init__(num_classes, variant.lower(), pretrained)

    def build_model(self) -> nn.Module:
        """Build ResNet model with custom classifier."""
        model_fn, weights = RESNET_VARIANTS[self.variant]

        if self.pretrained:
            model = model_fn(weights=weights)
        else:
            model = model_fn(weights=None)

        # Get the number of input features for the fc layer
        in_features = model.fc.in_features

        # Replace fc layer with custom classifier
        model.fc = nn.Linear(in_features, self.num_classes)

        return model

    def get_input_size(self) -> tuple[int, int]:
        """Get recommended input size for ResNet."""
        return (448, 448)

    def unfreeze_classifier(self) -> None:
        """Unfreeze the classifier head (fc layer)."""
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def get_feature_layer(self) -> nn.Module:
        """Get the last convolutional layer for feature extraction."""
        return self.model.layer4[-1]

    def get_selected_conv_layers_in_order(self) -> list[nn.Module]:
        """Get all convolutional layers in the model in forward order."""
        conv_layers = self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4
        return conv_layers
