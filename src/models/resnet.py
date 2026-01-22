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
    "resnet18": (models.resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (models.resnet34, ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, ResNet50_Weights.IMAGENET1K_V2),
    "resnet101": (models.resnet101, ResNet101_Weights.IMAGENET1K_V2),
    "resnet152": (models.resnet152, ResNet152_Weights.IMAGENET1K_V2),
}


class ResNetModel(BaseModel):
    """ResNet model wrapper for classification.

    Supports ResNet-18, 34, 50, 101, and 152 with ImageNet pretrained weights.
    """

    def __init__(self, num_classes: int, pretrained: bool = True, variant: str = "resnet18"):
        """Initialize ResNet model."""
        if variant.lower() not in RESNET_VARIANTS:
            raise ValueError(
                f"Unsupported variant: {variant}. Choose from: {list(RESNET_VARIANTS.keys())}"
            )
        self.variant = variant.lower()
        super().__init__(num_classes, pretrained)

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
        return (224, 224)

    def unfreeze_classifier(self) -> None:
        """Unfreeze the classifier head (fc layer)."""
        for param in self.model.fc.parameters():
            param.requires_grad = True
