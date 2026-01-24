"""ConvNeXt model implementation for retinal disease classification."""

import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ConvNeXt_Base_Weights,
    ConvNeXt_Large_Weights,
    ConvNeXt_Small_Weights,
    ConvNeXt_Tiny_Weights,
)

from models.base_model import BaseModel

# Mapping of variant names to model functions and weights
CONVNEXT_VARIANTS = {
    "tiny": (models.convnext_tiny, ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    "small": (models.convnext_small, ConvNeXt_Small_Weights.IMAGENET1K_V1),
    "base": (models.convnext_base, ConvNeXt_Base_Weights.IMAGENET1K_V1),
    "large": (models.convnext_large, ConvNeXt_Large_Weights.IMAGENET1K_V1),
}


class ConvNeXtModel(BaseModel):
    """ConvNeXt model wrapper for classification.

    ConvNeXt is a pure convolutional model that incorporates design choices
    from Vision Transformers, achieving state-of-the-art performance.

    Supports Tiny, Small, Base, and Large variants with ImageNet pretrained weights.
    """

    def __init__(self, num_classes: int, pretrained: bool = True, variant: str = "large"):
        """Initialize ConvNeXt model."""
        if variant.lower() not in CONVNEXT_VARIANTS:
            raise ValueError(f"Unsupported variant: {variant}. Choose from: {list(CONVNEXT_VARIANTS.keys())}")
        self.variant = variant.lower()
        super().__init__(num_classes, pretrained)

    def build_model(self) -> nn.Module:
        """Build ConvNeXt model with custom classifier."""
        model_fn, weights = CONVNEXT_VARIANTS[self.variant]

        if self.pretrained:
            model = model_fn(weights=weights)
        else:
            model = model_fn(weights=None)

        # Get the number of input features for the classifier
        # ConvNeXt classifier structure: Sequential(LayerNorm, Flatten, Linear)
        in_features = model.classifier[-1].in_features

        # Replace the last linear layer in classifier
        model.classifier[-1] = nn.Linear(in_features, self.num_classes)

        return model

    def get_input_size(self) -> tuple[int, int]:
        """Get recommended input size for ConvNeXt."""
        return (448, 448)

    def unfreeze_classifier(self) -> None:
        """Unfreeze the classifier head."""
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def get_feature_layer(self) -> nn.Module:
        """Get the last convolutional layer of the last block for feature extraction."""
        return self.model.features[-1][-1].block[0]
