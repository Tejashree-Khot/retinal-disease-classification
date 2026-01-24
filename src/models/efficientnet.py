"""EfficientNet model implementation for retinal disease classification."""

import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
    EfficientNet_B4_Weights,
    EfficientNet_B5_Weights,
    EfficientNet_B6_Weights,
    EfficientNet_B7_Weights,
)

from models.base_model import BaseModel

# Mapping of variant names to model functions and weights
EFFICIENTNET_VARIANTS = {
    "b0": (models.efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1),
    "b1": (models.efficientnet_b1, EfficientNet_B1_Weights.IMAGENET1K_V1),
    "b2": (models.efficientnet_b2, EfficientNet_B2_Weights.IMAGENET1K_V1),
    "b3": (models.efficientnet_b3, EfficientNet_B3_Weights.IMAGENET1K_V1),
    "b4": (models.efficientnet_b4, EfficientNet_B4_Weights.IMAGENET1K_V1),
    "b5": (models.efficientnet_b5, EfficientNet_B5_Weights.IMAGENET1K_V1),
    "b6": (models.efficientnet_b6, EfficientNet_B6_Weights.IMAGENET1K_V1),
    "b7": (models.efficientnet_b7, EfficientNet_B7_Weights.IMAGENET1K_V1),
}

# Recommended input sizes for each variant
EFFICIENTNET_INPUT_SIZES = {
    "b0": (224, 224),
    "b1": (240, 240),
    "b2": (260, 260),
    "b3": (300, 300),
    "b4": (380, 380),
    "b5": (456, 456),
    "b6": (528, 528),
    "b7": (600, 600),
}


class EfficientNetModel(BaseModel):
    """EfficientNet model wrapper for classification.

    Supports variants B0 through B7 with ImageNet pretrained weights.
    """

    def __init__(self, num_classes: int, pretrained: bool = True, variant: str = "b7"):
        """Initialize EfficientNet model."""
        if variant.lower() not in EFFICIENTNET_VARIANTS:
            raise ValueError(f"Unsupported variant: {variant}. Choose from: {list(EFFICIENTNET_VARIANTS.keys())}")
        self.variant = variant.lower()
        super().__init__(num_classes, pretrained)

    def build_model(self) -> nn.Module:
        """Build EfficientNet model with custom classifier."""
        model_fn, weights = EFFICIENTNET_VARIANTS[self.variant]

        if self.pretrained:
            model = model_fn(weights=weights)
        else:
            model = model_fn(weights=None)

        # Get the number of input features for the classifier
        in_features = model.classifier[1].in_features

        # Replace classifier with custom head
        model.classifier = nn.Sequential(nn.Dropout(p=0.5, inplace=True), nn.Linear(in_features, self.num_classes))

        return model

    def get_input_size(self) -> tuple[int, int]:
        """Get recommended input size for this variant."""
        return EFFICIENTNET_INPUT_SIZES[self.variant]

    def unfreeze_classifier(self) -> None:
        """Unfreeze the classifier head."""
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def get_feature_layer(self) -> nn.Module:
        """Get the last convolutional layer for feature extraction."""
        return self.model.features[-1]
