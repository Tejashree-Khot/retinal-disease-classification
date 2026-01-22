"""VGG model implementation for retinal disease classification."""

import torch.nn as nn
from torchvision import models
from torchvision.models import (
    VGG11_BN_Weights,
    VGG13_BN_Weights,
    VGG16_BN_Weights,
    VGG19_BN_Weights,
)

from models.base_model import BaseModel

# Mapping of variant names to model functions and weights
# Using batch normalization variants for better training stability
VGG_VARIANTS = {
    "vgg11": (models.vgg11_bn, VGG11_BN_Weights.IMAGENET1K_V1),
    "vgg13": (models.vgg13_bn, VGG13_BN_Weights.IMAGENET1K_V1),
    "vgg16": (models.vgg16_bn, VGG16_BN_Weights.IMAGENET1K_V1),
    "vgg19": (models.vgg19_bn, VGG19_BN_Weights.IMAGENET1K_V1),
}


class VGGModel(BaseModel):
    """VGG model wrapper for classification.

    Supports VGG-11, 13, 16, and 19 with batch normalization
    and ImageNet pretrained weights.
    """

    def __init__(self, num_classes: int, pretrained: bool = True, variant: str = "vgg16"):
        """Initialize VGG model."""
        if variant.lower() not in VGG_VARIANTS:
            raise ValueError(
                f"Unsupported variant: {variant}. Choose from: {list(VGG_VARIANTS.keys())}"
            )
        self.variant = variant.lower()
        super().__init__(num_classes, pretrained)

    def build_model(self) -> nn.Module:
        """Build VGG model with custom classifier."""
        model_fn, weights = VGG_VARIANTS[self.variant]

        if self.pretrained:
            model = model_fn(weights=weights)
        else:
            model = model_fn(weights=None)

        # Get the number of input features for the last classifier layer
        in_features = model.classifier[6].in_features

        # Replace the last layer of classifier
        model.classifier[6] = nn.Linear(in_features, self.num_classes)

        return model

    def get_input_size(self) -> tuple[int, int]:
        """Get recommended input size for VGG."""
        return (224, 224)

    def unfreeze_classifier(self) -> None:
        """Unfreeze the classifier head."""
        for param in self.model.classifier.parameters():
            param.requires_grad = True
