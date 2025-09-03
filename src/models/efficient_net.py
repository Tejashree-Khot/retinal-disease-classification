"""EfficientNet model definition for retinal disease classification."""

from pathlib import Path
from typing import Tuple, Callable, Optional

import torch
from torch import nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    EfficientNet_B7_Weights,
)


def get_efficientnet_model(
    model_name: str = "efficientnet-b0",
    num_classes: int = 5,
    pretrained: bool = True,
    fine_tune_all: bool = False,
    model_path: Optional[Path] = None,
) -> Tuple[nn.Module, Callable]:
    """
    Returns EfficientNet-B0 with a custom classifier and a function to unfreeze layers.
    Usage:
        model, unfreeze = get_efficientnet_model(...)
        unfreeze(0)      # only classifier
        unfreeze(1)      # classifier + last block
        unfreeze(100)    # all layers
    """
    if model_name == "efficientnet-b0":
        model = models.efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
    elif model_name == "efficientnet-b1":
        model = models.efficientnet_b1(
            weights=EfficientNet_B1_Weights.DEFAULT if pretrained else None
        )
    elif model_name == "efficientnet-b7":
        model = models.efficientnet_b2(
            weights=EfficientNet_B7_Weights.DEFAULT if pretrained else None
        )
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True), nn.Linear(model.classifier[-1].in_features, num_classes)
    )

    def unfreeze(stage=0):
        # Freeze all
        for p in model.parameters():
            p.requires_grad = False
        # Unfreeze classifier
        for p in model.classifier.parameters():
            p.requires_grad = True
        # Unfreeze last N blocks
        if stage > 0:
            blocks = list(model.features)
            for b in blocks[-stage:]:
                for p in b.parameters():
                    p.requires_grad = True
        # Unfreeze all if stage big
        if stage >= len(list(model.features)):
            for p in model.parameters():
                p.requires_grad = True

    # Apply unfreezing strategy
    # layer by layer unfreezing
    if fine_tune_all:
        unfreeze(len(list(model.features)) + 1)
    else:
        unfreeze(0)  # only classifier if training from scratch
    # else only classifier is unfrozen by default
    # Load weights if a path is provided
    if model_path and model_path.exists():
        print(f"Loading model weights from {model_path}")
        model.load_state_dict(torch.load(model_path))
    return model, unfreeze


if __name__ == "__main__":
    model, unfreeze = get_efficientnet_model()
