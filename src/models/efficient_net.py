"""EfficientNet model definition for retinal disease classification."""

from pathlib import Path
from typing import Tuple, Callable, Optional

import torch
from torch import nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


def get_efficientnet_model(
    num_classes: int = 2,
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

    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
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

    if fine_tune_all:
        unfreeze(100)
    else:
        unfreeze(0)
    if model_path and model_path.exists():
        print(f"Loading model weights from {model_path}")
        model.load_state_dict(torch.load(model_path))
    return model, unfreeze
