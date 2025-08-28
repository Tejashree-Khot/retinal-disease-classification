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
    Create an EfficientNet-B0 model with customized final classification layer.

    Args:
        num_classes: Number of output classes for classification
        pretrained: Whether to use pretrained weights from ImageNet
        fine_tune_all: Whether to make all parameters trainable immediately
        model_path: Optional path to load saved model weights

    Returns:
        Tuple containing:
        - The configured EfficientNet model
        - A function to gradually unfreeze layers during training
    """
    # Initialize model with or without pretrained weights
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)

    # Replace classifier with custom one for our task
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True), nn.Linear(model.classifier[-1].in_features, num_classes)
    )

    # Freeze layers initially
    for param in model.parameters():
        param.requires_grad = False

    # Always fine-tune the classifier
    for param in model.classifier.parameters():
        param.requires_grad = True

    def unfreeze_model_layers(stage: int = 0) -> None:
        """
        Gradually unfreeze model layers for fine-tuning.

        Implements gradual unfreezing strategy where we progressively
        unfreeze deeper layers of the network during training.

        Args:
            stage: Controls which layers to unfreeze:
                  0 = classifier only (default)
                  1 = classifier + last feature block
                  2 = classifier + last 2 feature blocks
                  etc.
        """
        # Reset - freeze all layers first
        for param in model.parameters():
            param.requires_grad = False

        # Always unfreeze classifier
        for param in model.classifier.parameters():
            param.requires_grad = True

        if stage > 0:
            # Unfreeze specified number of blocks from the end
            feature_blocks = list(model.features)
            blocks_to_unfreeze = min(stage, len(feature_blocks))

            for i in range(blocks_to_unfreeze):
                block_idx = len(feature_blocks) - 1 - i  # Start from the last block
                for param in feature_blocks[block_idx].parameters():
                    param.requires_grad = True

        # Log training status
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(
            f"Stage {stage}: {trainable_params:,} trainable parameters out of {total_params:,} ({trainable_params / total_params:.1%})"
        )

    # If fine_tune_all is True, unfreeze all parameters immediately
    if fine_tune_all:
        for param in model.parameters():
            param.requires_grad = True

    # Load weights from file if provided
    if model_path and model_path.exists():
        print(f"Loading model weights from {model_path}")
        model.load_state_dict(torch.load(model_path))

    return model, unfreeze_model_layers
