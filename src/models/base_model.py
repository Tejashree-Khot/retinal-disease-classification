"""Abstract base model class for all classification models."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class BaseModel(nn.Module, ABC):
    """Abstract base class defining the interface all models must implement.

    All concrete model implementations should inherit from this class and
    implement the required abstract methods.
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        """Initialize base model."""
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model = self.build_model()

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Build and return the model architecture."""
        pass

    @abstractmethod
    def get_input_size(self) -> tuple[int, int]:
        """Get the recommended input image size for this model."""
        pass

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Get the number of parameters in the model."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def freeze_backbone(self) -> None:
        """Freeze backbone layers, keeping only classifier trainable."""
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze classifier - subclasses should override if needed
        self._unfreeze_classifier()

    def unfreeze_all(self) -> None:
        """Unfreeze all layers for full fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True

    @abstractmethod
    def _unfreeze_classifier(self) -> None:
        """Unfreeze the classifier head. To be implemented by subclasses."""
        pass

    def predict(self, image: Tensor) -> Tensor:
        """Predict the class of an image."""
        # add softmax to the output
        with torch.no_grad():
            output = self.model(image)
            return torch.softmax(output, dim=1)
