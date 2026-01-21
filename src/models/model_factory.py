"""Model factory for creating classification models."""

from typing import Any

from models.base_model import BaseModel
from models.convnext import ConvNeXtModel
from models.efficientnet import EfficientNetModel
from models.resnet import ResNetModel
from models.vgg import VGGModel

# Registry mapping model names to their classes
MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "efficientnet": EfficientNetModel,
    "resnet": ResNetModel,
    "vgg": VGGModel,
    "convnext": ConvNeXtModel,
}


class ModelFactory:
    """Factory class for creating classification models.

    Provides a unified interface for creating different model architectures
    with consistent configuration options.
    """

    @staticmethod
    def create_model(
        name: str, num_classes: int, pretrained: bool = True, **kwargs: Any
    ) -> BaseModel:
        """Create a model instance by name."""
        name_lower = name.lower()
        if name_lower not in MODEL_REGISTRY:
            available = list(MODEL_REGISTRY.keys())
            raise ValueError(f"Unknown model: {name}. Available models: {available}")

        model_class = MODEL_REGISTRY[name_lower]
        return model_class(num_classes=num_classes, pretrained=pretrained, **kwargs)

    @staticmethod
    def get_available_models() -> list[str]:
        """Get list of available model names."""
        return list(MODEL_REGISTRY.keys())

    @staticmethod
    def register_model(name: str, model_class: type[BaseModel]) -> None:
        """Register a new model class."""
        if name.lower() in MODEL_REGISTRY:
            raise ValueError(f"Model {name} is already registered")
        MODEL_REGISTRY[name.lower()] = model_class


def create_model(name: str, num_classes: int, pretrained: bool = True, **kwargs: Any) -> BaseModel:
    """Convenience function to create a model."""
    return ModelFactory.create_model(name, num_classes, pretrained, **kwargs)


def get_available_models() -> list[str]:
    """Get list of available model names."""
    return ModelFactory.get_available_models()
