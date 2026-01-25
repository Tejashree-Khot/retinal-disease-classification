"""Inference script for running predictions on images."""

import argparse
import logging
import random
from pathlib import Path

import torch
from torch import Tensor

from dataloader.data_preprocessing import get_image_transforms, load_image
from dataloader.data_utils import CLASSES
from models.checkpoint import CheckpointManager
from models.model_factory import create_model
from utils.helper import get_device
from utils.logger import configure_logging

configure_logging()
LOGGER = logging.getLogger("inference")


def predict_image(model: torch.nn.Module, image_tensor: Tensor, device: torch.device) -> tuple[int, str, float]:
    """Predict class for a single image tensor."""
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)

    class_idx = predicted.item()
    class_name = CLASSES[class_idx]
    confidence_score = confidence.item()

    return class_idx, class_name, confidence_score


def predict_images(
    model: torch.nn.Module, image_paths: list[Path], input_size: tuple[int, int], device: torch.device
) -> list[tuple[Path, int, str, float]]:
    """Predict classes for all images in a directory."""
    results = []
    for image_path in image_paths:
        transform = get_image_transforms(input_size, data_type="test")
        image_tensor = load_image(image_path, transform)
        class_idx, class_name, confidence = predict_image(model, image_tensor, device)
        results.append((image_path, class_idx, class_name, confidence))
    return results


def make_argparser() -> argparse.ArgumentParser:
    """Create argument parser for test script."""
    parser = argparse.ArgumentParser(description="Test a trained model on the test dataset.")
    parser.add_argument("--model_name", type=str, default="convnext", help="Name of the model to test.")
    parser.add_argument("--variant", type=str, default="large", help="Variant of the model to test.")
    parser.add_argument("--image_path", type=Path, default=None, help="Path to the image to predict.")
    parser.add_argument("--image_dir", type=Path, default=None, help="Directory containing images to predict.")
    return parser


def main(args: argparse.Namespace) -> None:
    """Run inference on images."""
    root = Path(__file__).parent.parent

    model_name = args.model_name
    variant = args.variant

    checkpoint_path = root / "output" / "checkpoints" / f"{model_name}_{variant}_best_model.pt"
    model = create_model(model_name, variant, num_classes=len(CLASSES), pretrained=False)
    model = CheckpointManager.load_for_inference(model, checkpoint_path, device=get_device("auto"))
    LOGGER.info(f"Loaded model from {checkpoint_path}")

    if args.image_path:
        image_paths = [args.image_path]
    elif args.image_dir:
        image_paths = random.sample(list(args.image_dir.glob("*.jpg")), 10)
    else:
        image_dir = Path(__file__).parent.parent / "data" / "IDRiD" / "Test" / "images"
        image_paths = random.sample(list(image_dir.glob("*.jpg")), 10)

    results = predict_images(model, image_paths, model.get_input_size(), get_device("auto"))

    LOGGER.info(f"\nPredictions for {len(results)} images:")
    for img_path, class_idx, class_name, confidence in results:
        LOGGER.info(f"  {img_path.name}: {class_name} (index: {class_idx}, confidence: {confidence:.4f})")


if __name__ == "__main__":
    main(make_argparser().parse_args())
