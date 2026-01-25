"""Test script for evaluating trained models."""

import argparse
import logging
from pathlib import Path

import torch.nn as nn

from dataloader.data_loader import CustomDataset, get_data_loader
from dataloader.data_utils import CLASSES
from models.checkpoint import CheckpointManager
from models.model_factory import create_model
from utils.evaluate import evaluate_model
from utils.helper import get_device, log_metrics
from utils.logger import configure_logging

configure_logging()
LOGGER = logging.getLogger("test")


def make_argparser() -> argparse.ArgumentParser:
    """Create argument parser for test script."""
    parser = argparse.ArgumentParser(description="Test a trained model on the test dataset.")
    parser.add_argument("--model_name", type=str, default="convnext", help="Name of the model to test.")
    parser.add_argument("--variant", type=str, default="large", help="Variant of the model to test.")
    return parser


def main(args: argparse.Namespace) -> None:
    """Run model evaluation on test dataset."""
    root = Path(__file__).parent.parent

    model_name = args.model_name
    variant = args.variant

    checkpoint_path = root / "output" / "checkpoints" / f"{model_name}_{variant}_best_model.pt"
    model = create_model(model_name, variant, num_classes=len(CLASSES), pretrained=False)
    model = CheckpointManager.load_for_inference(model, checkpoint_path, device=get_device("auto"))
    LOGGER.info(f"Loaded model from {checkpoint_path}")

    test_path = root / "data" / "IDRiD" / "Test"
    test_dataset = CustomDataset(test_path, model.get_input_size(), "test")
    test_loader = get_data_loader(test_dataset, batch_size=8)

    LOGGER.info(f"Test samples: {len(test_dataset)}")

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate_model(model, test_loader, get_device("auto"), criterion=criterion, data_type="test")

    log_metrics(metrics, splits=("test",))


if __name__ == "__main__":
    main(make_argparser().parse_args())
