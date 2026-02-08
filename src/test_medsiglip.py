"""Test script for evaluating MedSigLIP models."""

import argparse
import logging
from pathlib import Path

import evaluate
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from dataloader.data_loader import MedSigLIPDataset, get_data_loader
from dataloader.data_utils import CLASSES
from utils.helper import get_device
from utils.logger import configure_logging

configure_logging()
LOGGER = logging.getLogger("test_medsiglip")


def make_argparser() -> argparse.ArgumentParser:
    """Create argument parser for test script."""
    parser = argparse.ArgumentParser(description="Test MedSigLIP model on the test dataset.")
    parser.add_argument("--model_path", type=str, default="google/medsiglip-448", help="Path to model or model ID.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    return parser


def run_inference(
    model: AutoModel, data_loader: DataLoader, classes: list[str], device: torch.device
) -> tuple[list[int], list[int]]:
    """Run zero-shot classification inference."""
    model.eval()
    predictions = []
    references = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Running inference..."):
            labels = batch["labels"]
            outputs = model(pixel_values=batch["pixel_values"], **batch)
            preds = outputs.logits_per_image.argmax(dim=1).cpu().tolist()
            predictions.extend(preds)
            references.extend(labels.tolist() if isinstance(labels, torch.Tensor) else labels)
    return predictions, references


def compute_metrics(predictions: list[int], references: list[int]) -> dict[str, float]:
    """Compute accuracy and F1 metrics."""
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    metrics = {}
    metrics.update(accuracy_metric.compute(predictions=predictions, references=references))
    metrics.update(f1_metric.compute(predictions=predictions, references=references, average="weighted"))
    return metrics


def main(args: argparse.Namespace) -> None:
    """Run model evaluation on test dataset."""
    root = Path(__file__).parent.parent
    device = get_device("auto")

    test_path = root / "data" / "IDRiD" / "Test"
    size = (448, 448)

    LOGGER.info(f"Loading test data from: {test_path}")
    processor = AutoProcessor.from_pretrained(args.model_path)
    test_dataset = MedSigLIPDataset(processor=processor, dataset_path=test_path, size=size, data_type="test")
    test_loader = get_data_loader(test_dataset, batch_size=args.batch_size)
    LOGGER.info(f"Test samples: {len(test_dataset)}")

    LOGGER.info(f"Loading model: {args.model_path}")
    model = AutoModel.from_pretrained(args.model_path).to(device)

    LOGGER.info("Running inference...")
    predictions, references = run_inference(model, test_loader, CLASSES, device)

    metrics = compute_metrics(predictions, references)
    LOGGER.info(f"Accuracy: {metrics['accuracy']:.4f}")
    LOGGER.info(f"F1 (weighted): {metrics['f1']:.4f}")


if __name__ == "__main__":
    main(make_argparser().parse_args())
