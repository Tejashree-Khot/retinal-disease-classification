"""Test script for evaluating MedSigLIP models with zero-shot classification."""

import argparse
import logging
from pathlib import Path

import evaluate
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from dataloader.data_loader import ZeroShotImageDataset, get_data_loader
from dataloader.data_utils import ZERO_SHOT_CLASS_PROMPTS
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


def tokenize_class_prompts(processor: AutoProcessor, device: torch.device) -> dict[str, torch.Tensor]:
    """Pre-tokenize zero-shot class text prompts."""
    text_inputs = processor(
        text=ZERO_SHOT_CLASS_PROMPTS,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in text_inputs.items()}


def run_inference(
    model: AutoModel,
    data_loader: DataLoader,
    text_inputs: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Run zero-shot classification inference."""
    model.eval()
    predictions: list[int] = []
    references: list[int] = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Running inference..."):
            labels = batch.pop("labels")
            pixel_values = batch["pixel_values"].to(device)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )
            preds = outputs.logits_per_image.argmax(dim=1).cpu().tolist()
            predictions.extend(preds)
            references.extend(labels.tolist() if isinstance(labels, torch.Tensor) else labels)

    df = pd.DataFrame({"predictions": predictions, "references": references})
    df.to_csv("test_predictions.csv", index=False)
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

    LOGGER.info(f"Loading test data from: {test_path}")
    processor = AutoProcessor.from_pretrained(args.model_path)
    test_dataset = ZeroShotImageDataset(processor=processor, dataset_path=test_path)
    test_loader = get_data_loader(test_dataset, batch_size=args.batch_size)
    LOGGER.info(f"Test samples: {len(test_dataset)}")

    LOGGER.info(f"Loading model: {args.model_path}")
    model = AutoModel.from_pretrained(args.model_path).to(device)

    LOGGER.info(f"Zero-shot class prompts: {ZERO_SHOT_CLASS_PROMPTS}")
    text_inputs = tokenize_class_prompts(processor, device)

    LOGGER.info("Running inference...")
    predictions, references = run_inference(model, test_loader, text_inputs, device)

    metrics = compute_metrics(predictions, references)
    LOGGER.info(f"Accuracy: {metrics['accuracy']:.4f}")
    LOGGER.info(f"F1 (weighted): {metrics['f1']:.4f}")


if __name__ == "__main__":
    main(make_argparser().parse_args())
