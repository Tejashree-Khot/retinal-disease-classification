"""Training script for MedSigLIP model."""

import argparse
import logging
import os
from pathlib import Path

from transformers import AutoModel, AutoProcessor, EvalPrediction, Trainer, TrainingArguments

from dataloader.data_loader import MedSigLIPDataset, collate_fn_text_image
from utils.logger import configure_logging

configure_logging()
LOGGER = logging.getLogger("train_medsiglip")

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


def make_argparser() -> argparse.ArgumentParser:
    """Create argument parser for training script."""
    parser = argparse.ArgumentParser(description="Train MedSigLIP model.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device.")
    return parser


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """Compute contrastive accuracy from evaluation predictions."""
    predictions = eval_pred.predictions
    print(predictions)
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    preds = predictions[:, 0]
    targets = predictions[:, 1]
    accuracy = (preds == targets).mean()
    return {"contrastive_accuracy": float(accuracy)}


def main(args: argparse.Namespace) -> None:
    """Run MedSigLIP training."""
    root = Path(__file__).parent.parent
    output_dir = root / "output" / "checkpoints" / "medsiglip-448-finetuned"

    model_id = "google/medsiglip-448"

    LOGGER.info(f"Loading model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # Freeze text encoder
    for p in model.text_model.parameters():
        p.requires_grad = False

    train_path = root / "data" / "IDRiD" / "Train"
    val_path = root / "data" / "IDRiD" / "Test"

    LOGGER.info(f"Loading training data from: {train_path}")
    train_dataset = MedSigLIPDataset(processor=processor, dataset_path=train_path)
    LOGGER.info(f"Training samples: {len(train_dataset)}")

    LOGGER.info(f"Loading validation data from: {val_path}")
    val_dataset = MedSigLIPDataset(processor=processor, dataset_path=val_path)
    LOGGER.info(f"Validation samples: {len(val_dataset)}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="steps",
        eval_steps=50,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=5,
        lr_scheduler_type="cosine",
        push_to_hub=False,
        report_to="tensorboard",
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn_text_image,
        compute_metrics=compute_metrics,
    )

    LOGGER.info("Starting training...")
    trainer.train()

    LOGGER.info(f"Saving model to: {output_dir}")
    trainer.save_model()

    LOGGER.info("Training completed.")


if __name__ == "__main__":
    main(make_argparser().parse_args())
