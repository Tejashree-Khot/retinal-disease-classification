"""Training script for MedSigLIP model."""

import argparse
import logging
from pathlib import Path

from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, Trainer, TrainingArguments

from dataloader.data_loader import MedSigLIPDataset, collate_fn_text_image
from utils.logger import configure_logging

configure_logging()
LOGGER = logging.getLogger("train_medsiglip")


def make_argparser() -> argparse.ArgumentParser:
    """Create argument parser for training script."""
    parser = argparse.ArgumentParser(description="Train MedSigLIP model.")
    parser.add_argument("--model_id", type=str, default="google/medsiglip-448", help="Model ID to load.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--output_dir", type=str, default="medsiglip-448-finetuned", help="Output directory.")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to Hub.")
    return parser


def main(args: argparse.Namespace) -> None:
    """Run MedSigLIP training."""
    root = Path(__file__).parent.parent

    LOGGER.info(f"Loading model: {args.model_id}")
    image_processor = AutoImageProcessor.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id)

    train_path = root / "data" / "IDRiD" / "Train"
    val_path = root / "data" / "IDRiD" / "Test"

    LOGGER.info(f"Loading training data from: {train_path}")
    train_dataset = MedSigLIPDataset(train_path, image_processor, tokenizer)
    LOGGER.info(f"Training samples: {len(train_dataset)}")

    LOGGER.info(f"Loading validation data from: {val_path}")
    val_dataset = MedSigLIPDataset(val_path, image_processor, tokenizer)
    LOGGER.info(f"Validation samples: {len(val_dataset)}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="steps",
        eval_steps=50,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=5,
        lr_scheduler_type="cosine",
        push_to_hub=args.push_to_hub,
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn_text_image,
    )

    LOGGER.info("Starting training...")
    trainer.train()

    LOGGER.info(f"Saving model to: {args.output_dir}")
    trainer.save_model()

    LOGGER.info("Training completed.")


if __name__ == "__main__":
    main(make_argparser().parse_args())
