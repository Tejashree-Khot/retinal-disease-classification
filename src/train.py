"""Training script entry point."""

import logging
from pathlib import Path

from trainer import Trainer, TrainerConfig
from utils.logger import configure_logging

configure_logging()
LOGGER = logging.getLogger("train")


def main() -> None:
    """Run training with specified configuration."""

    config = TrainerConfig(
        model_name="efficientnet",
        num_classes=5,
        train_path=Path(__file__).parent.parent / "data" / "IDRiD" / "Train",
        val_path=Path(__file__).parent.parent / "data" / "IDRiD" / "Test",
        epochs=50,
        batch_size=32,
        learning_rate=1e-4,
        weight_decay=1e-5,
        scheduler="cosine",
        early_stopping_patience=10,
        checkpoint_dir=Path(__file__).parent.parent / "output" / "checkpoints",
        use_wandb=True,
        wandb_project="retinal-classification",
        use_weighted_sampler=True,
        pretrained=True,
    )

    trainer = Trainer(config)
    results = trainer.train()
    LOGGER.info(f"Training completed. Best accuracy: {results['best_accuracy']:.4f}")


if __name__ == "__main__":
    main()
