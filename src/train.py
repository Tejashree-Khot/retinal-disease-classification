"""Training script entry point."""

import logging
from pathlib import Path

import wandb

from trainer import Trainer, TrainerConfig
from utils.logger import configure_logging

configure_logging()
LOGGER = logging.getLogger("train")


def setup_wandb(config: TrainerConfig) -> None:
    """Initialize wandb for experiment tracking."""
    wandb.init(
        project=config.wandb_project,
        config={
            "model": config.model_name,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "scheduler": config.scheduler,
        },
        dir=config.wandb_dir,
    )


def main() -> None:
    """Run training with specified configuration."""

    config = TrainerConfig(
        model_name="convnext",
        train_path=Path(__file__).parent.parent / "data" / "IDRiD" / "Train",
        val_path=Path(__file__).parent.parent / "data" / "IDRiD" / "Test",
    )
    setup_wandb(config)
    trainer = Trainer(config)
    results = trainer.train()
    LOGGER.info(f"Training completed. Best F1: {results['best_f1']:.4f}")


if __name__ == "__main__":
    main()
