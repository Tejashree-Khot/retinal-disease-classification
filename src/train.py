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

    model_name = "convnext"
    variant = "large"
    root_dir = Path(__file__).parent.parent

    config = TrainerConfig(
        model_name=model_name,
        variant=variant,
        train_path=root_dir / "data" / "IDRiD" / "Train",
        val_path=root_dir / "data" / "IDRiD" / "Test",
        resume_checkpoint_path=root_dir / "output" / "checkpoints" / f"{model_name}_{variant}_best_model.pt",
    )
    setup_wandb(config)
    trainer = Trainer(config)
    results = trainer.train()
    LOGGER.info(f"Training completed. Best F1: {results['best_f1']:.4f}")


if __name__ == "__main__":
    main()
