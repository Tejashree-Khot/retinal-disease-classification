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
        model_name="convnext",
        train_path=Path(__file__).parent.parent / "data" / "IDRiD" / "Train",
        val_path=Path(__file__).parent.parent / "data" / "IDRiD" / "Test",
    )

    trainer = Trainer(config)
    results = trainer.train()
    LOGGER.info(f"Training completed. Best F1: {results['best_f1']:.4f}")


if __name__ == "__main__":
    main()
