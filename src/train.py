"""Training script entry point."""

import argparse
import logging
from pathlib import Path

import wandb

from dataloader.data_utils import CLASSES, LABEL_COLUMN_NAME
from trainer import Trainer, TrainerConfig
from utils.logger import configure_logging

configure_logging()
LOGGER = logging.getLogger("train")


def setup_wandb(config: TrainerConfig, experiment_name: str) -> None:
    """Initialize wandb for experiment tracking."""
    wandb.init(
        project=config.wandb_project,
        name=experiment_name,
        config={
            "model": config.model_name,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "scheduler": config.scheduler,
            "classes": CLASSES,
            "label_column_name": LABEL_COLUMN_NAME,
        },
        dir=config.wandb_dir,
    )


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a model on the dataset.")
    parser.add_argument("--model_name", type=str, default="resnet", help="Name of the model to train.")
    parser.add_argument("--variant", type=str, default="50", help="Variant of the model to train.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("label_column_name", type=str, default="class", help="Label column name.")
    return parser


def main(args: argparse.Namespace) -> None:
    """Run training with specified configuration."""
    root_dir = Path(__file__).parent.parent

    model_name = args.model_name
    variant = args.variant

    param_dict = {}

    if param_dict:
        param_str = "_".join([f"{k}_{v}" for k, v in param_dict.items()])
        experiment_name = f"{model_name}_{variant}_{LABEL_COLUMN_NAME.split(' ')[-1]}_{param_str}"
    else:
        experiment_name = f"{model_name}_{variant}_{LABEL_COLUMN_NAME.split(' ')[-1]}"

    config = TrainerConfig(
        model_name=model_name,
        variant=variant,
        batch_size=args.batch_size,
        train_path=root_dir / "data" / "IDRiD" / "Train",
        val_path=root_dir / "data" / "IDRiD" / "Test",
        **param_dict,
        # resume_checkpoint_path=root_dir / "output" / "checkpoints" / f"{model_name}_{variant}_best_model.pt",
    )
    setup_wandb(config, experiment_name)
    trainer = Trainer(config)
    results = trainer.train()
    LOGGER.info(f"Training completed. Best F1: {results['best_f1']:.4f}")


if __name__ == "__main__":
    main(make_argparser().parse_args())
