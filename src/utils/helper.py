"""Helper functions for the project."""

import logging
import random
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import torch

from dataloader.data_utils import CLASSES, CLASSES_DICT
from utils.logger import configure_logging

configure_logging()
LOGGER = logging.getLogger("helper")


def get_device(device: str = "auto") -> torch.device:
    """Get the device to use for inference."""
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def log_metrics(metrics: dict, splits: tuple[str, str] = ("train", "val")) -> None:
    """Log training and validation metrics in a clean tabular format."""

    headers = ["Split", "Loss", "Acc", "F1", "Prec", "Recall", "QWK"]

    def row(split: str):
        return [
            split.capitalize(),
            f"{metrics.get(f'{split}_loss', 0.0):.4f}",
            f"{metrics.get(f'{split}_accuracy', 0.0):.4f}",
            f"{metrics.get(f'{split}_f1', 0.0):.4f}",
            f"{metrics.get(f'{split}_precision', 0.0):.4f}",
            f"{metrics.get(f'{split}_recall', 0.0):.4f}",
            f"{metrics.get(f'{split}_qwk', 0.0):.4f}",
        ]

    rows = [row(s) for s in splits]

    fmt = "{:<8} | {:<8} {:<6} {:<6} {:<6} {:<7} {:<6}"
    if "epoch" in metrics and "learning_rate" in metrics:
        LOGGER.info(f"\nEpoch {metrics['epoch']} | LR: {metrics['learning_rate']:.6f}")
    LOGGER.info("-" * 78)
    LOGGER.info(fmt.format(*headers))
    LOGGER.info("-" * 78)
    for r in rows:
        LOGGER.info(fmt.format(*r))
    LOGGER.info("-" * 78)


def plot_classwise_predictions(pred_csv_path: Path, image_dir: Path, pred_col="predictions"):
    """Plot classwise predictions for a given dataframe of predictions."""
    df = pd.read_csv(pred_csv_path)
    image_paths = sorted(list(image_dir.glob("*.jpg")))

    _, ax = plt.subplots(2, len(CLASSES), figsize=(4 * len(CLASSES), 8))
    random.seed(42)

    splits = [
        df[df.labels == df[pred_col]],  # correct
        df[df.labels != df[pred_col]],  # wrong
    ]

    for i, cls in enumerate(CLASSES):
        cls_id = CLASSES_DICT[cls]

        for r, sub_df in enumerate(splits):
            row = sub_df[sub_df.labels == cls_id].sample(1).iloc[0]
            img = mpimg.imread(image_paths[row.name])

            ax[r, i].imshow(img)
            ax[r, i].set_title(f"GT: {CLASSES[row.labels]} | Pred: {CLASSES[row[pred_col]]}", fontsize=9)
            ax[r, i].axis("off")

    ax[0, 0].set_ylabel("Correct")
    ax[1, 0].set_ylabel("Wrong")
    plt.tight_layout()
    plt.show()
