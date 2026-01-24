"""Main trainer class for model training."""

import logging

import torch.nn as nn
import wandb
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, StepLR
from tqdm import tqdm

from dataloader.data_loader import CustomDataset, get_data_loader
from models.checkpoint import CheckpointManager
from models.model_factory import create_model
from trainer.config import TrainerConfig
from trainer.early_stopping import EarlyStopping
from utils.evaluate import evaluate_model
from utils.helper import get_device, log_metrics
from utils.logger import configure_logging

configure_logging()

LOGGER = logging.getLogger("trainer")


class Trainer:
    """Trainer class for model training with validation, scheduling, and early stopping."""

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.device = get_device(self.config.device)
        self.model = self._setup_model()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.class_weights = None
        self.criterion = nn.CrossEntropyLoss()
        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience, mode="max")
        self.train_loader, self.val_loader = self._setup_dataloaders()
        self.best_f1 = 0.0

    def _setup_model(self) -> nn.Module:
        """Setup and return the model."""
        model = create_model(
            self.config.model_name, num_classes=self.config.num_classes, pretrained=self.config.pretrained
        )
        model.to(self.device)
        LOGGER.info(f"Model: {self.config.model_name}, Parameters: {model.get_num_parameters():,}")
        return model

    def _setup_optimizer(self) -> AdamW | SGD:
        """Setup and return the optimizer."""
        if self.config.optimizer == "sgd":
            optimizer = SGD(
                self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
            )
        else:
            optimizer = AdamW(
                self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
            )
        return optimizer

    def _setup_scheduler(self) -> LRScheduler | None:
        """Setup and return the learning rate scheduler."""
        if self.config.scheduler == "step":
            return StepLR(self.optimizer, step_size=10, gamma=0.1)
        if self.config.scheduler == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=self.config.epochs)
        return None

    def _setup_dataloaders(self) -> tuple:
        """Setup and return train and validation dataloaders."""
        input_size = self.model.get_input_size()

        train_dataset = CustomDataset(dataset_path=self.config.train_path, size=input_size, data_type="train")
        val_dataset = CustomDataset(dataset_path=self.config.val_path, size=input_size, data_type="val")

        train_loader = get_data_loader(
            train_dataset, batch_size=self.config.batch_size, use_weighted_sampler=self.config.use_weighted_sampler
        )
        val_loader = get_data_loader(val_dataset, batch_size=self.config.batch_size)

        LOGGER.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        return train_loader, val_loader

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False) -> None:
        """Save model checkpoint."""
        metrics["epoch"] = epoch
        suffix = "best_model.pt" if is_best else f"checkpoint_epoch_{epoch}.pt"
        path = self.config.checkpoint_dir / f"{self.config.model_name}_{self.model.variant}_{suffix}"

        CheckpointManager.save_checkpoint(
            model=self.model, optimizer=self.optimizer, epoch=epoch, metrics=metrics, path=path
        )

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            current_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "f1": f"{current_f1:.4f}"})

        metrics = {
            "train_loss": total_loss / len(self.train_loader),
            "train_accuracy": accuracy_score(all_labels, all_preds),
            "train_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
            "train_precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
            "train_recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
            "train_qwk": cohen_kappa_score(all_labels, all_preds, weights="quadratic"),
        }
        return metrics

    def train(self) -> dict:
        """Main training loop."""
        LOGGER.info(f"Starting training on {self.device}")
        self.model.freeze_backbone()

        for epoch in range(self.config.epochs):
            if epoch == self.config.unfreeze_epoch and self.config.unfreeze_all:
                # drop learning rate by 10x after unfreezing the backbone
                self.optimizer.param_groups[0]["lr"] /= 10
                self.model.unfreeze_all()

            train_metrics = self._train_epoch(epoch)
            val_metrics = evaluate_model(self.model, self.val_loader, self.device, self.criterion)

            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            metrics = {**train_metrics, **val_metrics}
            metrics.update({"learning_rate": current_lr, "epoch": epoch + 1})
            log_metrics(metrics)
            wandb.log(metrics)

            if val_metrics["val_f1"] > self.best_f1:
                self.best_f1 = val_metrics["val_f1"]
                self._save_checkpoint(epoch, val_metrics, is_best=True)

            if self.early_stopping(val_metrics["val_f1"]):
                LOGGER.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        wandb.finish()

        return {"best_f1": self.best_f1}
