"""Main trainer class for model training."""

import logging

import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, StepLR
from tqdm import tqdm

from dataloader.data_loader import CustomDataset, get_data_loader
from models.checkpoint import CheckpointManager
from models.model_factory import create_model
from trainer.config import TrainerConfig
from trainer.early_stopping import EarlyStopping
from utils.logger import configure_logging

configure_logging()

LOGGER = logging.getLogger("trainer")


class Trainer:
    """Trainer class for model training with validation, scheduling, and early stopping."""

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.device = self._get_device()
        self.model = self._setup_model()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.criterion = nn.CrossEntropyLoss()
        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience, mode="max")
        self.train_loader, self.val_loader = self._setup_dataloaders()
        self.best_accuracy = 0.0

        if config.use_wandb:
            self._setup_wandb()

    def _get_device(self) -> torch.device:
        """Get the device to use for training."""
        if self.config.device != "auto":
            return torch.device(self.config.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _setup_model(self) -> nn.Module:
        """Setup and return the model."""
        model = create_model(
            self.config.model_name,
            num_classes=self.config.num_classes,
            pretrained=self.config.pretrained,
        )
        model.to(self.device)
        LOGGER.info(f"Model: {self.config.model_name}, Parameters: {model.get_num_parameters():,}")
        return model

    def _setup_optimizer(self) -> AdamW:
        """Setup and return the optimizer."""
        return AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

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

        train_dataset = CustomDataset(
            dataset_path=self.config.train_path, size=input_size, data_type="train"
        )
        val_dataset = CustomDataset(
            dataset_path=self.config.val_path, size=input_size, data_type="val"
        )

        train_loader = get_data_loader(
            train_dataset,
            batch_size=self.config.batch_size,
            use_weighted_sampler=self.config.use_weighted_sampler,
        )
        val_loader = get_data_loader(val_dataset, batch_size=self.config.batch_size)

        LOGGER.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        return train_loader, val_loader

    def _setup_wandb(self) -> None:
        """Initialize wandb for experiment tracking."""
        wandb.init(
            project=self.config.wandb_project,
            config={
                "model": self.config.model_name,
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "scheduler": self.config.scheduler,
            },
        )

    def train(self) -> dict:
        """Main training loop."""
        LOGGER.info(f"Starting training on {self.device}")

        for epoch in range(self.config.epochs):
            train_loss, train_acc = self._train_epoch(epoch)
            val_loss, val_acc = self._validate(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            LOGGER.info(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - LR: {current_lr:.6f}"
            )

            if self.config.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "train_accuracy": train_acc,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                        "learning_rate": current_lr,
                    }
                )

            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self._save_checkpoint(epoch, val_acc, is_best=True)

            if self.early_stopping(val_acc):
                LOGGER.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        if self.config.use_wandb:
            wandb.finish()

        return {"best_accuracy": self.best_accuracy}

    def _train_epoch(self, epoch: int) -> tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [Train]")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{100.0 * correct / total:.2f}%"}
            )

        return total_loss / len(self.train_loader), correct / total

    def _validate(self, epoch: int) -> tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1} [Val]")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "acc": f"{100.0 * correct / total:.2f}%"}
                )

        return total_loss / len(self.val_loader), correct / total

    def _save_checkpoint(self, epoch: int, accuracy: float, is_best: bool = False) -> None:
        """Save model checkpoint."""
        metrics = {"accuracy": accuracy, "epoch": epoch}
        filename = "best_model.pt" if is_best else f"checkpoint_epoch_{epoch}.pt"
        path = self.config.checkpoint_dir / filename

        CheckpointManager.save_checkpoint(
            model=self.model, optimizer=self.optimizer, epoch=epoch, metrics=metrics, path=path
        )
