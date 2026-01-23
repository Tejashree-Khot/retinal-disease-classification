"""Main trainer class for model training."""

import logging

import torch
import torch.nn as nn
import wandb
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim import SGD, AdamW
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
        self.class_weights = None
        self.criterion = nn.CrossEntropyLoss(
            # weight=torch.tensor([0.37, 0.33, 0.16, 0.14])
        )  # from data analysis
        # use focal loss
        # self.criterion = FocalLoss(gamma=2.0)
        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience, mode="max")
        self.train_loader, self.val_loader = self._setup_dataloaders()
        self.best_f1 = 0.0

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

    def _setup_optimizer(self) -> AdamW | SGD:
        """Setup and return the optimizer."""
        if self.config.optimizer == "sgd":
            optimizer = SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
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
            dir=self.config.wandb_dir,
        )

    def train(self) -> dict:
        """Main training loop."""
        LOGGER.info(f"Starting training on {self.device}")
        self.model.freeze_backbone()

        for epoch in range(self.config.epochs):
            if epoch == self.config.unfreeze_epoch and self.config.unfreeze_all:
                # drop learning rate by 10x after unfreezing the backbone
                self.optimizer.param_groups[0]["lr"] /= 10
                self.model.unfreeze_all()

            train_loss, train_metrics = self._train_epoch(epoch)
            val_loss, val_metrics = self._validate(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            LOGGER.info(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"Train Loss: {train_loss:.4f}, Train F1: {train_metrics['f1']:.4f}, "
                f"Train Precision: {train_metrics['precision']:.4f}, Train Recall: {train_metrics['recall']:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val F1: {val_metrics['f1']:.4f}, "
                f"Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f} - "
                f"LR: {current_lr:.6f}"
            )

            if self.config.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "train_f1": train_metrics["f1"],
                        "train_precision": train_metrics["precision"],
                        "train_recall": train_metrics["recall"],
                        "val_loss": val_loss,
                        "val_f1": val_metrics["f1"],
                        "val_precision": val_metrics["precision"],
                        "val_recall": val_metrics["recall"],
                        "learning_rate": current_lr,
                    }
                )

            if val_metrics["f1"] > self.best_f1:
                self.best_f1 = val_metrics["f1"]
                self._save_checkpoint(epoch, val_metrics, is_best=True)

            if self.early_stopping(val_metrics["f1"]):
                LOGGER.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        if self.config.use_wandb:
            wandb.finish()

        return {"best_f1": self.best_f1}

    def _train_epoch(self, epoch: int) -> tuple[float, dict]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

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
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            current_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "f1": f"{current_f1:.4f}"})

        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

        return total_loss / len(self.train_loader), {
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    def _validate(self, epoch: int) -> tuple[float, dict]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1} [Val]")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                current_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "f1": f"{current_f1:.4f}"})

        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

        return total_loss / len(self.val_loader), {
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint_metrics = {
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "epoch": epoch,
        }
        filename = (
            f"{self.config.model_name}_best_model.pt"
            if is_best
            else f"{self.config.model_name}_checkpoint_epoch_{epoch}.pt"
        )
        path = self.config.checkpoint_dir / filename

        CheckpointManager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            metrics=checkpoint_metrics,
            path=path,
        )
