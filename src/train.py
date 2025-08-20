"""Train a SimpleCNN model on a dataset of retinal images."""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from dataloader.data_loader import get_data_loader
from dataloader.data_preprocessing import get_transforms, load_image
from models.efficient_net import get_efficientnet_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_optimizer(
    model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4
) -> optim.Optimizer:
    """Return Adam optimizer for the model."""
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_scheduler(
    optimizer: optim.Optimizer, step_size: int = 7, gamma: float = 0.1
) -> optim.lr_scheduler.StepLR:
    """Return a StepLR scheduler."""
    return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


def init_wandb():
    """Initialize Weights & Biases for experiment tracking."""
    name_ith_timestamp = f"EfficientNet_B0_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project="retinal-desease-classification",
        name=name_ith_timestamp,
        config={"epochs": 10, "batch_size": 16, "learning_rate": 0.001, "model": "EfficientNet-B0"},
        notes="Training a EfficientNet model on retinal images",
    )
    print("Weights & Biases initialized.")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device | str,
) -> tuple[float, float]:
    """Train the model for one epoch and return loss and accuracy."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch_inputs, batch_labels in tqdm(dataloader):
        inputs_dev, labels_dev = batch_inputs.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs_dev)
        loss = criterion(outputs, labels_dev)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs_dev.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels_dev).sum().item()
        total += labels_dev.size(0)
    return running_loss / total, correct / total


def evaluate(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device | str
) -> tuple[float, float]:
    """Evaluate the model and return loss and accuracy."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch_inputs, batch_labels in tqdm(dataloader):
            inputs_dev, labels_dev = batch_inputs.to(device), batch_labels.to(device)
            outputs = model(inputs_dev)
            loss = criterion(outputs, labels_dev)
            running_loss += loss.item() * inputs_dev.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels_dev).sum().item()
            total += labels_dev.size(0)
    return running_loss / total, correct / total


def inference(
    model: nn.Module, image_paths: list[str], transform: Callable, device: torch.device
) -> list[int]:
    """Run inference on a list of image paths and return predictions."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for path in tqdm(image_paths):
            img = load_image(Path(path), 448).to(device)
            outputs = model(img)
            _, pred = torch.max(outputs, 1)
            predictions.append(pred.item())
    return predictions


def train_model(data_dir: Path, epochs: int = 10, batch_size: int = 32, lr: float = 0.001):
    """Train a SimpleCNN model on the specified dataset.

    Args:
        data_dir (str): Path to the training data directory.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizer.
    """
    image_size = (448, 448)  # Resize images to this size
    # Load the training data
    print("Loading training data...")
    train_dir = data_dir.parent / "Train"
    train_loader = get_data_loader(train_dir, size=image_size, batch_size=batch_size, augment=True)
    print("Training data loaded successfully.")

    test_dir = data_dir.parent / "Test"
    test_loader = get_data_loader(test_dir, size=image_size, batch_size=batch_size, augment=False)

    # Initialize the model
    print("Initializing model...")
    model = get_efficientnet_model(num_classes=5, pretrained=True, fine_tune_all=False)
    model.to(DEVICE)
    # model = SimpleCNN().to(DEVICE)
    print(f"Using device: {DEVICE}")
    print(f"Batch size: {batch_size}, Learning rate: {lr}, Epochs: {epochs}")
    print("Model loaded and ready for training.")

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Changed for multi-class classification
    optimizer = get_optimizer(model, lr=lr)
    scheduler = get_scheduler(optimizer)
    # Initialize Weights & Biases
    init_wandb()
    print("Starting training...")

    best_val_acc = 0.0
    best_state_dict = None

    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, test_loader, criterion, DEVICE)
        scheduler.step()

        # save best weights
        if val_acc > best_val_acc:  # Example threshold for saving the model
            best_val_acc = val_acc
            best_state_dict = model.state_dict()

        # Log metrics to Weights & Biases
        wandb.log({"epoch": epoch + 1, "train loss": train_loss})
        wandb.log({"epoch": epoch + 1, "train accuracy": train_acc})
        wandb.log({"epoch": epoch + 1, "val loss": val_loss})
        wandb.log({"epoch": epoch + 1, "val accuracy": val_acc})

        print(f"Epoch {epoch + 1}/{epochs} completed.")

    print("Training completed.")
    # Finish the Weights & Biases run
    wandb.finish()

    # Save the trained model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print("Best model weights loaded.")
    else:
        print("No best model weights found, saving current model state.")

    torch.save(model.state_dict(), "../checkpoints/model.pth")
    print("Training done. Model saved as model.pth")


def make_parser():
    """Create an argument parser for command line arguments."""
    parser = argparse.ArgumentParser(description="Train a SimpleCNN model on retinal images.")
    parser.add_argument(
        "--data_dir", type=Path, required=True, help="Path to the training data directory."
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer.")
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    train_model(
        data_dir=Path(args.data_dir), epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
    )
