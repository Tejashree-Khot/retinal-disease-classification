"""Train a SimpleCNN model on a dataset of retinal images."""

from pathlib import Path

import torch
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets

from models.simple_model import SimpleCNN
from utils.data_preprocessing import get_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_wandb():
    """Initialize Weights & Biases for experiment tracking."""
    wandb.init(
        project="retinal-disease-classification",
        entity="tejashree",
        name="retinal_cnn_training",
        config={"epochs": 10, "batch_size": 16, "learning_rate": 0.001, "model": "SimpleCNN"},
    )
    print("Weights & Biases initialized.")


def train_model(data_dir: Path, epochs: int = 5, batch_size: int = 16, lr: float = 0.001):
    """Train a SimpleCNN model on the specified dataset.

    Args:
        data_dir (str): Path to the training data directory.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizer.
    """
    # Load the training data
    transform = get_transforms()
    print("Loading training data...")
    train_data = datasets.ImageFolder(str(data_dir), transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    print("Training data loaded successfully.")

    # Initialize the model
    print("Initializing model...")
    model = SimpleCNN().to(DEVICE)
    print(f"Using device: {DEVICE}")
    print(f"Number of training samples: {len(train_data)}")
    print(f"Batch size: {batch_size}, Learning rate: {lr}, Epochs: {epochs}")
    print("Model loaded and ready for training.")

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Changed for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize Weights & Biases
    init_wandb()
    print("Starting training...")

    for epoch in range(epochs):
        model.train()
        for iteration, batch in enumerate(train_loader):
            imgs, labels = batch
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            # labels should be LongTensor with class indices for CrossEntropyLoss
            optimizer.zero_grad()
            out = model(imgs)
            if iteration == 0:
                print(f"{out, labels}, Output shape: {out.shape}, Labels shape: {labels.shape}")
            loss = criterion(out, labels)
            # update model parameters for the current batch
            loss.backward()
            optimizer.step()
            print(
                f"Epoch {epoch + 1}, Iteration {iteration + 1}/{len(train_loader)}, "
                f"Loss: {loss.item():.4f}"
            )
            # if iteration >= 40:
            #     break
        # Log the loss to Weights & Biases
        wandb.log({"epoch": epoch + 1, "train loss": loss.item()})

        # Evaluate the model afer each epoch
        print("Evaluating model...")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                loss = criterion(out, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_val_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "val loss": avg_val_loss})

        print(f"Epoch {epoch + 1}/{epochs} completed.")
    print("Training completed.")
    # Finish the Weights & Biases run
    wandb.finish()

    # Save the trained model
    torch.save(model.state_dict(), "../../checkpoints/model.pth")
    print("Training done. Model saved as model.pth")


if __name__ == "__main__":
    train_model(data_dir=Path("../../data/train"), epochs=10, batch_size=16, lr=0.001)
