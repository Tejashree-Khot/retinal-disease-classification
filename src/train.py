"""Train a SimpleCNN model on a dataset of retinal images."""

from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets

from models.simple_model import SimpleCNN
from utils.data_preprocessing import get_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(data_dir: Path, epochs: int = 5, batch_size: int = 16, lr: float = 0.001):
    """Train a SimpleCNN model on the specified dataset.

    Args:
        data_dir (str): Path to the training data directory.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizer.
    """
    transform = get_transforms()
    print("Loading training data...")
    train_data = datasets.ImageFolder(str(data_dir), transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    print("Training data loaded successfully.")

    print("Initializing model...")
    model = SimpleCNN().to(DEVICE)
    print(f"Using device: {DEVICE}")
    print(f"Number of training samples: {len(train_data)}")
    print(f"Batch size: {batch_size}, Learning rate: {lr}, Epochs: {epochs}")
    print("Model loaded and ready for training.")
    criterion = nn.CrossEntropyLoss()  # Changed for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
            loss.backward()
            optimizer.step()
            print(
                f"Epoch {epoch + 1}, Iteration {iteration + 1}/{len(train_loader)}, "
                f"Loss: {loss.item():.4f}"
            )
            if iteration >= 40:
                break

        print(f"Epoch {epoch + 1}/{epochs} completed.")
    torch.save(model.state_dict(), "../checkpoints/model.pth")
    print("Training done. Model saved as model.pth")


if __name__ == "__main__":
    train_model(data_dir=Path("../data/train"), epochs=10, batch_size=16, lr=0.001)
