"""Evaluate a trained model on a test dataset."""

from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from termcolor import colored
from tqdm import tqdm

from dataloader.data_loader import get_data_loader
from src.models.model_utils import load_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_accuarcy_metrics(preds: list[int], labels: list[int]) -> None:
    """Calculate and print accuracy."""
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall = recall_score(labels, preds, average="weighted", zero_division=0)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    print(
        f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
        f"Recall: {recall:.4f}, F1 Score: {f1:.4f}"
    )
    if accuracy < 0.9:
        print(colored(f"Low accuracy: {accuracy:.4f}. Check your model and data.", "red"))
    else:
        print(
            colored(f"High accuracy: {accuracy:.4f}. Model seems to be performing well.", "green")
        )


def evaluate_model(model_path: Path, data_dir: Path, batch_size: int = 16) -> None:
    """Evaluate a trained SimpleCNN model on a test dataset and print accuracy.

    Args:
        model_path (str): Path to the trained model weights file.
        data_dir (str): Path to the test data directory.
        batch_size (int): Batch size for evaluation.
    """
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, DEVICE)

    if model is None:
        print(colored("Failed to load the model.", "red"))
        return
    # Load the training data
    print("Loading training data...")
    test_loader = get_data_loader(data_dir, size=(224, 224), batch_size=batch_size, augment=False)

    predictions = []
    gt_labels = []
    print(f"Number of training samples: {len(test_loader)}")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", unit="batch"):
            imgs, labels = batch
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)

            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            # round probabilities to 2 decimal places
            probabilities = [round(prob, 3) for prob in probabilities]

            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            gt_labels.extend(labels.cpu().numpy())

    calculate_accuarcy_metrics(predictions, gt_labels)


if __name__ == "__main__":
    evaluate_model(
        model_path=Path("../checkpoints/model.pth"), data_dir=Path("/workspace/data/IDRiD/train")
    )
