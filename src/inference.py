# Minimalistic inference script for cancer detection
import sys
from pathlib import Path

import torch
from termcolor import colored

from utils.data_preprocessing import load_image
from utils.model_utils import load_model

CLASSES = ["adenocarcinoma", "large.cell.carcinoma", "normal", "squamous.cell.carcinoma"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(image_path: Path, model_path: Path) -> None:
    """Predict the class of a given image using a trained SimpleCNN model.

    Args:
        image_path (str): Path to the image file to be predicted.
        model_path (str): Path to the trained model weights file.
    """
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, DEVICE)
    if model is None:
        print("Failed to load the model.")
        return
    print("Model loaded successfully.")

    # Load and preprocess the image
    image = load_image(image_path).to(DEVICE)
    print(f"Image loaded and preprocessed: {image.shape}")

    # Run inference
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
        # round probabilities to 2 decimal places
        probabilities = [round(prob, 3) for prob in probabilities]
        pred = int(torch.argmax(output, 1).item())
    print(colored(f"Prediction: {CLASSES[pred]}, Probabilities: {probabilities[:]}, green"))


def make_parser():
    """Create an argument parser for command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Cancer Detection Inference Script")
    parser.add_argument("image_path", type=Path, help="Path to the image file to be predicted")
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("../checkpoints/model.pth"),
        help="Path to the trained model weights file",
    )
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    if not args.image_path.exists():
        print(f"Image file not found at {args.image_path}. Please provide a valid image path.")
        sys.exit(1)

    predict(Path(sys.argv[1]), Path("../checkpoints/model.pth"))
