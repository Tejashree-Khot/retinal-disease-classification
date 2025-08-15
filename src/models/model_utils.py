"""Utility functions for loading and managing the SimpleCNN model."""

from collections import OrderedDict
from pathlib import Path

import torch
from termcolor import colored

from models.simple_model import SimpleCNN


def load_model(model_path: Path, device: torch.device | str) -> SimpleCNN | None:
    """Load the trained SimpleCNN model from the specified path.

    Args:
        model_path (str): Path to the trained model weights file.

    Returns:
        SimpleCNN: The loaded model.
    """  # Load the model
    model = SimpleCNN().to(device)
    if not model_path.exists():
        print(
            colored(
                f"Model file not found at {model_path}. Model weights are ramdomly initialized.",
                "red",
            )
        )
        return model
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as e:
        # Try to remove "module." prefix if present (from DataParallel)
        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")  # remove "module." if it exists
            new_state_dict[name] = v
        try:
            model.load_state_dict(new_state_dict)
        except Exception as e2:
            print(colored(f"Error(s) in loading state_dict: {e2}", "red"))
            print(colored(f"Original error:{e}", "red"))
            return None
    model.eval()
    return model
