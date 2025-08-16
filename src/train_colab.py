"""PyTorch custom data loader."""

from collections import OrderedDict
from pathlib import Path
from typing import Callable, cast

import numpy as np
import pandas as pd
import torch
import wandb
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from termcolor import colored
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_B0_Weights
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    "Clinically_Significant_Macular_Edema",
    "No_DR",
    "normal",
    "Mild_Moderate_NPDR",
    "Severe_PDR",
]

CLASSES_DICT = {
    "Clinically_Significant_Macular_Edema": 0,
    "No_DR": 1,
    "normal": 2,
    "Mild_Moderate_NPDR": 3,
    "Severe_PDR": 4,
}


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 54 * 54, 64), nn.ReLU(), nn.Linear(64, 4)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class CustomDataset(Dataset):
    """Pytorch custom dataloader."""

    def __init__(
        self,
        images: list | np.ndarray,
        labels: list | np.ndarray,
        image_transform: transforms.Compose,
    ):
        self.images = images  # Keep as list to handle different image sizes
        self.labels = np.array(labels)
        self.image_transform = image_transform

    def __len__(self) -> int:
        size = len(self.images)
        return size

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        label = self.labels[index]

        # Convert numpy array to PIL Image for transforms
        img_array = self.images[index]
        img_pil = Image.fromarray(img_array)
        transformed = self.image_transform(img_pil)
        if isinstance(transformed, Image.Image):
            transformed = transforms.ToTensor()(transformed)
        img_tensor = cast(Tensor, transformed)

        label = torch.tensor(label, dtype=torch.long)

        return img_tensor, label


def image_transform(
    size: tuple | list, augment: bool, arch: str = "efficientnet-b0"
) -> transforms.Compose:
    """Image transformation for training and validation."""
    if arch == "efficientnet-b0":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean, std = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

    data_transform = transforms.Compose(
        [
            transforms.Resize((size[0], size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return data_transform


def load_images(dataset_path: Path) -> tuple[list, list]:
    """Load images and labels from the dataset."""
    images = []
    labels = []
    image_paths = []

    data = pd.read_csv(dataset_path / "annotations.csv")[:30]

    dataset_path = Path(dataset_path)
    for row in tqdm(data.iterrows()):
        label = row[1]["class"]
        image_path = dataset_path / "images" / f"{row[1]['Image name']}.jpg"
        if image_path.exists():
            image_paths.append(image_path)
            image = Image.open(image_path).convert("RGB")
            image_array = np.array(image)
            images.append(image_array)
            labels.append(CLASSES_DICT[label])

    print(f"Loaded {len(images)} images from {dataset_path}.")
    return images, labels


def get_data_loader(
    dataset_path: Path, size: tuple | list, batch_size: int, augment: bool
) -> DataLoader:
    """Get data loader for the dataset."""
    # for faster training, we load data images first
    images, labels = load_images(dataset_path)

    data_transform = image_transform(size, augment)

    data = CustomDataset(images, labels, data_transform)

    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=5)
    return data_loader


def prepare_data_loaders(
    data_dir: Path, size: tuple | list, batch_size: int, augment: bool = True
) -> tuple[DataLoader, DataLoader]:
    """Prepare data loaders for training and validation."""
    train_dataset_path = data_dir / "train"
    val_dataset_path = data_dir / "val"
    train_loader = get_data_loader(train_dataset_path, size, batch_size, augment)
    val_loader = get_data_loader(val_dataset_path, size, batch_size, augment)
    return train_loader, val_loader


def get_efficient_net_data_transforms(img_size: int = 224) -> dict[str, Callable]:
    """Return data augmentation and normalization transforms."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
    }


def get_transforms(image_size: int = 224) -> transforms.Compose:
    """Get a torchvision transform pipeline for preprocessing images.

    Args:
        image_size (int): Size to which the image will be resized.

    Returns:
        transforms.Compose: Composed transform for preprocessing.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # Use 3-channel mean/std for RGB images
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def load_image(image_path: Path, image_size: int = 224) -> torch.Tensor:
    """Load and preprocess an image for model inference.

    Args:
        image_path (str): Path to the image file.
        image_size (int): Size to which the image will be resized.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    pil_image = Image.open(image_path).convert("RGB")
    transform = get_transforms(image_size)
    tensor_image = transform(pil_image)
    # Ensure the transformed result is a Tensor (some static analyzers may think it's a PIL Image)
    if isinstance(tensor_image, Image.Image):
        tensor_image = transforms.ToTensor()(tensor_image)
    tensor_image = tensor_image.unsqueeze(0)
    return tensor_image


def get_efficientnet_model(
    num_classes: int = 2, pretrained: bool = True, fine_tune_all: bool = False
) -> nn.Module:
    """Return EfficientNet-B0 with final layer adjusted for num_classes."""
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    # EfficientNet-B0's classifier is Sequential, last layer is Linear
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)  # type: ignore

    for param in model.parameters():
        param.requires_grad = fine_tune_all
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model


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
    wandb.init(
        project="retinal-desease-classification",
        name="retinal_cnn_training",
        config={"epochs": 10, "batch_size": 16, "learning_rate": 0.001, "model": "SimpleCNN"},
        notes="Training a SimpleCNN model on retinal images",
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
    for batch_inputs, batch_labels in dataloader:
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
        for batch_inputs, batch_labels in dataloader:
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
        for path in image_paths:
            img = load_image(Path(path), 224).to(device)
            outputs = model(img)
            _, pred = torch.max(outputs, 1)
            predictions.append(pred.item())
    return predictions


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
    train_loader = get_data_loader(data_dir, size=(224, 224), batch_size=batch_size, augment=True)
    print("Training data loaded successfully.")

    val_dir = data_dir.parent / "test"
    val_loader = get_data_loader(val_dir, size=(224, 224), batch_size=batch_size, augment=False)

    # Initialize the model
    print("Initializing model...")
    model = SimpleCNN().to(DEVICE)
    print(f"Using device: {DEVICE}")
    print(f"Number of training samples: {len(train_data)}")
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

    for epoch in range(epochs):
        model.train()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
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


if __name__ == "__main__":
    train_model(data_dir=Path("../../../../data/IDRiD/train"), epochs=10, batch_size=16, lr=0.001)
