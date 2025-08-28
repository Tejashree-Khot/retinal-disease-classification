"""
Train a model with gradual unfreezing of layers.

This script demonstrates how to use the gradual unfreezing technique to train an
EfficientNet model on retinal images. The approach:

1. First trains only the classifier layers (keep feature extractor frozen)
2. Then gradually unfreezes feature blocks from bottom to top
3. For each unfreezing stage, trains for a specified number of epochs

This approach often leads to better performance as it prevents catastrophic forgetting
of the pre-trained weights and allows the model to adapt more gradually.
"""

import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataloader.data_loader import get_data_loader
from models.efficient_net import get_efficientnet_model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    print("Training:")
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Print progress every 10 batches
        if batch_idx % 10 == 0:
            current_loss = running_loss / total if total > 0 else 0
            current_acc = 100.0 * correct / total if total > 0 else 0
            print(
                f"\rBatch {batch_idx}/{len(dataloader)}, "
                f"Loss: {current_loss:.4f}, Acc: {current_acc:.2f}%",
                end="",
                flush=True,
            )

    print()  # New line after the epoch
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    print("Validating:")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                current_loss = running_loss / total if total > 0 else 0
                current_acc = 100.0 * correct / total if total > 0 else 0
                print(
                    f"\rBatch {batch_idx}/{len(dataloader)}, "
                    f"Val Loss: {current_loss:.4f}, Val Acc: {current_acc:.2f}%",
                    end="",
                    flush=True,
                )

    print()  # New line after validation
    val_loss = running_loss / total
    val_acc = 100.0 * correct / total
    return val_loss, val_acc


def init_wandb(args):
    """Initialize Weights & Biases for experiment tracking."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"EfficientNet_B0_gradual_unfreeze_{timestamp}"

    # Initialize wandb with configuration
    wandb.init(
        project="retinal-disease-classification",
        name=experiment_name,
        config={
            "epochs_per_stage": args.epochs_per_stage,
            "stages": args.stages,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "model": "EfficientNet-B0",
            "num_classes": args.num_classes,
            "optimization": "Gradual Unfreezing",
        },
        notes="Training EfficientNet with gradual unfreezing of layers",
    )

    # Define custom charts
    wandb.define_metric("epoch")
    wandb.define_metric("stage")

    # Group train metrics
    wandb.define_metric("train/*", step_metric="epoch")

    # Group validation metrics
    wandb.define_metric("val/*", step_metric="epoch")

    # Create custom panels for gradual unfreezing visualization
    wandb.run.log_code(".")  # Log code snapshot

    print("Weights & Biases initialized.")


def main():
    parser = argparse.ArgumentParser(description="Train with gradual unfreezing")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-classes", type=int, default=5, help="Number of classes")
    parser.add_argument(
        "--epochs-per-stage",
        type=int,
        default=3,
        help="Number of epochs to train at each unfreezing stage",
    )
    parser.add_argument(
        "--stages",
        type=int,
        default=4,
        help="Number of unfreezing stages (0=classifier only, 1=classifier+last block, etc)",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda or cpu, default: auto-detect)"
    )
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases for logging")
    args = parser.parse_args()

    # Initialize wandb if requested
    if args.use_wandb:
        init_wandb(args)

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get data loaders
    train_loader = get_data_loader(
        Path(args.data_dir) / "train", batch_size=args.batch_size, shuffle=True
    )
    val_loader = get_data_loader(
        Path(args.data_dir) / "val", batch_size=args.batch_size, shuffle=False
    )

    # Get model and unfreezing function
    model, unfreeze_layers = get_efficientnet_model(num_classes=args.num_classes, pretrained=True)
    model = model.to(device)

    # If using wandb, log model architecture
    if args.use_wandb:
        wandb.watch(model, log="all")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Train with gradual unfreezing
    best_val_acc = 0.0

    for stage in range(args.stages + 1):  # +1 because we start with stage 0 (classifier only)
        print(f"\n{'=' * 50}")
        print(f"Training Stage {stage}: ", end="")

        # Unfreeze layers for this stage
        unfreeze_layers(stage)

        # Count trainable parameters for this stage
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_percentage = trainable_params / total_params * 100

        # Log unfreezing stats to wandb
        if args.use_wandb:
            wandb.log(
                {
                    "unfreezing/trainable_params": trainable_params,
                    "unfreezing/total_params": total_params,
                    "unfreezing/trainable_percentage": trainable_percentage,
                    "unfreezing/stage": stage,
                }
            )

        # Optimizer - we create a new optimizer for each stage
        # Use a smaller learning rate for later stages
        current_lr = args.lr / (2**stage) if stage > 0 else args.lr
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=current_lr)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=args.epochs_per_stage, eta_min=current_lr / 10
        )

        # Train for specified number of epochs
        for epoch in range(args.epochs_per_stage):
            global_epoch = stage * args.epochs_per_stage + epoch
            print(
                f"\nEpoch {epoch + 1}/{args.epochs_per_stage} - Stage {stage} - LR: {scheduler.get_last_lr()[0]:.1e}"
            )

            # Train one epoch
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

            # Validate
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Log metrics to wandb
            if args.use_wandb:
                # Create metrics dictionary
                metrics = {
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "stage": stage,
                    "unfrozen_blocks": stage,
                    "epoch": global_epoch,
                }

                # Log to wandb
                wandb.log(metrics)

            # Update learning rate
            scheduler.step()

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_save_path = "best_model_gradual_unfreeze.pth"
                torch.save(model.state_dict(), model_save_path)
                print(f"Saved new best model with val_acc: {val_acc:.2f}%")

                # Log best model to wandb
                if args.use_wandb:
                    wandb.save(model_save_path)
                    wandb.run.summary["best_val_accuracy"] = val_acc
                    wandb.run.summary["best_val_loss"] = val_loss
                    wandb.run.summary["best_model_stage"] = stage
                    wandb.run.summary["best_model_epoch"] = global_epoch

    print("\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    # Finish wandb run
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
