"""Training configuration dataclass."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainerConfig:
    """Configuration for model training."""

    model_name: str
    num_classes: int
    train_path: Path
    val_path: Path
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    scheduler: str = "cosine"
    early_stopping_patience: int = 10
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    use_wandb: bool = True
    wandb_project: str = "retinal-classification"
    device: str = "auto"
    use_weighted_sampler: bool = True
    pretrained: bool = True
