"""Early stopping callback for training."""


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value: float | None = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """Check if training should stop."""
        if self.best_value is None:
            self.best_value = value
            return False

        if self._is_improvement(value):
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def _is_improvement(self, value: float) -> bool:
        """Check if value is an improvement over best."""
        if self.mode == "max":
            return value > self.best_value + self.min_delta
        return value < self.best_value - self.min_delta

    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.best_value = None
        self.should_stop = False
