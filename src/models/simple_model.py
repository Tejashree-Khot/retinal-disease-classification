"""A simple CNN model for image classification."""

from torch import nn


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
            nn.Flatten(), nn.Linear(32 * 54 * 54, 64), nn.ReLU(), nn.Linear(64, 5)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
