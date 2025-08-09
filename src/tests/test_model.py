# tests/test_model.py
from models.simple_model import SimpleCNN


def test_simple_cnn():
    """Test the SimpleCNN model."""
    # Initialize the model
    model = SimpleCNN()
    assert model is not None, "Model initialization failed"
    assert isinstance(model, SimpleCNN), "Model is not an instance of SimpleCNN"
    assert len(model.conv) > 0, "Convolutional layers are not defined"
    assert len(model.fc) > 0, "Fully connected layers are not defined"
