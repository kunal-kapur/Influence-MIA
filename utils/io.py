from pathlib import Path
import numpy as np
import torch


def save_model(model, path):
    """Save model state dict to path, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path, device):
    """Load state dict from path onto device and return model."""
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def save_array(arr, path):
    """Save numpy array to path, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def load_array(path):
    """Load and return a numpy array from path."""
    return np.load(path)
