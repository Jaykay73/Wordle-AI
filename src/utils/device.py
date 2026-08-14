"""
Device Detection Utility for CPU/CUDA environments.
"""

import torch


def get_device() -> torch.device:
    """Returns CUDA device if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
