"""
Training & Synthetic Dataset Generation Module.
Provides synthetic trajectory generators, dataset wrappers, and training pipelines.
"""

from .generation import TrajectoryGenerator, TrajectorySample
from .dataset import WordleTrajectoryDataset, create_dataloaders

__all__ = [
    "TrajectoryGenerator",
    "TrajectorySample",
    "WordleTrajectoryDataset",
    "create_dataloaders",
]
