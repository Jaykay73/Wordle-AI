"""
Utilities package init.
"""

from .device import get_device
from .config import set_seed, load_config

__all__ = ["get_device", "set_seed", "load_config"]
