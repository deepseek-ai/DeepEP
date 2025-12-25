import torch

from .utils import EventOverlap
from .buffer import Buffer

# noinspection PyUnresolvedReferences
from deep_ep_cpp import Config, topk_idx_t

# Package version - matches setup.py
__version__ = "1.2.1"

# Public API exports
__all__ = [
    # Core classes
    "Buffer",
    "EventOverlap",
    "Config",
    # Types
    "topk_idx_t",
    # Version
    "__version__",
]
