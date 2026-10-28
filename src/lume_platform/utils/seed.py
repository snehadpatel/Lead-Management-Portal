"""Deterministic random seed utility for reproducible ML/DL experiments."""

from __future__ import annotations

import os
import random
import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Set seeds across Python, NumPy, and PyTorch for full reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Seed MPS if supported
            torch.mps.manual_seed(seed)
    except ImportError:
        pass


set_seed = seed_everything

