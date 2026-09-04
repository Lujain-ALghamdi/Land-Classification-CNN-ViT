"""Deterministic seeding helpers.

The original labs seeded each framework ad hoc in separate cells. This module
centralises it so a single call makes a run as reproducible as the framework
allows. Imports of TensorFlow and PyTorch are deferred so that seeding the one
framework you actually use does not force the other to load.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 7331, *, deterministic_torch: bool = True) -> int:
    """Seed Python, NumPy and (if importable) TensorFlow and PyTorch.

    Parameters
    ----------
    seed:
        The seed applied to every RNG.
    deterministic_torch:
        When ``True`` also switch cuDNN into deterministic mode. This trades a
        little GPU throughput for run-to-run reproducibility.

    Returns
    -------
    int
        The seed that was applied (handy for logging).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # TensorFlow is optional at import time.
        import tensorflow as tf

        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass

    try:  # PyTorch is optional at import time.
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed


def torch_worker_init_fn(worker_id: int, base_seed: int = 7331) -> None:
    """Re-seed a PyTorch ``DataLoader`` worker so worker RNGs do not collide."""
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    try:
        import torch

        torch.manual_seed(worker_seed)
    except ImportError:
        pass
