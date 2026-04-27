"""Helpers for deterministic, reproducible inference runs."""

from __future__ import annotations

import logging
import os
import random
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def set_global_seeds(seed: Optional[int], deterministic_torch: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) RNGs.

    Note: Monte Carlo Dropout is intentionally stochastic. Seeding here only
    fixes the *starting* state so that two runs with the same input produce
    the same MC samples; it does not disable dropout sampling.

    Args:
        seed: Integer seed. If None, this function is a no-op.
        deterministic_torch: If True, also enable cuDNN deterministic flags.
            Slower; useful for strict reproducibility benchmarks.
    """
    if seed is None:
        return

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # local import to keep this module light

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        logger.debug("torch not available; skipped torch seeding")

    logger.info("Global RNG seeded with %d (deterministic_torch=%s)", seed, deterministic_torch)
