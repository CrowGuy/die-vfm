from __future__ import annotations

import random

import torch


def set_global_seed(seed: int) -> None:
    """Sets the global random seed for supported libraries.

    Args:
      seed: Random seed value.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)