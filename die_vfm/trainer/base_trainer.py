from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainerState:
    """Minimal trainer state placeholder for future extensions."""

    epoch: int = 0
    global_step: int = 0