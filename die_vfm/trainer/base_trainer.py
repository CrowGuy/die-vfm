"""Base trainer state definitions for die_vfm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainerState:
    """Minimal trainer state shared across trainer entrypoints.

    This state is intentionally small for M1. It is designed to remain
    checkpoint-friendly while covering the minimum fields needed by the
    existing bootstrap flow and the new Round1 frozen orchestration.

    Attributes:
        epoch: Number of completed epochs.
        global_step: Monotonic trainer step counter.
        best_metric_name: Name of the metric used for best-checkpoint
            selection.
        best_metric_value: Best value observed for the selection metric.
    """

    epoch: int = 0
    global_step: int = 0
    best_metric_name: Optional[str] = None
    best_metric_value: Optional[float] = None