from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field


@dataclass(frozen=True)
class ResumeConfig:
    """Configuration for checkpoint resume behavior."""

    enabled: bool = False
    mode: str = "full_resume"
    checkpoint_path: str | None = None
    auto_resume_latest: bool = False


@dataclass(frozen=True)
class TrainConfig:
    """Training configuration."""

    max_epochs: int = 1
    log_every_n_steps: int = 10
    resume: ResumeConfig = field(default_factory=ResumeConfig)


@dataclass(frozen=True)
class RunConfig:
    """Run directory configuration."""

    output_root: str = "runs"
    run_name: str | None = None
    save_config_snapshot: bool = True