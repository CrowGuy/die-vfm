from __future__ import annotations

import datetime as dt
import pathlib


def create_run_dir(output_root: str, run_name: str | None = None) -> pathlib.Path:
    """Creates and returns the run directory path.

    Args:
      output_root: Root directory for experiment runs.
      run_name: Optional user-provided run name.

    Returns:
      Path to the created run directory.
    """
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_name = run_name or f"run_{timestamp}"
    run_dir = pathlib.Path(output_root) / final_name

    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    (run_dir / "embeddings").mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(parents=True, exist_ok=False)

    return run_dir