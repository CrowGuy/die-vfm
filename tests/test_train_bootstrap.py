from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_train_script_bootstraps(tmp_path) -> None:
    """Tests that the training bootstrap script runs successfully."""
    command = [
        sys.executable,
        "scripts/train.py",
        f"run.output_root={tmp_path}",
        "run.run_name=test_bootstrap",
    ]

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    run_dir = Path(tmp_path) / "test_bootstrap"
    assert run_dir.exists()
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "logs" / "run.log").exists()