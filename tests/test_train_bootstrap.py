from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_train_bootstrap_runs_dataloader_smoke_test(tmp_path: Path) -> None:
    """Tests that the training entrypoint completes bootstrap successfully."""
    run_name = "pytest-train-bootstrap"

    command = [
        sys.executable,
        "scripts/train.py",
        f"run.output_root={tmp_path}",
        f"run.run_name={run_name}",
        "system.num_workers=0",
        "train.run_dataloader_smoke_test=true",
    ]

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"train.py failed with stderr:\n{result.stderr}\n"
        f"stdout:\n{result.stdout}"
    )

    run_dir = tmp_path / run_name
    assert run_dir.exists()
    assert (run_dir / "config.yaml").exists()

    log_path = run_dir / "logs" / "run.log"
    assert log_path.exists()

    log_text = log_path.read_text(encoding="utf-8")
    assert "Starting training bootstrap." in log_text
    assert "Dataloader smoke test passed." in log_text
    assert "Batch image shape: (4, 3, 224, 224)" in log_text
    assert "Batch label shape: (4,)" in log_text
    assert "Training bootstrap completed successfully." in log_text

    dataset_metadata_path = run_dir / "dataset_metadata.yaml"
    assert dataset_metadata_path.exists()

    dataset_metadata_text = dataset_metadata_path.read_text(encoding="utf-8")
    assert "dataset_name: dummy" in dataset_metadata_text
    assert "split: train" in dataset_metadata_text
    assert "num_samples: 16" in dataset_metadata_text

    assert "Dataset metadata:" in log_text