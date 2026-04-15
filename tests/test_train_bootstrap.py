from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_train_bootstrap_runs_dataloader_and_model_smoke_test(
    tmp_path: Path,
) -> None:
    """Tests that the training entrypoint completes PR-3 bootstrap successfully."""
    run_name = "pytest-train-bootstrap"

    command = [
        sys.executable,
        "scripts/run.py",
        f"run.output_root={tmp_path}",
        f"run.run_name={run_name}",
        "system.num_workers=0",
        "system.device=cpu",
        "model/backbone=dummy",
        "model/pooler=mean",
        "dataset=dummy",
        "train.run_dataloader_smoke_test=true",
        "train.run_model_forward_smoke_test=true",
    ]

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"run.py failed with stderr:\n{result.stderr}\n"
        f"stdout:\n{result.stdout}"
    )

    run_dir = tmp_path / run_name
    assert run_dir.exists()
    assert (run_dir / "config.yaml").exists()

    log_path = run_dir / "logs" / "run.log"
    assert log_path.exists()

    log_text = log_path.read_text(encoding="utf-8")
    assert "Starting runtime entrypoint." in log_text
    assert "Starting bootstrap runtime." in log_text
    assert "Dataloader smoke test passed." in log_text
    assert "Batch image shape: (4, 3, 224, 224)" in log_text
    assert "Batch label shape: (4,)" in log_text

    assert "Built model: DieVFMModel" in log_text
    assert "Backbone: DummyBackbone" in log_text
    assert "Pooler: MeanPooler" in log_text
    assert "Model forward completed." in log_text
    assert "Embedding shape: (4, 192)" in log_text

    assert "Training bootstrap completed successfully." in log_text
    assert "Dataset metadata:" in log_text

    dataset_metadata_path = run_dir / "dataset_metadata.yaml"
    assert dataset_metadata_path.exists()

    dataset_metadata_text = dataset_metadata_path.read_text(encoding="utf-8")
    assert "dataset_name: dummy" in dataset_metadata_text
    assert "split: train" in dataset_metadata_text
    assert "num_samples: 16" in dataset_metadata_text

    model_smoke_path = run_dir / "model_smoke.yaml"
    assert model_smoke_path.exists()

    model_smoke_text = model_smoke_path.read_text(encoding="utf-8")
    assert "name: DieVFMModel" in model_smoke_text
    assert "embedding_dim: 192" in model_smoke_text
    assert "backbone: DummyBackbone" in model_smoke_text
    assert "pooler: MeanPooler" in model_smoke_text
    assert "- 4" in model_smoke_text
    assert "- 192" in model_smoke_text

def test_train_bootstrap_writes_checkpoint_set(tmp_path: Path) -> None:
  """Tests that bootstrap writes latest/best/epoch checkpoints."""
  run_name = "pytest-train-checkpoint-save"
  command = [
      sys.executable,
      "scripts/run.py",
      f"run.output_root={tmp_path}",
      f"run.run_name={run_name}",
      "system.num_workers=0",
      "system.device=cpu",
      "model/backbone=dummy",
      "model/pooler=mean",
      "dataset=dummy",
      "train.run_dataloader_smoke_test=true",
  ]

  result = subprocess.run(
      command,
      check=False,
      capture_output=True,
      text=True,
  )

  assert result.returncode == 0, (
      f"run.py failed with stderr:\n{result.stderr}\n"
      f"stdout:\n{result.stdout}"
  )

  checkpoint_dir = tmp_path / run_name / "checkpoints"
  assert checkpoint_dir.exists()
  assert (checkpoint_dir / "latest.pt").exists()
  assert (checkpoint_dir / "best.pt").exists()
  assert (checkpoint_dir / "epoch_0000.pt").exists()


def test_train_bootstrap_auto_resume_latest_full_resume(
    tmp_path: Path,
) -> None:
  """Tests that bootstrap can auto-resume from latest.pt."""
  run_name = "pytest-train-auto-resume-full"

  first_command = [
      sys.executable,
      "scripts/run.py",
      f"run.output_root={tmp_path}",
      f"run.run_name={run_name}",
      "system.num_workers=0",
      "system.device=cpu",
      "model/backbone=dummy",
      "model/pooler=mean",
      "dataset=dummy",
      "train.run_dataloader_smoke_test=true",
  ]
  first_result = subprocess.run(
      first_command,
      check=False,
      capture_output=True,
      text=True,
  )

  assert first_result.returncode == 0, (
      f"first run failed with stderr:\n{first_result.stderr}\n"
      f"stdout:\n{first_result.stdout}"
  )

  second_command = [
      sys.executable,
      "scripts/run.py",
      f"run.output_root={tmp_path}",
      f"run.run_name={run_name}",
      "system.num_workers=0",
      "system.device=cpu",
      "model/backbone=dummy",
      "model/pooler=mean",
      "dataset=dummy",
      "train.run_dataloader_smoke_test=true",
      "train.resume.enabled=true",
      "train.resume.mode=full_resume",
      "train.resume.auto_resume_latest=true",
      "train.resume.checkpoint_path=null",
  ]
  second_result = subprocess.run(
      second_command,
      check=False,
      capture_output=True,
      text=True,
  )

  assert second_result.returncode == 0, (
      f"second run failed with stderr:\n{second_result.stderr}\n"
      f"stdout:\n{second_result.stdout}"
  )

  log_path = tmp_path / run_name / "logs" / "run.log"
  log_text = log_path.read_text(encoding="utf-8")

  assert "Resolved resume checkpoint:" in log_text
  assert "Resume mode: full_resume" in log_text
  assert "Full resume completed." in log_text

  checkpoint_dir = tmp_path / run_name / "checkpoints"
  assert (checkpoint_dir / "latest.pt").exists()
  assert (checkpoint_dir / "epoch_0000.pt").exists()


def test_train_bootstrap_explicit_warm_start_checkpoint(
        tmp_path: Path,
    ) -> None:
    """Tests that bootstrap can warm start from an explicit checkpoint."""
    source_run_name = "pytest-train-warm-start-source"
    target_run_name = "pytest-train-warm-start-target"

    source_command = [
        sys.executable,
        "scripts/run.py",
        f"run.output_root={tmp_path}",
        f"run.run_name={source_run_name}",
        "system.num_workers=0",
        "system.device=cpu",
        "model/backbone=dummy",
        "model/pooler=mean",
        "dataset=dummy",
        "train.run_dataloader_smoke_test=true",
    ]
    source_result = subprocess.run(
        source_command,
        check=False,
        capture_output=True,
        text=True,
    )

    assert source_result.returncode == 0, (
        f"source run failed with stderr:\n{source_result.stderr}\n"
        f"stdout:\n{source_result.stdout}"
    )

    checkpoint_path = (
        tmp_path / source_run_name / "checkpoints" / "latest.pt"
    )
    assert checkpoint_path.exists()

    target_command = [
        sys.executable,
        "scripts/run.py",
        f"run.output_root={tmp_path}",
        f"run.run_name={target_run_name}",
        "system.num_workers=0",
        "system.device=cpu",
        "model/backbone=dummy",
        "model/pooler=mean",
        "dataset=dummy",
        "train.run_dataloader_smoke_test=true",
        "train.resume.enabled=true",
        "train.resume.mode=warm_start",
        f"train.resume.checkpoint_path={checkpoint_path}",
        "train.resume.auto_resume_latest=false",
    ]
    target_result = subprocess.run(
        target_command,
        check=False,
        capture_output=True,
        text=True,
    )

    assert target_result.returncode == 0, (
        f"target run failed with stderr:\n{target_result.stderr}\n"
        f"stdout:\n{target_result.stdout}"
    )

    log_path = tmp_path / target_run_name / "logs" / "run.log"
    log_text = log_path.read_text(encoding="utf-8")

    assert "Resolved resume checkpoint:" in log_text
    assert "Resume mode: warm_start" in log_text
    assert "Warm start completed from checkpoint." in log_text

    checkpoint_dir = tmp_path / target_run_name / "checkpoints"
    assert (checkpoint_dir / "latest.pt").exists()
    assert (checkpoint_dir / "best.pt").exists()
    assert (checkpoint_dir / "epoch_0000.pt").exists()
