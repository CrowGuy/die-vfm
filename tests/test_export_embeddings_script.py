"""Tests for scripts.export_embeddings."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
import yaml


def _run_script(repo_root: Path, overrides: list[str]) -> subprocess.CompletedProcess[str]:
    """Runs the export_embeddings script with Hydra overrides."""
    script_path = repo_root / "scripts" / "export_embeddings.py"

    command = [sys.executable, str(script_path), *overrides]
    return subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


def test_export_embeddings_script_noops_when_disabled(tmp_path: Path) -> None:
    """Exits cleanly and skips artifact export when disabled."""
    repo_root = Path(__file__).resolve().parents[1]
    run_name = "pytest-export-disabled"

    completed = _run_script(
        repo_root=repo_root,
        overrides=[
            f"run.output_root={tmp_path}",
            f"run.run_name={run_name}",
            "artifact.embedding.enabled=false",
        ],
    )

    assert completed.returncode == 0, completed.stderr

    run_dir = tmp_path / run_name
    log_path = run_dir / "logs" / "export_embeddings.log"
    config_path = run_dir / "config.yaml"

    assert run_dir.exists()
    assert log_path.exists()
    assert config_path.exists()
    assert not (run_dir / "embeddings").exists()

    log_text = log_path.read_text(encoding="utf-8")
    assert "Embedding artifact export disabled; skipping." in log_text
    assert "No embedding splits were exported." in log_text


def test_export_embeddings_script_writes_train_and_val_artifacts(
    tmp_path: Path,
) -> None:
    """Runs the script end to end and writes train/val embedding artifacts."""
    repo_root = Path(__file__).resolve().parents[1]
    run_name = "pytest-export-success"

    completed = _run_script(
        repo_root=repo_root,
        overrides=[
            f"run.output_root={tmp_path}",
            f"run.run_name={run_name}",
            "system.device=cpu",
            "system.num_workers=0",
            "model/backbone=dummy",
            "model/pooler=mean",
            "dataset=dummy",
        ],
    )

    assert completed.returncode == 0, completed.stderr

    run_dir = tmp_path / run_name
    log_path = run_dir / "logs" / "export_embeddings.log"
    config_path = run_dir / "config.yaml"
    train_dir = run_dir / "embeddings" / "train"
    val_dir = run_dir / "embeddings" / "val"

    assert run_dir.exists()
    assert log_path.exists()
    assert config_path.exists()
    assert train_dir.exists()
    assert val_dir.exists()

    train_manifest_path = train_dir / "manifest.yaml"
    val_manifest_path = val_dir / "manifest.yaml"
    train_shard_path = train_dir / "part-00000.pt"
    val_shard_path = val_dir / "part-00000.pt"

    assert train_manifest_path.exists()
    assert val_manifest_path.exists()
    assert train_shard_path.exists()
    assert val_shard_path.exists()

    with train_manifest_path.open("r", encoding="utf-8") as file:
        train_manifest = yaml.safe_load(file)
    with val_manifest_path.open("r", encoding="utf-8") as file:
        val_manifest = yaml.safe_load(file)

    train_payload = torch.load(train_shard_path)
    val_payload = torch.load(val_shard_path)

    assert train_manifest["split"] == "train"
    assert train_manifest["num_shards"] == 1
    assert train_manifest["num_samples"] > 0
    assert train_manifest["embedding_dim"] > 0

    assert val_manifest["split"] == "val"
    assert val_manifest["num_shards"] == 1
    assert val_manifest["num_samples"] > 0
    assert val_manifest["embedding_dim"] == train_manifest["embedding_dim"]

    assert train_payload["embeddings"].shape[0] == train_manifest["num_samples"]
    assert val_payload["embeddings"].shape[0] == val_manifest["num_samples"]
    assert len(train_payload["image_ids"]) == train_manifest["num_samples"]
    assert len(val_payload["image_ids"]) == val_manifest["num_samples"]
    assert len(train_payload["metadata"]) == train_manifest["num_samples"]
    assert len(val_payload["metadata"]) == val_manifest["num_samples"]
    assert train_payload["labels"] is not None
    assert val_payload["labels"] is not None

    log_text = log_path.read_text(encoding="utf-8")
    assert "Starting embedding export." in log_text
    assert "Export completed successfully for splits: train, val" in log_text
    assert "Embedding export job completed successfully." in log_text
