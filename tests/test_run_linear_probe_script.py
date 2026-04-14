"""Tests for scripts.run_linear_probe."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from die_vfm.artifacts.embedding_artifact import (
    EmbeddingManifest,
    EmbeddingShardInfo,
)


def _write_embedding_split(
    split_dir: Path,
    split_name: str,
    embeddings: torch.Tensor,
    labels: torch.Tensor | None,
    image_ids: list[str] | None = None,
    metadata: list[dict[str, Any]] | None = None,
) -> None:
    """Writes a minimal single-shard embedding artifact split for tests."""
    split_dir.mkdir(parents=True, exist_ok=True)

    num_samples = int(embeddings.shape[0])

    if image_ids is None:
        image_ids = [f"{split_name}_{index:04d}" for index in range(num_samples)]
    if metadata is None:
        metadata = [{"split": split_name, "index": index} for index in range(num_samples)]

    payload = {
        "embeddings": embeddings,
        "labels": labels,
        "image_ids": image_ids,
        "metadata": metadata,
    }
    torch.save(payload, split_dir / "part-00000.pt")

    manifest = EmbeddingManifest(
        split=split_name,
        num_samples=num_samples,
        embedding_dim=int(embeddings.shape[1]),
        dtype=str(embeddings.dtype).replace("torch.", ""),
        has_labels=labels is not None,
        num_shards=1,
        shards=[
            EmbeddingShardInfo(
                file_name="part-00000.pt",
                num_samples=num_samples,
            )
        ],
    )
    manifest.save_yaml(split_dir / "manifest.yaml")


def _make_artifacts(root_dir: Path) -> tuple[Path, Path]:
    """Creates easy train/val embedding artifacts for script tests."""
    train_dir = root_dir / "train"
    val_dir = root_dir / "val"

    train_embeddings = torch.tensor(
        [
            [-3.0, -2.5],
            [-2.5, -2.0],
            [-2.0, -3.0],
            [-1.5, -2.2],
            [2.0, 2.5],
            [2.5, 2.0],
            [3.0, 2.2],
            [1.8, 2.8],
        ],
        dtype=torch.float32,
    )
    train_labels = torch.tensor([10, 10, 10, 10, 20, 20, 20, 20], dtype=torch.long)

    val_embeddings = torch.tensor(
        [
            [-2.7, -2.1],
            [-1.8, -2.6],
            [2.2, 2.1],
            [2.9, 2.4],
        ],
        dtype=torch.float32,
    )
    val_labels = torch.tensor([10, 10, 20, 20], dtype=torch.long)

    _write_embedding_split(train_dir, "train", train_embeddings, train_labels)
    _write_embedding_split(val_dir, "val", val_embeddings, val_labels)

    return train_dir, val_dir


def _run_script(repo_root: Path, overrides: list[str]) -> subprocess.CompletedProcess[str]:
    """Runs the linear probe script with Hydra overrides."""
    script_path = repo_root / "scripts" / "run_linear_probe.py"

    command = [sys.executable, str(script_path), *overrides]
    return subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


def test_run_linear_probe_script_noops_when_disabled(tmp_path: Path) -> None:
    """Exits cleanly when evaluation.linear_probe.enabled is false."""
    repo_root = Path(__file__).resolve().parents[1]

    completed = _run_script(
        repo_root=repo_root,
        overrides=[
            "evaluation.linear_probe.enabled=false",
        ],
    )

    assert completed.returncode == 0
    assert "Linear probe evaluation is disabled. Skipping." in completed.stdout


def test_run_linear_probe_script_end_to_end(tmp_path: Path) -> None:
    """Runs the script end to end and writes expected outputs."""
    repo_root = Path(__file__).resolve().parents[1]
    train_dir, val_dir = _make_artifacts(tmp_path / "embeddings")
    output_dir = tmp_path / "outputs"
    hydra_run_dir = tmp_path / "hydra_run"

    completed = _run_script(
        repo_root=repo_root,
        overrides=[
            "evaluation.linear_probe.enabled=true",
            f"evaluation.linear_probe.input.train_split_dir={train_dir}",
            f"evaluation.linear_probe.input.val_split_dir={val_dir}",
            f"evaluation.linear_probe.output.output_dir={output_dir}",
            "evaluation.linear_probe.input.normalize_embeddings=false",
            "evaluation.linear_probe.model.bias=true",
            "evaluation.linear_probe.trainer.batch_size=4",
            "evaluation.linear_probe.trainer.num_epochs=20",
            "evaluation.linear_probe.trainer.learning_rate=0.05",
            "evaluation.linear_probe.trainer.optimizer_name=adamw",
            "evaluation.linear_probe.trainer.weight_decay=0.0",
            "evaluation.linear_probe.trainer.device=cpu",
            "evaluation.linear_probe.trainer.seed=123",
            "evaluation.linear_probe.trainer.selection_metric=val_accuracy",
            "evaluation.linear_probe.output.save_predictions=true",
            "evaluation.linear_probe.output.save_history=true",
            f"hydra.run.dir={hydra_run_dir}",
        ],
    )

    assert completed.returncode == 0, completed.stderr
    assert "Linear probe evaluation completed successfully." in completed.stdout
    assert f"Output directory: {output_dir}" in completed.stdout
    assert "Validation metrics:" in completed.stdout
    assert "Written files:" in completed.stdout
    assert "Resolved Hydra config saved to:" in completed.stdout

    metrics_path = output_dir / "metrics.yaml"
    summary_path = output_dir / "summary.yaml"
    config_path = output_dir / "config.yaml"
    history_path = output_dir / "history.yaml"
    predictions_path = output_dir / "predictions.pt"
    hydra_config_path = hydra_run_dir / "hydra_linear_probe_config.yaml"

    assert metrics_path.exists()
    assert summary_path.exists()
    assert config_path.exists()
    assert history_path.exists()
    assert predictions_path.exists()
    assert hydra_config_path.exists()

    with metrics_path.open("r", encoding="utf-8") as file:
        metrics_payload = yaml.safe_load(file)

    with summary_path.open("r", encoding="utf-8") as file:
        summary_payload = yaml.safe_load(file)

    with config_path.open("r", encoding="utf-8") as file:
        config_payload = yaml.safe_load(file)

    with history_path.open("r", encoding="utf-8") as file:
        history_payload = yaml.safe_load(file)

    predictions_payload = torch.load(predictions_path)

    assert metrics_payload["evaluator_type"] == "linear_probe"
    assert metrics_payload["input"]["train_split"] == "train"
    assert metrics_payload["input"]["val_split"] == "val"
    assert metrics_payload["input"]["train_num_samples"] == 8
    assert metrics_payload["input"]["val_num_samples"] == 4
    assert metrics_payload["input"]["embedding_dim"] == 2
    assert metrics_payload["input"]["num_classes"] == 2
    assert metrics_payload["input"]["class_ids"] == [10, 20]
    assert metrics_payload["best_epoch"] >= 1
    assert metrics_payload["train"]["loss"] >= 0.0
    assert metrics_payload["val"]["loss"] >= 0.0
    assert 0.0 <= metrics_payload["train"]["accuracy"] <= 1.0
    assert 0.0 <= metrics_payload["val"]["accuracy"] <= 1.0

    assert summary_payload["status"] == "success"
    assert summary_payload["evaluator"] == "linear_probe"
    assert summary_payload["output_dir"] == str(output_dir)
    assert summary_payload["val_accuracy"] == metrics_payload["val"]["accuracy"]
    assert summary_payload["val_loss"] == metrics_payload["val"]["loss"]

    assert config_payload["input"]["train_split_dir"] == str(train_dir)
    assert config_payload["input"]["val_split_dir"] == str(val_dir)
    assert config_payload["output"]["output_dir"] == str(output_dir)
    assert config_payload["trainer"]["optimizer_name"] == "adamw"
    assert config_payload["trainer"]["learning_rate"] == 0.05
    assert config_payload["trainer"]["num_epochs"] == 20
    assert config_payload["trainer"]["batch_size"] == 4
    assert config_payload["trainer"]["seed"] == 123

    assert "epochs" in history_payload
    assert len(history_payload["epochs"]) == 20

    assert predictions_payload["split"] == "val"
    assert len(predictions_payload["image_ids"]) == 4
    assert predictions_payload["labels"].shape == (4,)
    assert predictions_payload["pred_labels"].shape == (4,)
    assert predictions_payload["logits"].shape == (4, 2)
    assert torch.equal(
        predictions_payload["class_ids"],
        torch.tensor([10, 20], dtype=torch.long),
    )


def test_run_linear_probe_script_skips_optional_outputs(tmp_path: Path) -> None:
    """Does not write predictions/history when disabled."""
    repo_root = Path(__file__).resolve().parents[1]
    train_dir, val_dir = _make_artifacts(tmp_path / "embeddings")
    output_dir = tmp_path / "outputs"
    hydra_run_dir = tmp_path / "hydra_run"

    completed = _run_script(
        repo_root=repo_root,
        overrides=[
            "evaluation.linear_probe.enabled=true",
            f"evaluation.linear_probe.input.train_split_dir={train_dir}",
            f"evaluation.linear_probe.input.val_split_dir={val_dir}",
            f"evaluation.linear_probe.output.output_dir={output_dir}",
            "evaluation.linear_probe.trainer.batch_size=4",
            "evaluation.linear_probe.trainer.num_epochs=10",
            "evaluation.linear_probe.trainer.learning_rate=0.05",
            "evaluation.linear_probe.trainer.optimizer_name=adamw",
            "evaluation.linear_probe.trainer.weight_decay=0.0",
            "evaluation.linear_probe.trainer.device=cpu",
            "evaluation.linear_probe.trainer.seed=1",
            "evaluation.linear_probe.output.save_predictions=false",
            "evaluation.linear_probe.output.save_history=false",
            f"hydra.run.dir={hydra_run_dir}",
        ],
    )

    assert completed.returncode == 0, completed.stderr

    assert (output_dir / "metrics.yaml").exists()
    assert (output_dir / "summary.yaml").exists()
    assert (output_dir / "config.yaml").exists()
    assert not (output_dir / "history.yaml").exists()
    assert not (output_dir / "predictions.pt").exists()
    assert (hydra_run_dir / "hydra_linear_probe_config.yaml").exists()
