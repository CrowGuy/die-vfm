"""Tests for die_vfm.evaluator.linear_probe_runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from die_vfm.artifacts.embedding_artifact import (
    EmbeddingManifest,
    EmbeddingShardInfo,
)
from die_vfm.evaluator.linear_probe_runner import (
    LinearProbeInputConfig,
    LinearProbeModelConfig,
    LinearProbeOutputConfig,
    LinearProbeRunConfig,
    LinearProbeRunResult,
    build_linear_probe_run_config,
    resolve_linear_probe_run_config,
    run_linear_probe,
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
    """Creates easy train/val embedding artifacts for runner tests."""
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

    _write_embedding_split(
        split_dir=train_dir,
        split_name="train",
        embeddings=train_embeddings,
        labels=train_labels,
    )
    _write_embedding_split(
        split_dir=val_dir,
        split_name="val",
        embeddings=val_embeddings,
        labels=val_labels,
    )

    return train_dir, val_dir


def test_build_linear_probe_run_config_builds_expected_structure(
    tmp_path: Path,
) -> None:
    """Builds a typed runner config from explicit arguments."""
    train_dir, val_dir = _make_artifacts(tmp_path / "embeddings")
    output_dir = tmp_path / "outputs"

    config = build_linear_probe_run_config(
        train_split_dir=train_dir,
        val_split_dir=val_dir,
        output_dir=output_dir,
        normalize_embeddings=True,
        map_location="cpu",
        bias=False,
        batch_size=8,
        num_epochs=12,
        learning_rate=0.05,
        weight_decay=0.01,
        optimizer_name="adamw",
        momentum=0.0,
        device="cpu",
        seed=7,
        selection_metric="val_accuracy",
        save_predictions=False,
        save_history=False,
    )

    assert isinstance(config, LinearProbeRunConfig)
    assert config.input.train_split_dir == train_dir
    assert config.input.val_split_dir == val_dir
    assert config.input.normalize_embeddings is True
    assert config.input.map_location == "cpu"

    assert config.output.output_dir == output_dir
    assert config.output.save_predictions is False
    assert config.output.save_history is False

    assert config.model.bias is False

    assert config.trainer.batch_size == 8
    assert config.trainer.num_epochs == 12
    assert config.trainer.learning_rate == 0.05
    assert config.trainer.weight_decay == 0.01
    assert config.trainer.optimizer_name == "adamw"
    assert config.trainer.device == "cpu"
    assert config.trainer.seed == 7
    assert config.trainer.selection_metric == "val_accuracy"


def test_resolve_linear_probe_run_config_accepts_dict(tmp_path: Path) -> None:
    """Resolves a plain dict into a typed runner config."""
    train_dir, val_dir = _make_artifacts(tmp_path / "embeddings")
    output_dir = tmp_path / "outputs"

    raw_config = {
        "input": {
            "train_split_dir": train_dir,
            "val_split_dir": val_dir,
            "normalize_embeddings": True,
            "map_location": "cpu",
        },
        "output": {
            "output_dir": output_dir,
            "save_predictions": False,
            "save_history": True,
        },
        "model": {
            "bias": False,
        },
        "trainer": {
            "batch_size": 16,
            "num_epochs": 9,
            "learning_rate": 0.03,
            "weight_decay": 0.0,
            "optimizer_name": "adamw",
            "device": "cpu",
            "seed": 5,
            "selection_metric": "val_loss",
        },
    }

    config = resolve_linear_probe_run_config(raw_config)

    assert isinstance(config, LinearProbeRunConfig)
    assert config.input.train_split_dir == train_dir
    assert config.input.val_split_dir == val_dir
    assert config.input.normalize_embeddings is True
    assert config.output.output_dir == output_dir
    assert config.output.save_predictions is False
    assert config.output.save_history is True
    assert config.model.bias is False
    assert config.trainer.batch_size == 16
    assert config.trainer.num_epochs == 9
    assert config.trainer.learning_rate == 0.03
    assert config.trainer.optimizer_name == "adamw"
    assert config.trainer.selection_metric == "val_loss"


def test_resolve_linear_probe_run_config_accepts_typed_config(
    tmp_path: Path,
) -> None:
    """Returns typed config unchanged."""
    train_dir, val_dir = _make_artifacts(tmp_path / "embeddings")
    output_dir = tmp_path / "outputs"

    config = LinearProbeRunConfig(
        input=LinearProbeInputConfig(
            train_split_dir=train_dir,
            val_split_dir=val_dir,
            normalize_embeddings=False,
            map_location="cpu",
        ),
        output=LinearProbeOutputConfig(
            output_dir=output_dir,
            save_predictions=True,
            save_history=True,
        ),
        model=LinearProbeModelConfig(
            bias=True,
        ),
        trainer=build_linear_probe_run_config(
            train_split_dir=train_dir,
            val_split_dir=val_dir,
            output_dir=output_dir,
        ).trainer,
    )

    resolved = resolve_linear_probe_run_config(config)

    assert resolved is config


def test_run_linear_probe_writes_expected_outputs(tmp_path: Path) -> None:
    """Runs the full evaluator pipeline and writes all expected files."""
    train_dir, val_dir = _make_artifacts(tmp_path / "embeddings")
    output_dir = tmp_path / "outputs"

    config = build_linear_probe_run_config(
        train_split_dir=train_dir,
        val_split_dir=val_dir,
        output_dir=output_dir,
        normalize_embeddings=False,
        bias=True,
        batch_size=4,
        num_epochs=30,
        learning_rate=0.05,
        optimizer_name="adamw",
        weight_decay=0.0,
        device="cpu",
        seed=123,
        selection_metric="val_accuracy",
        save_predictions=True,
        save_history=True,
    )

    result = run_linear_probe(config)

    assert isinstance(result, LinearProbeRunResult)
    assert result.output_dir == output_dir
    assert result.best_epoch >= 1
    assert result.bundle.embedding_dim == 2
    assert result.bundle.num_classes == 2
    assert result.model.input_dim == 2
    assert result.model.num_classes == 2

    expected_keys = {"metrics", "summary", "config", "history", "predictions"}
    assert set(result.written_paths.keys()) == expected_keys

    for path in result.written_paths.values():
        assert path.exists()
        assert path.is_file()

    metrics_path = result.written_paths["metrics"]
    summary_path = result.written_paths["summary"]
    config_path = result.written_paths["config"]
    history_path = result.written_paths["history"]
    predictions_path = result.written_paths["predictions"]

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
    assert metrics_payload["evaluator_version"] == "v1"
    assert metrics_payload["input"]["train_split"] == "train"
    assert metrics_payload["input"]["val_split"] == "val"
    assert metrics_payload["input"]["train_num_samples"] == 8
    assert metrics_payload["input"]["val_num_samples"] == 4
    assert metrics_payload["input"]["embedding_dim"] == 2
    assert metrics_payload["input"]["num_classes"] == 2
    assert metrics_payload["input"]["class_ids"] == [10, 20]
    assert metrics_payload["best_epoch"] == result.best_epoch
    assert metrics_payload["train"]["loss"] >= 0.0
    assert metrics_payload["val"]["loss"] >= 0.0
    assert 0.0 <= metrics_payload["train"]["accuracy"] <= 1.0
    assert 0.0 <= metrics_payload["val"]["accuracy"] <= 1.0

    assert summary_payload["status"] == "success"
    assert summary_payload["evaluator"] == "linear_probe"
    assert summary_payload["best_epoch"] == result.best_epoch
    assert summary_payload["train_split"] == "train"
    assert summary_payload["val_split"] == "val"
    assert summary_payload["train_num_samples"] == 8
    assert summary_payload["val_num_samples"] == 4
    assert summary_payload["embedding_dim"] == 2
    assert summary_payload["num_classes"] == 2
    assert summary_payload["output_dir"] == str(output_dir)
    assert summary_payload["val_accuracy"] == metrics_payload["val"]["accuracy"]
    assert summary_payload["val_loss"] == metrics_payload["val"]["loss"]

    assert config_payload["input"]["train_split_dir"] == str(train_dir)
    assert config_payload["input"]["val_split_dir"] == str(val_dir)
    assert config_payload["input"]["normalize_embeddings"] is False
    assert config_payload["output"]["output_dir"] == str(output_dir)
    assert config_payload["output"]["save_predictions"] is True
    assert config_payload["output"]["save_history"] is True
    assert config_payload["model"]["bias"] is True
    assert config_payload["trainer"]["batch_size"] == 4
    assert config_payload["trainer"]["num_epochs"] == 30
    assert config_payload["trainer"]["learning_rate"] == 0.05
    assert config_payload["trainer"]["optimizer_name"] == "adamw"
    assert config_payload["trainer"]["device"] == "cpu"
    assert config_payload["trainer"]["seed"] == 123

    assert "epochs" in history_payload
    assert len(history_payload["epochs"]) == 30
    for index, epoch_record in enumerate(history_payload["epochs"], start=1):
        assert epoch_record["epoch"] == index
        assert epoch_record["train_loss"] >= 0.0
        assert 0.0 <= epoch_record["train_accuracy"] <= 1.0
        assert epoch_record["val_loss"] >= 0.0
        assert 0.0 <= epoch_record["val_accuracy"] <= 1.0

    assert predictions_payload["split"] == "val"
    assert predictions_payload["image_ids"] == result.bundle.val.image_ids
    assert torch.equal(
        predictions_payload["labels"],
        result.training_result.val_output.labels,
    )
    assert torch.equal(
        predictions_payload["pred_labels"],
        result.training_result.val_output.predictions,
    )
    assert torch.equal(
        predictions_payload["logits"],
        result.training_result.val_output.logits,
    )
    assert torch.equal(
        predictions_payload["class_ids"],
        torch.tensor([10, 20], dtype=torch.long),
    )

    pred_accuracy = float(
        (
            predictions_payload["pred_labels"] == predictions_payload["labels"]
        ).sum().item()
    ) / float(predictions_payload["labels"].shape[0])
    assert pred_accuracy == metrics_payload["val"]["accuracy"]


def test_run_linear_probe_respects_disabled_optional_outputs(
    tmp_path: Path,
) -> None:
    """Skips predictions/history files when disabled."""
    train_dir, val_dir = _make_artifacts(tmp_path / "embeddings")
    output_dir = tmp_path / "outputs"

    config = build_linear_probe_run_config(
        train_split_dir=train_dir,
        val_split_dir=val_dir,
        output_dir=output_dir,
        batch_size=4,
        num_epochs=10,
        learning_rate=0.05,
        optimizer_name="adamw",
        weight_decay=0.0,
        device="cpu",
        seed=1,
        save_predictions=False,
        save_history=False,
    )

    result = run_linear_probe(config)

    expected_keys = {"metrics", "summary", "config"}
    assert set(result.written_paths.keys()) == expected_keys

    assert (output_dir / "metrics.yaml").exists()
    assert (output_dir / "summary.yaml").exists()
    assert (output_dir / "config.yaml").exists()
    assert not (output_dir / "history.yaml").exists()
    assert not (output_dir / "predictions.pt").exists()


def test_run_linear_probe_with_normalization_still_succeeds(
    tmp_path: Path,
) -> None:
    """Runs successfully when embedding normalization is enabled."""
    train_dir, val_dir = _make_artifacts(tmp_path / "embeddings")
    output_dir = tmp_path / "outputs"

    config = build_linear_probe_run_config(
        train_split_dir=train_dir,
        val_split_dir=val_dir,
        output_dir=output_dir,
        normalize_embeddings=True,
        batch_size=4,
        num_epochs=20,
        learning_rate=0.05,
        optimizer_name="adamw",
        weight_decay=0.0,
        device="cpu",
        seed=99,
        selection_metric="val_loss",
    )

    result = run_linear_probe(config)

    assert result.best_epoch >= 1
    assert result.bundle.embedding_dim == 2
    assert result.bundle.num_classes == 2
    assert result.val_metrics["loss"] >= 0.0
    assert 0.0 <= result.val_metrics["accuracy"] <= 1.0
    assert (output_dir / "metrics.yaml").exists()
    assert (output_dir / "summary.yaml").exists()
    assert (output_dir / "config.yaml").exists()