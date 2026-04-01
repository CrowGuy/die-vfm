"""Tests for die_vfm.evaluator.knn_runner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from die_vfm.evaluator.knn_evaluator import (
    KnnEvaluationOutput,
    KnnEvaluatorConfig,
)
from die_vfm.evaluator.knn_runner import (
    KnnInputConfig,
    KnnOutputConfig,
    KnnRunConfig,
    build_knn_run_config,
    resolve_knn_run_config,
    run_knn,
)


def _make_split(
    embeddings: list[list[float]],
    labels: list[int],
    image_id_prefix: str,
    split_name: str,
) -> SimpleNamespace:
    """Builds a minimal split object compatible with runner dependencies."""
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return SimpleNamespace(
        embeddings=embeddings_tensor,
        labels=labels_tensor,
        image_ids=[
            f"{image_id_prefix}_{index}" for index in range(len(labels))
        ],
        num_samples=len(labels),
        split_name=split_name,
    )


def _make_bundle() -> SimpleNamespace:
    """Builds a minimal bundle object compatible with runner dependencies."""
    train_split = _make_split(
        embeddings=[
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        labels=[0, 0, 1, 1],
        image_id_prefix="train",
        split_name="train",
    )
    val_split = _make_split(
        embeddings=[
            [0.95, 0.05],
            [0.05, 0.95],
        ],
        labels=[0, 1],
        image_id_prefix="val",
        split_name="val",
    )
    return SimpleNamespace(
        train=train_split,
        val=val_split,
        embedding_dim=2,
        num_classes=2,
        class_ids=[0, 1],
    )


def _make_evaluation_output() -> KnnEvaluationOutput:
    """Builds a minimal deterministic kNN evaluation output."""
    return KnnEvaluationOutput(
        predictions=torch.tensor([0, 1], dtype=torch.long),
        labels=torch.tensor([0, 1], dtype=torch.long),
        logits=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=torch.float32,
        ),
        neighbor_indices=torch.tensor(
            [[0, 1], [2, 3]],
            dtype=torch.long,
        ),
        neighbor_labels=torch.tensor(
            [[0, 0], [1, 1]],
            dtype=torch.long,
        ),
        neighbor_scores=torch.tensor(
            [[0.99, 0.95], [0.98, 0.96]],
            dtype=torch.float32,
        ),
        image_ids=["val_0", "val_1"],
        metrics={
            "accuracy": 1.0,
            "top1_accuracy": 1.0,
        },
    )


def test_build_knn_run_config_returns_typed_config(tmp_path: Path) -> None:
    output_dir = tmp_path / "knn_eval"

    config = build_knn_run_config(
        train_split_dir="artifacts/train",
        val_split_dir="artifacts/val",
        output_dir=output_dir,
        normalize_embeddings=True,
        map_location="cpu",
        save_predictions=False,
        k=7,
        metric="cosine",
        weighting="distance",
        temperature=0.2,
        batch_size=64,
        device="cpu",
        topk=(1, 5),
    )

    assert isinstance(config, KnnRunConfig)
    assert isinstance(config.input, KnnInputConfig)
    assert isinstance(config.output, KnnOutputConfig)
    assert isinstance(config.evaluator, KnnEvaluatorConfig)

    assert config.input.train_split_dir == "artifacts/train"
    assert config.input.val_split_dir == "artifacts/val"
    assert config.input.normalize_embeddings is True
    assert config.input.map_location == "cpu"

    assert config.output.output_dir == output_dir
    assert config.output.save_predictions is False

    assert config.evaluator.k == 7
    assert config.evaluator.metric == "cosine"
    assert config.evaluator.weighting == "distance"
    assert config.evaluator.temperature == pytest.approx(0.2)
    assert config.evaluator.batch_size == 64
    assert config.evaluator.device == "cpu"
    assert config.evaluator.topk == (1, 5)


def test_resolve_knn_run_config_accepts_dict() -> None:
    raw_config = {
        "input": {
            "train_split_dir": "artifacts/train",
            "val_split_dir": "artifacts/val",
            "normalize_embeddings": True,
            "map_location": "cpu",
        },
        "output": {
            "output_dir": "outputs/knn",
            "save_predictions": False,
        },
        "evaluator": {
            "k": 9,
            "metric": "l2",
            "weighting": "uniform",
            "temperature": 0.5,
            "batch_size": 128,
            "device": "cpu",
            "topk": [1, 3],
        },
    }

    resolved = resolve_knn_run_config(raw_config)

    assert isinstance(resolved, KnnRunConfig)
    assert resolved.input.train_split_dir == "artifacts/train"
    assert resolved.input.val_split_dir == "artifacts/val"
    assert resolved.input.normalize_embeddings is True
    assert resolved.output.output_dir == "outputs/knn"
    assert resolved.output.save_predictions is False
    assert resolved.evaluator.k == 9
    assert resolved.evaluator.metric == "l2"
    assert resolved.evaluator.weighting == "uniform"
    assert resolved.evaluator.temperature == pytest.approx(0.5)
    assert resolved.evaluator.batch_size == 128
    assert resolved.evaluator.device == "cpu"
    assert resolved.evaluator.topk == (1, 3)


def test_resolve_knn_run_config_accepts_namespace_like_object() -> None:
    raw_config = SimpleNamespace(
        input=SimpleNamespace(
            train_split_dir="artifacts/train",
            val_split_dir="artifacts/val",
            normalize_embeddings=False,
            map_location="cpu",
        ),
        output=SimpleNamespace(
            output_dir="outputs/knn",
            save_predictions=True,
        ),
        evaluator=SimpleNamespace(
            k=5,
            metric="cosine",
            weighting="uniform",
            temperature=0.07,
            batch_size=32,
            device="cpu",
            topk=(1,),
        ),
    )

    resolved = resolve_knn_run_config(raw_config)

    assert isinstance(resolved, KnnRunConfig)
    assert resolved.input.train_split_dir == "artifacts/train"
    assert resolved.output.output_dir == "outputs/knn"
    assert resolved.evaluator.k == 5
    assert resolved.evaluator.topk == (1,)


def test_run_knn_calls_loader_and_evaluator_with_expected_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _make_bundle()
    evaluation_output = _make_evaluation_output()
    captured: dict[str, object] = {}

    def _fake_load_linear_probe_bundle(**kwargs: object) -> SimpleNamespace:
        captured["load_kwargs"] = kwargs
        return bundle

    def _fake_evaluate_knn(
        bundle: object,
        config: KnnEvaluatorConfig,
    ) -> KnnEvaluationOutput:
        captured["evaluate_bundle"] = bundle
        captured["evaluate_config"] = config
        return evaluation_output

    monkeypatch.setattr(
        "die_vfm.evaluator.knn_runner.load_linear_probe_bundle",
        _fake_load_linear_probe_bundle,
    )
    monkeypatch.setattr(
        "die_vfm.evaluator.knn_runner.evaluate_knn",
        _fake_evaluate_knn,
    )

    output_dir = tmp_path / "knn_eval"
    config = build_knn_run_config(
        train_split_dir="artifacts/train",
        val_split_dir="artifacts/val",
        output_dir=output_dir,
        normalize_embeddings=True,
        map_location="cpu",
        save_predictions=True,
        k=3,
        metric="cosine",
        weighting="distance",
        temperature=0.1,
        batch_size=16,
        device="cpu",
        topk=(1, 5),
    )

    result = run_knn(config)

    assert captured["load_kwargs"] == {
        "train_split_dir": "artifacts/train",
        "val_split_dir": "artifacts/val",
        "normalize_embeddings": True,
        "map_location": "cpu",
    }
    assert captured["evaluate_bundle"] is bundle

    evaluate_config = captured["evaluate_config"]
    assert isinstance(evaluate_config, KnnEvaluatorConfig)
    assert evaluate_config.k == 3
    assert evaluate_config.metric == "cosine"
    assert evaluate_config.weighting == "distance"
    assert evaluate_config.temperature == pytest.approx(0.1)
    assert evaluate_config.batch_size == 16
    assert evaluate_config.topk == (1, 5)

    assert result.output_dir == output_dir
    assert result.bundle is bundle
    assert result.evaluation_output == evaluation_output
    assert result.val_metrics["accuracy"] == pytest.approx(1.0)


def test_run_knn_writes_expected_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _make_bundle()
    evaluation_output = _make_evaluation_output()

    monkeypatch.setattr(
        "die_vfm.evaluator.knn_runner.load_linear_probe_bundle",
        lambda **_: bundle,
    )
    monkeypatch.setattr(
        "die_vfm.evaluator.knn_runner.evaluate_knn",
        lambda **_: evaluation_output,
    )

    output_dir = tmp_path / "knn_eval"
    config = build_knn_run_config(
        train_split_dir="artifacts/train",
        val_split_dir="artifacts/val",
        output_dir=output_dir,
        save_predictions=True,
        k=3,
    )

    result = run_knn(config)

    assert (output_dir / "metrics.yaml").is_file()
    assert (output_dir / "summary.yaml").is_file()
    assert (output_dir / "config.yaml").is_file()
    assert (output_dir / "predictions.pt").is_file()

    assert result.written_paths["metrics"] == output_dir / "metrics.yaml"
    assert result.written_paths["summary"] == output_dir / "summary.yaml"
    assert result.written_paths["config"] == output_dir / "config.yaml"
    assert result.written_paths["predictions"] == output_dir / "predictions.pt"


def test_run_knn_writes_metrics_yaml_with_expected_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _make_bundle()
    evaluation_output = _make_evaluation_output()

    monkeypatch.setattr(
        "die_vfm.evaluator.knn_runner.load_linear_probe_bundle",
        lambda **_: bundle,
    )
    monkeypatch.setattr(
        "die_vfm.evaluator.knn_runner.evaluate_knn",
        lambda **_: evaluation_output,
    )

    output_dir = tmp_path / "knn_eval"
    config = build_knn_run_config(
        train_split_dir="artifacts/train",
        val_split_dir="artifacts/val",
        output_dir=output_dir,
        k=2,
        metric="cosine",
        weighting="uniform",
    )

    run_knn(config)

    with (output_dir / "metrics.yaml").open("r", encoding="utf-8") as file_obj:
        metrics = yaml.safe_load(file_obj)

    assert metrics["evaluator_type"] == "knn"
    assert metrics["evaluator_version"] == "v1"
    assert metrics["input"]["train_split"] == "train"
    assert metrics["input"]["val_split"] == "val"
    assert metrics["input"]["train_num_samples"] == 4
    assert metrics["input"]["val_num_samples"] == 2
    assert metrics["input"]["embedding_dim"] == 2
    assert metrics["input"]["num_classes"] == 2
    assert metrics["input"]["class_ids"] == [0, 1]
    assert metrics["val"]["accuracy"] == pytest.approx(1.0)
    assert metrics["val"]["top1_accuracy"] == pytest.approx(1.0)


def test_run_knn_writes_summary_yaml_with_expected_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _make_bundle()
    evaluation_output = _make_evaluation_output()

    monkeypatch.setattr(
        "die_vfm.evaluator.knn_runner.load_linear_probe_bundle",
        lambda **_: bundle,
    )
    monkeypatch.setattr(
        "die_vfm.evaluator.knn_runner.evaluate_knn",
        lambda **_: evaluation_output,
    )

    output_dir = tmp_path / "knn_eval"
    config = build_knn_run_config(
        train_split_dir="artifacts/train",
        val_split_dir="artifacts/val",
        output_dir=output_dir,
        k=2,
    )

    run_knn(config)

    with (output_dir / "summary.yaml").open("r", encoding="utf-8") as file_obj:
        summary = yaml.safe_load(file_obj)

    assert summary["status"] == "success"
    assert summary["evaluator"] == "knn"
    assert summary["train_split"] == "train"
    assert summary["val_split"] == "val"
    assert summary["train_num_samples"] == 4
    assert summary["val_num_samples"] == 2
    assert summary["embedding_dim"] == 2
    assert summary["num_classes"] == 2
    assert summary["val_accuracy"] == pytest.approx(1.0)
    assert summary["top1_accuracy"] == pytest.approx(1.0)
    assert Path(summary["output_dir"]) == output_dir


def test_run_knn_writes_config_yaml_with_expected_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _make_bundle()
    evaluation_output = _make_evaluation_output()

    monkeypatch.setattr(
        "die_vfm.evaluator.knn_runner.load_linear_probe_bundle",
        lambda **_: bundle,
    )
    monkeypatch.setattr(
        "die_vfm.evaluator.knn_runner.evaluate_knn",
        lambda **_: evaluation_output,
    )

    output_dir = tmp_path / "knn_eval"
    config = build_knn_run_config(
        train_split_dir="artifacts/train",
        val_split_dir="artifacts/val",
        output_dir=output_dir,
        normalize_embeddings=True,
        map_location="cpu",
        save_predictions=True,
        k=7,
        metric="l2",
        weighting="distance",
        temperature=0.3,
        batch_size=64,
        device="cpu",
        topk=(1, 3),
    )

    run_knn(config)

    with (output_dir / "config.yaml").open("r", encoding="utf-8") as file_obj:
        written_config = yaml.safe_load(file_obj)

    assert written_config["input"]["train_split_dir"] == "artifacts/train"
    assert written_config["input"]["val_split_dir"] == "artifacts/val"
    assert written_config["input"]["normalize_embeddings"] is True
    assert written_config["input"]["map_location"] == "cpu"

    assert written_config["output"]["output_dir"] == str(output_dir)
    assert written_config["output"]["save_predictions"] is True

    assert written_config["evaluator"]["k"] == 7
    assert written_config["evaluator"]["metric"] == "l2"
    assert written_config["evaluator"]["weighting"] == "distance"
    assert written_config["evaluator"]["temperature"] == pytest.approx(0.3)
    assert written_config["evaluator"]["batch_size"] == 64
    assert written_config["evaluator"]["device"] == "cpu"
    assert written_config["evaluator"]["topk"] == [1, 3]


def test_run_knn_writes_predictions_payload_with_neighbor_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _make_bundle()
    evaluation_output = _make_evaluation_output()

    monkeypatch.setattr(
        "die_vfm.evaluator.knn_runner.load_linear_probe_bundle",
        lambda **_: bundle,
    )
    monkeypatch.setattr(
        "die_vfm.evaluator.knn_runner.evaluate_knn",
        lambda **_: evaluation_output,
    )

    output_dir = tmp_path / "knn_eval"
    config = build_knn_run_config(
        train_split_dir="artifacts/train",
        val_split_dir="artifacts/val",
        output_dir=output_dir,
        save_predictions=True,
        k=2,
    )

    run_knn(config)

    predictions = torch.load(output_dir / "predictions.pt")

    assert predictions["split"] == "val"
    assert predictions["image_ids"] == ["val_0", "val_1"]

    assert torch.equal(
        predictions["labels"],
        torch.tensor([0, 1], dtype=torch.long),
    )
    assert torch.equal(
        predictions["pred_labels"],
        torch.tensor([0, 1], dtype=torch.long),
    )
    assert torch.equal(
        predictions["logits"],
        torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
    )
    assert torch.equal(
        predictions["class_ids"],
        torch.tensor([0, 1], dtype=torch.long),
    )
    assert torch.equal(
        predictions["neighbor_indices"],
        torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
    )
    assert torch.equal(
        predictions["neighbor_labels"],
        torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
    )
    assert torch.equal(
        predictions["neighbor_scores"],
        torch.tensor(
            [[0.99, 0.95], [0.98, 0.96]],
            dtype=torch.float32,
        ),
    )


def test_run_knn_respects_save_predictions_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _make_bundle()
    evaluation_output = _make_evaluation_output()

    monkeypatch.setattr(
        "die_vfm.evaluator.knn_runner.load_linear_probe_bundle",
        lambda **_: bundle,
    )
    monkeypatch.setattr(
        "die_vfm.evaluator.knn_runner.evaluate_knn",
        lambda **_: evaluation_output,
    )

    output_dir = tmp_path / "knn_eval"
    config = build_knn_run_config(
        train_split_dir="artifacts/train",
        val_split_dir="artifacts/val",
        output_dir=output_dir,
        save_predictions=False,
        k=2,
    )

    result = run_knn(config)

    assert (output_dir / "metrics.yaml").is_file()
    assert (output_dir / "summary.yaml").is_file()
    assert (output_dir / "config.yaml").is_file()
    assert not (output_dir / "predictions.pt").exists()
    assert "predictions" not in result.written_paths