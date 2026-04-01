"""Tests for centroid runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from die_vfm.evaluator.centroid_evaluator import (
    CentroidEvaluationOutput,
    CentroidEvaluatorConfig,
)
from die_vfm.evaluator.centroid_runner import (
    CentroidInputConfig,
    CentroidOutputConfig,
    CentroidRunConfig,
    build_centroid_run_config,
    resolve_centroid_run_config,
    run_centroid,
)


@dataclass(frozen=True)
class _FakeSplit:
    """Minimal split object for centroid runner tests."""

    split_name: str
    embeddings: torch.Tensor
    labels: torch.Tensor
    image_ids: list[str]

    @property
    def num_samples(self) -> int:
        return int(self.labels.shape[0])


@dataclass(frozen=True)
class _FakeBundle:
    """Minimal bundle object for centroid runner tests."""

    train: _FakeSplit
    val: _FakeSplit
    num_classes: int
    embedding_dim: int
    class_ids: list[int]


def _make_fake_bundle() -> _FakeBundle:
    """Builds a fake train/val embedding bundle."""
    train = _FakeSplit(
        split_name="train",
        embeddings=torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=torch.float32,
        ),
        labels=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        image_ids=["train_0", "train_1", "train_2", "train_3"],
    )
    val = _FakeSplit(
        split_name="val",
        embeddings=torch.tensor(
            [
                [0.95, 0.05],
                [0.05, 0.95],
            ],
            dtype=torch.float32,
        ),
        labels=torch.tensor([0, 1], dtype=torch.long),
        image_ids=["val_0", "val_1"],
    )
    return _FakeBundle(
        train=train,
        val=val,
        num_classes=2,
        embedding_dim=2,
        class_ids=[10, 20],
    )


def _make_fake_evaluation_output() -> CentroidEvaluationOutput:
    """Builds a fake centroid evaluation output."""
    return CentroidEvaluationOutput(
        predictions=torch.tensor([0, 1], dtype=torch.long),
        labels=torch.tensor([0, 1], dtype=torch.long),
        logits=torch.tensor(
            [
                [0.9, 0.1],
                [0.2, 0.8],
            ],
            dtype=torch.float32,
        ),
        prototype_labels=torch.tensor([0, 1], dtype=torch.long),
        prototypes=torch.tensor(
            [
                [0.95, 0.05],
                [0.05, 0.95],
            ],
            dtype=torch.float32,
        ),
        image_ids=["val_0", "val_1"],
        metrics={
            "accuracy": 1.0,
            "top1_accuracy": 1.0,
        },
    )


def test_build_centroid_run_config_returns_expected_dataclasses() -> None:
    """build_centroid_run_config should build a typed run config."""
    config = build_centroid_run_config(
        train_split_dir="/tmp/train",
        val_split_dir="/tmp/val",
        output_dir="/tmp/out",
        normalize_embeddings=True,
        map_location="cpu",
        save_predictions=False,
        metric="l2",
        batch_size=32,
        device="cuda",
        topk=(1, 2),
    )

    assert isinstance(config, CentroidRunConfig)
    assert config.input == CentroidInputConfig(
        train_split_dir="/tmp/train",
        val_split_dir="/tmp/val",
        normalize_embeddings=True,
        map_location="cpu",
    )
    assert config.output == CentroidOutputConfig(
        output_dir="/tmp/out",
        save_predictions=False,
    )
    assert config.evaluator == CentroidEvaluatorConfig(
        metric="l2",
        batch_size=32,
        device="cuda",
        topk=(1, 2),
    )


def test_resolve_centroid_run_config_returns_same_object_for_typed_config() -> None:
    """resolve_centroid_run_config should return typed config unchanged."""
    config = CentroidRunConfig(
        input=CentroidInputConfig(
            train_split_dir="/tmp/train",
            val_split_dir="/tmp/val",
            normalize_embeddings=False,
            map_location="cpu",
        ),
        output=CentroidOutputConfig(
            output_dir="/tmp/out",
            save_predictions=True,
        ),
        evaluator=CentroidEvaluatorConfig(
            metric="cosine",
            batch_size=64,
            device="cpu",
            topk=(1, 5),
        ),
    )

    resolved = resolve_centroid_run_config(config)
    assert resolved is config


def test_resolve_centroid_run_config_accepts_plain_dict() -> None:
    """resolve_centroid_run_config should parse plain dict configs."""
    raw_config = {
        "input": {
            "train_split_dir": "/tmp/train",
            "val_split_dir": "/tmp/val",
            "normalize_embeddings": True,
            "map_location": "cpu",
        },
        "output": {
            "output_dir": "/tmp/out",
            "save_predictions": False,
        },
        "evaluator": {
            "metric": "l2",
            "batch_size": 16,
            "device": "cuda",
            "topk": [1, 2],
        },
    }

    resolved = resolve_centroid_run_config(raw_config)

    assert resolved == CentroidRunConfig(
        input=CentroidInputConfig(
            train_split_dir="/tmp/train",
            val_split_dir="/tmp/val",
            normalize_embeddings=True,
            map_location="cpu",
        ),
        output=CentroidOutputConfig(
            output_dir="/tmp/out",
            save_predictions=False,
        ),
        evaluator=CentroidEvaluatorConfig(
            metric="l2",
            batch_size=16,
            device="cuda",
            topk=(1, 2),
        ),
    )


def test_resolve_centroid_run_config_fills_defaults() -> None:
    """resolve_centroid_run_config should fill optional default values."""
    raw_config = {
        "input": {
            "train_split_dir": "/tmp/train",
            "val_split_dir": "/tmp/val",
        },
        "output": {
            "output_dir": "/tmp/out",
        },
        "evaluator": {},
    }

    resolved = resolve_centroid_run_config(raw_config)

    assert resolved == CentroidRunConfig(
        input=CentroidInputConfig(
            train_split_dir="/tmp/train",
            val_split_dir="/tmp/val",
            normalize_embeddings=False,
            map_location="cpu",
        ),
        output=CentroidOutputConfig(
            output_dir="/tmp/out",
            save_predictions=True,
        ),
        evaluator=CentroidEvaluatorConfig(
            metric="cosine",
            batch_size=1024,
            device="cpu",
            topk=(1, 5),
        ),
    )


def test_run_centroid_orchestrates_loader_evaluator_and_writer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """run_centroid should wire loader, evaluator, and writer together."""
    bundle = _make_fake_bundle()
    evaluation_output = _make_fake_evaluation_output()
    written_paths = {
        "metrics": tmp_path / "metrics.yaml",
        "summary": tmp_path / "summary.yaml",
        "config": tmp_path / "config.yaml",
        "predictions": tmp_path / "predictions.pt",
    }

    calls: dict[str, object] = {}

    def fake_load_linear_probe_bundle(
        *,
        train_split_dir: str | Path,
        val_split_dir: str | Path,
        normalize_embeddings: bool,
        map_location: str,
    ) -> _FakeBundle:
        calls["loader"] = {
            "train_split_dir": train_split_dir,
            "val_split_dir": val_split_dir,
            "normalize_embeddings": normalize_embeddings,
            "map_location": map_location,
        }
        return bundle

    def fake_evaluate_centroid(
        *,
        bundle: _FakeBundle,
        config: CentroidEvaluatorConfig,
    ) -> CentroidEvaluationOutput:
        calls["evaluator"] = {
            "bundle": bundle,
            "config": config,
        }
        return evaluation_output

    def fake_write_centroid_outputs(
        *,
        output_dir: str | Path,
        result: CentroidEvaluationOutput,
        bundle: _FakeBundle,
        config: dict,
        save_predictions: bool,
    ) -> dict[str, Path]:
        calls["writer"] = {
            "output_dir": output_dir,
            "result": result,
            "bundle": bundle,
            "config": config,
            "save_predictions": save_predictions,
        }
        return written_paths

    monkeypatch.setattr(
        "die_vfm.evaluator.centroid_runner.load_linear_probe_bundle",
        fake_load_linear_probe_bundle,
    )
    monkeypatch.setattr(
        "die_vfm.evaluator.centroid_runner.evaluate_centroid",
        fake_evaluate_centroid,
    )
    monkeypatch.setattr(
        "die_vfm.evaluator.centroid_runner.write_centroid_outputs",
        fake_write_centroid_outputs,
    )

    config = CentroidRunConfig(
        input=CentroidInputConfig(
            train_split_dir="/data/train_embeddings",
            val_split_dir="/data/val_embeddings",
            normalize_embeddings=True,
            map_location="cpu",
        ),
        output=CentroidOutputConfig(
            output_dir=tmp_path,
            save_predictions=False,
        ),
        evaluator=CentroidEvaluatorConfig(
            metric="cosine",
            batch_size=8,
            device="cpu",
            topk=(1,),
        ),
    )

    result = run_centroid(config)

    assert calls["loader"] == {
        "train_split_dir": "/data/train_embeddings",
        "val_split_dir": "/data/val_embeddings",
        "normalize_embeddings": True,
        "map_location": "cpu",
    }
    assert calls["evaluator"] == {
        "bundle": bundle,
        "config": config.evaluator,
    }

    writer_call = calls["writer"]
    assert writer_call["output_dir"] == Path(tmp_path)
    assert writer_call["result"] == evaluation_output
    assert writer_call["bundle"] == bundle
    assert writer_call["save_predictions"] is False
    assert writer_call["config"] == {
        "input": {
            "train_split_dir": "/data/train_embeddings",
            "val_split_dir": "/data/val_embeddings",
            "normalize_embeddings": True,
            "map_location": "cpu",
        },
        "output": {
            "output_dir": str(tmp_path),
            "save_predictions": False,
        },
        "evaluator": {
            "metric": "cosine",
            "batch_size": 8,
            "device": "cpu",
            "topk": (1,),
        },
    }

    assert result.output_dir == Path(tmp_path)
    assert result.written_paths == written_paths
    assert result.bundle == bundle
    assert result.evaluation_output == evaluation_output
    assert result.val_metrics == evaluation_output.metrics


def test_run_centroid_passes_through_writer_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """run_centroid should return writer-produced artifact paths unchanged."""
    bundle = _make_fake_bundle()
    evaluation_output = _make_fake_evaluation_output()
    written_paths = {
        "metrics": tmp_path / "nested" / "metrics.yaml",
        "summary": tmp_path / "nested" / "summary.yaml",
        "config": tmp_path / "nested" / "config.yaml",
    }

    monkeypatch.setattr(
        "die_vfm.evaluator.centroid_runner.load_linear_probe_bundle",
        lambda **_: bundle,
    )
    monkeypatch.setattr(
        "die_vfm.evaluator.centroid_runner.evaluate_centroid",
        lambda **_: evaluation_output,
    )
    monkeypatch.setattr(
        "die_vfm.evaluator.centroid_runner.write_centroid_outputs",
        lambda **_: written_paths,
    )

    config = build_centroid_run_config(
        train_split_dir="/data/train",
        val_split_dir="/data/val",
        output_dir=tmp_path,
    )
    result = run_centroid(config)

    assert result.written_paths == written_paths
    assert result.output_dir == Path(tmp_path)