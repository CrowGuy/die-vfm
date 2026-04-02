"""Tests for retrieval runner."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from die_vfm.evaluator.io import LinearProbeDataBundle
from die_vfm.evaluator.io import LinearProbeSplitData
from die_vfm.evaluator.retrieval_evaluator import RetrievalEvaluationOutput
from die_vfm.evaluator.retrieval_evaluator import RetrievalEvaluatorConfig
from die_vfm.evaluator.retrieval_runner import RetrievalInputConfig
from die_vfm.evaluator.retrieval_runner import RetrievalOutputConfig
from die_vfm.evaluator.retrieval_runner import RetrievalRunConfig
from die_vfm.evaluator.retrieval_runner import build_retrieval_run_config
from die_vfm.evaluator.retrieval_runner import resolve_retrieval_run_config
from die_vfm.evaluator.retrieval_runner import run_retrieval


class _DummyManifest:
    """Minimal manifest stub for split construction."""

    has_labels = True


def _make_split(
    split_name: str,
    embeddings: list[list[float]],
    labels: list[int],
    image_ids: list[str] | None = None,
) -> LinearProbeSplitData:
    """Builds a minimal validated split."""
    num_samples = len(embeddings)
    if image_ids is None:
        image_ids = [f"{split_name}_{index}" for index in range(num_samples)]

    return LinearProbeSplitData(
        split_name=split_name,
        embeddings=torch.tensor(embeddings, dtype=torch.float32),
        labels=torch.tensor(labels, dtype=torch.long),
        original_labels=torch.tensor(labels, dtype=torch.long),
        image_ids=image_ids,
        metadata=[{} for _ in range(num_samples)],
        manifest=_DummyManifest(),
    )


def _make_bundle() -> LinearProbeDataBundle:
    """Builds a minimal train/val bundle for retrieval runner tests."""
    train_split = _make_split(
        split_name="train",
        embeddings=[
            [1.0, 0.0],
            [0.9, 0.0],
            [0.0, 1.0],
            [0.0, 0.9],
        ],
        labels=[0, 0, 1, 1],
    )
    val_split = _make_split(
        split_name="val",
        embeddings=[
            [0.95, 0.0],
            [0.0, 0.95],
        ],
        labels=[0, 1],
    )
    return LinearProbeDataBundle(
        train=train_split,
        val=val_split,
        class_ids=[0, 1],
        class_to_index={0: 0, 1: 1},
    )


def _make_evaluation_output() -> RetrievalEvaluationOutput:
    """Builds a minimal retrieval evaluation output."""
    return RetrievalEvaluationOutput(
        query_labels=torch.tensor([0, 1], dtype=torch.long),
        topk_indices=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        topk_labels=torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
        topk_scores=torch.tensor(
            [[0.99, 0.98], [0.97, 0.96]], dtype=torch.float32
        ),
        topk_matches=torch.tensor(
            [[True, True], [True, True]], dtype=torch.bool
        ),
        image_ids=["val_0", "val_1"],
        topk_image_ids=[["train_0", "train_1"], ["train_2", "train_3"]],
        metrics={
            "recall_at_1": 1.0,
            "recall_at_2": 1.0,
            "map_at_1": 1.0,
            "map_at_2": 1.0,
            "num_queries": 2.0,
            "num_gallery": 4.0,
            "num_valid_queries_for_map": 2.0,
            "num_queries_without_positive": 0.0,
        },
    )


def test_run_retrieval_calls_loader_and_evaluator_and_writes_outputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """run_retrieval should orchestrate io -> evaluator -> real writer."""
    bundle = _make_bundle()
    evaluation_output = _make_evaluation_output()
    output_dir = tmp_path / "retrieval_eval"

    config = RetrievalRunConfig(
        input=RetrievalInputConfig(
            train_split_dir="runs/demo/embeddings/train",
            val_split_dir="runs/demo/embeddings/val",
            normalize_embeddings=True,
            map_location="cpu",
        ),
        output=RetrievalOutputConfig(
            output_dir=output_dir,
            save_predictions=True,
        ),
        evaluator=RetrievalEvaluatorConfig(
            metric="cosine",
            batch_size=32,
            device="cpu",
            topk=(1, 2),
            save_predictions_topk=2,
            exclude_same_image_id=False,
        ),
    )

    captured: dict[str, object] = {}

    def _fake_load_linear_probe_bundle(
        train_split_dir: str,
        val_split_dir: str,
        normalize_embeddings: bool,
        map_location: str,
    ) -> LinearProbeDataBundle:
        captured["load_args"] = {
            "train_split_dir": train_split_dir,
            "val_split_dir": val_split_dir,
            "normalize_embeddings": normalize_embeddings,
            "map_location": map_location,
        }
        return bundle

    def _fake_evaluate_retrieval(
        bundle: LinearProbeDataBundle,
        config: RetrievalEvaluatorConfig,
    ) -> RetrievalEvaluationOutput:
        captured["evaluate_args"] = {
            "bundle": bundle,
            "config": config,
        }
        return evaluation_output

    monkeypatch.setattr(
        "die_vfm.evaluator.retrieval_runner.load_linear_probe_bundle",
        _fake_load_linear_probe_bundle,
    )
    monkeypatch.setattr(
        "die_vfm.evaluator.retrieval_runner.evaluate_retrieval",
        _fake_evaluate_retrieval,
    )

    result = run_retrieval(config)

    assert captured["load_args"] == {
        "train_split_dir": "runs/demo/embeddings/train",
        "val_split_dir": "runs/demo/embeddings/val",
        "normalize_embeddings": True,
        "map_location": "cpu",
    }
    assert captured["evaluate_args"]["bundle"] is bundle
    assert captured["evaluate_args"]["config"] == config.evaluator

    assert result.output_dir == output_dir
    assert result.bundle is bundle
    assert result.evaluation_output is evaluation_output
    assert result.val_metrics == evaluation_output.metrics

    assert result.written_paths["metrics"] == output_dir / "metrics.yaml"
    assert result.written_paths["summary"] == output_dir / "summary.yaml"
    assert result.written_paths["config"] == output_dir / "config.yaml"
    assert result.written_paths["predictions"] == output_dir / "predictions.pt"

    assert (output_dir / "metrics.yaml").is_file()
    assert (output_dir / "summary.yaml").is_file()
    assert (output_dir / "config.yaml").is_file()
    assert (output_dir / "predictions.pt").is_file()

    metrics_payload = yaml.safe_load((output_dir / "metrics.yaml").read_text())
    assert metrics_payload["evaluator_type"] == "retrieval"
    assert metrics_payload["input"]["gallery_split"] == "train"
    assert metrics_payload["input"]["query_split"] == "val"
    assert metrics_payload["input"]["gallery_num_samples"] == 4
    assert metrics_payload["input"]["query_num_samples"] == 2
    assert metrics_payload["query"]["recall_at_1"] == 1.0
    assert metrics_payload["query"]["map_at_2"] == 1.0

    summary_payload = yaml.safe_load((output_dir / "summary.yaml").read_text())
    assert summary_payload["status"] == "success"
    assert summary_payload["evaluator"] == "retrieval"
    assert summary_payload["gallery_split"] == "train"
    assert summary_payload["query_split"] == "val"
    assert summary_payload["recall_at_1"] == 1.0
    assert summary_payload["map_at_1"] == 1.0

    config_payload = yaml.safe_load((output_dir / "config.yaml").read_text())
    assert config_payload["input"]["train_split_dir"] == "runs/demo/embeddings/train"
    assert config_payload["input"]["val_split_dir"] == "runs/demo/embeddings/val"
    assert config_payload["output"]["save_predictions"] is True
    assert config_payload["evaluator"]["metric"] == "cosine"
    assert config_payload["evaluator"]["topk"] == [1, 2]

    predictions_payload = torch.load(output_dir / "predictions.pt")
    assert predictions_payload["query_split"] == "val"
    assert predictions_payload["gallery_split"] == "train"
    assert predictions_payload["query_image_ids"] == ["val_0", "val_1"]
    assert predictions_payload["topk_image_ids"] == [
        ["train_0", "train_1"],
        ["train_2", "train_3"],
    ]
    assert torch.equal(
        predictions_payload["query_labels"],
        torch.tensor([0, 1], dtype=torch.long),
    )
    assert torch.equal(
        predictions_payload["topk_indices"],
        torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
    )
    assert torch.equal(
        predictions_payload["topk_labels"],
        torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
    )
    assert torch.equal(
        predictions_payload["topk_matches"],
        torch.tensor([[True, True], [True, True]], dtype=torch.bool),
    )


def test_run_retrieval_does_not_write_predictions_when_disabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """run_retrieval should respect save_predictions=False."""
    bundle = _make_bundle()
    evaluation_output = _make_evaluation_output()
    output_dir = tmp_path / "retrieval_eval"

    config = RetrievalRunConfig(
        input=RetrievalInputConfig(
            train_split_dir="runs/demo/embeddings/train",
            val_split_dir="runs/demo/embeddings/val",
        ),
        output=RetrievalOutputConfig(
            output_dir=output_dir,
            save_predictions=False,
        ),
        evaluator=RetrievalEvaluatorConfig(
            metric="cosine",
            batch_size=16,
            device="cpu",
            topk=(1, 2),
            save_predictions_topk=2,
        ),
    )

    monkeypatch.setattr(
        "die_vfm.evaluator.retrieval_runner.load_linear_probe_bundle",
        lambda **_: bundle,
    )
    monkeypatch.setattr(
        "die_vfm.evaluator.retrieval_runner.evaluate_retrieval",
        lambda **_: evaluation_output,
    )

    result = run_retrieval(config)

    assert (output_dir / "metrics.yaml").is_file()
    assert (output_dir / "summary.yaml").is_file()
    assert (output_dir / "config.yaml").is_file()
    assert not (output_dir / "predictions.pt").exists()
    assert "predictions" not in result.written_paths


def test_build_retrieval_run_config_builds_expected_dataclass() -> None:
    """build_retrieval_run_config should assemble nested typed config."""
    config = build_retrieval_run_config(
        train_split_dir="runs/demo/embeddings/train",
        val_split_dir="runs/demo/embeddings/val",
        output_dir="runs/demo/eval/retrieval",
        normalize_embeddings=True,
        map_location="cpu",
        save_predictions=False,
        metric="l2",
        batch_size=64,
        device="cuda",
        topk=(1, 5, 10),
        save_predictions_topk=5,
        exclude_same_image_id=True,
    )

    assert config.input.train_split_dir == "runs/demo/embeddings/train"
    assert config.input.val_split_dir == "runs/demo/embeddings/val"
    assert config.input.normalize_embeddings is True
    assert config.input.map_location == "cpu"

    assert config.output.output_dir == "runs/demo/eval/retrieval"
    assert config.output.save_predictions is False

    assert config.evaluator.metric == "l2"
    assert config.evaluator.batch_size == 64
    assert config.evaluator.device == "cuda"
    assert config.evaluator.topk == (1, 5, 10)
    assert config.evaluator.save_predictions_topk == 5
    assert config.evaluator.exclude_same_image_id is True


def test_resolve_retrieval_run_config_returns_typed_config_unchanged() -> None:
    """resolve_retrieval_run_config should accept typed configs directly."""
    config = RetrievalRunConfig(
        input=RetrievalInputConfig(
            train_split_dir="train_dir",
            val_split_dir="val_dir",
            normalize_embeddings=False,
            map_location="cpu",
        ),
        output=RetrievalOutputConfig(
            output_dir="output_dir",
            save_predictions=True,
        ),
        evaluator=RetrievalEvaluatorConfig(
            metric="cosine",
            batch_size=128,
            device="cpu",
            topk=(1, 5),
            save_predictions_topk=5,
            exclude_same_image_id=False,
        ),
    )

    resolved = resolve_retrieval_run_config(config)
    assert resolved is config


def test_resolve_retrieval_run_config_accepts_plain_dict() -> None:
    """resolve_retrieval_run_config should convert nested dict configs."""
    raw_config = {
        "input": {
            "train_split_dir": "runs/demo/embeddings/train",
            "val_split_dir": "runs/demo/embeddings/val",
            "normalize_embeddings": True,
            "map_location": "cpu",
        },
        "output": {
            "output_dir": "runs/demo/eval/retrieval",
            "save_predictions": True,
        },
        "evaluator": {
            "metric": "cosine",
            "batch_size": 32,
            "device": "cpu",
            "topk": (1, 2, 4),
            "save_predictions_topk": 4,
            "exclude_same_image_id": True,
        },
    }

    resolved = resolve_retrieval_run_config(raw_config)

    assert isinstance(resolved, RetrievalRunConfig)
    assert resolved.input.train_split_dir == "runs/demo/embeddings/train"
    assert resolved.input.val_split_dir == "runs/demo/embeddings/val"
    assert resolved.input.normalize_embeddings is True
    assert resolved.input.map_location == "cpu"

    assert resolved.output.output_dir == "runs/demo/eval/retrieval"
    assert resolved.output.save_predictions is True

    assert resolved.evaluator.metric == "cosine"
    assert resolved.evaluator.batch_size == 32
    assert resolved.evaluator.device == "cpu"
    assert resolved.evaluator.topk == (1, 2, 4)
    assert resolved.evaluator.save_predictions_topk == 4
    assert resolved.evaluator.exclude_same_image_id is True


def test_resolve_retrieval_run_config_accepts_list_topk_and_normalizes_tuple() -> None:
    """resolve_retrieval_run_config should normalize list topk to tuple."""
    raw_config = {
        "input": {
            "train_split_dir": "train_dir",
            "val_split_dir": "val_dir",
        },
        "output": {
            "output_dir": "output_dir",
        },
        "evaluator": {
            "metric": "l2",
            "batch_size": 8,
            "device": "cpu",
            "topk": [1, 5, 10],
            "save_predictions_topk": 5,
            "exclude_same_image_id": False,
        },
    }

    resolved = resolve_retrieval_run_config(raw_config)

    assert resolved.evaluator.topk == (1, 5, 10)
    assert resolved.evaluator.metric == "l2"


def test_resolve_retrieval_run_config_raises_on_missing_required_section() -> None:
    """resolve_retrieval_run_config should fail clearly on invalid structure."""
    raw_config = {
        "input": {
            "train_split_dir": "train_dir",
            "val_split_dir": "val_dir",
        },
        "output": {
            "output_dir": "output_dir",
        },
    }

    try:
        resolve_retrieval_run_config(raw_config)
    except KeyError as exc:
        assert "evaluator" in str(exc)
    else:
        raise AssertionError("Expected KeyError for missing evaluator section.")