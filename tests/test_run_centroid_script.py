"""Tests for run_centroid script."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts.run_centroid import (
    _extract_centroid_config,
    _format_metrics,
    _to_plain_object,
    main,
)


@dataclass(frozen=True)
class _FakeRunResult:
    """Minimal run result for script tests."""

    output_dir: Path
    val_metrics: dict[str, float]


def test_extract_centroid_config_returns_expected_subtree() -> None:
    """Should extract evaluation.centroid subtree."""
    config = OmegaConf.create(
        {
            "evaluation": {
                "centroid": {
                    "input": {
                        "train_split_dir": "/tmp/train",
                        "val_split_dir": "/tmp/val",
                    },
                    "output": {
                        "output_dir": "/tmp/out",
                    },
                    "evaluator": {
                        "metric": "cosine",
                    },
                }
            }
        }
    )

    centroid_config = _extract_centroid_config(config)

    assert centroid_config.input.train_split_dir == "/tmp/train"
    assert centroid_config.input.val_split_dir == "/tmp/val"
    assert centroid_config.output.output_dir == "/tmp/out"
    assert centroid_config.evaluator.metric == "cosine"


def test_extract_centroid_config_raises_when_evaluation_is_missing() -> None:
    """Should fail clearly if evaluation section is missing."""
    config = OmegaConf.create({})

    with pytest.raises(KeyError, match="Missing required Hydra config section: evaluation"):
        _extract_centroid_config(config)


def test_extract_centroid_config_raises_when_centroid_is_missing() -> None:
    """Should fail clearly if evaluation.centroid is missing."""
    config = OmegaConf.create({"evaluation": {}})

    with pytest.raises(
        KeyError,
        match="Missing required Hydra config section: evaluation.centroid",
    ):
        _extract_centroid_config(config)


def test_to_plain_object_returns_plain_dict() -> None:
    """Should convert DictConfig subtree to plain Python dict."""
    config = OmegaConf.create(
        {
            "input": {
                "train_split_dir": "/tmp/train",
                "val_split_dir": "/tmp/val",
            },
            "output": {
                "output_dir": "/tmp/out",
                "save_predictions": True,
            },
            "evaluator": {
                "metric": "cosine",
                "batch_size": 32,
                "device": "cpu",
                "topk": [1, 5],
            },
        }
    )

    plain = _to_plain_object(config)

    assert plain == {
        "input": {
            "train_split_dir": "/tmp/train",
            "val_split_dir": "/tmp/val",
        },
        "output": {
            "output_dir": "/tmp/out",
            "save_predictions": True,
        },
        "evaluator": {
            "metric": "cosine",
            "batch_size": 32,
            "device": "cpu",
            "topk": [1, 5],
        },
    }


def test_to_plain_object_raises_for_non_mapping() -> None:
    """Should reject configs that do not resolve to a dict."""
    config = OmegaConf.create([1, 2, 3])

    with pytest.raises(TypeError, match="Expected evaluation.centroid to resolve to a dict"):
        _to_plain_object(config)


def test_format_metrics_prefers_standard_keys() -> None:
    """Should print common metric keys first in a stable order."""
    metrics = {
        "top5_accuracy": 1.0,
        "accuracy": 0.75,
        "top1_accuracy": 0.75,
        "other_metric": 123.0,
    }

    formatted = _format_metrics(metrics)

    assert formatted == (
        "accuracy=0.750000, top1_accuracy=0.750000, top5_accuracy=1.000000"
    )


def test_format_metrics_falls_back_to_sorted_items() -> None:
    """Should fall back to sorted full metric dump if standard keys are absent."""
    metrics = {
        "b_metric": 2.0,
        "a_metric": 1.0,
    }

    formatted = _format_metrics(metrics)

    assert formatted == "a_metric=1.000000, b_metric=2.000000"


def test_main_orchestrates_resolve_and_run(monkeypatch, capsys, tmp_path: Path) -> None:
    """main should resolve config, run evaluation, and print summary."""
    hydra_config = OmegaConf.create(
        {
            "evaluation": {
                "centroid": {
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
                        "topk": [1],
                    },
                }
            }
        }
    )

    resolved_config = object()
    fake_result = _FakeRunResult(
        output_dir=tmp_path,
        val_metrics={
            "accuracy": 1.0,
            "top1_accuracy": 1.0,
        },
    )

    calls: dict[str, object] = {}

    def fake_resolve_centroid_run_config(config: dict) -> object:
        calls["resolve"] = config
        return resolved_config

    def fake_run_centroid(config: object) -> _FakeRunResult:
        calls["run"] = config
        return fake_result

    monkeypatch.setattr(
        "scripts.run_centroid.resolve_centroid_run_config",
        fake_resolve_centroid_run_config,
    )
    monkeypatch.setattr(
        "scripts.run_centroid.run_centroid",
        fake_run_centroid,
    )

    main.__wrapped__(hydra_config)

    assert calls["resolve"] == {
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
            "topk": [1],
        },
    }
    assert calls["run"] is resolved_config

    stdout = capsys.readouterr().out
    assert "Centroid evaluation completed." in stdout
    assert f"Output directory: {tmp_path}" in stdout
    assert "accuracy=1.000000" in stdout
    assert "top1_accuracy=1.000000" in stdout


def test_main_propagates_missing_centroid_section(monkeypatch) -> None:
    """main should surface config structure errors."""
    config = OmegaConf.create({"evaluation": {}})

    monkeypatch.setattr(
        "scripts.run_centroid.resolve_centroid_run_config",
        lambda _: pytest.fail("resolve_centroid_run_config should not be called"),
    )
    monkeypatch.setattr(
        "scripts.run_centroid.run_centroid",
        lambda _: pytest.fail("run_centroid should not be called"),
    )

    with pytest.raises(
        KeyError,
        match="Missing required Hydra config section: evaluation.centroid",
    ):
        main.__wrapped__(config)