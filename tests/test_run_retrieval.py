"""Tests for retrieval evaluation script."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts import run_retrieval


def test_extract_retrieval_config_returns_expected_subtree() -> None:
    """Should extract evaluation.retrieval subtree from Hydra config."""
    config = OmegaConf.create(
        {
            "evaluation": {
                "retrieval": {
                    "enabled": True,
                    "input": {
                        "train_split_dir": "runs/demo/embeddings/train",
                        "val_split_dir": "runs/demo/embeddings/val",
                    },
                    "output": {
                        "output_dir": "runs/demo/eval/retrieval",
                    },
                }
            }
        }
    )

    retrieval_cfg = run_retrieval._extract_retrieval_config(config)

    assert retrieval_cfg.enabled is True
    assert (
        retrieval_cfg.input.train_split_dir
        == "runs/demo/embeddings/train"
    )
    assert retrieval_cfg.output.output_dir == "runs/demo/eval/retrieval"


def test_extract_retrieval_config_raises_when_evaluation_missing() -> None:
    """Should fail clearly when evaluation section is missing."""
    config = OmegaConf.create({})

    with pytest.raises(KeyError, match="evaluation"):
        run_retrieval._extract_retrieval_config(config)


def test_extract_retrieval_config_raises_when_retrieval_missing() -> None:
    """Should fail clearly when evaluation.retrieval section is missing."""
    config = OmegaConf.create({"evaluation": {}})

    with pytest.raises(KeyError, match="evaluation.retrieval"):
        run_retrieval._extract_retrieval_config(config)


def test_is_enabled_defaults_to_true_when_missing() -> None:
    """Retrieval evaluator should default to enabled=True."""
    retrieval_cfg = OmegaConf.create(
        {
            "input": {
                "train_split_dir": "train_dir",
                "val_split_dir": "val_dir",
            },
            "output": {
                "output_dir": "output_dir",
            },
        }
    )

    assert run_retrieval._is_enabled(retrieval_cfg) is True


def test_is_enabled_returns_false_when_disabled() -> None:
    """Should detect enabled=False correctly."""
    retrieval_cfg = OmegaConf.create(
        {
            "enabled": False,
            "input": {
                "train_split_dir": "train_dir",
                "val_split_dir": "val_dir",
            },
            "output": {
                "output_dir": "output_dir",
            },
        }
    )

    assert run_retrieval._is_enabled(retrieval_cfg) is False


def test_validate_retrieval_config_accepts_valid_config() -> None:
    """Validation should pass for complete retrieval config."""
    retrieval_cfg = OmegaConf.create(
        {
            "enabled": True,
            "input": {
                "train_split_dir": "runs/demo/embeddings/train",
                "val_split_dir": "runs/demo/embeddings/val",
            },
            "output": {
                "output_dir": "runs/demo/eval/retrieval",
            },
        }
    )

    run_retrieval._validate_retrieval_config(retrieval_cfg)


def test_validate_retrieval_config_raises_when_input_missing() -> None:
    """Should fail when input section is missing."""
    retrieval_cfg = OmegaConf.create(
        {
            "output": {
                "output_dir": "runs/demo/eval/retrieval",
            },
        }
    )

    with pytest.raises(ValueError, match="evaluation.retrieval.input"):
        run_retrieval._validate_retrieval_config(retrieval_cfg)


def test_validate_retrieval_config_raises_when_output_missing() -> None:
    """Should fail when output section is missing."""
    retrieval_cfg = OmegaConf.create(
        {
            "input": {
                "train_split_dir": "runs/demo/embeddings/train",
                "val_split_dir": "runs/demo/embeddings/val",
            },
        }
    )

    with pytest.raises(ValueError, match="evaluation.retrieval.output"):
        run_retrieval._validate_retrieval_config(retrieval_cfg)


def test_validate_retrieval_config_raises_when_train_split_dir_missing() -> None:
    """Should fail when train_split_dir is missing."""
    retrieval_cfg = OmegaConf.create(
        {
            "input": {
                "val_split_dir": "runs/demo/embeddings/val",
            },
            "output": {
                "output_dir": "runs/demo/eval/retrieval",
            },
        }
    )

    with pytest.raises(ValueError, match="train_split_dir"):
        run_retrieval._validate_retrieval_config(retrieval_cfg)


def test_validate_retrieval_config_raises_when_val_split_dir_missing() -> None:
    """Should fail when val_split_dir is missing."""
    retrieval_cfg = OmegaConf.create(
        {
            "input": {
                "train_split_dir": "runs/demo/embeddings/train",
            },
            "output": {
                "output_dir": "runs/demo/eval/retrieval",
            },
        }
    )

    with pytest.raises(ValueError, match="val_split_dir"):
        run_retrieval._validate_retrieval_config(retrieval_cfg)


def test_validate_retrieval_config_raises_when_output_dir_missing() -> None:
    """Should fail when output_dir is missing."""
    retrieval_cfg = OmegaConf.create(
        {
            "input": {
                "train_split_dir": "runs/demo/embeddings/train",
                "val_split_dir": "runs/demo/embeddings/val",
            },
            "output": {},
        }
    )

    with pytest.raises(ValueError, match="output_dir"):
        run_retrieval._validate_retrieval_config(retrieval_cfg)


def test_to_plain_object_converts_omegaconf_to_plain_dict() -> None:
    """Should convert OmegaConf subtree to plain dict."""
    retrieval_cfg = OmegaConf.create(
        {
            "enabled": True,
            "input": {
                "train_split_dir": "train_dir",
                "val_split_dir": "val_dir",
            },
            "output": {
                "output_dir": "output_dir",
            },
            "evaluator": {
                "metric": "cosine",
                "topk": [1, 5],
            },
        }
    )

    plain = run_retrieval._to_plain_object(retrieval_cfg)

    assert isinstance(plain, dict)
    assert plain["enabled"] is True
    assert plain["input"]["train_split_dir"] == "train_dir"
    assert plain["evaluator"]["topk"] == [1, 5]


def test_format_metrics_prefers_common_keys() -> None:
    """Should format the preferred retrieval metrics first."""
    metrics = {
        "recall_at_1": 0.8,
        "recall_at_5": 0.9,
        "map_at_1": 0.7,
        "map_at_5": 0.85,
        "num_queries": 100.0,
    }

    summary = run_retrieval._format_metrics(metrics)

    assert "recall_at_1=0.800000" in summary
    assert "recall_at_5=0.900000" in summary
    assert "map_at_1=0.700000" in summary
    assert "map_at_5=0.850000" in summary


def test_print_run_summary_prints_expected_lines(capsys) -> None:
    """Should print a compact retrieval run summary."""
    run_retrieval._print_run_summary(
        result_output_dir=Path("runs/demo/eval/retrieval"),
        metrics={
            "recall_at_1": 0.8,
            "map_at_1": 0.7,
        },
    )

    captured = capsys.readouterr()
    assert "Retrieval evaluation completed." in captured.out
    assert "runs/demo/eval/retrieval" in captured.out
    assert "recall_at_1=0.800000" in captured.out
    assert "map_at_1=0.700000" in captured.out


def test_main_skips_when_disabled(monkeypatch, capsys) -> None:
    """Main should skip execution when evaluator is disabled."""
    config = OmegaConf.create(
        {
            "evaluation": {
                "retrieval": {
                    "enabled": False,
                    "input": {
                        "train_split_dir": "train_dir",
                        "val_split_dir": "val_dir",
                    },
                    "output": {
                        "output_dir": "output_dir",
                    },
                }
            }
        }
    )

    called = {"resolve": False, "run": False}

    def _fake_resolve(_config):
        called["resolve"] = True
        return _config

    def _fake_run(_config):
        called["run"] = True
        return _config

    monkeypatch.setattr(
        run_retrieval,
        "resolve_retrieval_run_config",
        _fake_resolve,
    )
    monkeypatch.setattr(
        run_retrieval,
        "run_retrieval",
        _fake_run,
    )

    run_retrieval.main.__wrapped__(config)

    captured = capsys.readouterr()
    assert "Retrieval evaluation is disabled. Skipping." in captured.out
    assert called["resolve"] is False
    assert called["run"] is False


def test_main_resolves_and_runs(monkeypatch, capsys) -> None:
    """Main should resolve config, run evaluator, and print summary."""
    config = OmegaConf.create(
        {
            "evaluation": {
                "retrieval": {
                    "enabled": True,
                    "input": {
                        "train_split_dir": "runs/demo/embeddings/train",
                        "val_split_dir": "runs/demo/embeddings/val",
                    },
                    "output": {
                        "output_dir": "runs/demo/eval/retrieval",
                    },
                    "evaluator": {
                        "metric": "cosine",
                        "topk": [1, 5],
                    },
                }
            }
        }
    )

    captured: dict[str, object] = {}

    class _DummyResult:
        output_dir = Path("runs/demo/eval/retrieval")
        val_metrics = {
            "recall_at_1": 0.81,
            "map_at_1": 0.75,
        }

    def _fake_resolve(raw_config):
        captured["resolved_input"] = raw_config
        return "typed-config"

    def _fake_run(typed_config):
        captured["run_input"] = typed_config
        return _DummyResult()

    monkeypatch.setattr(
        run_retrieval,
        "resolve_retrieval_run_config",
        _fake_resolve,
    )
    monkeypatch.setattr(
        run_retrieval,
        "run_retrieval",
        _fake_run,
    )

    run_retrieval.main.__wrapped__(config)

    assert isinstance(captured["resolved_input"], dict)
    assert captured["resolved_input"]["input"]["train_split_dir"] == (
        "runs/demo/embeddings/train"
    )
    assert captured["resolved_input"]["evaluator"]["topk"] == [1, 5]
    assert captured["run_input"] == "typed-config"

    stdout = capsys.readouterr().out
    assert "Retrieval evaluation completed." in stdout
    assert "runs/demo/eval/retrieval" in stdout
    assert "recall_at_1=0.810000" in stdout
    assert "map_at_1=0.750000" in stdout


def test_main_raises_for_invalid_config() -> None:
    """Main should raise when required retrieval config is incomplete."""
    config = OmegaConf.create(
        {
            "evaluation": {
                "retrieval": {
                    "enabled": True,
                    "input": {
                        "train_split_dir": "runs/demo/embeddings/train",
                    },
                    "output": {
                        "output_dir": "runs/demo/eval/retrieval",
                    },
                }
            }
        }
    )

    with pytest.raises(ValueError, match="val_split_dir"):
        run_retrieval.main.__wrapped__(config)