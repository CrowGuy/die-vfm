"""Tests for scripts.run_knn."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from types import SimpleNamespace

from tests._artifact_test_utils import make_artifacts
from die_vfm.evaluator.knn_runner import KnnRunResult
from scripts.run_knn import (
    _get_knn_config,
    _is_enabled,
    _print_run_summary,
    _to_plain_config,
    _validate_knn_config,
    main,
)


def _run_script(repo_root: Path, overrides: list[str]) -> subprocess.CompletedProcess[str]:
    """Runs the kNN script with Hydra overrides."""
    script_path = repo_root / "scripts" / "run_knn.py"

    command = [sys.executable, str(script_path), *overrides]
    return subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


def _make_root_config() -> object:
    """Builds a minimal Hydra-like root config for kNN evaluation."""
    return OmegaConf.create(
        {
            "evaluation": {
                "knn": {
                    "enabled": True,
                    "input": {
                        "train_split_dir": "artifacts/train",
                        "val_split_dir": "artifacts/val",
                        "normalize_embeddings": True,
                        "map_location": "cpu",
                    },
                    "output": {
                        "output_dir": "outputs/knn",
                        "save_predictions": True,
                    },
                    "evaluator": {
                        "k": 5,
                        "metric": "cosine",
                        "weighting": "uniform",
                        "temperature": 0.07,
                        "batch_size": 64,
                        "device": "cpu",
                        "topk": [1, 5],
                    },
                }
            }
        }
    )


def test_get_knn_config_returns_evaluation_knn_subtree() -> None:
    cfg = _make_root_config()

    knn_cfg = _get_knn_config(cfg)

    assert knn_cfg is cfg.evaluation.knn
    assert knn_cfg.input.train_split_dir == "artifacts/train"
    assert knn_cfg.output.output_dir == "outputs/knn"


def test_get_knn_config_raises_when_evaluation_section_is_missing() -> None:
    cfg = OmegaConf.create({})

    with pytest.raises(
        ValueError,
        match="Missing required config section: evaluation.",
    ):
        _get_knn_config(cfg)


def test_get_knn_config_raises_when_knn_section_is_missing() -> None:
    cfg = OmegaConf.create({"evaluation": {}})

    with pytest.raises(
        ValueError,
        match="Missing required config section: evaluation.knn.",
    ):
        _get_knn_config(cfg)


def test_is_enabled_returns_true_by_default() -> None:
    knn_cfg = OmegaConf.create({})

    assert _is_enabled(knn_cfg) is True


def test_is_enabled_returns_false_when_disabled() -> None:
    knn_cfg = OmegaConf.create({"enabled": False})

    assert _is_enabled(knn_cfg) is False


def test_validate_knn_config_accepts_valid_config() -> None:
    cfg = _make_root_config()

    _validate_knn_config(cfg.evaluation.knn)


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (
            {"output": {"output_dir": "outputs/knn"}},
            "Missing required config section: evaluation.knn.input.",
        ),
        (
            {"input": {"train_split_dir": "a", "val_split_dir": "b"}},
            "Missing required config section: evaluation.knn.output.",
        ),
        (
            {
                "input": {
                    "train_split_dir": None,
                    "val_split_dir": "artifacts/val",
                },
                "output": {"output_dir": "outputs/knn"},
            },
            "Missing required config: evaluation.knn.input.train_split_dir.",
        ),
        (
            {
                "input": {
                    "train_split_dir": "artifacts/train",
                    "val_split_dir": None,
                },
                "output": {"output_dir": "outputs/knn"},
            },
            "Missing required config: evaluation.knn.input.val_split_dir.",
        ),
        (
            {
                "input": {
                    "train_split_dir": "artifacts/train",
                    "val_split_dir": "artifacts/val",
                },
                "output": {"output_dir": None},
            },
            "Missing required config: evaluation.knn.output.output_dir.",
        ),
    ],
)
def test_validate_knn_config_raises_for_missing_required_fields(
    cfg: dict[str, object],
    message: str,
) -> None:
    knn_cfg = OmegaConf.create(cfg)

    with pytest.raises(ValueError, match=message):
        _validate_knn_config(knn_cfg)


def test_to_plain_config_returns_plain_dict() -> None:
    cfg = _make_root_config()

    plain_cfg = _to_plain_config(cfg.evaluation.knn)

    assert isinstance(plain_cfg, dict)
    assert plain_cfg["input"]["train_split_dir"] == "artifacts/train"
    assert plain_cfg["output"]["output_dir"] == "outputs/knn"
    assert plain_cfg["evaluator"]["topk"] == [1, 5]


def test_print_run_summary_prints_expected_lines(capsys: pytest.CaptureFixture[str]) -> None:
    output_dir = Path("outputs/knn")
    metrics = {
        "accuracy": 0.9,
        "top1_accuracy": 0.9,
        "top5_accuracy": 1.0,
    }

    _print_run_summary(output_dir=output_dir, metrics=metrics)

    captured = capsys.readouterr()
    assert "kNN evaluation completed." in captured.out
    assert "Output directory: outputs/knn" in captured.out
    assert "Validation accuracy: 0.900000" in captured.out
    assert "Validation top-1 accuracy: 0.900000" in captured.out
    assert "Validation top-5 accuracy: 1.000000" in captured.out


def test_main_skips_when_knn_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _make_root_config()
    cfg.evaluation.knn.enabled = False

    def _unexpected_call(*args: object, **kwargs: object) -> None:
        raise AssertionError("run_knn should not be called when disabled.")

    monkeypatch.setattr("scripts.run_knn.run_knn", _unexpected_call)

    main.__wrapped__(cfg)

    captured = capsys.readouterr()
    assert "kNN evaluation is disabled. Skipping." in captured.out


def test_main_calls_runner_on_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _make_root_config()
    captured: dict[str, object] = {}

    def _fake_resolve_knn_run_config(config: dict[str, object]) -> str:
        captured["resolved_input"] = config
        return "resolved-run-config"

    def _fake_run_knn(config: object) -> KnnRunResult:
        captured["run_config"] = config
        return KnnRunResult(
            output_dir=Path("outputs/knn"),
            written_paths={
                "metrics": Path("outputs/knn/metrics.yaml"),
                "summary": Path("outputs/knn/summary.yaml"),
                "config": Path("outputs/knn/config.yaml"),
                "predictions": Path("outputs/knn/predictions.pt"),
            },
            bundle=SimpleNamespace(),
            evaluation_output=SimpleNamespace(
                metrics={
                    "accuracy": 1.0,
                    "top1_accuracy": 1.0,
                }
            ),
        )

    monkeypatch.setattr(
        "scripts.run_knn.resolve_knn_run_config",
        _fake_resolve_knn_run_config,
    )
    monkeypatch.setattr(
        "scripts.run_knn.run_knn",
        _fake_run_knn,
    )

    main.__wrapped__(cfg)

    assert captured["run_config"] == "resolved-run-config"
    resolved_input = captured["resolved_input"]
    assert isinstance(resolved_input, dict)
    assert resolved_input["input"]["train_split_dir"] == "artifacts/train"
    assert resolved_input["input"]["val_split_dir"] == "artifacts/val"
    assert resolved_input["output"]["output_dir"] == "outputs/knn"
    assert resolved_input["evaluator"]["k"] == 5
    assert resolved_input["evaluator"]["topk"] == [1, 5]

    output = capsys.readouterr().out
    assert "kNN evaluation completed." in output
    assert "Output directory: outputs/knn" in output


def test_main_raises_for_missing_required_config() -> None:
    cfg = OmegaConf.create(
        {
            "evaluation": {
                "knn": {
                    "enabled": True,
                    "input": {
                        "train_split_dir": None,
                        "val_split_dir": "artifacts/val",
                    },
                    "output": {
                        "output_dir": "outputs/knn",
                    },
                    "evaluator": {
                        "k": 5,
                    },
                }
            }
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing required config: evaluation.knn.input.train_split_dir.",
    ):
        main.__wrapped__(cfg)


def test_run_knn_script_end_to_end(tmp_path: Path) -> None:
    """Runs the kNN script end to end from saved embedding artifacts."""
    repo_root = Path(__file__).resolve().parents[1]
    train_dir, val_dir = make_artifacts(tmp_path / "embeddings")
    output_dir = tmp_path / "outputs"

    completed = _run_script(
        repo_root=repo_root,
        overrides=[
            "evaluation.knn.enabled=true",
            f"evaluation.knn.input.train_split_dir={train_dir}",
            f"evaluation.knn.input.val_split_dir={val_dir}",
            f"evaluation.knn.output.output_dir={output_dir}",
            "evaluation.knn.input.normalize_embeddings=false",
            "evaluation.knn.evaluator.k=3",
            "evaluation.knn.evaluator.metric=cosine",
            "evaluation.knn.evaluator.weighting=uniform",
            "evaluation.knn.evaluator.temperature=0.07",
            "evaluation.knn.evaluator.batch_size=8",
            "evaluation.knn.evaluator.device=cpu",
            "evaluation.knn.evaluator.topk=[1]",
            "evaluation.knn.output.save_predictions=true",
        ],
    )

    assert completed.returncode == 0, completed.stderr
    assert "kNN evaluation completed." in completed.stdout
    assert f"Output directory: {output_dir}" in completed.stdout

    metrics_path = output_dir / "metrics.yaml"
    summary_path = output_dir / "summary.yaml"
    config_path = output_dir / "config.yaml"
    predictions_path = output_dir / "predictions.pt"

    assert metrics_path.exists()
    assert summary_path.exists()
    assert config_path.exists()
    assert predictions_path.exists()

    with metrics_path.open("r", encoding="utf-8") as file:
        metrics_payload = yaml.safe_load(file)
    with summary_path.open("r", encoding="utf-8") as file:
        summary_payload = yaml.safe_load(file)
    with config_path.open("r", encoding="utf-8") as file:
        config_payload = yaml.safe_load(file)

    assert metrics_payload["evaluator_type"] == "knn"
    assert metrics_payload["input"]["train_split"] == "train"
    assert metrics_payload["input"]["val_split"] == "val"
    assert summary_payload["evaluator"] == "knn"
    assert summary_payload["output_dir"] == str(output_dir)
    assert config_payload["input"]["train_split_dir"] == str(train_dir)
    assert config_payload["input"]["val_split_dir"] == str(val_dir)
