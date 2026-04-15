from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from omegaconf import OmegaConf

from die_vfm.trainer import Round1FrozenRunner
from die_vfm.trainer.round1_runner import Round1RunArtifacts
import die_vfm.trainer.round1_runner as round1_runner_module


def _build_cfg(tmp_path: Path):
    """Builds the minimal current-spec config needed by Round1 tests."""
    return OmegaConf.create(
        {
            "run": {
                "run_name": "round1_test",
                "output_root": str(tmp_path / "runs"),
                "save_config_snapshot": True,
            },
            "system": {
                "seed": 123,
                "device": "cpu",
                "num_workers": 0,
            },
            "dataloader": {
                "batch_size": 4,
                "persistent_workers": False,
                "drop_last": False,
                "pin_memory": False,
            },
            "train": {
                "mode": "round1_frozen",
                "num_epochs": 1,
                "freeze_backbone": True,
                "freeze_pooler": True,
                "selection_metric": "linear_probe.accuracy",
                "resume": {
                    "enabled": False,
                    "checkpoint_path": None,
                    "mode": "full_resume",
                    "auto_resume_latest": False,
                },
            },
            "dataset": {
                "name": "dummy",
                "train_size": 16,
                "val_size": 12,
                "split_seed": {
                    "train": 123,
                    "val": 456,
                },
                "image_size": [32, 32],
                "num_channels": 3,
                "num_classes": 4,
                "label_offset": 0,
                "batch_size": 4,
                "num_workers": 0,
                "pin_memory": False,
            },
            "model": {
                "backbone": {
                    "name": "dummy",
                    "image_size": 32,
                    "patch_size": 16,
                    "in_channels": 3,
                    "embed_dim": 8,
                },
                "pooler": {
                    "name": "attn_pooler_v1",
                    "hidden_dim": 16,
                },
            },
            "evaluation": {
                "run_linear_probe": True,
                "run_knn": True,
                "run_retrieval": True,
                "linear_probe": {
                    "input": {
                        "normalize_embeddings": False,
                        "map_location": "cpu",
                    },
                    "model": {
                        "bias": True,
                    },
                    "trainer": {
                        "batch_size": 4,
                        "num_epochs": 2,
                        "learning_rate": 0.01,
                        "weight_decay": 0.0,
                        "optimizer_name": "sgd",
                        "momentum": 0.9,
                        "device": "cpu",
                        "seed": 123,
                        "selection_metric": "val_accuracy",
                    },
                    "output": {
                        "save_predictions": False,
                        "save_history": False,
                    },
                },
                "knn": {
                    "enabled": True,
                    "input": {
                        "normalize_embeddings": False,
                        "map_location": "cpu",
                    },
                    "evaluator": {
                        "k": 3,
                        "metric": "cosine",
                        "weighting": "uniform",
                        "temperature": 0.07,
                        "batch_size": 8,
                        "device": "cpu",
                        "topk": [1],
                    },
                    "output": {
                        "save_predictions": False,
                    },
                },
                "retrieval": {
                    "enabled": True,
                    "input": {
                        "normalize_embeddings": False,
                        "map_location": "cpu",
                    },
                    "evaluator": {
                        "metric": "cosine",
                        "batch_size": 8,
                        "device": "cpu",
                        "topk": [1],
                        "save_predictions_topk": 1,
                        "exclude_same_image_id": False,
                    },
                    "output": {
                        "save_predictions": False,
                    },
                },
            },
        }
    )


def _build_run_artifacts(tmp_path: Path) -> Round1RunArtifacts:
    round1_dir = tmp_path / "round1"
    return Round1RunArtifacts(
        round1_dir=round1_dir,
        train_embedding_dir=round1_dir / "embeddings" / "train",
        val_embedding_dir=round1_dir / "embeddings" / "val",
        linear_probe_dir=round1_dir / "evaluation" / "linear_probe",
        knn_dir=round1_dir / "evaluation" / "knn",
        retrieval_dir=round1_dir / "evaluation" / "retrieval",
        summary_yaml_path=round1_dir / "round1_summary.yaml",
        summary_json_path=round1_dir / "round1_summary.json",
    )


def test_round1_runner_writes_run_level_artifacts_and_summary(
    tmp_path: Path,
) -> None:
    cfg = _build_cfg(tmp_path)

    run_dir = Path(cfg.run.output_root) / cfg.run.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = Round1FrozenRunner(
        cfg=cfg,
        run_dir=run_dir,
    )

    metrics = runner.run()

    round1_dir = run_dir / "round1"

    assert "linear_probe.accuracy" in metrics
    assert (round1_dir / "embeddings" / "train").exists()
    assert (round1_dir / "embeddings" / "val").exists()
    assert (round1_dir / "evaluation" / "linear_probe").exists()
    assert (round1_dir / "evaluation" / "knn").exists()
    assert (round1_dir / "evaluation" / "retrieval").exists()
    assert (round1_dir / "round1_summary.yaml").exists()
    assert (round1_dir / "round1_summary.json").exists()
    assert not (run_dir / "checkpoints").exists()

    with (round1_dir / "round1_summary.yaml").open("r", encoding="utf-8") as file:
        summary = yaml.safe_load(file)

    assert summary["runtime_semantics"]["mode"] == "single_shot_inference_evaluation"
    assert summary["runtime_semantics"]["uses_gradient_updates"] is False
    assert summary["runtime_semantics"]["uses_epoch_loop"] is False
    assert summary["runtime_semantics"]["supports_resume"] is False
    assert summary["runtime_semantics"]["supports_checkpoint_continuation"] is False
    assert summary["execution"]["enabled_evaluators"] == [
        "knn",
        "linear_probe",
        "retrieval",
    ]
    assert summary["manifests"]["train"]["num_samples"] == 16
    assert summary["manifests"]["val"]["num_samples"] == 12


def test_round1_runner_skips_linear_probe_when_root_flag_is_disabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg = _build_cfg(tmp_path)
    cfg.evaluation.run_linear_probe = False

    runner = Round1FrozenRunner(
        cfg=cfg,
        run_dir=tmp_path / "runs" / "round1_test",
    )
    artifacts = _build_run_artifacts(tmp_path)

    def _unexpected_linear_probe(_: object) -> None:
        raise AssertionError("run_linear_probe should not be called when disabled.")

    monkeypatch.setattr(
        round1_runner_module,
        "run_linear_probe",
        _unexpected_linear_probe,
    )
    monkeypatch.setattr(
        round1_runner_module,
        "run_knn",
        lambda _: SimpleNamespace(val_metrics={"top1_accuracy": 0.75}),
    )
    monkeypatch.setattr(
        round1_runner_module,
        "run_retrieval",
        lambda _: SimpleNamespace(val_metrics={"recall@1": 0.5}),
    )

    metrics = runner._run_evaluators(artifacts)

    assert "linear_probe.accuracy" not in metrics
    assert metrics["knn.top1_accuracy"] == 0.75
    assert metrics["retrieval.recall@1"] == 0.5


def test_round1_runner_skips_knn_when_root_flag_is_disabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg = _build_cfg(tmp_path)
    cfg.evaluation.run_knn = False

    runner = Round1FrozenRunner(
        cfg=cfg,
        run_dir=tmp_path / "runs" / "round1_test",
    )
    artifacts = _build_run_artifacts(tmp_path)

    def _unexpected_knn(_: object) -> None:
        raise AssertionError("run_knn should not be called when disabled.")

    monkeypatch.setattr(
        round1_runner_module,
        "run_linear_probe",
        lambda _: SimpleNamespace(val_metrics={"accuracy": 1.0}),
    )
    monkeypatch.setattr(
        round1_runner_module,
        "run_knn",
        _unexpected_knn,
    )
    monkeypatch.setattr(
        round1_runner_module,
        "run_retrieval",
        lambda _: SimpleNamespace(val_metrics={"recall@1": 0.5}),
    )

    metrics = runner._run_evaluators(artifacts)

    assert metrics["linear_probe.accuracy"] == 1.0
    assert "knn.top1_accuracy" not in metrics
    assert metrics["retrieval.recall@1"] == 0.5


def test_round1_runner_skips_retrieval_when_root_flag_is_disabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg = _build_cfg(tmp_path)
    cfg.evaluation.run_retrieval = False

    runner = Round1FrozenRunner(
        cfg=cfg,
        run_dir=tmp_path / "runs" / "round1_test",
    )
    artifacts = _build_run_artifacts(tmp_path)

    def _unexpected_retrieval(_: object) -> None:
        raise AssertionError("run_retrieval should not be called when disabled.")

    monkeypatch.setattr(
        round1_runner_module,
        "run_linear_probe",
        lambda _: SimpleNamespace(val_metrics={"accuracy": 1.0}),
    )
    monkeypatch.setattr(
        round1_runner_module,
        "run_knn",
        lambda _: SimpleNamespace(val_metrics={"top1_accuracy": 0.75}),
    )
    monkeypatch.setattr(
        round1_runner_module,
        "run_retrieval",
        _unexpected_retrieval,
    )

    metrics = runner._run_evaluators(artifacts)

    assert metrics["linear_probe.accuracy"] == 1.0
    assert metrics["knn.top1_accuracy"] == 0.75
    assert "retrieval.recall@1" not in metrics


def test_round1_runner_fails_fast_when_no_round1_evaluator_enabled(
    tmp_path: Path,
) -> None:
    cfg = _build_cfg(tmp_path)
    cfg.evaluation.run_linear_probe = False
    cfg.evaluation.run_knn = False
    cfg.evaluation.run_retrieval = False

    run_dir = Path(cfg.run.output_root) / cfg.run.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = Round1FrozenRunner(
        cfg=cfg,
        run_dir=run_dir,
    )

    with pytest.raises(ValueError, match="at least one enabled evaluator"):
        runner.run()


def test_round1_runner_fails_fast_when_centroid_flag_is_enabled(
    tmp_path: Path,
) -> None:
    cfg = _build_cfg(tmp_path)
    cfg.evaluation.run_centroid = True

    run_dir = Path(cfg.run.output_root) / cfg.run.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = Round1FrozenRunner(
        cfg=cfg,
        run_dir=run_dir,
    )

    with pytest.raises(ValueError, match="does not orchestrate centroid"):
        runner.run()


def test_round1_runner_rejects_resume_semantics(
    tmp_path: Path,
) -> None:
    cfg = _build_cfg(tmp_path)
    cfg.train.resume.enabled = True

    run_dir = Path(cfg.run.output_root) / cfg.run.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = Round1FrozenRunner(
        cfg=cfg,
        run_dir=run_dir,
    )

    with pytest.raises(ValueError, match="does not support train.resume"):
        runner.run()


def test_round1_runner_does_not_depend_on_num_epochs_or_selection_metric(
    tmp_path: Path,
) -> None:
    cfg = _build_cfg(tmp_path)
    cfg.train.num_epochs = 0
    cfg.train.selection_metric = "unsupported.metric"
    cfg.evaluation.run_knn = False
    cfg.evaluation.run_retrieval = False

    run_dir = Path(cfg.run.output_root) / cfg.run.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = Round1FrozenRunner(
        cfg=cfg,
        run_dir=run_dir,
    )
    metrics = runner.run()

    assert metrics["linear_probe.accuracy"] >= 0.0
