from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml
from omegaconf import OmegaConf

from die_vfm.trainer import CheckpointManager
from die_vfm.trainer.round2_runner import Round2SSLRunner
import die_vfm.trainer.round2_runner as round2_runner_module


def _build_cfg(tmp_path: Path) -> Any:
    return OmegaConf.create(
        {
            "run": {
                "run_name": "round2_test",
                "output_root": str(tmp_path / "runs"),
                "save_config_snapshot": True,
            },
            "system": {
                "seed": 123,
                "device": "cpu",
                "num_workers": 0,
            },
            "dataloader": {
                "batch_size": 2,
                "persistent_workers": False,
                "drop_last": False,
                "pin_memory": False,
            },
            "train": {
                "mode": "round2_ssl",
                "num_epochs": 1,
                "update_mode": "projector_pooler_only",
                "last_n_blocks": None,
                "precision_mode": "fp32",
                "log_every_n_steps": 10,
                "freeze_backbone": False,
                "freeze_pooler": False,
            },
            "dataset": {
                "name": "dummy",
                "train_size": 8,
                "val_size": 4,
                "split_seed": {"train": 123, "val": 456},
                "image_size": [32, 32],
                "num_channels": 3,
                "num_classes": 4,
                "label_offset": 0,
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
                    "name": "mean",
                    "l2_norm": False,
                },
                "return_debug_outputs": True,
            },
            "evaluation": {
                "run_linear_probe": False,
                "run_knn": False,
                "run_centroid": False,
                "run_retrieval": False,
            },
            "round2": {
                "optimizer": {
                    "name": "adamw",
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "betas": [0.9, 0.999],
                },
                "scheduler": {
                    "name": "none",
                    "min_learning_rate": 0.0,
                },
                "ema": {
                    "policy": "fixed",
                    "momentum": 0.996,
                    "final_momentum": 0.999,
                },
                "projector": {
                    "hidden_dim": None,
                    "output_dim": None,
                    "num_layers": 2,
                },
                "token_projector": {
                    "hidden_dim": None,
                    "output_dim": None,
                    "num_layers": 2,
                },
                "loss": {
                    "token_loss_enabled": False,
                    "token_loss_weight": 0.2,
                },
                "augmentation": {
                    "horizontal_flip_prob": 0.5,
                    "vertical_flip_prob": 0.5,
                },
                "postprocess": {
                    "mode": "in_process",
                    "checkpoint_path": None,
                    "auto_use_latest_checkpoint": True,
                },
                "distributed": {
                    "strategy": "ddp",
                    "backend": None,
                    "find_unused_parameters": False,
                },
                "evaluation": {
                    "cadence": "end_only",
                    "run_pair_benchmark": True,
                    "run_slicing_analysis": True,
                    "pair_benchmark": {
                        "pair_candidates_path": str(tmp_path / "pair_candidates.csv"),
                        "annotations_path": str(tmp_path / "annotations.csv"),
                        "join_key": "did",
                        "output_subdir": "pair_benchmark",
                        "hard_limit": 5,
                        "map_location": "cpu",
                        "embedding_splits": ["val"],
                    },
                    "slicing": {
                        "output_subdir": "slicing",
                        "confidences": ["high"],
                        "hard_limit": 5,
                    },
                },
            },
        }
    )


def test_round2_runner_writes_summary_with_pair_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pair_candidates_path = tmp_path / "pair_candidates.csv"
    annotations_path = tmp_path / "annotations.csv"
    pair_candidates_path.write_text("pair_id,did_a,did_b\n", encoding="utf-8")
    annotations_path.write_text("pair_id,review_status,visual_relation\n", encoding="utf-8")

    def _fake_pair_benchmark(**kwargs):
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        pair_scores_path = output_dir / "pair_scores.csv"
        pair_scores_path.write_text("pair_id,visual_relation,cosine_similarity\n", encoding="utf-8")
        return {
            "summary": {
                "coverage": {
                    "annotated_reviewed_pairs": 1,
                    "matched_pairs": 1,
                    "unmatched_pairs": 0,
                },
                "separation": {
                    "same_vs_different_cosine_auc_like": 0.81,
                },
            },
            "pair_scores_path": pair_scores_path,
            "summary_yaml_path": output_dir / "pair_metrics_summary.yaml",
            "summary_json_path": output_dir / "pair_metrics_summary.json",
            "unmatched_pairs_path": output_dir / "unmatched_pairs.csv",
            "hard_same_far_path": output_dir / "hard_same_far.csv",
            "hard_different_close_path": output_dir / "hard_different_close.csv",
            "uncertain_high_similarity_path": output_dir / "uncertain_high_similarity.csv",
        }

    def _fake_slicing(**kwargs):
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "summary": {
                "coverage": {
                    "input_pair_scores": 1,
                    "filtered_pair_scores": 1,
                },
                "slices": {
                    "source_slice": {
                        "cross_source": {
                            "same_vs_different_cosine_auc_like": 0.77,
                        }
                    }
                },
            },
            "slice_summary_path": output_dir / "slice_summary.csv",
            "relation_stats_path": output_dir / "relation_stats.csv",
            "hard_cases_path": output_dir / "slice_hard_cases.csv",
            "summary_yaml_path": output_dir / "slice_analysis_summary.yaml",
            "summary_json_path": output_dir / "slice_analysis_summary.json",
            "enriched_scores_path": output_dir / "pair_scores_enriched.csv",
            "filtered_scores_path": output_dir / "pair_scores_filtered.csv",
        }

    monkeypatch.setattr(round2_runner_module, "run_pair_benchmark", _fake_pair_benchmark)
    monkeypatch.setattr(round2_runner_module, "run_pair_slicing_analysis", _fake_slicing)

    cfg = _build_cfg(tmp_path)
    run_dir = Path(cfg.run.output_root) / str(cfg.run.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = Round2SSLRunner(cfg=cfg, run_dir=run_dir)
    metrics = runner.run()

    assert metrics["pair_benchmark.same_vs_different_cosine_auc_like"] == 0.81
    assert metrics["pair_benchmark.cross_source_auc_like.high"] == 0.77

    summary_path = run_dir / "round2" / "round2_summary.yaml"
    assert summary_path.exists()
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = yaml.safe_load(handle)

    assert summary["phase"] == "round2_ssl"
    assert summary["execution"]["runtime_default_update_mode"] == "full_backbone"
    assert summary["execution"]["planned_experiment_order"] == [
        "projector_pooler_only",
        "last_n_blocks",
        "full_backbone",
    ]
    assert "student_token_projector" not in summary["execution"]["resolved_trainable_module_names"]


def test_round2_runner_summary_records_effective_ema_momentum_for_schedule(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pair_candidates_path = tmp_path / "pair_candidates.csv"
    annotations_path = tmp_path / "annotations.csv"
    pair_candidates_path.write_text("pair_id,did_a,did_b\n", encoding="utf-8")
    annotations_path.write_text("pair_id,review_status,visual_relation\n", encoding="utf-8")

    monkeypatch.setattr(
        round2_runner_module,
        "run_pair_benchmark",
        lambda **kwargs: {
            "summary": {
                "coverage": {
                    "annotated_reviewed_pairs": 1,
                    "matched_pairs": 1,
                    "unmatched_pairs": 0,
                },
                "separation": {
                    "same_vs_different_cosine_auc_like": 0.81,
                },
            },
            "pair_scores_path": Path(kwargs["output_dir"]) / "pair_scores.csv",
            "summary_yaml_path": Path(kwargs["output_dir"]) / "pair_metrics_summary.yaml",
            "summary_json_path": Path(kwargs["output_dir"]) / "pair_metrics_summary.json",
            "unmatched_pairs_path": Path(kwargs["output_dir"]) / "unmatched_pairs.csv",
            "hard_same_far_path": Path(kwargs["output_dir"]) / "hard_same_far.csv",
            "hard_different_close_path": Path(kwargs["output_dir"]) / "hard_different_close.csv",
            "uncertain_high_similarity_path": Path(kwargs["output_dir"]) / "uncertain_high_similarity.csv",
        },
    )
    monkeypatch.setattr(
        round2_runner_module,
        "run_pair_slicing_analysis",
        lambda **kwargs: {
            "summary": {
                "coverage": {
                    "input_pair_scores": 1,
                    "filtered_pair_scores": 1,
                },
                "slices": {},
            },
            "slice_summary_path": Path(kwargs["output_dir"]) / "slice_summary.csv",
            "relation_stats_path": Path(kwargs["output_dir"]) / "relation_stats.csv",
            "hard_cases_path": Path(kwargs["output_dir"]) / "slice_hard_cases.csv",
            "summary_yaml_path": Path(kwargs["output_dir"]) / "slice_analysis_summary.yaml",
            "summary_json_path": Path(kwargs["output_dir"]) / "slice_analysis_summary.json",
            "enriched_scores_path": Path(kwargs["output_dir"]) / "pair_scores_enriched.csv",
            "filtered_scores_path": Path(kwargs["output_dir"]) / "pair_scores_filtered.csv",
        },
    )

    cfg = _build_cfg(tmp_path)
    cfg.train.num_epochs = 3
    cfg.round2.ema.policy = "schedule"
    cfg.round2.ema.momentum = 0.9
    cfg.round2.ema.final_momentum = 0.99
    run_dir = Path(cfg.run.output_root) / str(cfg.run.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = Round2SSLRunner(cfg=cfg, run_dir=run_dir)
    metrics = runner.run()
    assert metrics["pair_benchmark.same_vs_different_cosine_auc_like"] == 0.81

    summary_path = run_dir / "round2" / "round2_summary.yaml"
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = yaml.safe_load(handle)

    execution = summary["execution"]
    assert execution["ema_policy"] == "schedule"
    assert execution["ema_base_momentum"] == pytest.approx(0.9)
    assert execution["ema_final_momentum"] == pytest.approx(0.99)
    assert execution["ema_momentum"] == pytest.approx(0.99)


def test_round2_runner_non_rank_zero_skips_post_training_writers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _build_cfg(tmp_path)
    run_dir = Path(cfg.run.output_root) / str(cfg.run.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = Round2SSLRunner(cfg=cfg, run_dir=run_dir)
    runner._distributed_enabled = True
    runner._world_size = 2
    runner._rank = 1

    counters = {
        "export": 0,
        "eval": 0,
        "summary": 0,
        "barrier": 0,
    }

    monkeypatch.setattr(runner, "_setup_distributed_if_needed", lambda: None)
    monkeypatch.setattr(runner, "_cleanup_distributed", lambda: None)
    monkeypatch.setattr(
        runner,
        "_barrier_if_distributed",
        lambda: counters.__setitem__("barrier", counters["barrier"] + 1),
    )
    monkeypatch.setattr(
        runner,
        "_train_loop",
        lambda **kwargs: {"train.total_loss": 1.0},
    )
    monkeypatch.setattr(
        runner,
        "_broadcast_final_metrics",
        lambda metrics: {"train.total_loss": 1.0},
    )
    monkeypatch.setattr(
        runner,
        "_export_split",
        lambda **kwargs: counters.__setitem__("export", counters["export"] + 1),
    )
    monkeypatch.setattr(
        runner,
        "_run_end_only_evaluation",
        lambda **kwargs: counters.__setitem__("eval", counters["eval"] + 1),
    )
    monkeypatch.setattr(
        runner,
        "_write_run_summary",
        lambda **kwargs: counters.__setitem__("summary", counters["summary"] + 1),
    )

    metrics = runner.run()
    assert metrics == {"train.total_loss": 1.0}
    assert counters["export"] == 0
    assert counters["eval"] == 0
    assert counters["summary"] == 0
    assert counters["barrier"] >= 2


def test_round2_runner_separate_step_training_then_postprocess(
    tmp_path: Path,
) -> None:
    cfg = _build_cfg(tmp_path)
    cfg.round2.postprocess.mode = "separate_step"
    cfg.round2.evaluation.run_pair_benchmark = False
    cfg.round2.evaluation.run_slicing_analysis = False
    run_dir = Path(cfg.run.output_root) / str(cfg.run.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = Round2SSLRunner(cfg=cfg, run_dir=run_dir)
    train_metrics = runner.run()
    assert "train.total_loss" in train_metrics

    round2_dir = run_dir / "round2"
    assert (round2_dir / "checkpoints" / "latest.pt").exists()
    assert not (round2_dir / "embeddings" / "train" / "manifest.yaml").exists()

    with (round2_dir / "round2_summary.yaml").open("r", encoding="utf-8") as handle:
        train_summary = yaml.safe_load(handle)
    assert train_summary["execution"]["postprocess_mode"] == "separate_step"
    assert train_summary["execution"]["postprocess_completed"] is False

    postprocess_metrics = runner.run_postprocess()
    assert "train.total_loss" in postprocess_metrics
    assert (round2_dir / "embeddings" / "train" / "manifest.yaml").exists()
    assert (round2_dir / "embeddings" / "val" / "manifest.yaml").exists()

    with (round2_dir / "round2_summary.yaml").open("r", encoding="utf-8") as handle:
        postprocess_summary = yaml.safe_load(handle)
    assert postprocess_summary["execution"]["postprocess_completed"] is True
    assert postprocess_summary["manifests"]["train"]["available"] is True
    assert postprocess_summary["execution"]["update_mode"] == "projector_pooler_only"
    assert postprocess_summary["execution"]["resolved_trainable_module_names"] == [
        "student_encoder.pooler",
        "student_global_projector",
    ]


def test_round2_runner_postprocess_preserves_training_execution_metadata(
    tmp_path: Path,
) -> None:
    cfg = _build_cfg(tmp_path)
    cfg.round2.postprocess.mode = "separate_step"
    cfg.round2.evaluation.run_pair_benchmark = False
    cfg.round2.evaluation.run_slicing_analysis = False
    run_dir = Path(cfg.run.output_root) / str(cfg.run.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = Round2SSLRunner(cfg=cfg, run_dir=run_dir)
    runner.run()

    cfg.train.update_mode = "full_backbone"
    mutated_runner = Round2SSLRunner(cfg=cfg, run_dir=run_dir)
    mutated_runner.run_postprocess()

    with (run_dir / "round2" / "round2_summary.yaml").open("r", encoding="utf-8") as handle:
        summary = yaml.safe_load(handle)

    assert summary["execution"]["update_mode"] == "projector_pooler_only"
    assert summary["execution"]["last_n_blocks"] is None
    assert summary["execution"]["resolved_trainable_module_names"] == [
        "student_encoder.pooler",
        "student_global_projector",
    ]


def test_round2_runner_postprocess_allows_val_only_pair_benchmark_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _build_cfg(tmp_path)
    cfg.round2.evaluation.run_pair_benchmark = True
    cfg.round2.evaluation.run_slicing_analysis = False
    cfg.round2.evaluation.pair_benchmark.embedding_splits = ["val"]
    run_dir = Path(cfg.run.output_root) / str(cfg.run.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = Round2SSLRunner(cfg=cfg, run_dir=run_dir)
    artifacts = runner._resolve_run_artifacts()
    artifacts.summary_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.summary_yaml_path.write_text("metrics:\n  train.total_loss: 1.0\n", encoding="utf-8")

    export_calls: list[str] = []
    summary_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        runner,
        "_resolve_postprocess_checkpoint_path",
        lambda **kwargs: artifacts.checkpoint_dir / "latest.pt",
    )
    monkeypatch.setattr(
        CheckpointManager,
        "load_full_resume",
        lambda self, **kwargs: None,
    )

    def _fake_split_loader(split: str) -> Any | None:
        if split == "train":
            return None
        if split == "val":
            return SimpleNamespace(dataset=[1, 2, 3])
        raise AssertionError(f"unexpected split: {split}")

    monkeypatch.setattr(runner, "_build_split_dataloader_or_none", _fake_split_loader)

    def _fake_export_split(**kwargs):
        export_calls.append(str(kwargs["split"]))
        return SimpleNamespace(num_samples=3, embedding_dim=8, has_labels=False)

    monkeypatch.setattr(runner, "_export_split", _fake_export_split)
    monkeypatch.setattr(
        runner,
        "_run_end_only_evaluation",
        lambda **kwargs: {
            "pair_benchmark.same_vs_different_cosine_auc_like": 0.9,
        },
    )
    monkeypatch.setattr(
        runner,
        "_write_run_summary",
        lambda **kwargs: summary_calls.append(kwargs),
    )

    metrics = runner.run_postprocess()

    assert metrics["train.total_loss"] == 1.0
    assert metrics["pair_benchmark.same_vs_different_cosine_auc_like"] == 0.9
    assert export_calls == ["val"]
    assert summary_calls[-1]["train_manifest"] is None
    assert summary_calls[-1]["val_manifest"] is not None


def test_round2_runner_postprocess_rejects_artifact_evaluators_without_train_split(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _build_cfg(tmp_path)
    cfg.evaluation.run_knn = True
    cfg.round2.evaluation.run_pair_benchmark = False
    cfg.round2.evaluation.run_slicing_analysis = False
    run_dir = Path(cfg.run.output_root) / str(cfg.run.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = Round2SSLRunner(cfg=cfg, run_dir=run_dir)
    artifacts = runner._resolve_run_artifacts()

    monkeypatch.setattr(
        runner,
        "_resolve_postprocess_checkpoint_path",
        lambda **kwargs: artifacts.checkpoint_dir / "latest.pt",
    )
    monkeypatch.setattr(
        CheckpointManager,
        "load_full_resume",
        lambda self, **kwargs: None,
    )
    monkeypatch.setattr(
        runner,
        "_build_split_dataloader_or_none",
        lambda split: None if split == "train" else SimpleNamespace(dataset=[1]),
    )
    monkeypatch.setattr(
        runner,
        "_export_split",
        lambda **kwargs: SimpleNamespace(num_samples=1, embedding_dim=8, has_labels=False),
    )

    with pytest.raises(ValueError, match="require a non-empty train split export"):
        runner.run_postprocess()


def test_round2_runner_distributed_preflight_runs_only_on_rank_zero(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _build_cfg(tmp_path)
    run_dir = Path(cfg.run.output_root) / str(cfg.run.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = Round2SSLRunner(cfg=cfg, run_dir=run_dir)
    runner._distributed_enabled = True
    runner._world_size = 2
    runner._rank = 1

    counters = {
        "preflight": 0,
        "barrier": 0,
        "broadcast": 0,
    }

    monkeypatch.setattr(
        round2_runner_module.torch.distributed,
        "is_initialized",
        lambda: True,
    )

    def _fake_broadcast(payload, src: int) -> None:
        assert src == 0
        counters["broadcast"] += 1

    monkeypatch.setattr(
        round2_runner_module.torch.distributed,
        "broadcast_object_list",
        _fake_broadcast,
    )
    monkeypatch.setattr(
        runner,
        "_run_preflight_validation",
        lambda **kwargs: counters.__setitem__("preflight", counters["preflight"] + 1),
    )
    monkeypatch.setattr(
        runner,
        "_barrier_if_distributed",
        lambda: counters.__setitem__("barrier", counters["barrier"] + 1),
    )

    artifacts = runner._resolve_run_artifacts()
    resume_state = round2_runner_module.Round2ResumeState(
        enabled=False,
        mode="full_resume",
        resolved_checkpoint_path=None,
        restored_from_checkpoint=False,
    )

    runner._run_distributed_preflight_validation(
        artifacts=artifacts,
        resume_state=resume_state,
    )

    assert counters["preflight"] == 0
    assert counters["broadcast"] == 1
    assert counters["barrier"] == 1


def test_round2_runner_resume_helpers_restore_full_and_warm_start(
    tmp_path: Path,
) -> None:
    cfg = _build_cfg(tmp_path)
    run_dir = Path(cfg.run.output_root) / str(cfg.run.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = Round2SSLRunner(cfg=cfg, run_dir=run_dir)
    artifacts = runner._resolve_run_artifacts()
    manager = CheckpointManager(artifacts.checkpoint_dir)

    ssl_module, _ = runner._build_ssl_module()
    optimizer = runner._build_optimizer(ssl_module)
    scheduler = runner._build_scheduler(optimizer)
    trainer_state = round2_runner_module.TrainerState(
        epoch=1,
        global_step=7,
        best_metric_name="train.total_loss",
        best_metric_value=0.5,
    )

    manager.save(
        model=ssl_module,
        trainer_state=trainer_state,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        epoch=trainer_state.epoch,
        global_step=trainer_state.global_step,
        is_best=True,
    )

    resumed_module, _ = runner._build_ssl_module()
    resumed_optimizer = runner._build_optimizer(resumed_module)
    resumed_scheduler = runner._build_scheduler(resumed_optimizer)
    resumed_state = round2_runner_module.TrainerState()

    full_resume_state = runner._maybe_resume_round2(
        checkpoint_manager=manager,
        ssl_module=resumed_module,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        trainer_state=resumed_state,
        resume_state=round2_runner_module.Round2ResumeState(
            enabled=True,
            mode="full_resume",
            resolved_checkpoint_path=manager.get_latest_checkpoint_path(),
            restored_from_checkpoint=False,
        ),
    )

    assert full_resume_state.restored_from_checkpoint is True
    assert resumed_state.epoch == 1
    assert resumed_state.global_step == 7
    assert resumed_state.best_metric_value == 0.5

    warm_module, _ = runner._build_ssl_module()
    warm_optimizer = runner._build_optimizer(warm_module)
    warm_scheduler = runner._build_scheduler(warm_optimizer)
    warm_state = round2_runner_module.TrainerState()
    original_weight = next(
        warm_module.student_global_projector.parameters()
    ).detach().clone()

    with torch.no_grad():
        for parameter in ssl_module.parameters():
            parameter.add_(0.25)
    manager.save(
        model=ssl_module,
        trainer_state=trainer_state,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        epoch=2,
        global_step=10,
        is_best=False,
    )

    warm_resume_state = runner._maybe_resume_round2(
        checkpoint_manager=manager,
        ssl_module=warm_module,
        optimizer=warm_optimizer,
        scheduler=warm_scheduler,
        trainer_state=warm_state,
        resume_state=round2_runner_module.Round2ResumeState(
            enabled=True,
            mode="warm_start",
            resolved_checkpoint_path=manager.get_latest_checkpoint_path(),
            restored_from_checkpoint=False,
        ),
    )

    resumed_weight = next(
        warm_module.student_global_projector.parameters()
    ).detach().clone()
    assert warm_resume_state.restored_from_checkpoint is True
    assert warm_state.epoch == 0
    assert warm_state.global_step == 0
    assert not torch.allclose(original_weight, resumed_weight)


def test_round2_runner_resume_unwraps_wrapped_module_before_load(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _build_cfg(tmp_path)
    run_dir = Path(cfg.run.output_root) / str(cfg.run.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = Round2SSLRunner(cfg=cfg, run_dir=run_dir)
    artifacts = runner._resolve_run_artifacts()
    manager = CheckpointManager(artifacts.checkpoint_dir)

    saved_module, _ = runner._build_ssl_module()
    saved_optimizer = runner._build_optimizer(saved_module)
    saved_scheduler = runner._build_scheduler(saved_optimizer)
    saved_state = round2_runner_module.TrainerState(
        epoch=1,
        global_step=5,
        best_metric_name="train.total_loss",
        best_metric_value=0.25,
    )

    with torch.no_grad():
        for parameter in saved_module.parameters():
            parameter.add_(0.5)

    manager.save(
        model=saved_module,
        trainer_state=saved_state,
        optimizer=saved_optimizer,
        lr_scheduler=saved_scheduler,
        epoch=saved_state.epoch,
        global_step=saved_state.global_step,
        is_best=False,
    )

    class _FakeDDP(torch.nn.Module):
        def __init__(self, module: torch.nn.Module) -> None:
            super().__init__()
            self.module = module

    monkeypatch.setattr(round2_runner_module, "DistributedDataParallel", _FakeDDP)

    resumed_module, _ = runner._build_ssl_module()
    resumed_optimizer = runner._build_optimizer(resumed_module)
    resumed_scheduler = runner._build_scheduler(resumed_optimizer)
    resumed_state = round2_runner_module.TrainerState()
    before_resume_weight = next(
        resumed_module.student_global_projector.parameters()
    ).detach().clone()

    resume_outcome = runner._maybe_resume_round2(
        checkpoint_manager=manager,
        ssl_module=_FakeDDP(resumed_module),
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        trainer_state=resumed_state,
        resume_state=round2_runner_module.Round2ResumeState(
            enabled=True,
            mode="full_resume",
            resolved_checkpoint_path=manager.get_latest_checkpoint_path(),
            restored_from_checkpoint=False,
        ),
    )

    after_resume_weight = next(
        resumed_module.student_global_projector.parameters()
    ).detach().clone()
    assert resume_outcome.restored_from_checkpoint is True
    assert resumed_state.epoch == 1
    assert resumed_state.global_step == 5
    assert not torch.allclose(before_resume_weight, after_resume_weight)


def test_round2_runner_full_resume_reconciles_cosine_scheduler_horizon(
    tmp_path: Path,
) -> None:
    cfg = _build_cfg(tmp_path)
    cfg.round2.scheduler.name = "cosine"
    cfg.round2.scheduler.min_learning_rate = 0.0
    cfg.train.num_epochs = 1
    run_dir = Path(cfg.run.output_root) / str(cfg.run.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = Round2SSLRunner(cfg=cfg, run_dir=run_dir)
    artifacts = runner._resolve_run_artifacts()
    manager = CheckpointManager(artifacts.checkpoint_dir)

    saved_module, _ = runner._build_ssl_module()
    saved_optimizer = runner._build_optimizer(saved_module)
    saved_scheduler = runner._build_scheduler(saved_optimizer)
    assert saved_scheduler is not None
    saved_state = round2_runner_module.TrainerState(
        epoch=1,
        global_step=4,
        best_metric_name="train.total_loss",
        best_metric_value=0.3,
    )

    manager.save(
        model=saved_module,
        trainer_state=saved_state,
        optimizer=saved_optimizer,
        lr_scheduler=saved_scheduler,
        epoch=saved_state.epoch,
        global_step=saved_state.global_step,
        is_best=False,
    )

    cfg.train.num_epochs = 3
    resumed_runner = Round2SSLRunner(cfg=cfg, run_dir=run_dir)
    resumed_module, _ = resumed_runner._build_ssl_module()
    resumed_optimizer = resumed_runner._build_optimizer(resumed_module)
    resumed_scheduler = resumed_runner._build_scheduler(resumed_optimizer)
    resumed_state = round2_runner_module.TrainerState()

    resume_outcome = resumed_runner._maybe_resume_round2(
        checkpoint_manager=manager,
        ssl_module=resumed_module,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        trainer_state=resumed_state,
        resume_state=round2_runner_module.Round2ResumeState(
            enabled=True,
            mode="full_resume",
            resolved_checkpoint_path=manager.get_latest_checkpoint_path(),
            restored_from_checkpoint=False,
        ),
    )

    assert resume_outcome.restored_from_checkpoint is True
    assert resumed_scheduler is not None
    assert resumed_scheduler.last_epoch == saved_scheduler.last_epoch
    assert resumed_scheduler.T_max == 3


def test_round2_runner_build_train_dataloader_uses_same_dataset_instance_for_ddp(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _build_cfg(tmp_path)
    run_dir = Path(cfg.run.output_root) / str(cfg.run.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = Round2SSLRunner(cfg=cfg, run_dir=run_dir)
    runner._distributed_enabled = True
    runner._world_size = 2
    runner._rank = 1

    dataset = [0, 1, 2, 3]

    monkeypatch.setattr(round2_runner_module, "build_dataset", lambda cfg, split: dataset)

    def _fake_build_dataloader(
        cfg: Any,
        split: str,
        *,
        dataset: Any = None,
        sampler: Any = None,
        shuffle: bool | None = None,
    ) -> Any:
        assert split == "train"
        assert dataset is not None
        assert dataset is sampler.dataset
        assert shuffle is False
        return SimpleNamespace(dataset=dataset, sampler=sampler)

    monkeypatch.setattr(round2_runner_module, "build_dataloader", _fake_build_dataloader)

    dataloader = runner._build_train_dataloader()
    assert dataloader.dataset is dataset
    assert dataloader.sampler.dataset is dataset


def test_round2_runner_preflight_rejects_cross_run_full_resume(
    tmp_path: Path,
) -> None:
    source_cfg = _build_cfg(tmp_path)
    source_cfg.run.run_name = "source_run"
    source_run_dir = Path(source_cfg.run.output_root) / str(source_cfg.run.run_name)
    source_run_dir.mkdir(parents=True, exist_ok=True)
    source_runner = Round2SSLRunner(cfg=source_cfg, run_dir=source_run_dir)
    source_artifacts = source_runner._resolve_run_artifacts()
    source_manager = CheckpointManager(source_artifacts.checkpoint_dir)

    source_module, _ = source_runner._build_ssl_module()
    source_optimizer = source_runner._build_optimizer(source_module)
    source_scheduler = source_runner._build_scheduler(source_optimizer)
    source_state = round2_runner_module.TrainerState(epoch=1, global_step=3)
    source_manager.save(
        model=source_module,
        trainer_state=source_state,
        optimizer=source_optimizer,
        lr_scheduler=source_scheduler,
        epoch=source_state.epoch,
        global_step=source_state.global_step,
        is_best=False,
    )

    target_cfg = _build_cfg(tmp_path)
    target_cfg.run.run_name = "target_run"
    target_cfg.train.resume = {
        "enabled": True,
        "mode": "full_resume",
        "checkpoint_path": str(source_manager.get_latest_checkpoint_path()),
        "auto_resume_latest": False,
    }
    target_run_dir = Path(target_cfg.run.output_root) / str(target_cfg.run.run_name)
    target_run_dir.mkdir(parents=True, exist_ok=True)
    target_runner = Round2SSLRunner(cfg=target_cfg, run_dir=target_run_dir)
    target_artifacts = target_runner._resolve_run_artifacts()
    resume_state = target_runner._resolve_resume_state(target_artifacts.checkpoint_dir)

    with pytest.raises(FileExistsError, match="full_resume requires a checkpoint from the same run lineage"):
        target_runner._run_preflight_validation(
            artifacts=target_artifacts,
            resume_state=resume_state,
        )


def test_round2_runner_preflight_allows_cross_run_warm_start(
    tmp_path: Path,
) -> None:
    source_cfg = _build_cfg(tmp_path)
    source_cfg.run.run_name = "source_run"
    source_run_dir = Path(source_cfg.run.output_root) / str(source_cfg.run.run_name)
    source_run_dir.mkdir(parents=True, exist_ok=True)
    source_runner = Round2SSLRunner(cfg=source_cfg, run_dir=source_run_dir)
    source_artifacts = source_runner._resolve_run_artifacts()
    source_manager = CheckpointManager(source_artifacts.checkpoint_dir)

    source_module, _ = source_runner._build_ssl_module()
    source_optimizer = source_runner._build_optimizer(source_module)
    source_scheduler = source_runner._build_scheduler(source_optimizer)
    source_state = round2_runner_module.TrainerState(epoch=1, global_step=3)
    source_manager.save(
        model=source_module,
        trainer_state=source_state,
        optimizer=source_optimizer,
        lr_scheduler=source_scheduler,
        epoch=source_state.epoch,
        global_step=source_state.global_step,
        is_best=False,
    )

    target_cfg = _build_cfg(tmp_path)
    target_cfg.run.run_name = "target_run"
    target_cfg.train.resume = {
        "enabled": True,
        "mode": "warm_start",
        "checkpoint_path": str(source_manager.get_latest_checkpoint_path()),
        "auto_resume_latest": False,
    }
    target_run_dir = Path(target_cfg.run.output_root) / str(target_cfg.run.run_name)
    target_run_dir.mkdir(parents=True, exist_ok=True)
    target_runner = Round2SSLRunner(cfg=target_cfg, run_dir=target_run_dir)
    target_artifacts = target_runner._resolve_run_artifacts()
    resume_state = target_runner._resolve_resume_state(target_artifacts.checkpoint_dir)

    target_runner._run_preflight_validation(
        artifacts=target_artifacts,
        resume_state=resume_state,
    )


def test_round2_runner_pair_benchmark_fails_fast_when_embedding_manifest_missing(
    tmp_path: Path,
) -> None:
    cfg = _build_cfg(tmp_path)
    cfg.round2.evaluation.run_pair_benchmark = True
    cfg.round2.evaluation.run_slicing_analysis = False
    cfg.round2.evaluation.pair_benchmark.embedding_splits = ["val"]
    run_dir = Path(cfg.run.output_root) / str(cfg.run.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = Round2SSLRunner(cfg=cfg, run_dir=run_dir)
    artifacts = runner._resolve_run_artifacts()

    with pytest.raises(FileNotFoundError, match="embedding manifests"):
        runner._run_pair_benchmark_suite(artifacts)


def test_round2_runner_pair_benchmark_rejects_unsupported_embedding_split(
    tmp_path: Path,
) -> None:
    cfg = _build_cfg(tmp_path)
    cfg.round2.evaluation.run_pair_benchmark = True
    cfg.round2.evaluation.run_slicing_analysis = False
    cfg.round2.evaluation.pair_benchmark.embedding_splits = ["query"]
    run_dir = Path(cfg.run.output_root) / str(cfg.run.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = Round2SSLRunner(cfg=cfg, run_dir=run_dir)
    artifacts = runner._resolve_run_artifacts()

    with pytest.raises(ValueError, match="unsupported embedding_splits"):
        runner._run_pair_benchmark_suite(artifacts)
