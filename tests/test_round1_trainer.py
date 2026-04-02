from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from die_vfm.trainer import CheckpointManager
from die_vfm.trainer import Round1FrozenTrainer


def _build_cfg(tmp_path: Path):
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
            "artifact": {
                "embedding": {
                    "save_token_weights": True,
                },
            },
            "evaluation": {
                "run_linear_probe": True,
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

# -----------------------------
# Test 1: full round1 run
# -----------------------------
def test_round1_trainer_writes_artifacts_and_checkpoints(
    tmp_path: Path,
) -> None:
    cfg = _build_cfg(tmp_path)

    run_dir = Path(cfg.run.output_root) / cfg.run.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_manager = CheckpointManager(run_dir / "checkpoints")

    trainer = Round1FrozenTrainer(
        cfg=cfg,
        run_dir=run_dir,
        checkpoint_manager=checkpoint_manager,
    )

    metrics = trainer.run()

    epoch_dir = run_dir / "round1" / "epoch_0000"

    # ---- metrics ----
    assert "linear_probe.accuracy" in metrics

    # ---- embedding artifacts ----
    assert (epoch_dir / "embeddings" / "train").exists()
    assert (epoch_dir / "embeddings" / "val").exists()

    # ---- evaluator outputs ----
    assert (epoch_dir / "evaluation" / "linear_probe").exists()
    assert (epoch_dir / "evaluation" / "knn").exists()
    assert (epoch_dir / "evaluation" / "retrieval").exists()

    # ---- summary ----
    assert (epoch_dir / "round1_summary.yaml").exists()
    assert (epoch_dir / "round1_summary.json").exists()

    # ---- checkpoints ----
    assert checkpoint_manager.get_latest_checkpoint_path().exists()
    assert checkpoint_manager.get_best_checkpoint_path().exists()
    assert checkpoint_manager.get_epoch_checkpoint_path(0).exists()


# -----------------------------
# Test 2: full_resume correctness
# -----------------------------
def test_round1_trainer_full_resume_continues_from_next_epoch(
    tmp_path: Path,
) -> None:
    cfg = _build_cfg(tmp_path)
    cfg.train.num_epochs = 1

    run_dir = Path(cfg.run.output_root) / cfg.run.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_manager = CheckpointManager(run_dir / "checkpoints")

    # first run (epoch 0)
    trainer = Round1FrozenTrainer(
        cfg=cfg,
        run_dir=run_dir,
        checkpoint_manager=checkpoint_manager,
    )
    trainer.run()

    # resume run (epoch 1)
    resume_cfg = _build_cfg(tmp_path)
    resume_cfg.train.num_epochs = 2
    resume_cfg.train.resume.enabled = True
    resume_cfg.train.resume.mode = "full_resume"
    resume_cfg.train.resume.auto_resume_latest = True

    resumed_trainer = Round1FrozenTrainer(
        cfg=resume_cfg,
        run_dir=run_dir,
        checkpoint_manager=checkpoint_manager,
    )
    resumed_trainer.run()

    # ---- new epoch exists ----
    assert checkpoint_manager.get_epoch_checkpoint_path(1).exists()

    epoch1_dir = run_dir / "round1" / "epoch_0001"
    assert epoch1_dir.exists()
    assert (epoch1_dir / "round1_summary.yaml").exists()