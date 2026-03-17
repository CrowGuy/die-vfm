from __future__ import annotations

from pathlib import Path

from hydra import compose
from hydra import initialize_config_dir


from pathlib import Path

from hydra import compose, initialize_config_dir


def test_base_config_loads() -> None:
    """Tests that the Hydra config composes successfully."""
    config_dir = str(Path("configs").resolve())

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config")

    assert cfg.project.name == "die_vfm"
    assert cfg.run.output_root == "runs"
    assert cfg.system.seed == 42
    assert cfg.system.num_workers == 4

    assert cfg.experiment.mode == "round1_frozen"
    assert cfg.model.backbone.name == "dinov2"
    assert cfg.model.pooler.name == "attn_pooler_v1"

    assert cfg.train.max_epochs == 1
    assert cfg.train.log_every_n_steps == 10
    assert cfg.train.run_dataloader_smoke_test is True

    assert cfg.dataset.name == "dummy"
    assert list(cfg.dataset.image_size) == [224, 224]
    assert cfg.dataset.num_channels == 3
    assert cfg.dataset.num_classes == 5
    assert cfg.dataset.train_size == 16
    assert cfg.dataset.val_size == 8
    assert cfg.dataset.label_offset == 0
    assert cfg.dataset.split_seed.train == 101
    assert cfg.dataset.split_seed.val == 202

    assert cfg.dataloader.batch_size == 4
    assert cfg.dataloader.drop_last is False
    assert cfg.dataloader.pin_memory is False
    assert cfg.dataloader.persistent_workers is False

    assert cfg.evaluation.run_linear_probe is True
    assert cfg.evaluation.run_knn is True