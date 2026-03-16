from __future__ import annotations

from pathlib import Path

from hydra import compose
from hydra import initialize_config_dir


def test_base_config_loads() -> None:
    """Tests that the Hydra config composes successfully."""
    config_dir = str(Path("configs").resolve())

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config")

    assert cfg.project.name == "die_vfm"
    assert cfg.run.output_root == "runs"
    assert cfg.system.seed == 42
    assert cfg.experiment.mode == "round1_frozen"
    assert cfg.model.backbone.name == "dinov2"
    assert cfg.model.pooler.name == "attn_pooler_v1"
    assert cfg.train.max_epochs == 1
    assert cfg.evaluation.run_linear_probe is True