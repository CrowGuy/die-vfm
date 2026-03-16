from __future__ import annotations

import pathlib

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

from die_vfm.utils.logging_utils import configure_logging
from die_vfm.utils.run_dir import create_run_dir
from die_vfm.utils.seed import set_global_seed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Bootstraps a training run."""
    set_global_seed(int(cfg.system.seed))

    run_directory = create_run_dir(
        output_root=str(cfg.run.output_root),
        run_name=cfg.run.run_name,
    )
    logger = configure_logging(run_directory / "logs")

    logger.info("Starting Die VFM run bootstrap.")
    logger.info("Project name: %s", cfg.project.name)
    logger.info("Experiment mode: %s", cfg.experiment.mode)
    logger.info("Backbone: %s", cfg.model.backbone.name)
    logger.info("Pooler: %s", cfg.model.pooler.name)
    logger.info("Output run directory: %s", run_directory)

    if bool(cfg.run.save_config_snapshot):
        config_path = pathlib.Path(run_directory) / "config.yaml"
        config_path.write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
        logger.info("Saved config snapshot to %s", config_path)

    logger.info("PR-1 skeleton bootstrapped successfully.")


if __name__ == "__main__":
    main()