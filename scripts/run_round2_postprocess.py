from __future__ import annotations

import logging
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf

from die_vfm.trainer.round2_runner import Round2SSLRunner

LOGGER = logging.getLogger(__name__)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_run_dir(cfg: DictConfig) -> Path:
    output_root = Path(cfg.run.output_root)
    run_name = cfg.run.run_name
    if run_name is None:
        run_dir = output_root / "default"
    else:
        run_dir = output_root / str(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logging(run_dir: Path) -> None:
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "round2_postprocess.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)


def save_config_snapshot(cfg: DictConfig, run_dir: Path) -> None:
    if not bool(cfg.run.save_config_snapshot):
        return
    config_path = run_dir / "round2_postprocess_config.yaml"
    OmegaConf.save(cfg, config_path)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_dir = resolve_run_dir(cfg)
    setup_logging(run_dir)

    LOGGER.info("Starting Round2 postprocess.")
    LOGGER.info("Resolved run directory: %s", run_dir)

    set_random_seed(int(cfg.system.seed))
    LOGGER.info("Random seed set to %d.", int(cfg.system.seed))

    save_config_snapshot(cfg, run_dir)
    LOGGER.info("Saved config snapshot to run directory.")

    runner = Round2SSLRunner(cfg=cfg, run_dir=run_dir)
    metrics = runner.run_postprocess()

    LOGGER.info("Round2 postprocess completed successfully.")
    LOGGER.info("Metric keys: %s", sorted(metrics.keys()))


if __name__ == "__main__":
    main()
