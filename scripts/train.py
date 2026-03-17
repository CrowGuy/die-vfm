from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from die_vfm.datasets.builder import build_dataloader


LOGGER = logging.getLogger(__name__)


def set_random_seed(seed: int) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(run_dir: Path) -> None:
    """Sets up logging for console and file output."""
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "run.log"

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


def resolve_run_dir(cfg: DictConfig) -> Path:
    """Resolves the output run directory."""
    output_root = Path(cfg.run.output_root)

    run_name = cfg.run.run_name
    if run_name is None:
        run_dir = output_root / "default"
    else:
        run_dir = output_root / str(run_name)

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config_snapshot(cfg: DictConfig, run_dir: Path) -> None:
    """Saves the resolved config snapshot into the run directory."""
    if not bool(cfg.run.save_config_snapshot):
        return

    config_path = run_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)


def _format_label_shape(label: Optional[torch.Tensor]) -> str:
    """Formats label tensor shape for logging."""
    if label is None:
        return "None"
    return str(tuple(label.shape))

def log_dataset_metadata(dataset: Any) -> None:
    """Logs dataset-level metadata when available."""
    if not hasattr(dataset, "get_dataset_metadata"):
        LOGGER.info("Dataset metadata: unavailable")
        return

    metadata = dataset.get_dataset_metadata()
    LOGGER.info("Dataset metadata: %s", metadata)


def save_dataset_metadata(dataset: Any, run_dir: Path) -> None:
    """Saves dataset-level metadata into the run directory."""
    if not hasattr(dataset, "get_dataset_metadata"):
        return

    metadata = dataset.get_dataset_metadata()
    metadata_path = run_dir / "dataset_metadata.yaml"
    OmegaConf.save(config=OmegaConf.create(metadata), f=metadata_path)

def run_dataloader_smoke_test(cfg: DictConfig, run_dir: Path) -> None:
    """Builds the train dataloader and reads one batch."""
    train_loader = build_dataloader(cfg, split="train")

    log_dataset_metadata(train_loader.dataset)
    save_dataset_metadata(train_loader.dataset, run_dir)

    batch = next(iter(train_loader))
    image = batch["image"]
    label = batch["label"]
    image_ids = batch["image_id"]

    LOGGER.info("Dataloader smoke test passed.")
    LOGGER.info("Batch image shape: %s", tuple(image.shape))
    LOGGER.info("Batch label shape: %s", _format_label_shape(label))
    LOGGER.info("Batch image ids: %s", image_ids)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training entrypoint."""
    run_dir = resolve_run_dir(cfg)
    setup_logging(run_dir)

    LOGGER.info("Starting training bootstrap.")
    LOGGER.info("Resolved run directory: %s", run_dir)

    set_random_seed(int(cfg.system.seed))
    LOGGER.info("Random seed set to %d.", int(cfg.system.seed))

    save_config_snapshot(cfg, run_dir)
    LOGGER.info("Saved config snapshot to run directory.")

    if bool(cfg.train.run_dataloader_smoke_test):
        run_dataloader_smoke_test(cfg, run_dir)

    LOGGER.info("Training bootstrap completed successfully.")


if __name__ == "__main__":
    main()