from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from die_vfm.artifacts import export_split_embeddings
from die_vfm.datasets.builder import build_dataloader
from die_vfm.models.builder import build_model

LOGGER = logging.getLogger(__name__)


def set_random_seed(seed: int) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def setup_logging(run_dir: Path) -> None:
    """Sets up logging for console and file output."""
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "export_embeddings.log"

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
    """Saves the resolved config snapshot into the run directory."""
    if not bool(cfg.run.save_config_snapshot):
        return

    config_path = run_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)


def _resolve_export_splits(cfg: DictConfig) -> list[str]:
    """Resolves export splits from artifact config."""
    splits = [str(split) for split in cfg.artifact.embedding.export_splits]

    if bool(cfg.artifact.embedding.include_test_split) and "test" not in splits:
        splits.append("test")

    if not splits:
        raise ValueError("cfg.artifact.embedding.export_splits must be non-empty.")

    deduped_splits: list[str] = []
    for split in splits:
        if split not in deduped_splits:
            deduped_splits.append(split)
    return deduped_splits


def _resolve_embeddings_root(cfg: DictConfig, run_dir: Path) -> Path:
    """Resolves the root directory where embedding artifacts are stored."""
    output_subdir = str(cfg.artifact.embedding.output_subdir)
    embeddings_root = run_dir / output_subdir
    embeddings_root.mkdir(parents=True, exist_ok=True)
    return embeddings_root


def _resolve_device(cfg: DictConfig) -> torch.device:
    """Resolves torch device from config."""
    device = torch.device(str(cfg.system.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "cfg.system.device is set to CUDA, but CUDA is not available."
        )
    return device


def export_embeddings(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    """Builds model and exports configured embedding splits."""
    if not bool(cfg.artifact.embedding.enabled):
        LOGGER.info("Embedding artifact export disabled; skipping.")
        return {}

    device = _resolve_device(cfg)
    splits = _resolve_export_splits(cfg)
    embeddings_root = _resolve_embeddings_root(cfg, run_dir)

    LOGGER.info("Building model for embedding export.")
    model = build_model(cfg.model)
    model = model.to(device)

    LOGGER.info("Model built: %s", model.__class__.__name__)
    LOGGER.info("Export splits: %s", splits)
    LOGGER.info("Embeddings root: %s", embeddings_root)

    manifests: dict[str, Any] = {}
    for split in splits:
        LOGGER.info("Building dataloader for split=%s", split)
        dataloader = build_dataloader(cfg, split=split)

        split_dir = embeddings_root / split
        LOGGER.info("Exporting embeddings for split=%s to %s", split, split_dir)

        manifest = export_split_embeddings(
            model=model,
            dataloader=dataloader,
            output_dir=split_dir,
            split=split,
            device=device,
        )
        manifests[split] = manifest

        LOGGER.info(
            "Finished split=%s | num_samples=%d | embedding_dim=%d | has_labels=%s",
            split,
            manifest.num_samples,
            manifest.embedding_dim,
            manifest.has_labels,
        )

    return manifests


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Embedding artifact export entrypoint."""
    run_dir = resolve_run_dir(cfg)
    setup_logging(run_dir)

    LOGGER.info("Starting embedding export.")
    LOGGER.info("Resolved run directory: %s", run_dir)

    set_random_seed(int(cfg.system.seed))
    LOGGER.info("Random seed set to %d.", int(cfg.system.seed))

    save_config_snapshot(cfg, run_dir)
    LOGGER.info("Saved config snapshot to run directory.")

    manifests = export_embeddings(cfg, run_dir)

    if manifests:
        exported = ", ".join(sorted(manifests.keys()))
        LOGGER.info("Export completed successfully for splits: %s", exported)
    else:
        LOGGER.info("No embedding splits were exported.")

    LOGGER.info("Embedding export job completed successfully.")


if __name__ == "__main__":
    main()