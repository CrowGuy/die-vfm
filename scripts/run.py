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
from die_vfm.models.builder import build_model
from die_vfm.trainer import CheckpointManager
from die_vfm.trainer import Round1FrozenRunner
from die_vfm.trainer import TrainerState

LOGGER = logging.getLogger(__name__)


def save_model_smoke_artifact(
    model: Any,
    batch: dict[str, Any],
    output: Any,
    run_dir: Path,
) -> None:
    """Saves model smoke test metadata into the run directory."""
    artifact = {
        "model": {
            "name": model.__class__.__name__,
            "embedding_dim": int(model.embedding_dim),
            "backbone": model.backbone.__class__.__name__,
            "pooler": model.pooler.__class__.__name__,
        },
        "batch": {
            "image_shape": list(batch["image"].shape),
            "batch_size": int(batch["image"].shape[0]),
        },
        "output": {
            "embedding_shape": list(output.embedding.shape),
        },
    }

    if output.backbone is not None:
        artifact["output"]["backbone"] = {
            "patch_tokens_shape": list(output.backbone.patch_tokens.shape),
            "feature_dim": int(output.backbone.feature_dim),
            "patch_grid": (
                list(output.backbone.patch_grid)
                if output.backbone.patch_grid is not None
                else None
            ),
        }

    artifact_path = run_dir / "model_smoke.yaml"
    OmegaConf.save(config=OmegaConf.create(artifact), f=artifact_path)


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


def maybe_resume_model(
    cfg: DictConfig,
    checkpoint_manager: CheckpointManager,
    model: torch.nn.Module,
) -> TrainerState:
    """Resolves and applies resume policy for the model."""
    trainer_state = TrainerState()

    if not bool(cfg.train.resume.enabled):
        LOGGER.info("Resume disabled.")
        return trainer_state

    checkpoint_path = checkpoint_manager.resolve_resume_path(
        checkpoint_path=cfg.train.resume.checkpoint_path,
        auto_resume_latest=bool(cfg.train.resume.auto_resume_latest),
    )
    if checkpoint_path is None:
        LOGGER.info("Resume requested but no checkpoint was resolved.")
        return trainer_state

    resume_mode = str(cfg.train.resume.mode)
    LOGGER.info("Resolved resume checkpoint: %s", checkpoint_path)
    LOGGER.info("Resume mode: %s", resume_mode)

    if resume_mode == "warm_start":
        checkpoint_manager.load_warm_start(
            checkpoint_path=checkpoint_path,
            model=model,
            strict=True,
        )
        LOGGER.info("Warm start completed from checkpoint.")
        return trainer_state

    if resume_mode == "full_resume":
        checkpoint_manager.load_full_resume(
            checkpoint_path=checkpoint_path,
            model=model,
            trainer_state=trainer_state,
            strict=True,
        )
        LOGGER.info(
            "Full resume completed. Restored epoch=%d global_step=%d.",
            trainer_state.epoch,
            trainer_state.global_step,
        )
        return trainer_state

    raise ValueError(f"Unsupported resume mode: {resume_mode}")


def save_bootstrap_checkpoint(
    cfg: DictConfig,
    checkpoint_manager: CheckpointManager,
    model: torch.nn.Module,
    trainer_state: TrainerState,
) -> None:
    """Saves bootstrap checkpoints for M1 closeout."""
    is_best = not checkpoint_manager.get_best_checkpoint_path().exists()

    written_paths = checkpoint_manager.save(
        model=model,
        trainer_state=trainer_state,
        epoch=trainer_state.epoch,
        global_step=trainer_state.global_step,
        is_best=is_best,
        extra_metadata={
            "phase": "bootstrap",
            "run_name": cfg.run.run_name,
            "resume_enabled": bool(cfg.train.resume.enabled),
            "resume_mode": str(cfg.train.resume.mode),
        },
    )

    LOGGER.info("Saved latest checkpoint: %s", written_paths["latest"])
    LOGGER.info("Saved epoch checkpoint: %s", written_paths["epoch"])
    if "best" in written_paths:
        LOGGER.info("Saved best checkpoint: %s", written_paths["best"])


def run_dataloader_smoke_test(
    cfg: DictConfig,
    run_dir: Path,
    checkpoint_manager: CheckpointManager,
) -> None:
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

    model = build_model(cfg.model)
    LOGGER.info("Built model: %s", model.__class__.__name__)
    LOGGER.info("Backbone: %s", model.backbone.__class__.__name__)
    LOGGER.info("Pooler: %s", model.pooler.__class__.__name__)

    trainer_state = maybe_resume_model(
        cfg=cfg,
        checkpoint_manager=checkpoint_manager,
        model=model,
    )

    output = model(image)

    LOGGER.info("Model forward completed.")
    LOGGER.info("Embedding shape: %s", tuple(output.embedding.shape))

    save_model_smoke_artifact(
        model=model,
        batch=batch,
        output=output,
        run_dir=run_dir,
    )

    if output.backbone is not None:
        LOGGER.info(
            "Patch tokens shape: %s",
            tuple(output.backbone.patch_tokens.shape),
        )

    trainer_state.global_step += 1
    save_bootstrap_checkpoint(
        cfg=cfg,
        checkpoint_manager=checkpoint_manager,
        model=model,
        trainer_state=trainer_state,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main runtime entrypoint."""
    run_dir = resolve_run_dir(cfg)
    setup_logging(run_dir)

    LOGGER.info("Starting runtime entrypoint.")
    LOGGER.info("Resolved run directory: %s", run_dir)

    set_random_seed(int(cfg.system.seed))
    LOGGER.info("Random seed set to %d.", int(cfg.system.seed))

    save_config_snapshot(cfg, run_dir)
    LOGGER.info("Saved config snapshot to run directory.")

    mode = str(getattr(cfg.train, "mode", "bootstrap"))

    if mode == "bootstrap":
        LOGGER.info("Starting bootstrap runtime.")
        checkpoint_manager = CheckpointManager(run_dir / "checkpoints")
        if bool(cfg.train.run_dataloader_smoke_test):
            run_dataloader_smoke_test(
                cfg=cfg,
                run_dir=run_dir,
                checkpoint_manager=checkpoint_manager,
            )
        LOGGER.info("Training bootstrap completed successfully.")
        return

    if mode == "round1_frozen":
        runner = Round1FrozenRunner(
            cfg=cfg,
            run_dir=run_dir,
        )
        runner.run()
        LOGGER.info("Round1 frozen runner completed successfully.")
        return

    raise ValueError(f"Unsupported train.mode: {mode}")


if __name__ == "__main__":
    main()
