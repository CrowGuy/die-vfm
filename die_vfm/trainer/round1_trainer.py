"""Round1 trainer orchestration for die_vfm."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from die_vfm.artifacts import export_split_embeddings
from die_vfm.datasets.builder import build_dataloader
from die_vfm.evaluator.knn_runner import build_knn_run_config
from die_vfm.evaluator.knn_runner import run_knn
from die_vfm.evaluator.linear_probe_runner import build_linear_probe_run_config
from die_vfm.evaluator.linear_probe_runner import run_linear_probe
from die_vfm.evaluator.retrieval_runner import build_retrieval_run_config
from die_vfm.evaluator.retrieval_runner import run_retrieval
from die_vfm.models.builder import build_model
from die_vfm.trainer.base_trainer import TrainerState
from die_vfm.trainer.checkpoint_manager import CheckpointManager

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Round1EpochArtifacts:
    """Resolved output locations for one Round1 epoch."""

    epoch_dir: Path
    train_embedding_dir: Path
    val_embedding_dir: Path
    linear_probe_dir: Path
    knn_dir: Path
    retrieval_dir: Path
    summary_yaml_path: Path
    summary_json_path: Path


class Round1FrozenTrainer:
    """Minimal Round1 frozen trainer orchestration.

    Scope is intentionally narrow:

    - Build model and dataloaders
    - Freeze modules according to Round1 policy
    - Export embedding artifacts
    - Run artifact-driven evaluators
    - Save summary + checkpoints
    - Support warm_start / full_resume

    This is not a generalized trainer framework.
    """

    def __init__(
        self,
        cfg: DictConfig,
        run_dir: str | Path,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        """Initializes the Round1 trainer."""
        self._cfg = cfg
        self._run_dir = Path(run_dir)
        self._checkpoint_manager = checkpoint_manager
        self._device = torch.device(str(cfg.system.device))

    def run(self) -> dict[str, float]:
        """Runs the Round1 frozen experiment end to end."""
        model = build_model(self._cfg.model)
        model.to(self._device)

        self._apply_freeze_policy(model)

        trainer_state = TrainerState()
        trainer_state = self._maybe_resume(
            model=model,
            trainer_state=trainer_state,
        )

        start_epoch = int(trainer_state.epoch)
        num_epochs = int(self._cfg.train.num_epochs)

        LOGGER.info("Round1 frozen trainer started.")
        LOGGER.info("Resolved start_epoch=%d num_epochs=%d", start_epoch, num_epochs)

        final_metrics: dict[str, float] = {}

        for epoch in range(start_epoch, num_epochs):
            LOGGER.info("Round1 epoch %d started.", epoch)

            epoch_artifacts = self._resolve_epoch_artifacts(epoch)
            self._ensure_epoch_dirs(epoch_artifacts)

            train_loader = build_dataloader(self._cfg, split="train")
            val_loader = build_dataloader(self._cfg, split="val")

            train_manifest = self._export_split(
                model=model,
                dataloader=train_loader,
                split="train",
                output_dir=epoch_artifacts.train_embedding_dir,
            )
            val_manifest = self._export_split(
                model=model,
                dataloader=val_loader,
                split="val",
                output_dir=epoch_artifacts.val_embedding_dir,
            )

            LOGGER.info(
                "Embedding export finished. "
                "train_samples=%d val_samples=%d embedding_dim=%d",
                int(train_manifest.num_samples),
                int(val_manifest.num_samples),
                int(train_manifest.embedding_dim),
            )

            metrics = self._run_evaluators(epoch_artifacts)
            final_metrics = metrics

            # print("DEBUG metrics keys:", sorted(metrics.keys()))
            # print("DEBUG metrics:", metrics)

            is_best = self._update_best_metric(
                trainer_state=trainer_state,
                metrics=metrics,
            )

            trainer_state.epoch = epoch + 1
            trainer_state.global_step += 1

            self._write_epoch_summary(
                epoch=epoch,
                trainer_state=trainer_state,
                metrics=metrics,
                epoch_artifacts=epoch_artifacts,
            )

            checkpoint_paths = self._checkpoint_manager.save(
                model=model,
                trainer_state=trainer_state,
                epoch=epoch,
                global_step=trainer_state.global_step,
                is_best=is_best,
                extra_metadata={
                    "phase": "round1_frozen",
                    "selection_metric": str(self._cfg.train.selection_metric),
                    "metrics": metrics,
                },
            )

            LOGGER.info("Round1 epoch %d completed.", epoch)
            LOGGER.info("Checkpoint paths: %s", checkpoint_paths)

        LOGGER.info("Round1 frozen trainer finished.")
        return final_metrics

    def _apply_freeze_policy(self, model: torch.nn.Module) -> None:
        """Applies the minimal Round1 freeze policy."""
        freeze_backbone = bool(self._cfg.train.freeze_backbone)
        freeze_pooler = bool(self._cfg.train.freeze_pooler)

        if freeze_backbone and hasattr(model, "backbone"):
            for parameter in model.backbone.parameters():
                parameter.requires_grad = False

        if freeze_pooler and hasattr(model, "pooler"):
            for parameter in model.pooler.parameters():
                parameter.requires_grad = False

        model.eval()

        LOGGER.info(
            "Applied freeze policy: freeze_backbone=%s freeze_pooler=%s",
            freeze_backbone,
            freeze_pooler,
        )

    def _maybe_resume(
        self,
        model: torch.nn.Module,
        trainer_state: TrainerState,
    ) -> TrainerState:
        """Applies resume policy using the existing checkpoint manager."""
        if not bool(self._cfg.train.resume.enabled):
            LOGGER.info("Resume disabled.")
            return trainer_state

        checkpoint_path = self._checkpoint_manager.resolve_resume_path(
            checkpoint_path=self._cfg.train.resume.checkpoint_path,
            auto_resume_latest=bool(self._cfg.train.resume.auto_resume_latest),
        )
        if checkpoint_path is None:
            LOGGER.info("Resume enabled but no checkpoint was resolved.")
            return trainer_state

        resume_mode = str(self._cfg.train.resume.mode)
        LOGGER.info("Resolved resume checkpoint: %s", checkpoint_path)
        LOGGER.info("Resume mode: %s", resume_mode)

        if resume_mode == "warm_start":
            self._checkpoint_manager.load_warm_start(
                checkpoint_path=checkpoint_path,
                model=model,
                strict=True,
                map_location=self._device,
            )
            LOGGER.info("Warm start completed.")
            return trainer_state

        if resume_mode == "full_resume":
            self._checkpoint_manager.load_full_resume(
                checkpoint_path=checkpoint_path,
                model=model,
                trainer_state=trainer_state,
                strict=True,
                map_location=self._device,
            )
            LOGGER.info(
                "Full resume completed. resumed_epoch=%d resumed_global_step=%d",
                int(trainer_state.epoch),
                int(trainer_state.global_step),
            )
            return trainer_state

        raise ValueError(f"Unsupported resume mode: {resume_mode}")

    def _resolve_epoch_artifacts(self, epoch: int) -> Round1EpochArtifacts:
        """Builds the Round1 epoch artifact layout."""
        epoch_dir = self._run_dir / "round1" / f"epoch_{epoch:04d}"
        return Round1EpochArtifacts(
            epoch_dir=epoch_dir,
            train_embedding_dir=epoch_dir / "embeddings" / "train",
            val_embedding_dir=epoch_dir / "embeddings" / "val",
            linear_probe_dir=epoch_dir / "evaluation" / "linear_probe",
            knn_dir=epoch_dir / "evaluation" / "knn",
            retrieval_dir=epoch_dir / "evaluation" / "retrieval",
            summary_yaml_path=epoch_dir / "round1_summary.yaml",
            summary_json_path=epoch_dir / "round1_summary.json",
        )

    def _ensure_epoch_dirs(self, artifacts: Round1EpochArtifacts) -> None:
        """Creates required epoch directories."""
        artifacts.train_embedding_dir.mkdir(parents=True, exist_ok=True)
        artifacts.val_embedding_dir.mkdir(parents=True, exist_ok=True)
        artifacts.linear_probe_dir.mkdir(parents=True, exist_ok=True)
        artifacts.knn_dir.mkdir(parents=True, exist_ok=True)
        artifacts.retrieval_dir.mkdir(parents=True, exist_ok=True)

    def _export_split(
        self,
        model: torch.nn.Module,
        dataloader: Any,
        split: str,
        output_dir: str | Path,
    ) -> Any:
        """Exports one split as an embedding artifact."""
        LOGGER.info("Exporting %s embeddings to %s", split, output_dir)
        manifest = export_split_embeddings(
            model=model,
            dataloader=dataloader,
            output_dir=output_dir,
            split=split,
            device=str(self._device),
        )
        return manifest

    def _run_evaluators(
        self,
        artifacts: Round1EpochArtifacts,
    ) -> dict[str, float]:
        """Runs enabled artifact-driven evaluators."""
        metrics: dict[str, float] = {}

        evaluation_cfg = self._cfg.evaluation

        if bool(evaluation_cfg.run_linear_probe):
            linear_probe_cfg = evaluation_cfg.linear_probe
            linear_probe_run_cfg = build_linear_probe_run_config(
                train_split_dir=artifacts.train_embedding_dir,
                val_split_dir=artifacts.val_embedding_dir,
                output_dir=artifacts.linear_probe_dir,
                normalize_embeddings=bool(
                    linear_probe_cfg.input.normalize_embeddings
                ),
                map_location=str(linear_probe_cfg.input.map_location),
                bias=bool(linear_probe_cfg.model.bias),
                batch_size=int(linear_probe_cfg.trainer.batch_size),
                num_epochs=int(linear_probe_cfg.trainer.num_epochs),
                learning_rate=float(linear_probe_cfg.trainer.learning_rate),
                weight_decay=float(linear_probe_cfg.trainer.weight_decay),
                optimizer_name=str(linear_probe_cfg.trainer.optimizer_name),
                momentum=float(linear_probe_cfg.trainer.momentum),
                device=str(linear_probe_cfg.trainer.device),
                seed=int(linear_probe_cfg.trainer.seed),
                selection_metric=str(
                    linear_probe_cfg.trainer.selection_metric
                ),
                save_predictions=bool(
                    linear_probe_cfg.output.save_predictions
                ),
                save_history=bool(linear_probe_cfg.output.save_history),
            )
            linear_probe_result = run_linear_probe(linear_probe_run_cfg)
            for key, value in linear_probe_result.val_metrics.items():
                metrics[f"linear_probe.{key}"] = float(value)

        if bool(evaluation_cfg.run_knn):
            knn_cfg = evaluation_cfg.knn
            knn_run_cfg = build_knn_run_config(
                train_split_dir=artifacts.train_embedding_dir,
                val_split_dir=artifacts.val_embedding_dir,
                output_dir=artifacts.knn_dir,
                normalize_embeddings=bool(knn_cfg.input.normalize_embeddings),
                map_location=str(knn_cfg.input.map_location),
                save_predictions=bool(knn_cfg.output.save_predictions),
                k=int(knn_cfg.evaluator.k),
                metric=str(knn_cfg.evaluator.metric),
                weighting=str(knn_cfg.evaluator.weighting),
                temperature=float(knn_cfg.evaluator.temperature),
                batch_size=int(knn_cfg.evaluator.batch_size),
                device=str(knn_cfg.evaluator.device),
                topk=tuple(knn_cfg.evaluator.topk),
            )
            knn_result = run_knn(knn_run_cfg)
            for key, value in knn_result.val_metrics.items():
                metrics[f"knn.{key}"] = float(value)

        if bool(evaluation_cfg.run_retrieval):
            retrieval_cfg = evaluation_cfg.retrieval
            retrieval_run_cfg = build_retrieval_run_config(
                train_split_dir=artifacts.train_embedding_dir,
                val_split_dir=artifacts.val_embedding_dir,
                output_dir=artifacts.retrieval_dir,
                normalize_embeddings=bool(
                    retrieval_cfg.input.normalize_embeddings
                ),
                map_location=str(retrieval_cfg.input.map_location),
                save_predictions=bool(retrieval_cfg.output.save_predictions),
                metric=str(retrieval_cfg.evaluator.metric),
                batch_size=int(retrieval_cfg.evaluator.batch_size),
                device=str(retrieval_cfg.evaluator.device),
                topk=tuple(retrieval_cfg.evaluator.topk),
                save_predictions_topk=int(
                    retrieval_cfg.evaluator.save_predictions_topk
                ),
                exclude_same_image_id=bool(
                    retrieval_cfg.evaluator.exclude_same_image_id
                ),
            )
            retrieval_result = run_retrieval(retrieval_run_cfg)
            for key, value in retrieval_result.val_metrics.items():
                metrics[f"retrieval.{key}"] = float(value)

        LOGGER.info("Round1 metrics: %s", metrics)
        return metrics

    def _update_best_metric(
        self,
        trainer_state: TrainerState,
        metrics: dict[str, float],
    ) -> bool:
        """Updates best metric tracking and returns whether current epoch is best."""
        metric_name = str(self._cfg.train.selection_metric)
        if metric_name not in metrics:
            raise KeyError(
                f"Selection metric not found in metrics: {metric_name}"
            )

        current_value = float(metrics[metric_name])
        best_value = trainer_state.best_metric_value

        if best_value is None or current_value > best_value:
            trainer_state.best_metric_name = metric_name
            trainer_state.best_metric_value = current_value
            LOGGER.info(
                "New best metric: %s=%.6f",
                metric_name,
                current_value,
            )
            return True
        return False

    def _write_epoch_summary(
        self,
        epoch: int,
        trainer_state: TrainerState,
        metrics: dict[str, float],
        epoch_artifacts: Round1EpochArtifacts,
    ) -> None:
        """Writes a compact Round1 epoch summary."""
        summary = {
            "phase": "round1_frozen",
            "epoch": int(epoch),
            "global_step": int(trainer_state.global_step),
            "selection_metric": str(self._cfg.train.selection_metric),
            "best_metric_name": trainer_state.best_metric_name,
            "best_metric_value": trainer_state.best_metric_value,
            "metrics": metrics,
            "artifacts": {
                "train_embedding_dir": str(epoch_artifacts.train_embedding_dir),
                "val_embedding_dir": str(epoch_artifacts.val_embedding_dir),
                "linear_probe_dir": str(epoch_artifacts.linear_probe_dir),
                "knn_dir": str(epoch_artifacts.knn_dir),
                "retrieval_dir": str(epoch_artifacts.retrieval_dir),
            },
        }

        OmegaConf.save(
            config=OmegaConf.create(summary),
            f=epoch_artifacts.summary_yaml_path,
        )
        with epoch_artifacts.summary_json_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(summary, file, indent=2, sort_keys=True)
