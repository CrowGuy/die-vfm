"""Round1 single-shot runner for die_vfm."""

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

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Round1RunArtifacts:
    """Resolved output locations for one Round1 single-shot run."""

    round1_dir: Path
    train_embedding_dir: Path
    val_embedding_dir: Path
    linear_probe_dir: Path
    knn_dir: Path
    retrieval_dir: Path
    summary_yaml_path: Path
    summary_json_path: Path


class Round1FrozenRunner:
    """Single-shot Round1 frozen runner.

    Scope is intentionally narrow:

    - Build model and dataloaders
    - Freeze modules according to Round1 policy
    - Export embedding artifacts
    - Run artifact-driven evaluators
    - Save run-level summary artifacts

    This is a frozen inference/evaluation runner, not a training loop.
    """

    def __init__(
        self,
        cfg: DictConfig,
        run_dir: str | Path,
    ) -> None:
        """Initializes the Round1 runner."""
        self._cfg = cfg
        self._run_dir = Path(run_dir)
        self._device = torch.device(str(cfg.system.device))

    def run(self) -> dict[str, float]:
        """Runs the Round1 frozen experiment end to end."""
        enabled_evaluators = self._run_preflight_validation()

        model = build_model(self._cfg.model)
        model.to(self._device)
        self._apply_freeze_policy(model)

        artifacts = self._resolve_run_artifacts()
        self._ensure_run_dirs(
            artifacts=artifacts,
            enabled_evaluators=enabled_evaluators,
        )

        LOGGER.info("Round1 frozen runner started.")
        LOGGER.info("Enabled Round1 evaluators: %s", enabled_evaluators)

        train_loader = build_dataloader(self._cfg, split="train")
        val_loader = build_dataloader(self._cfg, split="val")

        train_manifest = self._export_split(
            model=model,
            dataloader=train_loader,
            split="train",
            output_dir=artifacts.train_embedding_dir,
        )
        val_manifest = self._export_split(
            model=model,
            dataloader=val_loader,
            split="val",
            output_dir=artifacts.val_embedding_dir,
        )

        LOGGER.info(
            "Embedding export finished. train_samples=%d val_samples=%d embedding_dim=%d",
            int(train_manifest.num_samples),
            int(val_manifest.num_samples),
            int(train_manifest.embedding_dim),
        )

        metrics = self._run_evaluators(artifacts)
        self._write_run_summary(
            metrics=metrics,
            artifacts=artifacts,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            enabled_evaluators=enabled_evaluators,
        )

        LOGGER.info("Round1 frozen runner finished.")
        return metrics

    def _run_preflight_validation(self) -> list[str]:
        """Validates Round1 runtime assumptions before any heavy work starts."""
        if bool(OmegaConf.select(self._cfg, "train.resume.enabled", default=False)):
            raise ValueError(
                "round1_frozen does not support train.resume.*. "
                "Use a fresh run_name for Round1, or use bootstrap / future training rounds "
                "for checkpoint-resume behavior."
            )

        if bool(OmegaConf.select(self._cfg, "evaluation.run_centroid", default=False)):
            raise ValueError(
                "round1_frozen does not orchestrate centroid evaluation. "
                "Use standalone centroid script or disable evaluation.run_centroid."
            )

        evaluator_flags = self._resolve_round1_evaluator_flags()
        enabled_evaluators = [
            name for name, is_enabled in evaluator_flags.items() if is_enabled
        ]
        if not enabled_evaluators:
            raise ValueError(
                "Round1 requires at least one enabled evaluator. "
                "Set one of evaluation.run_linear_probe / run_knn / run_retrieval."
            )

        round1_dir = self._resolve_run_artifacts().round1_dir
        if round1_dir.exists() and any(round1_dir.iterdir()):
            raise FileExistsError(
                "Round1 single-shot run would overwrite existing outputs. "
                f"path={round1_dir}. Use a new run_name or clean the previous Round1 outputs."
            )

        return enabled_evaluators

    def _resolve_round1_evaluator_flags(self) -> dict[str, bool]:
        """Returns root-level evaluator enable flags used by Round1 orchestration."""
        return {
            "linear_probe": bool(
                OmegaConf.select(self._cfg, "evaluation.run_linear_probe", default=False)
            ),
            "knn": bool(OmegaConf.select(self._cfg, "evaluation.run_knn", default=False)),
            "retrieval": bool(
                OmegaConf.select(self._cfg, "evaluation.run_retrieval", default=False)
            ),
        }

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

    def _resolve_run_artifacts(self) -> Round1RunArtifacts:
        """Builds the Round1 run-level artifact layout."""
        round1_dir = self._run_dir / "round1"
        return Round1RunArtifacts(
            round1_dir=round1_dir,
            train_embedding_dir=round1_dir / "embeddings" / "train",
            val_embedding_dir=round1_dir / "embeddings" / "val",
            linear_probe_dir=round1_dir / "evaluation" / "linear_probe",
            knn_dir=round1_dir / "evaluation" / "knn",
            retrieval_dir=round1_dir / "evaluation" / "retrieval",
            summary_yaml_path=round1_dir / "round1_summary.yaml",
            summary_json_path=round1_dir / "round1_summary.json",
        )

    def _ensure_run_dirs(
        self,
        artifacts: Round1RunArtifacts,
        enabled_evaluators: list[str],
    ) -> None:
        """Creates required Round1 output directories."""
        artifacts.train_embedding_dir.mkdir(parents=True, exist_ok=True)
        artifacts.val_embedding_dir.mkdir(parents=True, exist_ok=True)

        enabled_set = set(enabled_evaluators)
        if "linear_probe" in enabled_set:
            artifacts.linear_probe_dir.mkdir(parents=True, exist_ok=True)
        if "knn" in enabled_set:
            artifacts.knn_dir.mkdir(parents=True, exist_ok=True)
        if "retrieval" in enabled_set:
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
        return export_split_embeddings(
            model=model,
            dataloader=dataloader,
            output_dir=output_dir,
            split=split,
            device=str(self._device),
        )

    def _run_evaluators(
        self,
        artifacts: Round1RunArtifacts,
    ) -> dict[str, float]:
        """Runs enabled artifact-driven evaluators."""
        metrics: dict[str, float] = {}

        evaluation_cfg = self._cfg.evaluation
        evaluator_flags = self._resolve_round1_evaluator_flags()

        if evaluator_flags["linear_probe"]:
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
                selection_metric=str(linear_probe_cfg.trainer.selection_metric),
                save_predictions=bool(linear_probe_cfg.output.save_predictions),
                save_history=bool(linear_probe_cfg.output.save_history),
            )
            linear_probe_result = run_linear_probe(linear_probe_run_cfg)
            for key, value in linear_probe_result.val_metrics.items():
                metrics[f"linear_probe.{key}"] = float(value)

        if evaluator_flags["knn"]:
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

        if evaluator_flags["retrieval"]:
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

    def _write_run_summary(
        self,
        *,
        metrics: dict[str, float],
        artifacts: Round1RunArtifacts,
        train_manifest: Any,
        val_manifest: Any,
        enabled_evaluators: list[str],
    ) -> None:
        """Writes a compact Round1 run summary."""
        enabled_set = set(enabled_evaluators)
        evaluation_dirs: dict[str, str] = {}
        if "linear_probe" in enabled_set:
            evaluation_dirs["linear_probe"] = str(artifacts.linear_probe_dir)
        if "knn" in enabled_set:
            evaluation_dirs["knn"] = str(artifacts.knn_dir)
        if "retrieval" in enabled_set:
            evaluation_dirs["retrieval"] = str(artifacts.retrieval_dir)

        summary = {
            "phase": "round1_frozen",
            "runtime_semantics": {
                "mode": "single_shot_inference_evaluation",
                "uses_gradient_updates": False,
                "uses_epoch_loop": False,
                "supports_resume": False,
                "supports_checkpoint_continuation": False,
            },
            "execution": {
                "enabled_evaluators": sorted(enabled_evaluators),
                "freeze_backbone": bool(self._cfg.train.freeze_backbone),
                "freeze_pooler": bool(self._cfg.train.freeze_pooler),
            },
            "manifests": {
                "train": {
                    "num_samples": int(train_manifest.num_samples),
                    "embedding_dim": int(train_manifest.embedding_dim),
                    "has_labels": bool(train_manifest.has_labels),
                },
                "val": {
                    "num_samples": int(val_manifest.num_samples),
                    "embedding_dim": int(val_manifest.embedding_dim),
                    "has_labels": bool(val_manifest.has_labels),
                },
            },
            "metrics": metrics,
            "artifacts": {
                "round1_dir": str(artifacts.round1_dir),
                "train_embedding_dir": str(artifacts.train_embedding_dir),
                "val_embedding_dir": str(artifacts.val_embedding_dir),
                "evaluation_dirs": evaluation_dirs,
            },
        }

        OmegaConf.save(
            config=OmegaConf.create(summary),
            f=artifacts.summary_yaml_path,
        )
        with artifacts.summary_json_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, sort_keys=True)
