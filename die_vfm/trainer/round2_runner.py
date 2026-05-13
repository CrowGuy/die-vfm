"""Round2 SSL trainer and end-only evaluator orchestration."""

from __future__ import annotations

import json
import importlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.distributed import DistributedSampler

from die_vfm.artifacts import default_manifest_path
from die_vfm.artifacts import export_split_embeddings
from die_vfm.datasets.builder import build_dataset
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
from die_vfm.trainer.checkpoint_manager import LATEST_CHECKPOINT_NAME
from die_vfm.trainer.round2_ssl import FlipMetadata
from die_vfm.trainer.round2_ssl import Round2SSLModule
from die_vfm.trainer.round2_ssl import UpdateModeResolution
from die_vfm.trainer.round2_ssl import apply_update_mode
from die_vfm.trainer.round2_ssl import autocast_context
from die_vfm.trainer.round2_ssl import canonicalize_patch_tokens
from die_vfm.trainer.round2_ssl import generate_augmented_view
from die_vfm.trainer.round2_ssl import projected_cosine_loss
from die_vfm.trainer.round2_ssl import reconcile_token_loss_trainability
from die_vfm.trainer.round2_ssl import resolve_ema_momentum
from die_vfm.trainer.round2_ssl import resolve_trainable_parameters
from die_vfm.trainer.round2_ssl import update_teacher_ema
from die_vfm.trainer.round2_ssl import validate_round2_train_contract

LOGGER = logging.getLogger(__name__)


def run_pair_benchmark(**kwargs: Any) -> dict[str, Any]:
    """Lazy-loads the pair benchmark helper for runtime and tests."""
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in os.sys.path:
        os.sys.path.insert(0, str(repo_root))

    pair_module = importlib.import_module("scripts.evaluate_pair_benchmark")
    helper = getattr(pair_module, "run_pair_benchmark")
    return helper(**kwargs)


def run_pair_slicing_analysis(**kwargs: Any) -> dict[str, Any]:
    """Lazy-loads the pair slicing helper for runtime and tests."""
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in os.sys.path:
        os.sys.path.insert(0, str(repo_root))

    slicing_module = importlib.import_module(
        "scripts.analyze_pair_benchmark_slices"
    )
    helper = getattr(slicing_module, "run_pair_slicing_analysis")
    return helper(**kwargs)


@dataclass(frozen=True)
class Round2RunArtifacts:
    round2_dir: Path
    checkpoint_dir: Path
    train_embedding_dir: Path
    val_embedding_dir: Path
    linear_probe_dir: Path
    knn_dir: Path
    retrieval_dir: Path
    pair_benchmark_dir: Path
    slicing_root_dir: Path
    summary_yaml_path: Path
    summary_json_path: Path


@dataclass(frozen=True)
class Round2ResumeState:
    """Resolved Round2 resume intent for one run."""

    enabled: bool
    mode: str
    resolved_checkpoint_path: Path | None
    restored_from_checkpoint: bool = False


class Round2SSLRunner:
    """Round2 SSL training runner with end-only evaluation."""

    def __init__(
        self,
        cfg: DictConfig,
        run_dir: str | Path,
    ) -> None:
        self._cfg = cfg
        self._run_dir = Path(run_dir)
        self._device = torch.device(str(cfg.system.device))
        self._world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self._rank = int(os.environ.get("RANK", "0"))
        self._local_rank = int(os.environ.get("LOCAL_RANK", str(self._rank)))
        self._distributed_enabled = self._world_size > 1
        self._process_group_initialized = False

    def run(self) -> dict[str, float]:
        """Runs Round2 SSL end to end."""
        try:
            validate_round2_train_contract(self._cfg)
            artifacts = self._resolve_run_artifacts()
            resume_state = self._resolve_resume_state(artifacts.checkpoint_dir)
            self._setup_distributed_if_needed()
            self._run_distributed_preflight_validation(
                artifacts=artifacts,
                resume_state=resume_state,
            )
            checkpoint_manager = CheckpointManager(artifacts.checkpoint_dir)
            ssl_module, update_resolution = self._build_ssl_module()
            if not isinstance(ssl_module, DistributedDataParallel):
                ssl_module.to(self._device)

            optimizer = self._build_optimizer(ssl_module)
            scheduler = self._build_scheduler(optimizer)
            trainer_state = TrainerState(
                best_metric_name="train.total_loss",
                best_metric_value=None,
            )
            resume_state = self._maybe_resume_round2(
                checkpoint_manager=checkpoint_manager,
                ssl_module=ssl_module,
                optimizer=optimizer,
                scheduler=scheduler,
                trainer_state=trainer_state,
                resume_state=resume_state,
            )
            train_loader = self._build_train_dataloader()
            train_metrics = self._train_loop(
                ssl_module=ssl_module,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                trainer_state=trainer_state,
                checkpoint_manager=checkpoint_manager,
                update_resolution=update_resolution,
            )
            self._barrier_if_distributed()

            if self._postprocess_mode() == "separate_step":
                metrics = dict(train_metrics)
                self._run_rank_zero_only(
                    lambda: self._write_run_summary(
                        artifacts=artifacts,
                        metrics=metrics,
                        train_manifest=None,
                        val_manifest=None,
                        trainer_state=trainer_state,
                        update_resolution=update_resolution,
                        resume_state=resume_state,
                        postprocess_completed=False,
                        postprocess_checkpoint_path=None,
                    ),
                    default=None,
                    barrier_before=False,
                    barrier_after=True,
                )
                return self._broadcast_final_metrics(metrics)

            student_encoder = self._unwrap_model(ssl_module).student_encoder
            train_manifest = self._run_rank_zero_only(
                lambda: self._export_split(
                    model=student_encoder,
                    split="train",
                    output_dir=artifacts.train_embedding_dir,
                ),
                default=None,
                barrier_before=False,
                barrier_after=True,
            )

            val_loader = self._build_split_dataloader_or_none(split="val")
            val_manifest = self._run_rank_zero_only(
                lambda: (
                    self._export_split(
                        model=student_encoder,
                        split="val",
                        output_dir=artifacts.val_embedding_dir,
                        dataloader=val_loader,
                    )
                    if val_loader is not None
                    else None
                ),
                default=None,
                barrier_before=False,
                barrier_after=True,
            )

            metrics = self._run_rank_zero_only(
                lambda: self._build_rank_zero_metrics(
                    artifacts=artifacts,
                    train_metrics=train_metrics,
                    train_manifest=train_manifest,
                    val_manifest=val_manifest,
                ),
                default={},
                barrier_before=False,
                barrier_after=True,
            )

            self._run_rank_zero_only(
                lambda: self._write_run_summary(
                    artifacts=artifacts,
                    metrics=metrics,
                    train_manifest=train_manifest,
                    val_manifest=val_manifest,
                    trainer_state=trainer_state,
                    update_resolution=update_resolution,
                    resume_state=resume_state,
                    postprocess_completed=True,
                    postprocess_checkpoint_path=None,
                ),
                default=None,
                barrier_before=False,
                barrier_after=True,
            )

            return self._broadcast_final_metrics(metrics)
        finally:
            self._cleanup_distributed()

    def run_postprocess(self) -> dict[str, float]:
        """Runs Round2 postprocessing as a separate single-process step."""
        validate_round2_train_contract(self._cfg)
        artifacts = self._resolve_run_artifacts()
        checkpoint_manager = CheckpointManager(artifacts.checkpoint_dir)
        checkpoint_path = self._resolve_postprocess_checkpoint_path(
            checkpoint_manager=checkpoint_manager
        )

        ssl_module, update_resolution = self._build_ssl_module()
        if isinstance(ssl_module, DistributedDataParallel):
            raise RuntimeError(
                "Round2 postprocess must run in single-process mode."
            )
        ssl_module.to(self._device)

        trainer_state = TrainerState(
            best_metric_name="train.total_loss",
            best_metric_value=None,
        )
        checkpoint_manager.load_full_resume(
            checkpoint_path=checkpoint_path,
            model=self._unwrap_model(ssl_module),
            trainer_state=trainer_state,
            strict=True,
            map_location=self._device,
        )

        student_encoder = self._unwrap_model(ssl_module).student_encoder
        train_manifest = None
        val_manifest = None
        if bool(
            OmegaConf.select(
                self._cfg,
                "artifact.embedding.enabled",
                default=True,
            )
        ):
            train_loader = self._build_split_dataloader_or_none(split="train")
            if train_loader is not None:
                train_manifest = self._export_split(
                    model=student_encoder,
                    split="train",
                    output_dir=artifacts.train_embedding_dir,
                    dataloader=train_loader,
                )
            else:
                LOGGER.info(
                    "Round2 postprocess skipped train embedding export because the "
                    "configured dataset manifest does not contain a non-empty train split."
                )
            val_loader = self._build_split_dataloader_or_none(split="val")
            val_manifest = (
                self._export_split(
                    model=student_encoder,
                    split="val",
                    output_dir=artifacts.val_embedding_dir,
                    dataloader=val_loader,
                )
                if val_loader is not None
                else None
            )

        metrics = self._load_existing_summary_metrics(artifacts=artifacts)
        preserved_execution = self._load_existing_summary_execution(
            artifacts=artifacts
        )
        metrics.update(
            self._run_end_only_evaluation(
                artifacts=artifacts,
                train_manifest=train_manifest,
                val_manifest=val_manifest,
            )
        )
        self._write_run_summary(
            artifacts=artifacts,
            metrics=metrics,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            trainer_state=trainer_state,
            update_resolution=update_resolution,
            resume_state=Round2ResumeState(
                enabled=False,
                mode="full_resume",
                resolved_checkpoint_path=None,
                restored_from_checkpoint=False,
            ),
            postprocess_completed=True,
            postprocess_checkpoint_path=checkpoint_path,
            preserved_execution=preserved_execution,
        )
        return metrics

    def _run_preflight_validation(
        self,
        *,
        artifacts: Round2RunArtifacts,
        resume_state: Round2ResumeState,
    ) -> None:
        round2_dir = artifacts.round2_dir
        has_contents = round2_dir.exists() and any(round2_dir.iterdir())
        if has_contents:
            if not resume_state.enabled:
                raise FileExistsError(
                    "Round2 run would overwrite existing outputs. "
                    f"path={round2_dir}. Use a new run_name, enable train.resume.*, "
                    "or clean the previous Round2 outputs."
                )
            if resume_state.resolved_checkpoint_path is None:
                raise FileExistsError(
                    "Round2 resume was requested for an existing run directory, "
                    "but no resume checkpoint was resolved."
                )
            if resume_state.mode != "full_resume":
                raise FileExistsError(
                    "Existing Round2 outputs may only be continued with "
                    "train.resume.mode=full_resume."
                )
            if not self._checkpoint_belongs_to_round2_run(
                checkpoint_path=resume_state.resolved_checkpoint_path,
                checkpoint_dir=artifacts.checkpoint_dir,
            ):
                raise FileExistsError(
                    "Existing Round2 outputs require a checkpoint from the same "
                    f"run checkpoint directory. resolved_checkpoint={resume_state.resolved_checkpoint_path} "
                    f"checkpoint_dir={artifacts.checkpoint_dir}"
                )
        self._validate_resume_lineage(
            checkpoint_dir=artifacts.checkpoint_dir,
            resume_state=resume_state,
        )
        if bool(OmegaConf.select(self._cfg, "evaluation.run_centroid", default=False)):
            raise ValueError("Round2 v1 does not orchestrate centroid evaluation.")

    def _run_distributed_preflight_validation(
        self,
        *,
        artifacts: Round2RunArtifacts,
        resume_state: Round2ResumeState,
    ) -> None:
        if not self._distributed_enabled or not torch.distributed.is_initialized():
            self._run_preflight_validation(
                artifacts=artifacts,
                resume_state=resume_state,
            )
            return

        payload: list[dict[str, str] | None] = [None]
        if self._is_rank_zero():
            try:
                self._run_preflight_validation(
                    artifacts=artifacts,
                    resume_state=resume_state,
                )
            except Exception as exc:  # pragma: no cover - exercised via broadcast contract.
                payload[0] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }

        torch.distributed.broadcast_object_list(payload, src=0)
        error = payload[0]
        if error is not None:
            exception_type = str(error["type"])
            message = (
                "Round2 distributed preflight failed on rank 0: "
                f"{error['message']}"
            )
            if exception_type == "FileExistsError":
                raise FileExistsError(message)
            if exception_type == "FileNotFoundError":
                raise FileNotFoundError(message)
            if exception_type == "ValueError":
                raise ValueError(message)
            raise RuntimeError(message)

        self._barrier_if_distributed()

    def _resolve_run_artifacts(self) -> Round2RunArtifacts:
        round2_dir = self._run_dir / "round2"
        return Round2RunArtifacts(
            round2_dir=round2_dir,
            checkpoint_dir=round2_dir / "checkpoints",
            train_embedding_dir=round2_dir / "embeddings" / "train",
            val_embedding_dir=round2_dir / "embeddings" / "val",
            linear_probe_dir=round2_dir / "evaluation" / "linear_probe",
            knn_dir=round2_dir / "evaluation" / "knn",
            retrieval_dir=round2_dir / "evaluation" / "retrieval",
            pair_benchmark_dir=round2_dir
            / "evaluation"
            / str(self._cfg.round2.evaluation.pair_benchmark.output_subdir),
            slicing_root_dir=round2_dir
            / "evaluation"
            / str(self._cfg.round2.evaluation.slicing.output_subdir),
            summary_yaml_path=round2_dir / "round2_summary.yaml",
            summary_json_path=round2_dir / "round2_summary.json",
        )

    def _build_ssl_module(self) -> tuple[nn.Module, UpdateModeResolution]:
        student_encoder = build_model(self._cfg.model)
        ssl_module = Round2SSLModule.from_student_encoder(
            student_encoder=student_encoder,
            global_hidden_dim=self._cfg.round2.projector.hidden_dim,
            global_output_dim=self._cfg.round2.projector.output_dim,
            global_num_layers=int(self._cfg.round2.projector.num_layers),
            token_hidden_dim=self._cfg.round2.token_projector.hidden_dim,
            token_output_dim=self._cfg.round2.token_projector.output_dim,
            token_num_layers=int(self._cfg.round2.token_projector.num_layers),
        )

        update_resolution = apply_update_mode(
            ssl_module,
            update_mode=str(self._cfg.train.update_mode),
            last_n_blocks=OmegaConf.select(self._cfg, "train.last_n_blocks"),
        )
        update_resolution = reconcile_token_loss_trainability(
            ssl_module,
            update_resolution=update_resolution,
            token_loss_enabled=bool(self._cfg.round2.loss.token_loss_enabled),
        )

        if self._distributed_enabled and torch.distributed.is_initialized():
            ssl_module = DistributedDataParallel(
                ssl_module.to(self._device),
                device_ids=[self._local_rank] if self._device.type == "cuda" else None,
                output_device=self._local_rank if self._device.type == "cuda" else None,
                find_unused_parameters=bool(
                    self._cfg.round2.distributed.find_unused_parameters
                ),
            )

        return ssl_module, update_resolution

    def _resolve_resume_state(
        self,
        checkpoint_dir: Path,
    ) -> Round2ResumeState:
        resume_enabled = bool(
            OmegaConf.select(self._cfg, "train.resume.enabled", default=False)
        )
        resume_mode = str(
            OmegaConf.select(self._cfg, "train.resume.mode", default="full_resume")
        )
        resolved_checkpoint_path = None

        if resume_enabled:
            explicit_checkpoint = OmegaConf.select(
                self._cfg,
                "train.resume.checkpoint_path",
                default=None,
            )
            auto_resume_latest = bool(
                OmegaConf.select(
                    self._cfg,
                    "train.resume.auto_resume_latest",
                    default=False,
                )
            )
            if explicit_checkpoint is not None:
                resolved_checkpoint_path = Path(str(explicit_checkpoint))
                if not resolved_checkpoint_path.exists():
                    raise FileNotFoundError(
                        f"Checkpoint path does not exist: {resolved_checkpoint_path}"
                    )
            elif auto_resume_latest:
                latest_path = checkpoint_dir / LATEST_CHECKPOINT_NAME
                if latest_path.exists():
                    resolved_checkpoint_path = latest_path
            if resolved_checkpoint_path is None:
                raise FileNotFoundError(
                    "Round2 resume was enabled but no checkpoint could be resolved. "
                    "Set train.resume.checkpoint_path or enable "
                    "train.resume.auto_resume_latest with an existing latest.pt."
                )

        return Round2ResumeState(
            enabled=resume_enabled,
            mode=resume_mode,
            resolved_checkpoint_path=resolved_checkpoint_path,
            restored_from_checkpoint=False,
        )

    def _postprocess_mode(self) -> str:
        return str(
            OmegaConf.select(
                self._cfg,
                "round2.postprocess.mode",
                default="in_process",
            )
        )

    def _resolve_postprocess_checkpoint_path(
        self,
        *,
        checkpoint_manager: CheckpointManager,
    ) -> Path:
        explicit_checkpoint = OmegaConf.select(
            self._cfg,
            "round2.postprocess.checkpoint_path",
            default=None,
        )
        auto_use_latest = bool(
            OmegaConf.select(
                self._cfg,
                "round2.postprocess.auto_use_latest_checkpoint",
                default=True,
            )
        )
        checkpoint_path = checkpoint_manager.resolve_resume_path(
            checkpoint_path=explicit_checkpoint,
            auto_resume_latest=auto_use_latest,
        )
        if checkpoint_path is None:
            raise FileNotFoundError(
                "Round2 postprocess could not resolve a checkpoint. "
                "Set round2.postprocess.checkpoint_path or enable "
                "round2.postprocess.auto_use_latest_checkpoint with an existing latest.pt."
            )
        return checkpoint_path

    def _load_existing_summary_metrics(
        self,
        *,
        artifacts: Round2RunArtifacts,
    ) -> dict[str, float]:
        if not artifacts.summary_yaml_path.exists():
            return {}
        summary = OmegaConf.load(artifacts.summary_yaml_path)
        metrics = OmegaConf.to_container(
            OmegaConf.select(summary, "metrics", default={}),
            resolve=True,
        )
        if not isinstance(metrics, dict):
            return {}
        return dict(metrics)

    def _load_existing_summary_execution(
        self,
        *,
        artifacts: Round2RunArtifacts,
    ) -> dict[str, Any] | None:
        if not artifacts.summary_yaml_path.exists():
            return None
        summary = OmegaConf.load(artifacts.summary_yaml_path)
        execution_cfg = OmegaConf.select(summary, "execution", default=None)
        if execution_cfg is None:
            return None
        execution = OmegaConf.to_container(
            execution_cfg,
            resolve=True,
        )
        if not isinstance(execution, dict):
            return None
        return dict(execution)

    def _maybe_resume_round2(
        self,
        *,
        checkpoint_manager: CheckpointManager,
        ssl_module: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        trainer_state: TrainerState,
        resume_state: Round2ResumeState,
    ) -> Round2ResumeState:
        if not resume_state.enabled:
            return resume_state

        if resume_state.resolved_checkpoint_path is None:
            raise FileNotFoundError(
                "Round2 resume was enabled but no checkpoint path was resolved."
            )

        wrapped = self._unwrap_model(ssl_module)

        if resume_state.mode == "warm_start":
            checkpoint_manager.load_warm_start(
                checkpoint_path=resume_state.resolved_checkpoint_path,
                model=wrapped,
                strict=True,
                map_location="cpu",
            )
            return Round2ResumeState(
                enabled=True,
                mode="warm_start",
                resolved_checkpoint_path=resume_state.resolved_checkpoint_path,
                restored_from_checkpoint=True,
            )

        if resume_state.mode == "full_resume":
            checkpoint_manager.load_full_resume(
                checkpoint_path=resume_state.resolved_checkpoint_path,
                model=wrapped,
                trainer_state=trainer_state,
                optimizer=optimizer,
                lr_scheduler=scheduler,
                grad_scaler=None,
                strict=True,
                map_location="cpu",
            )
            self._reconcile_scheduler_after_full_resume(
                scheduler=scheduler,
            )
            return Round2ResumeState(
                enabled=True,
                mode="full_resume",
                resolved_checkpoint_path=resume_state.resolved_checkpoint_path,
                restored_from_checkpoint=True,
            )

        raise ValueError(f"Unsupported Round2 resume mode: {resume_state.mode!r}")

    def _build_optimizer(self, ssl_module: nn.Module) -> torch.optim.Optimizer:
        parameters = resolve_trainable_parameters(self._unwrap_model(ssl_module))
        if not parameters:
            raise ValueError("Round2 resolved zero trainable parameters.")

        optimizer_name = str(self._cfg.round2.optimizer.name).lower()
        learning_rate = float(self._cfg.round2.optimizer.learning_rate)
        weight_decay = float(self._cfg.round2.optimizer.weight_decay)
        if optimizer_name == "adamw":
            betas = tuple(float(value) for value in self._cfg.round2.optimizer.betas)
            return AdamW(
                parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=betas,
            )
        if optimizer_name == "sgd":
            momentum = float(
                OmegaConf.select(self._cfg, "round2.optimizer.momentum", default=0.9)
            )
            return SGD(
                parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=momentum,
            )
        raise ValueError(f"Unsupported Round2 optimizer: {optimizer_name!r}")

    def _build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler._LRScheduler | None:
        scheduler_name = str(self._cfg.round2.scheduler.name).lower()
        if scheduler_name == "none":
            return None
        if scheduler_name == "cosine":
            return CosineAnnealingLR(
                optimizer,
                T_max=max(int(self._cfg.train.num_epochs), 1),
                eta_min=float(self._cfg.round2.scheduler.min_learning_rate),
            )
        raise ValueError(f"Unsupported Round2 scheduler: {scheduler_name!r}")

    def _reconcile_scheduler_after_full_resume(
        self,
        *,
        scheduler: Any,
    ) -> None:
        if scheduler is None:
            return

        scheduler_name = str(self._cfg.round2.scheduler.name).lower()
        if scheduler_name == "cosine" and isinstance(scheduler, CosineAnnealingLR):
            scheduler.T_max = max(int(self._cfg.train.num_epochs), 1)

    def _validate_resume_lineage(
        self,
        *,
        checkpoint_dir: Path,
        resume_state: Round2ResumeState,
    ) -> None:
        if not resume_state.enabled:
            return
        if resume_state.mode != "full_resume":
            return
        if resume_state.resolved_checkpoint_path is None:
            return
        if self._checkpoint_belongs_to_round2_run(
            checkpoint_path=resume_state.resolved_checkpoint_path,
            checkpoint_dir=checkpoint_dir,
        ):
            return
        raise FileExistsError(
            "Round2 full_resume requires a checkpoint from the same run lineage. "
            "Use train.resume.mode=warm_start for cross-run initialization. "
            f"resolved_checkpoint={resume_state.resolved_checkpoint_path} "
            f"checkpoint_dir={checkpoint_dir}"
        )

    def _build_train_dataloader(self) -> Any:
        if self._distributed_enabled:
            dataset = build_dataset(self._cfg, split="train")
            sampler = DistributedSampler(
                dataset,
                num_replicas=self._world_size,
                rank=self._rank,
                shuffle=True,
            )
            return build_dataloader(
                self._cfg,
                split="train",
                dataset=dataset,
                sampler=sampler,
                shuffle=False,
            )
        return build_dataloader(self._cfg, split="train")

    def _build_split_dataloader_or_none(self, split: str) -> Any | None:
        try:
            dataloader = build_dataloader(self._cfg, split=split)
        except ValueError as exc:
            dataset_name = str(OmegaConf.select(self._cfg, "dataset.name", default=""))
            if (
                split == "train"
                and dataset_name == "domain"
                and "Filtered train split is empty" in str(exc)
            ):
                return None
            if (
                split == "val"
                and dataset_name == "domain"
                and "Filtered val split is empty" in str(exc)
            ):
                return None
            raise
        if len(dataloader.dataset) == 0:
            return None
        return dataloader

    def _train_loop(
        self,
        *,
        ssl_module: nn.Module,
        train_loader: Any,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        trainer_state: TrainerState,
        checkpoint_manager: CheckpointManager,
        update_resolution: UpdateModeResolution,
    ) -> dict[str, float]:
        wrapped = self._unwrap_model(ssl_module)
        num_epochs = int(self._cfg.train.num_epochs)
        precision_mode = str(self._cfg.train.precision_mode)
        token_loss_enabled = bool(self._cfg.round2.loss.token_loss_enabled)
        token_loss_weight = float(self._cfg.round2.loss.token_loss_weight)
        log_every_n_steps = int(self._cfg.train.log_every_n_steps)
        final_epoch_metrics: dict[str, float] = {}

        for epoch_index in range(int(trainer_state.epoch), num_epochs):
            if isinstance(getattr(train_loader, "sampler", None), DistributedSampler):
                train_loader.sampler.set_epoch(epoch_index)

            wrapped.student_encoder.train()
            wrapped.student_global_projector.train()
            wrapped.student_token_projector.train()
            wrapped.teacher_encoder.eval()
            wrapped.teacher_global_projector.eval()
            wrapped.teacher_token_projector.eval()

            epoch_total_loss = 0.0
            epoch_global_loss = 0.0
            epoch_token_loss = 0.0
            num_batches = 0
            ema_momentum = resolve_ema_momentum(
                cfg=self._cfg,
                epoch_index=epoch_index,
                num_epochs=num_epochs,
            )

            for batch in train_loader:
                images = batch["image"].to(self._device, non_blocking=True)
                view_a, flip_a = generate_augmented_view(
                    images,
                    horizontal_flip_prob=float(
                        self._cfg.round2.augmentation.horizontal_flip_prob
                    ),
                    vertical_flip_prob=float(
                        self._cfg.round2.augmentation.vertical_flip_prob
                    ),
                )
                view_b, flip_b = generate_augmented_view(
                    images,
                    horizontal_flip_prob=float(
                        self._cfg.round2.augmentation.horizontal_flip_prob
                    ),
                    vertical_flip_prob=float(
                        self._cfg.round2.augmentation.vertical_flip_prob
                    ),
                )

                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device=self._device, precision_mode=precision_mode):
                    student_a = wrapped.student_encoder(view_a)
                    student_b = wrapped.student_encoder(view_b)
                    with torch.no_grad():
                        teacher_a = wrapped.teacher_encoder(view_a)
                        teacher_b = wrapped.teacher_encoder(view_b)

                    student_a_proj = wrapped.student_global_projector(student_a.embedding)
                    student_b_proj = wrapped.student_global_projector(student_b.embedding)
                    with torch.no_grad():
                        teacher_a_proj = wrapped.teacher_global_projector(
                            teacher_a.embedding
                        )
                        teacher_b_proj = wrapped.teacher_global_projector(
                            teacher_b.embedding
                        )

                    global_loss = 0.5 * (
                        projected_cosine_loss(student_a_proj, teacher_b_proj.detach())
                        + projected_cosine_loss(
                            student_b_proj,
                            teacher_a_proj.detach(),
                        )
                    )

                    token_loss = torch.zeros((), device=self._device, dtype=torch.float32)
                    if token_loss_enabled:
                        token_loss = self._compute_token_loss(
                            wrapped=wrapped,
                            student_a=student_a,
                            student_b=student_b,
                            teacher_a=teacher_a,
                            teacher_b=teacher_b,
                            flip_a=flip_a,
                            flip_b=flip_b,
                        )

                    total_loss = global_loss + token_loss_weight * token_loss

                total_loss.backward()
                optimizer.step()
                update_teacher_ema(wrapped, momentum=ema_momentum)

                trainer_state.global_step += 1
                epoch_total_loss += float(total_loss.detach().cpu().item())
                epoch_global_loss += float(global_loss.detach().cpu().item())
                epoch_token_loss += float(token_loss.detach().cpu().item())
                num_batches += 1

                if trainer_state.global_step % log_every_n_steps == 0 and self._is_rank_zero():
                    LOGGER.info(
                        "Round2 step=%d epoch=%d total_loss=%.6f global_loss=%.6f token_loss=%.6f lr=%.8f",
                        trainer_state.global_step,
                        epoch_index,
                        float(total_loss.detach().cpu().item()),
                        float(global_loss.detach().cpu().item()),
                        float(token_loss.detach().cpu().item()),
                        self._current_learning_rate(optimizer),
                    )

            if scheduler is not None:
                scheduler.step()

            trainer_state.epoch = epoch_index + 1
            epoch_metrics = {
                "train.total_loss": epoch_total_loss / max(num_batches, 1),
                "train.global_loss": epoch_global_loss / max(num_batches, 1),
                "train.token_loss": epoch_token_loss / max(num_batches, 1),
                "train.learning_rate": self._current_learning_rate(optimizer),
            }
            final_epoch_metrics = epoch_metrics

            current_metric = epoch_metrics["train.total_loss"]
            is_best = (
                trainer_state.best_metric_value is None
                or current_metric < float(trainer_state.best_metric_value)
            )
            if is_best:
                trainer_state.best_metric_value = current_metric

            if self._is_rank_zero():
                checkpoint_manager.save(
                    model=wrapped,
                    trainer_state=trainer_state,
                    optimizer=optimizer,
                    lr_scheduler=scheduler,
                    epoch=trainer_state.epoch,
                    global_step=trainer_state.global_step,
                    is_best=is_best,
                    extra_metadata={
                        "phase": "round2_ssl",
                        "run_name": self._cfg.run.run_name,
                        "update_mode": update_resolution.update_mode,
                        "last_n_blocks": update_resolution.last_n_blocks,
                        "resolved_trainable_module_names": update_resolution.trainable_module_names,
                        "resolved_trainable_block_indices": update_resolution.trainable_block_indices,
                        "precision_mode": precision_mode,
                        "ema_policy": str(self._cfg.round2.ema.policy),
                        "ema_momentum": ema_momentum,
                        "token_loss_enabled": token_loss_enabled,
                        "token_loss_weight": token_loss_weight,
                    },
                )

        return final_epoch_metrics

    def _compute_token_loss(
        self,
        *,
        wrapped: Round2SSLModule,
        student_a: Any,
        student_b: Any,
        teacher_a: Any,
        teacher_b: Any,
        flip_a: FlipMetadata,
        flip_b: FlipMetadata,
    ) -> torch.Tensor:
        if student_a.backbone is None or student_b.backbone is None:
            raise ValueError(
                "Round2 token loss requires model.return_debug_outputs=true."
            )
        if teacher_a.backbone is None or teacher_b.backbone is None:
            raise ValueError(
                "Round2 token loss requires teacher encoder debug outputs."
            )

        student_a_tokens = canonicalize_patch_tokens(
            wrapped.student_token_projector(student_a.backbone.patch_tokens),
            patch_grid=student_a.backbone.patch_grid,
            flip_metadata=flip_a,
        )
        student_b_tokens = canonicalize_patch_tokens(
            wrapped.student_token_projector(student_b.backbone.patch_tokens),
            patch_grid=student_b.backbone.patch_grid,
            flip_metadata=flip_b,
        )
        with torch.no_grad():
            teacher_a_tokens = canonicalize_patch_tokens(
                wrapped.teacher_token_projector(teacher_a.backbone.patch_tokens),
                patch_grid=teacher_a.backbone.patch_grid,
                flip_metadata=flip_a,
            )
            teacher_b_tokens = canonicalize_patch_tokens(
                wrapped.teacher_token_projector(teacher_b.backbone.patch_tokens),
                patch_grid=teacher_b.backbone.patch_grid,
                flip_metadata=flip_b,
            )

        return 0.5 * (
            projected_cosine_loss(student_a_tokens, teacher_b_tokens.detach())
            + projected_cosine_loss(student_b_tokens, teacher_a_tokens.detach())
        )

    def _build_rank_zero_metrics(
        self,
        *,
        artifacts: Round2RunArtifacts,
        train_metrics: dict[str, float],
        train_manifest: Any,
        val_manifest: Any | None,
    ) -> dict[str, Any]:
        metrics = dict(train_metrics)
        metrics.update(
            self._run_end_only_evaluation(
                artifacts=artifacts,
                train_manifest=train_manifest,
                val_manifest=val_manifest,
            )
        )
        return metrics

    def _export_split(
        self,
        *,
        model: nn.Module,
        split: str,
        output_dir: Path,
        dataloader: Any | None = None,
    ) -> Any:
        resolved_loader = dataloader
        if resolved_loader is None:
            resolved_loader = build_dataloader(
                self._cfg,
                split=split,
                shuffle=False,
            )
        return export_split_embeddings(
            model=model,
            dataloader=resolved_loader,
            output_dir=output_dir,
            split=split,
            device=str(self._device),
        )

    def _run_end_only_evaluation(
        self,
        *,
        artifacts: Round2RunArtifacts,
        train_manifest: Any,
        val_manifest: Any | None,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}

        if self._artifact_evaluators_enabled() and train_manifest is None:
            raise ValueError(
                "Round2 artifact evaluators require a non-empty train split export, "
                "but postprocess did not produce train embeddings. Use a manifest "
                "with Train rows for linear_probe/knn/retrieval, or disable those evaluators."
            )

        if val_manifest is not None:
            metrics.update(self._run_artifact_evaluators(artifacts))

        if bool(self._cfg.round2.evaluation.run_pair_benchmark):
            metrics.update(self._run_pair_benchmark_suite(artifacts))

        return metrics

    def _run_artifact_evaluators(self, artifacts: Round2RunArtifacts) -> dict[str, float]:
        metrics: dict[str, float] = {}

        if bool(self._cfg.evaluation.run_linear_probe):
            cfg = self._cfg.evaluation.linear_probe
            result = run_linear_probe(
                build_linear_probe_run_config(
                    train_split_dir=artifacts.train_embedding_dir,
                    val_split_dir=artifacts.val_embedding_dir,
                    output_dir=artifacts.linear_probe_dir,
                    normalize_embeddings=bool(cfg.input.normalize_embeddings),
                    map_location=str(cfg.input.map_location),
                    bias=bool(cfg.model.bias),
                    batch_size=int(cfg.trainer.batch_size),
                    num_epochs=int(cfg.trainer.num_epochs),
                    learning_rate=float(cfg.trainer.learning_rate),
                    weight_decay=float(cfg.trainer.weight_decay),
                    optimizer_name=str(cfg.trainer.optimizer_name),
                    momentum=float(cfg.trainer.momentum),
                    device=str(cfg.trainer.device),
                    seed=int(cfg.trainer.seed),
                    selection_metric=str(cfg.trainer.selection_metric),
                    save_predictions=bool(cfg.output.save_predictions),
                    save_history=bool(cfg.output.save_history),
                )
            )
            for key, value in result.val_metrics.items():
                metrics[f"linear_probe.{key}"] = float(value)

        if bool(self._cfg.evaluation.run_knn):
            cfg = self._cfg.evaluation.knn
            result = run_knn(
                build_knn_run_config(
                    train_split_dir=artifacts.train_embedding_dir,
                    val_split_dir=artifacts.val_embedding_dir,
                    output_dir=artifacts.knn_dir,
                    normalize_embeddings=bool(cfg.input.normalize_embeddings),
                    map_location=str(cfg.input.map_location),
                    save_predictions=bool(cfg.output.save_predictions),
                    k=int(cfg.evaluator.k),
                    metric=str(cfg.evaluator.metric),
                    weighting=str(cfg.evaluator.weighting),
                    temperature=float(cfg.evaluator.temperature),
                    batch_size=int(cfg.evaluator.batch_size),
                    device=str(cfg.evaluator.device),
                    topk=tuple(cfg.evaluator.topk),
                )
            )
            for key, value in result.val_metrics.items():
                metrics[f"knn.{key}"] = float(value)

        if bool(self._cfg.evaluation.run_retrieval):
            cfg = self._cfg.evaluation.retrieval
            result = run_retrieval(
                build_retrieval_run_config(
                    train_split_dir=artifacts.train_embedding_dir,
                    val_split_dir=artifacts.val_embedding_dir,
                    output_dir=artifacts.retrieval_dir,
                    normalize_embeddings=bool(cfg.input.normalize_embeddings),
                    map_location=str(cfg.input.map_location),
                    save_predictions=bool(cfg.output.save_predictions),
                    metric=str(cfg.evaluator.metric),
                    batch_size=int(cfg.evaluator.batch_size),
                    device=str(cfg.evaluator.device),
                    topk=tuple(cfg.evaluator.topk),
                    save_predictions_topk=int(cfg.evaluator.save_predictions_topk),
                    exclude_same_image_id=bool(cfg.evaluator.exclude_same_image_id),
                )
            )
            for key, value in result.val_metrics.items():
                metrics[f"retrieval.{key}"] = float(value)

        return metrics

    def _run_pair_benchmark_suite(
        self,
        artifacts: Round2RunArtifacts,
    ) -> dict[str, float]:
        pair_cfg = self._cfg.round2.evaluation.pair_benchmark
        pair_candidates_path = OmegaConf.select(
            self._cfg, "round2.evaluation.pair_benchmark.pair_candidates_path"
        )
        annotations_path = OmegaConf.select(
            self._cfg, "round2.evaluation.pair_benchmark.annotations_path"
        )
        if pair_candidates_path is None or annotations_path is None:
            raise ValueError(
                "Round2 pair benchmark requires both "
                "round2.evaluation.pair_benchmark.pair_candidates_path and "
                "round2.evaluation.pair_benchmark.annotations_path."
            )

        split_dir_lookup = {
            "train": artifacts.train_embedding_dir,
            "val": artifacts.val_embedding_dir,
        }
        self._validate_pair_benchmark_embedding_splits(
            split_dir_lookup=split_dir_lookup,
            embedding_splits=pair_cfg.embedding_splits,
        )
        embedding_split_dirs = [
            split_dir_lookup[str(split_name)]
            for split_name in pair_cfg.embedding_splits
        ]
        result = run_pair_benchmark(
            pair_candidates_path=Path(str(pair_candidates_path)),
            annotations_path=Path(str(annotations_path)),
            embedding_split_dirs=embedding_split_dirs,
            join_key=str(pair_cfg.join_key),
            output_dir=artifacts.pair_benchmark_dir,
            hard_limit=int(pair_cfg.hard_limit),
            map_location=str(pair_cfg.map_location),
        )
        metrics = {
            "pair_benchmark.same_vs_different_cosine_auc_like": float(
                result["summary"]["separation"]["same_vs_different_cosine_auc_like"]
            )
        }

        if bool(self._cfg.round2.evaluation.run_slicing_analysis):
            slicing_cfg = self._cfg.round2.evaluation.slicing
            for confidence in slicing_cfg.confidences:
                slice_output_dir = artifacts.slicing_root_dir / str(confidence)
                slice_result = run_pair_slicing_analysis(
                    pair_scores_path=result["pair_scores_path"],
                    pair_candidates_path=Path(str(pair_candidates_path)),
                    output_dir=slice_output_dir,
                    confidence=str(confidence),
                    hard_limit=int(slicing_cfg.hard_limit),
                )
                source_slice = (
                    slice_result["summary"]
                    .get("slices", {})
                    .get("source_slice", {})
                    .get("cross_source", {})
                    .get("same_vs_different_cosine_auc_like")
                )
                if source_slice is not None:
                    metrics[
                        f"pair_benchmark.cross_source_auc_like.{confidence}"
                    ] = float(source_slice)

        return metrics

    def _validate_pair_benchmark_embedding_splits(
        self,
        *,
        split_dir_lookup: dict[str, Path],
        embedding_splits: Any,
    ) -> None:
        missing_targets: list[str] = []
        unknown_splits: list[str] = []

        for raw_split_name in embedding_splits:
            split_name = str(raw_split_name)
            split_dir = split_dir_lookup.get(split_name)
            if split_dir is None:
                unknown_splits.append(split_name)
                continue

            manifest_path = default_manifest_path(split_dir)
            if not split_dir.exists() or not manifest_path.exists():
                missing_targets.append(
                    f"{split_name} (dir={split_dir}, manifest={manifest_path})"
                )

        if unknown_splits:
            raise ValueError(
                "Round2 pair benchmark received unsupported embedding_splits. "
                f"Got {unknown_splits!r}; supported splits are {sorted(split_dir_lookup.keys())}."
            )

        if missing_targets:
            raise FileNotFoundError(
                "Round2 pair benchmark requires exported embedding manifests for all "
                "configured round2.evaluation.pair_benchmark.embedding_splits, but "
                f"the following targets are missing: {missing_targets}. "
                "Export the requested split embeddings first, or change "
                "round2.evaluation.pair_benchmark.embedding_splits to only use available splits."
            )

    def _write_run_summary(
        self,
        *,
        artifacts: Round2RunArtifacts,
        metrics: dict[str, float],
        train_manifest: Any,
        val_manifest: Any | None,
        trainer_state: TrainerState,
        update_resolution: UpdateModeResolution,
        resume_state: Round2ResumeState,
        postprocess_completed: bool,
        postprocess_checkpoint_path: Path | None,
        preserved_execution: dict[str, Any] | None = None,
    ) -> None:
        ema_metadata = self._resolve_summary_ema_metadata(trainer_state=trainer_state)
        execution = {
            "runtime_default_update_mode": "full_backbone",
            "planned_experiment_order": [
                "projector_pooler_only",
                "last_n_blocks",
                "full_backbone",
            ],
            "update_mode": update_resolution.update_mode,
            "last_n_blocks": update_resolution.last_n_blocks,
            "resolved_trainable_module_names": update_resolution.trainable_module_names,
            "resolved_trainable_block_indices": update_resolution.trainable_block_indices,
            "precision_mode": str(self._cfg.train.precision_mode),
            "token_loss_enabled": bool(self._cfg.round2.loss.token_loss_enabled),
            "token_loss_weight": float(self._cfg.round2.loss.token_loss_weight),
            **ema_metadata,
            "resume_enabled": bool(resume_state.enabled),
            "resume_mode": str(resume_state.mode),
            "resolved_resume_checkpoint": (
                str(resume_state.resolved_checkpoint_path)
                if resume_state.resolved_checkpoint_path is not None
                else None
            ),
            "restored_from_checkpoint": bool(
                resume_state.restored_from_checkpoint
            ),
            "postprocess_mode": self._postprocess_mode(),
            "postprocess_completed": bool(postprocess_completed),
            "postprocess_checkpoint_path": (
                str(postprocess_checkpoint_path)
                if postprocess_checkpoint_path is not None
                else None
            ),
        }
        if preserved_execution is not None:
            execution.update(preserved_execution)
            execution["postprocess_mode"] = self._postprocess_mode()
            execution["postprocess_completed"] = bool(postprocess_completed)
            execution["postprocess_checkpoint_path"] = (
                str(postprocess_checkpoint_path)
                if postprocess_checkpoint_path is not None
                else None
            )
        summary = {
            "phase": "round2_ssl",
            "runtime_semantics": {
                "mode": "ssl_training",
                "uses_gradient_updates": True,
                "uses_epoch_loop": True,
                "supports_resume": True,
                "supports_checkpoint_continuation": True,
            },
            "execution": execution,
            "trainer_state": {
                "epoch": int(trainer_state.epoch),
                "global_step": int(trainer_state.global_step),
                "best_metric_name": trainer_state.best_metric_name,
                "best_metric_value": trainer_state.best_metric_value,
            },
            "manifests": {
                "train": self._manifest_summary(train_manifest),
                "val": self._manifest_summary(val_manifest),
            },
            "metrics": metrics,
            "artifacts": {
                "round2_dir": str(artifacts.round2_dir),
                "checkpoint_dir": str(artifacts.checkpoint_dir),
                "train_embedding_dir": str(artifacts.train_embedding_dir),
                "val_embedding_dir": str(artifacts.val_embedding_dir),
                "linear_probe_dir": str(artifacts.linear_probe_dir),
                "knn_dir": str(artifacts.knn_dir),
                "retrieval_dir": str(artifacts.retrieval_dir),
                "pair_benchmark_dir": str(artifacts.pair_benchmark_dir),
                "slicing_root_dir": str(artifacts.slicing_root_dir),
            },
        }
        OmegaConf.save(config=OmegaConf.create(summary), f=artifacts.summary_yaml_path)
        artifacts.summary_json_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _resolve_summary_ema_metadata(
        self,
        *,
        trainer_state: TrainerState,
    ) -> dict[str, Any]:
        ema_policy = str(self._cfg.round2.ema.policy)
        base_momentum = float(self._cfg.round2.ema.momentum)
        final_momentum = float(self._cfg.round2.ema.final_momentum)
        if int(trainer_state.epoch) <= 0:
            effective_momentum = base_momentum
        else:
            effective_momentum = resolve_ema_momentum(
                cfg=self._cfg,
                epoch_index=max(int(trainer_state.epoch) - 1, 0),
                num_epochs=max(int(self._cfg.train.num_epochs), 1),
            )
        return {
            "ema_policy": ema_policy,
            "ema_momentum": float(effective_momentum),
            "ema_base_momentum": base_momentum,
            "ema_final_momentum": final_momentum,
        }

    def _manifest_summary(self, manifest: Any | None) -> dict[str, Any]:
        if manifest is None:
            return {
                "available": False,
                "num_samples": 0,
                "embedding_dim": None,
                "has_labels": False,
            }
        return {
            "available": True,
            "num_samples": int(manifest.num_samples),
            "embedding_dim": int(manifest.embedding_dim),
            "has_labels": bool(manifest.has_labels),
        }

    def _artifact_evaluators_enabled(self) -> bool:
        return bool(
            self._cfg.evaluation.run_linear_probe
            or self._cfg.evaluation.run_knn
            or self._cfg.evaluation.run_retrieval
        )

    def _setup_distributed_if_needed(self) -> None:
        if not self._distributed_enabled:
            return
        backend = OmegaConf.select(self._cfg, "round2.distributed.backend")
        if backend is None:
            backend = "nccl" if self._device.type == "cuda" else "gloo"
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=str(backend))
            self._process_group_initialized = True
        if self._device.type == "cuda":
            torch.cuda.set_device(self._local_rank)

    def _cleanup_distributed(self) -> None:
        if self._process_group_initialized and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            self._process_group_initialized = False

    def _barrier_if_distributed(self) -> None:
        if self._distributed_enabled and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _run_rank_zero_only(
        self,
        operation: Any,
        *,
        default: Any,
        barrier_before: bool = True,
        barrier_after: bool = True,
    ) -> Any:
        if barrier_before:
            self._barrier_if_distributed()
        if self._is_rank_zero():
            result = operation()
        else:
            result = default
        if barrier_after:
            self._barrier_if_distributed()
        return result

    def _broadcast_final_metrics(
        self,
        metrics: dict[str, float],
    ) -> dict[str, float]:
        if not self._distributed_enabled or not torch.distributed.is_initialized():
            return metrics
        payload = [metrics if self._is_rank_zero() else None]
        torch.distributed.broadcast_object_list(payload, src=0)
        return dict(payload[0] or {})

    def _unwrap_model(self, model: nn.Module) -> Round2SSLModule:
        if isinstance(model, DistributedDataParallel):
            return model.module  # type: ignore[return-value]
        return model  # type: ignore[return-value]

    def _current_learning_rate(self, optimizer: torch.optim.Optimizer) -> float:
        return float(optimizer.param_groups[0]["lr"])

    def _is_rank_zero(self) -> bool:
        return self._rank == 0

    def _checkpoint_belongs_to_round2_run(
        self,
        *,
        checkpoint_path: Path,
        checkpoint_dir: Path,
    ) -> bool:
        resolved_checkpoint = checkpoint_path.resolve()
        resolved_dir = checkpoint_dir.resolve()
        try:
            resolved_checkpoint.relative_to(resolved_dir)
            return True
        except ValueError:
            return False
