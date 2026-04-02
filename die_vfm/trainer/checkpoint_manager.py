"""Checkpoint manager for Die VFM M1 closeout."""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

import torch


LATEST_CHECKPOINT_NAME = "latest.pt"
BEST_CHECKPOINT_NAME = "best.pt"
CHECKPOINT_VERSION = "v1"


class CheckpointValidationError(ValueError):
  """Raised when a checkpoint payload is invalid or incompatible."""


class CheckpointManager:
  """Handles checkpoint save/load and resume path resolution."""

  def __init__(self, checkpoint_dir: str | Path) -> None:
    self._checkpoint_dir = Path(checkpoint_dir)
    self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

  @property
  def checkpoint_dir(self) -> Path:
    """Returns the checkpoint directory."""
    return self._checkpoint_dir

  def get_latest_checkpoint_path(self) -> Path:
    """Returns the canonical latest checkpoint path."""
    return self._checkpoint_dir / LATEST_CHECKPOINT_NAME

  def get_best_checkpoint_path(self) -> Path:
    """Returns the canonical best checkpoint path."""
    return self._checkpoint_dir / BEST_CHECKPOINT_NAME

  def get_epoch_checkpoint_path(self, epoch: int) -> Path:
    """Returns the per-epoch checkpoint path."""
    return self._checkpoint_dir / f"epoch_{epoch:04d}.pt"

  def has_latest_checkpoint(self) -> bool:
    """Returns whether latest.pt exists."""
    return self.get_latest_checkpoint_path().exists()

  def resolve_resume_path(
      self,
      checkpoint_path: str | Path | None,
      auto_resume_latest: bool,
  ) -> Path | None:
    """Resolves the effective checkpoint path for resume.

    Priority:
      1. Explicit checkpoint_path
      2. latest.pt when auto_resume_latest=True
      3. None

    Args:
      checkpoint_path: Explicit checkpoint path from config.
      auto_resume_latest: Whether to fall back to latest.pt.

    Returns:
      Resolved checkpoint path, or None if no checkpoint is available.

    Raises:
      FileNotFoundError: If an explicit checkpoint path is provided but missing.
    """
    if checkpoint_path is not None:
      resolved_path = Path(checkpoint_path)
      if not resolved_path.exists():
        raise FileNotFoundError(
            f"Checkpoint path does not exist: {resolved_path}"
        )
      return resolved_path

    if auto_resume_latest:
      latest_path = self.get_latest_checkpoint_path()
      if latest_path.exists():
        return latest_path

    return None

  def save(
      self,
      *,
      model: torch.nn.Module,
      trainer_state: Any,
      epoch: int,
      global_step: int,
      optimizer: Any | None = None,
      lr_scheduler: Any | None = None,
      grad_scaler: Any | None = None,
      is_best: bool = False,
      extra_metadata: Optional[dict[str, Any]] = None,
  ) -> dict[str, Path]:
    """Saves checkpoint payloads atomically.

    Always writes:
      - epoch_xxxx.pt
      - latest.pt

    Optionally writes:
      - best.pt when is_best=True

    Args:
      model: Model to serialize.
      trainer_state: Trainer state object.
      epoch: Current epoch.
      global_step: Current global step.
      optimizer: Optional optimizer object.
      lr_scheduler: Optional LR scheduler object.
      grad_scaler: Optional grad scaler object.
      is_best: Whether this checkpoint should also update best.pt.
      extra_metadata: Optional metadata to include in the payload.

    Returns:
      Mapping of checkpoint roles to written paths.
    """
    payload = self._build_payload(
        model=model,
        trainer_state=trainer_state,
        epoch=epoch,
        global_step=global_step,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        grad_scaler=grad_scaler,
        extra_metadata=extra_metadata,
    )

    written_paths: dict[str, Path] = {}

    epoch_path = self.get_epoch_checkpoint_path(epoch)
    self._atomic_torch_save(payload, epoch_path)
    written_paths["epoch"] = epoch_path

    latest_path = self.get_latest_checkpoint_path()
    self._atomic_torch_save(payload, latest_path)
    written_paths["latest"] = latest_path

    if is_best:
      best_path = self.get_best_checkpoint_path()
      self._atomic_torch_save(payload, best_path)
      written_paths["best"] = best_path

    return written_paths

  def load(
      self,
      checkpoint_path: str | Path,
      map_location: str | torch.device = "cpu",
  ) -> dict[str, Any]:
    """Loads and validates a checkpoint payload.

    Args:
      checkpoint_path: Path to the checkpoint.
      map_location: Torch map_location.

    Returns:
      Validated checkpoint payload.

    Raises:
      CheckpointValidationError: If payload is malformed or incompatible.
    """
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    self._validate_payload(payload, checkpoint_path=path)
    return payload

  def load_warm_start(
      self,
      *,
      checkpoint_path: str | Path,
      model: torch.nn.Module,
      strict: bool = True,
      map_location: str | torch.device = "cpu",
  ) -> dict[str, Any]:
    """Loads model-only state for warm start.

    Args:
      checkpoint_path: Path to the checkpoint.
      model: Model to restore.
      strict: Whether to enforce exact key matching.
      map_location: Torch map_location.

    Returns:
      Loaded checkpoint payload.
    """
    payload = self.load(
        checkpoint_path=checkpoint_path,
        map_location=map_location,
    )
    model.load_state_dict(payload["model_state_dict"], strict=strict)
    return payload

  def load_full_resume(
      self,
      *,
      checkpoint_path: str | Path,
      model: torch.nn.Module,
      trainer_state: Any,
      optimizer: Any | None = None,
      lr_scheduler: Any | None = None,
      grad_scaler: Any | None = None,
      strict: bool = True,
      map_location: str | torch.device = "cpu",
  ) -> dict[str, Any]:
    """Loads full resume state into runtime objects.

    Args:
      checkpoint_path: Path to the checkpoint.
      model: Model to restore.
      trainer_state: Trainer state object to restore in place.
      optimizer: Optional optimizer to restore.
      lr_scheduler: Optional LR scheduler to restore.
      grad_scaler: Optional grad scaler to restore.
      strict: Whether to enforce exact model key matching.
      map_location: Torch map_location.

    Returns:
      Loaded checkpoint payload.

    Raises:
      CheckpointValidationError: If required resume state is missing.
    """
    payload = self.load(
        checkpoint_path=checkpoint_path,
        map_location=map_location,
    )

    model.load_state_dict(payload["model_state_dict"], strict=strict)
    self._restore_trainer_state(
        trainer_state=trainer_state,
        state_dict=payload["trainer_state"],
    )

    optimizer_state = payload.get("optimizer_state_dict")
    if optimizer is not None:
      if optimizer_state is None:
        raise CheckpointValidationError(
            "Full resume requested but optimizer state is missing."
        )
      optimizer.load_state_dict(optimizer_state)

    scheduler_state = payload.get("lr_scheduler_state_dict")
    if lr_scheduler is not None:
      if scheduler_state is None:
        raise CheckpointValidationError(
            "Full resume requested but lr_scheduler state is missing."
        )
      lr_scheduler.load_state_dict(scheduler_state)

    scaler_state = payload.get("grad_scaler_state_dict")
    if grad_scaler is not None:
      if scaler_state is None:
        raise CheckpointValidationError(
            "Full resume requested but grad_scaler state is missing."
        )
      grad_scaler.load_state_dict(scaler_state)

    return payload

  def _build_payload(
      self,
      *,
      model: torch.nn.Module,
      trainer_state: Any,
      epoch: int,
      global_step: int,
      optimizer: Any | None,
      lr_scheduler: Any | None,
      grad_scaler: Any | None,
      extra_metadata: Optional[dict[str, Any]],
  ) -> dict[str, Any]:
    """Builds the stable checkpoint payload."""
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "lr_scheduler_state_dict": (
            lr_scheduler.state_dict() if lr_scheduler is not None else None
        ),
        "grad_scaler_state_dict": (
            grad_scaler.state_dict() if grad_scaler is not None else None
        ),
        "trainer_state": self._trainer_state_to_dict(trainer_state),
        "metadata": dict(extra_metadata or {}),
    }

  def _trainer_state_to_dict(self, trainer_state: Any) -> dict[str, Any]:
    """Converts trainer state into a serializable dict."""
    if isinstance(trainer_state, dict):
      return dict(trainer_state)

    if is_dataclass(trainer_state):
      return asdict(trainer_state)

    raise TypeError(
        "trainer_state must be a dict or dataclass-compatible object."
    )

  def _restore_trainer_state(
      self,
      *,
      trainer_state: Any,
      state_dict: dict[str, Any],
  ) -> None:
    """Restores trainer state in place."""
    if isinstance(trainer_state, dict):
      trainer_state.clear()
      trainer_state.update(state_dict)
      return

    for key, value in state_dict.items():
      if hasattr(trainer_state, key):
        setattr(trainer_state, key, value)

  def _validate_payload(
      self,
      payload: Any,
      *,
      checkpoint_path: Path,
  ) -> None:
    """Validates the minimal checkpoint contract."""
    if not isinstance(payload, dict):
      raise CheckpointValidationError(
          f"Checkpoint payload must be a dict: {checkpoint_path}"
      )

    required_keys = (
        "checkpoint_version",
        "epoch",
        "global_step",
        "model_state_dict",
        "trainer_state",
        "metadata",
    )
    missing_keys = [key for key in required_keys if key not in payload]
    if missing_keys:
      raise CheckpointValidationError(
          f"Checkpoint is missing required keys {missing_keys}: "
          f"{checkpoint_path}"
      )

    if payload["checkpoint_version"] != CHECKPOINT_VERSION:
      raise CheckpointValidationError(
          "Unsupported checkpoint version: "
          f"{payload['checkpoint_version']}"
      )

    if not isinstance(payload["epoch"], int):
      raise CheckpointValidationError("Checkpoint field 'epoch' must be int.")

    if not isinstance(payload["global_step"], int):
      raise CheckpointValidationError(
          "Checkpoint field 'global_step' must be int."
      )

    if not isinstance(payload["trainer_state"], dict):
      raise CheckpointValidationError(
          "Checkpoint field 'trainer_state' must be a dict."
      )

    if not isinstance(payload["metadata"], dict):
      raise CheckpointValidationError(
          "Checkpoint field 'metadata' must be a dict."
      )

  def _atomic_torch_save(
      self,
      payload: dict[str, Any],
      target_path: Path,
  ) -> None:
    """Saves checkpoint atomically with temp file + replace."""
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        dir=target_path.parent,
        prefix=f".{target_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temp_file:
      temp_path = Path(temp_file.name)

    try:
      torch.save(payload, temp_path)
      os.replace(temp_path, target_path)
    finally:
      temp_path.unlink(missing_ok=True)