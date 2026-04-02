from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from die_vfm.trainer.checkpoint_manager import (
    BEST_CHECKPOINT_NAME,
    LATEST_CHECKPOINT_NAME,
    CheckpointManager,
    CheckpointValidationError,
)


@dataclass
class DummyTrainerState:
  epoch: int = 0
  global_step: int = 0


def _build_model() -> torch.nn.Module:
  model = torch.nn.Linear(4, 2)
  with torch.no_grad():
    model.weight.fill_(1.5)
    model.bias.fill_(0.25)
  return model


def test_save_writes_latest_best_and_epoch_checkpoints(
    tmp_path: Path,
) -> None:
  manager = CheckpointManager(tmp_path / "checkpoints")
  model = _build_model()
  trainer_state = DummyTrainerState(epoch=3, global_step=12)

  written_paths = manager.save(
      model=model,
      trainer_state=trainer_state,
      epoch=3,
      global_step=12,
      is_best=True,
      extra_metadata={"phase": "unit_test"},
  )

  latest_path = manager.get_latest_checkpoint_path()
  best_path = manager.get_best_checkpoint_path()
  epoch_path = manager.get_epoch_checkpoint_path(3)

  assert written_paths["latest"] == latest_path
  assert written_paths["best"] == best_path
  assert written_paths["epoch"] == epoch_path

  assert latest_path.exists()
  assert best_path.exists()
  assert epoch_path.exists()

  assert latest_path.name == LATEST_CHECKPOINT_NAME
  assert best_path.name == BEST_CHECKPOINT_NAME
  assert epoch_path.name == "epoch_0003.pt"


def test_save_without_is_best_does_not_write_best_checkpoint(
    tmp_path: Path,
) -> None:
  manager = CheckpointManager(tmp_path / "checkpoints")
  model = _build_model()

  written_paths = manager.save(
      model=model,
      trainer_state=DummyTrainerState(),
      epoch=1,
      global_step=5,
      is_best=False,
  )

  assert "best" not in written_paths
  assert manager.get_latest_checkpoint_path().exists()
  assert manager.get_epoch_checkpoint_path(1).exists()
  assert not manager.get_best_checkpoint_path().exists()


def test_has_latest_checkpoint_reflects_checkpoint_presence(
    tmp_path: Path,
) -> None:
  manager = CheckpointManager(tmp_path / "checkpoints")
  assert not manager.has_latest_checkpoint()

  manager.save(
      model=_build_model(),
      trainer_state=DummyTrainerState(),
      epoch=0,
      global_step=1,
  )

  assert manager.has_latest_checkpoint()


def test_resolve_resume_path_returns_explicit_checkpoint_path(
    tmp_path: Path,
) -> None:
  manager = CheckpointManager(tmp_path / "checkpoints")
  explicit_path = tmp_path / "external_checkpoint.pt"
  explicit_path.write_bytes(b"placeholder")

  resolved = manager.resolve_resume_path(
      checkpoint_path=explicit_path,
      auto_resume_latest=False,
  )

  assert resolved == explicit_path


def test_resolve_resume_path_raises_for_missing_explicit_checkpoint(
    tmp_path: Path,
) -> None:
  manager = CheckpointManager(tmp_path / "checkpoints")

  missing_path = tmp_path / "missing.pt"
  with pytest.raises(FileNotFoundError):
    manager.resolve_resume_path(
        checkpoint_path=missing_path,
        auto_resume_latest=False,
    )


def test_resolve_resume_path_returns_latest_when_enabled(
    tmp_path: Path,
) -> None:
  manager = CheckpointManager(tmp_path / "checkpoints")
  manager.save(
      model=_build_model(),
      trainer_state=DummyTrainerState(),
      epoch=2,
      global_step=9,
  )

  resolved = manager.resolve_resume_path(
      checkpoint_path=None,
      auto_resume_latest=True,
  )

  assert resolved == manager.get_latest_checkpoint_path()


def test_resolve_resume_path_returns_none_when_no_checkpoint_available(
    tmp_path: Path,
) -> None:
  manager = CheckpointManager(tmp_path / "checkpoints")

  resolved = manager.resolve_resume_path(
      checkpoint_path=None,
      auto_resume_latest=True,
  )

  assert resolved is None


def test_load_returns_validated_payload(tmp_path: Path) -> None:
  manager = CheckpointManager(tmp_path / "checkpoints")
  model = _build_model()

  manager.save(
      model=model,
      trainer_state=DummyTrainerState(epoch=4, global_step=22),
      epoch=4,
      global_step=22,
      extra_metadata={"phase": "unit_test"},
  )

  payload = manager.load(manager.get_latest_checkpoint_path())

  assert payload["checkpoint_version"] == "v1"
  assert payload["epoch"] == 4
  assert payload["global_step"] == 22
  assert isinstance(payload["model_state_dict"], dict)
  assert payload["trainer_state"] == {
      "epoch": 4,
      "global_step": 22,
  }
  assert payload["metadata"] == {"phase": "unit_test"}


def test_load_warm_start_restores_model_state_only(
    tmp_path: Path,
) -> None:
  manager = CheckpointManager(tmp_path / "checkpoints")
  source_model = _build_model()

  manager.save(
      model=source_model,
      trainer_state=DummyTrainerState(epoch=5, global_step=99),
      epoch=5,
      global_step=99,
  )

  target_model = torch.nn.Linear(4, 2)
  original_weight = target_model.weight.detach().clone()
  original_bias = target_model.bias.detach().clone()

  payload = manager.load_warm_start(
      checkpoint_path=manager.get_latest_checkpoint_path(),
      model=target_model,
  )

  assert not torch.equal(original_weight, target_model.weight)
  assert not torch.equal(original_bias, target_model.bias)
  assert torch.equal(target_model.weight, source_model.weight)
  assert torch.equal(target_model.bias, source_model.bias)
  assert payload["epoch"] == 5
  assert payload["global_step"] == 99


def test_load_full_resume_restores_model_trainer_and_optimizer_state(
    tmp_path: Path,
) -> None:
  manager = CheckpointManager(tmp_path / "checkpoints")
  model = _build_model()
  optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
  trainer_state = DummyTrainerState(epoch=7, global_step=123)

  manager.save(
      model=model,
      trainer_state=trainer_state,
      epoch=7,
      global_step=123,
      optimizer=optimizer,
  )

  resumed_model = torch.nn.Linear(4, 2)
  resumed_optimizer = torch.optim.SGD(
      resumed_model.parameters(),
      lr=1.0,
      momentum=0.0,
  )
  resumed_trainer_state = DummyTrainerState()

  payload = manager.load_full_resume(
      checkpoint_path=manager.get_latest_checkpoint_path(),
      model=resumed_model,
      trainer_state=resumed_trainer_state,
      optimizer=resumed_optimizer,
  )

  assert payload["epoch"] == 7
  assert payload["global_step"] == 123
  assert torch.equal(resumed_model.weight, model.weight)
  assert torch.equal(resumed_model.bias, model.bias)
  assert resumed_trainer_state.epoch == 7
  assert resumed_trainer_state.global_step == 123
  assert resumed_optimizer.state_dict()["param_groups"][0]["lr"] == 0.1
  assert resumed_optimizer.state_dict()["param_groups"][0]["momentum"] == 0.9


def test_load_full_resume_raises_when_optimizer_state_missing(
    tmp_path: Path,
) -> None:
  manager = CheckpointManager(tmp_path / "checkpoints")
  model = _build_model()

  manager.save(
      model=model,
      trainer_state=DummyTrainerState(epoch=1, global_step=2),
      epoch=1,
      global_step=2,
      optimizer=None,
  )

  resumed_model = torch.nn.Linear(4, 2)
  resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=0.1)

  with pytest.raises(CheckpointValidationError):
    manager.load_full_resume(
        checkpoint_path=manager.get_latest_checkpoint_path(),
        model=resumed_model,
        trainer_state=DummyTrainerState(),
        optimizer=resumed_optimizer,
    )


def test_load_raises_for_missing_required_payload_keys(
    tmp_path: Path,
) -> None:
  manager = CheckpointManager(tmp_path / "checkpoints")
  broken_path = manager.get_latest_checkpoint_path()

  torch.save(
      {
          "checkpoint_version": "v1",
          "epoch": 1,
          "global_step": 2,
          "model_state_dict": {},
      },
      broken_path,
  )

  with pytest.raises(CheckpointValidationError):
    manager.load(broken_path)


def test_load_raises_for_unsupported_checkpoint_version(
    tmp_path: Path,
) -> None:
  manager = CheckpointManager(tmp_path / "checkpoints")
  broken_path = manager.get_latest_checkpoint_path()

  torch.save(
      {
          "checkpoint_version": "v999",
          "epoch": 1,
          "global_step": 2,
          "model_state_dict": {},
          "trainer_state": {},
          "metadata": {},
      },
      broken_path,
  )

  with pytest.raises(CheckpointValidationError):
    manager.load(broken_path)


def test_load_raises_when_trainer_state_is_not_dict(
    tmp_path: Path,
) -> None:
  manager = CheckpointManager(tmp_path / "checkpoints")
  broken_path = manager.get_latest_checkpoint_path()

  torch.save(
      {
          "checkpoint_version": "v1",
          "epoch": 1,
          "global_step": 2,
          "model_state_dict": {},
          "trainer_state": [],
          "metadata": {},
      },
      broken_path,
  )

  with pytest.raises(CheckpointValidationError):
    manager.load(broken_path)


def test_save_raises_for_unsupported_trainer_state_type(
    tmp_path: Path,
) -> None:
  manager = CheckpointManager(tmp_path / "checkpoints")
  model = _build_model()

  with pytest.raises(TypeError):
    manager.save(
        model=model,
        trainer_state=object(),
        epoch=1,
        global_step=1,
    )