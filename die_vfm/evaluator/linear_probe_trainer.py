"""Training and evaluation loops for linear probe classifiers."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from die_vfm.evaluator.io import LinearProbeDataBundle, LinearProbeSplitData
from die_vfm.evaluator.linear_probe import LinearProbeClassifier
from die_vfm.evaluator.metrics import (
    AverageMeter,
    compute_accuracy,
    compute_predictions,
)


@dataclass(frozen=True)
class LinearProbeTrainerConfig:
    """Configuration for linear probe training.

    Attributes:
        batch_size: Mini-batch size used for tensor slicing.
        num_epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.
        weight_decay: Optimizer weight decay.
        optimizer_name: Optimizer name. Supported: "sgd", "adamw".
        momentum: SGD momentum. Ignored for non-SGD optimizers.
        device: Torch device string.
        seed: Random seed for reproducibility.
        selection_metric: Metric used for best-checkpoint selection.
            Supported: "val_accuracy", "val_loss".
    """

    batch_size: int = 256
    num_epochs: int = 50
    learning_rate: float = 1e-2
    weight_decay: float = 0.0
    optimizer_name: str = "sgd"
    momentum: float = 0.9
    device: str = "cpu"
    seed: int = 0
    selection_metric: str = "val_accuracy"

    def __post_init__(self) -> None:
        """Validates trainer configuration."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}.")
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}.")
        if self.learning_rate <= 0.0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}."
            )
        if self.weight_decay < 0.0:
            raise ValueError(
                f"weight_decay must be non-negative, got {self.weight_decay}."
            )

        optimizer_name = self.optimizer_name.lower()
        if optimizer_name not in {"sgd", "adamw"}:
            raise ValueError(
                "optimizer_name must be one of {'sgd', 'adamw'}, "
                f"got {self.optimizer_name!r}."
            )

        if self.selection_metric not in {"val_accuracy", "val_loss"}:
            raise ValueError(
                "selection_metric must be one of {'val_accuracy', 'val_loss'}, "
                f"got {self.selection_metric!r}."
            )


@dataclass(frozen=True)
class SplitEvaluationOutput:
    """Evaluation output for one split."""

    loss: float
    accuracy: float
    logits: torch.Tensor
    predictions: torch.Tensor
    labels: torch.Tensor
    image_ids: list[str]


@dataclass(frozen=True)
class LinearProbeTrainingResult:
    """Full result of linear probe training and validation."""

    best_epoch: int
    best_state_dict: dict[str, Any]
    history: list[dict[str, float]]
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    val_output: SplitEvaluationOutput


def train_linear_probe(
    model: LinearProbeClassifier,
    bundle: LinearProbeDataBundle,
    config: LinearProbeTrainerConfig,
) -> LinearProbeTrainingResult:
    """Trains a linear probe on train embeddings and evaluates on val embeddings.

    Args:
        model: Linear probe classifier.
        bundle: Prepared train/val bundle from embedding artifacts.
        config: Trainer configuration.

    Returns:
        Full training result including best state, metrics, and history.
    """
    _set_random_seed(config.seed)

    device = torch.device(config.device)
    model = model.to(device)

    optimizer = _build_optimizer(model=model, config=config)
    criterion = nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    best_epoch = -1
    best_state_dict: dict[str, Any] | None = None
    best_score: float | None = None
    best_val_output: SplitEvaluationOutput | None = None
    best_train_metrics: dict[str, float] | None = None
    best_val_metrics: dict[str, float] | None = None

    for epoch in range(1, config.num_epochs + 1):
        train_metrics = _run_train_epoch(
            model=model,
            split=bundle.train,
            optimizer=optimizer,
            criterion=criterion,
            batch_size=config.batch_size,
            device=device,
            epoch_seed=config.seed + epoch,
        )

        val_output = evaluate_linear_probe(
            model=model,
            split=bundle.val,
            criterion=criterion,
            batch_size=config.batch_size,
            device=device,
        )
        val_metrics = {
            "loss": val_output.loss,
            "accuracy": val_output.accuracy,
        }

        epoch_record = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(epoch_record)

        current_score = _select_score(
            metrics=val_metrics,
            selection_metric=config.selection_metric,
        )
        if _is_better_score(
            current_score=current_score,
            best_score=best_score,
            selection_metric=config.selection_metric,
        ):
            best_epoch = epoch
            best_score = current_score
            best_state_dict = copy.deepcopy(model.state_dict())
            best_val_output = val_output
            best_train_metrics = train_metrics
            best_val_metrics = val_metrics

    if best_state_dict is None:
        raise RuntimeError("Training completed without producing a best_state_dict.")

    if best_val_output is None or best_train_metrics is None or best_val_metrics is None:
        raise RuntimeError("Training completed without producing best metrics.")

    return LinearProbeTrainingResult(
        best_epoch=best_epoch,
        best_state_dict=best_state_dict,
        history=history,
        train_metrics=best_train_metrics,
        val_metrics=best_val_metrics,
        val_output=best_val_output,
    )


@torch.no_grad()
def evaluate_linear_probe(
    model: LinearProbeClassifier,
    split: LinearProbeSplitData,
    criterion: nn.Module,
    batch_size: int,
    device: torch.device,
) -> SplitEvaluationOutput:
    """Evaluates a linear probe on one labeled split.

    Args:
        model: Linear probe classifier.
        split: Prepared split data.
        criterion: Loss module.
        batch_size: Mini-batch size.
        device: Target device.

    Returns:
        Aggregated evaluation output for the split.
    """
    model.eval()

    loss_meter = AverageMeter()
    logits_batches: list[torch.Tensor] = []
    labels_batches: list[torch.Tensor] = []

    for batch_indices in _iter_batch_indices(
        num_samples=split.num_samples,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    ):
        batch_embeddings = split.embeddings[batch_indices].to(device)
        batch_labels = split.labels[batch_indices].to(device)

        logits = model(batch_embeddings)
        loss = criterion(logits, batch_labels)

        batch_size_actual = int(batch_labels.shape[0])
        loss_meter = loss_meter.update(loss.item(), n=batch_size_actual)

        logits_batches.append(logits.detach().cpu())
        labels_batches.append(batch_labels.detach().cpu())

    all_logits = torch.cat(logits_batches, dim=0)
    all_labels = torch.cat(labels_batches, dim=0)
    all_predictions = compute_predictions(all_logits)
    accuracy = compute_accuracy(all_logits, all_labels)

    return SplitEvaluationOutput(
        loss=loss_meter.average,
        accuracy=accuracy,
        logits=all_logits,
        predictions=all_predictions,
        labels=all_labels,
        image_ids=list(split.image_ids),
    )


def _run_train_epoch(
    model: LinearProbeClassifier,
    split: LinearProbeSplitData,
    optimizer: Optimizer,
    criterion: nn.Module,
    batch_size: int,
    device: torch.device,
    epoch_seed: int,
) -> dict[str, float]:
    """Runs one training epoch and returns aggregate metrics."""
    model.train()

    loss_meter = AverageMeter()
    total_correct = 0
    total_samples = 0

    for batch_indices in _iter_batch_indices(
        num_samples=split.num_samples,
        batch_size=batch_size,
        shuffle=True,
        seed=epoch_seed,
    ):
        batch_embeddings = split.embeddings[batch_indices].to(device)
        batch_labels = split.labels[batch_indices].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(batch_embeddings)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()

        batch_size_actual = int(batch_labels.shape[0])
        batch_predictions = compute_predictions(logits.detach())
        batch_correct = int((batch_predictions == batch_labels).sum().item())

        loss_meter = loss_meter.update(loss.item(), n=batch_size_actual)
        total_correct += batch_correct
        total_samples += batch_size_actual

    if total_samples == 0:
        raise ValueError("Training split must contain at least one sample.")

    return {
        "loss": loss_meter.average,
        "accuracy": float(total_correct) / float(total_samples),
    }


def _build_optimizer(
    model: LinearProbeClassifier,
    config: LinearProbeTrainerConfig,
) -> Optimizer:
    """Builds the requested optimizer."""
    optimizer_name = config.optimizer_name.lower()

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    raise ValueError(f"Unsupported optimizer_name: {config.optimizer_name!r}.")


def _iter_batch_indices(
    num_samples: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> list[torch.Tensor]:
    """Builds a list of index tensors for mini-batch slicing."""
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    indices = torch.arange(num_samples, dtype=torch.long)
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
        permutation = torch.randperm(num_samples, generator=generator)
        indices = indices[permutation]

    batches: list[torch.Tensor] = []
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batches.append(indices[start:end])

    return batches


def _select_score(
    metrics: dict[str, float],
    selection_metric: str,
) -> float:
    """Returns the scalar score used for best-model selection."""
    if selection_metric == "val_accuracy":
        return metrics["accuracy"]
    if selection_metric == "val_loss":
        return metrics["loss"]
    raise ValueError(f"Unsupported selection_metric: {selection_metric!r}.")


def _is_better_score(
    current_score: float,
    best_score: float | None,
    selection_metric: str,
) -> bool:
    """Determines whether the current score improves over the best score."""
    if best_score is None:
        return True

    if selection_metric == "val_accuracy":
        return current_score > best_score

    if selection_metric == "val_loss":
        return current_score < best_score

    raise ValueError(f"Unsupported selection_metric: {selection_metric!r}.")


def _set_random_seed(seed: int) -> None:
    """Sets Python and torch random seeds."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)