"""Metric helpers for artifact-driven evaluators."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AverageMeter:
    """Accumulates a weighted average for scalar values.

    Attributes:
        total: Weighted sum of all observed values.
        count: Total weight (usually number of samples).
    """

    total: float = 0.0
    count: int = 0

    @property
    def average(self) -> float:
        """Returns the accumulated average.

        Returns:
            The weighted average. Returns 0.0 when no samples were added.
        """
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def update(self, value: float, n: int = 1) -> "AverageMeter":
        """Returns a new meter updated with one observation.

        Args:
            value: Scalar value to accumulate.
            n: Weight for the observation, typically batch size.

        Returns:
            A new AverageMeter with updated total and count.

        Raises:
            ValueError: If n is negative.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}.")

        return AverageMeter(
            total=self.total + float(value) * n,
            count=self.count + n,
        )


def compute_predictions(logits: torch.Tensor) -> torch.Tensor:
    """Computes argmax class predictions from logits.

    Args:
        logits: Logits with shape [N, C].

    Returns:
        Predicted class indices with shape [N].

    Raises:
        ValueError: If logits does not have shape [N, C].
    """
    _validate_logits(logits)
    return torch.argmax(logits, dim=1)


def compute_num_correct(logits: torch.Tensor, labels: torch.Tensor) -> int:
    """Counts the number of correct predictions.

    Args:
        logits: Logits with shape [N, C].
        labels: Integer class indices with shape [N].

    Returns:
        Number of correct predictions as a Python int.

    Raises:
        ValueError: If tensor shapes are invalid or batch sizes mismatch.
        TypeError: If labels are not integer-valued.
    """
    predictions = compute_predictions(logits)
    canonical_labels = _canonicalize_labels(labels)

    if predictions.shape[0] != canonical_labels.shape[0]:
        raise ValueError(
            "Predictions and labels must have the same batch size. "
            f"Got predictions={int(predictions.shape[0])}, "
            f"labels={int(canonical_labels.shape[0])}."
        )

    return int((predictions == canonical_labels).sum().item())


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Computes top-1 accuracy from logits and labels.

    Args:
        logits: Logits with shape [N, C].
        labels: Integer class indices with shape [N].

    Returns:
        Accuracy in [0.0, 1.0].

    Raises:
        ValueError: If tensor shapes are invalid or batch sizes mismatch.
        TypeError: If labels are not integer-valued.
    """
    canonical_labels = _canonicalize_labels(labels)
    num_samples = int(canonical_labels.shape[0])
    if num_samples == 0:
        raise ValueError("labels must be non-empty.")

    num_correct = compute_num_correct(logits, canonical_labels)
    return float(num_correct) / float(num_samples)


def compute_topk_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int,
) -> float:
    """Computes top-k accuracy from logits and labels.

    Args:
        logits: Logits with shape [N, C].
        labels: Integer class indices with shape [N].
        k: Top-k value.

    Returns:
        Top-k accuracy in [0.0, 1.0].

    Raises:
        ValueError: If shapes are invalid, labels are empty, or k is invalid.
        TypeError: If labels are not integer-valued.
    """
    _validate_logits(logits)
    canonical_labels = _canonicalize_labels(labels)

    num_samples = int(canonical_labels.shape[0])
    num_classes = int(logits.shape[1])

    if num_samples == 0:
        raise ValueError("labels must be non-empty.")

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")

    if k > num_classes:
        raise ValueError(
            f"k must be <= num_classes, got k={k}, num_classes={num_classes}."
        )

    if int(logits.shape[0]) != num_samples:
        raise ValueError(
            "Logits and labels must have the same batch size. "
            f"Got logits={int(logits.shape[0])}, labels={num_samples}."
        )

    topk_indices = torch.topk(logits, k=k, dim=1).indices
    matches = topk_indices.eq(canonical_labels.unsqueeze(1))
    num_correct = int(matches.any(dim=1).sum().item())
    return float(num_correct) / float(num_samples)


def summarize_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss: float | None = None,
    topk: tuple[int, ...] = (),
) -> dict[str, float]:
    """Builds a flat metric dictionary for classification outputs.

    Args:
        logits: Logits with shape [N, C].
        labels: Integer class indices with shape [N].
        loss: Optional precomputed scalar loss.
        topk: Optional top-k values to compute.

    Returns:
        Flat metric dictionary, e.g.:
            {
                "loss": 0.123,
                "accuracy": 0.95,
                "top1_accuracy": 0.95,
                "top5_accuracy": 0.99,
            }

    Raises:
        ValueError: If metric inputs are invalid.
        TypeError: If labels are not integer-valued.
    """
    metrics = {
        "accuracy": compute_accuracy(logits=logits, labels=labels),
    }

    for k in topk:
        metrics[f"top{k}_accuracy"] = compute_topk_accuracy(
            logits=logits,
            labels=labels,
            k=k,
        )

    if loss is not None:
        metrics["loss"] = float(loss)

    return metrics


def _validate_logits(logits: torch.Tensor) -> None:
    """Validates classification logits."""
    if logits.ndim != 2:
        raise ValueError(
            f"logits must have shape [N, C], got ndim={logits.ndim}."
        )

    if int(logits.shape[0]) == 0:
        raise ValueError("logits batch dimension must be non-zero.")

    if int(logits.shape[1]) == 0:
        raise ValueError("logits class dimension must be non-zero.")


def _canonicalize_labels(labels: torch.Tensor) -> torch.Tensor:
    """Validates labels and converts them to torch.long."""
    if labels.ndim != 1:
        raise ValueError(
            f"labels must have shape [N], got ndim={labels.ndim}."
        )

    if labels.dtype.is_floating_point or labels.dtype.is_complex:
        raise TypeError(
            f"labels must be integer-valued, got dtype={labels.dtype}."
        )

    return labels.to(dtype=torch.long)