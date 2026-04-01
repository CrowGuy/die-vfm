"""Tests for die_vfm.evaluator.metrics."""

from __future__ import annotations

import pytest
import torch

from die_vfm.evaluator.metrics import (
    AverageMeter,
    compute_accuracy,
    compute_num_correct,
    compute_predictions,
    compute_topk_accuracy,
    summarize_classification_metrics,
)


def test_average_meter_returns_zero_when_empty() -> None:
    """Returns 0.0 average when no values were accumulated."""
    meter = AverageMeter()

    assert meter.total == 0.0
    assert meter.count == 0
    assert meter.average == 0.0


def test_average_meter_update_accumulates_weighted_average() -> None:
    """Accumulates weighted totals and counts correctly."""
    meter = AverageMeter()
    meter = meter.update(2.0, n=3)
    meter = meter.update(4.0, n=1)

    assert meter.total == pytest.approx(10.0)
    assert meter.count == 4
    assert meter.average == pytest.approx(2.5)


def test_average_meter_update_allows_zero_weight() -> None:
    """A zero-weight update should not change the aggregate."""
    meter = AverageMeter(total=6.0, count=3)

    updated = meter.update(123.0, n=0)

    assert updated.total == pytest.approx(6.0)
    assert updated.count == 3
    assert updated.average == pytest.approx(2.0)


def test_average_meter_update_raises_for_negative_weight() -> None:
    """Rejects negative weights."""
    meter = AverageMeter()

    with pytest.raises(ValueError, match="non-negative"):
        meter.update(1.0, n=-1)


def test_compute_predictions_returns_argmax_indices() -> None:
    """Computes per-row argmax predictions."""
    logits = torch.tensor(
        [
            [0.1, 0.9, 0.0],
            [2.0, 1.0, 3.0],
            [-1.0, -0.5, -2.0],
        ],
        dtype=torch.float32,
    )

    predictions = compute_predictions(logits)

    expected = torch.tensor([1, 2, 1], dtype=torch.long)
    assert torch.equal(predictions, expected)


def test_compute_predictions_raises_for_invalid_rank() -> None:
    """Rejects logits that do not have shape [N, C]."""
    logits = torch.randn(2, 3, 4)

    with pytest.raises(ValueError, match="shape \\[N, C\\]"):
        compute_predictions(logits)


def test_compute_num_correct_counts_matching_predictions() -> None:
    """Counts the number of correct predictions."""
    logits = torch.tensor(
        [
            [3.0, 1.0],
            [0.2, 0.8],
            [0.6, 0.4],
            [0.1, 0.9],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 1, 1], dtype=torch.long)

    num_correct = compute_num_correct(logits, labels)

    assert num_correct == 3


def test_compute_num_correct_raises_for_batch_size_mismatch() -> None:
    """Rejects mismatched logits and labels batch sizes."""
    logits = torch.randn(3, 2)
    labels = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="same batch size"):
        compute_num_correct(logits, labels)


def test_compute_accuracy_returns_fraction_correct() -> None:
    """Computes top-1 accuracy in [0, 1]."""
    logits = torch.tensor(
        [
            [4.0, 1.0],
            [0.1, 0.9],
            [0.7, 0.3],
            [0.2, 0.8],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 1, 1], dtype=torch.long)

    accuracy = compute_accuracy(logits, labels)

    assert accuracy == pytest.approx(0.75)


def test_compute_accuracy_accepts_non_long_integer_labels() -> None:
    """Canonicalizes integer labels to torch.long internally."""
    logits = torch.tensor(
        [
            [0.9, 0.1],
            [0.1, 0.9],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1], dtype=torch.int32)

    accuracy = compute_accuracy(logits, labels)

    assert accuracy == pytest.approx(1.0)


def test_compute_accuracy_raises_for_empty_labels() -> None:
    """Rejects empty label tensors."""
    logits = torch.empty(0, 2, dtype=torch.float32)
    labels = torch.empty(0, dtype=torch.long)

    with pytest.raises(ValueError, match="non-zero|non-empty"):
        compute_accuracy(logits, labels)


def test_compute_accuracy_raises_for_floating_point_labels() -> None:
    """Rejects floating-point labels."""
    logits = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0.0, 1.0], dtype=torch.float32)

    with pytest.raises(TypeError, match="integer-valued"):
        compute_accuracy(logits, labels)


def test_compute_topk_accuracy_returns_expected_value() -> None:
    """Computes top-k accuracy correctly."""
    logits = torch.tensor(
        [
            [0.70, 0.20, 0.10],  # label 0 in top-1
            [0.45, 0.40, 0.15],  # label 1 in top-2, not top-1
            [0.10, 0.30, 0.60],  # label 2 in top-1
            [0.50, 0.30, 0.20],  # label 2 not in top-2
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 2, 2], dtype=torch.long)

    top1 = compute_topk_accuracy(logits, labels, k=1)
    top2 = compute_topk_accuracy(logits, labels, k=2)

    assert top1 == pytest.approx(0.5)
    assert top2 == pytest.approx(0.75)


def test_compute_topk_accuracy_raises_for_invalid_k() -> None:
    """Rejects invalid k values."""
    logits = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    with pytest.raises(ValueError, match="positive"):
        compute_topk_accuracy(logits, labels, k=0)

    with pytest.raises(ValueError, match="<= num_classes"):
        compute_topk_accuracy(logits, labels, k=4)


def test_compute_topk_accuracy_raises_for_batch_size_mismatch() -> None:
    """Rejects mismatched logits and labels batch sizes."""
    logits = torch.randn(3, 5)
    labels = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="same batch size"):
        compute_topk_accuracy(logits, labels, k=1)


def test_summarize_classification_metrics_returns_accuracy_and_loss() -> None:
    """Builds a flat metric dictionary with accuracy and loss."""
    logits = torch.tensor(
        [
            [2.0, 1.0],
            [0.1, 0.9],
            [0.8, 0.2],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 1], dtype=torch.long)

    metrics = summarize_classification_metrics(
        logits=logits,
        labels=labels,
        loss=0.25,
    )

    assert set(metrics.keys()) == {"accuracy", "loss"}
    assert metrics["accuracy"] == pytest.approx(2.0 / 3.0)
    assert metrics["loss"] == pytest.approx(0.25)


def test_summarize_classification_metrics_includes_requested_topk() -> None:
    """Adds top-k metrics when requested."""
    logits = torch.tensor(
        [
            [0.80, 0.10, 0.10],
            [0.40, 0.35, 0.25],
            [0.10, 0.20, 0.70],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 2], dtype=torch.long)

    metrics = summarize_classification_metrics(
        logits=logits,
        labels=labels,
        loss=1.5,
        topk=(1, 2),
    )

    assert metrics["loss"] == pytest.approx(1.5)
    assert metrics["accuracy"] == pytest.approx(2.0 / 3.0)
    assert metrics["top1_accuracy"] == pytest.approx(2.0 / 3.0)
    assert metrics["top2_accuracy"] == pytest.approx(1.0)