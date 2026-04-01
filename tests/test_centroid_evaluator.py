"""Tests for centroid evaluator."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from die_vfm.evaluator.centroid_evaluator import (
    CentroidEvaluatorConfig,
    _build_class_prototypes,
    _compute_similarity,
    _resolve_topk,
    evaluate_centroid,
)


@dataclass(frozen=True)
class _FakeSplit:
    """Minimal split object for centroid evaluator tests."""

    embeddings: torch.Tensor
    labels: torch.Tensor
    image_ids: list[str]

    @property
    def num_samples(self) -> int:
        return int(self.labels.shape[0])


@dataclass(frozen=True)
class _FakeBundle:
    """Minimal bundle object for centroid evaluator tests."""

    train: _FakeSplit
    val: _FakeSplit
    num_classes: int


def _make_split(
    embeddings: list[list[float]],
    labels: list[int],
    prefix: str,
) -> _FakeSplit:
    """Builds a fake split for testing."""
    return _FakeSplit(
        embeddings=torch.tensor(embeddings, dtype=torch.float32),
        labels=torch.tensor(labels, dtype=torch.long),
        image_ids=[f"{prefix}_{index}" for index in range(len(labels))],
    )


def _make_separable_bundle() -> _FakeBundle:
    """Builds a simple 2-class linearly separable centroid test bundle."""
    train = _make_split(
        embeddings=[
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        labels=[0, 0, 1, 1],
        prefix="train",
    )
    val = _make_split(
        embeddings=[
            [0.95, 0.05],
            [0.05, 0.95],
            [0.80, 0.20],
            [0.20, 0.80],
        ],
        labels=[0, 1, 0, 1],
        prefix="val",
    )
    return _FakeBundle(train=train, val=val, num_classes=2)


def _extract_accuracy(metrics: dict[str, float]) -> float:
    """Returns top-1 style accuracy from metric dict."""
    for key in ("accuracy", "top1_accuracy", "top_1_accuracy"):
        if key in metrics:
            return float(metrics[key])
    raise AssertionError(f"Could not find accuracy key in metrics: {metrics}")


def test_centroid_evaluator_config_rejects_invalid_metric() -> None:
    """Config should reject unsupported metric."""
    with pytest.raises(ValueError, match="metric must be"):
        CentroidEvaluatorConfig(metric="dot")


def test_centroid_evaluator_config_rejects_non_positive_batch_size() -> None:
    """Config should reject invalid batch size."""
    with pytest.raises(ValueError, match="batch_size must be positive"):
        CentroidEvaluatorConfig(batch_size=0)


def test_centroid_evaluator_config_rejects_empty_topk() -> None:
    """Config should reject empty top-k."""
    with pytest.raises(ValueError, match="topk must be non-empty"):
        CentroidEvaluatorConfig(topk=())


def test_build_class_prototypes_returns_class_means() -> None:
    """Prototype builder should compute one mean vector per class."""
    embeddings = torch.tensor(
        [
            [1.0, 1.0],
            [3.0, 3.0],
            [10.0, 0.0],
            [14.0, 4.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    prototypes = _build_class_prototypes(
        embeddings=embeddings,
        labels=labels,
        num_classes=2,
    )

    expected = torch.tensor(
        [
            [2.0, 2.0],
            [12.0, 2.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(prototypes, expected)


def test_build_class_prototypes_raises_when_a_class_is_missing() -> None:
    """Prototype builder should fail fast if train split misses a class."""
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0], dtype=torch.long)

    with pytest.raises(
        ValueError,
        match="train split must contain at least one sample for every class",
    ):
        _build_class_prototypes(
            embeddings=embeddings,
            labels=labels,
            num_classes=2,
        )


def test_compute_similarity_cosine_returns_expected_scores() -> None:
    """Cosine similarity should match normalized dot product."""
    query_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    prototype_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    logits = _compute_similarity(
        query_embeddings=query_embeddings,
        prototype_embeddings=prototype_embeddings,
        metric="cosine",
    )

    expected = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0**-0.5, 2.0**-0.5],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(logits, expected, atol=1e-6, rtol=1e-6)


def test_compute_similarity_l2_returns_negative_distances() -> None:
    """L2 mode should return negative Euclidean distances."""
    query_embeddings = torch.tensor(
        [
            [0.0, 0.0],
            [3.0, 4.0],
        ],
        dtype=torch.float32,
    )
    prototype_embeddings = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 4.0],
        ],
        dtype=torch.float32,
    )

    logits = _compute_similarity(
        query_embeddings=query_embeddings,
        prototype_embeddings=prototype_embeddings,
        metric="l2",
    )

    expected = torch.tensor(
        [
            [-0.0, -4.0],
            [-5.0, -3.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(logits, expected, atol=1e-6, rtol=1e-6)


def test_resolve_topk_sorts_and_deduplicates_values() -> None:
    """Top-k resolver should sort and deduplicate requested values."""
    resolved = _resolve_topk((5, 1, 3, 3), num_classes=5)
    assert resolved == (1, 3, 5)


def test_resolve_topk_raises_when_value_exceeds_num_classes() -> None:
    """Top-k resolver should reject invalid k > num_classes."""
    with pytest.raises(ValueError, match="topk must be <= num_classes"):
        _resolve_topk((1, 3), num_classes=2)


def test_evaluate_centroid_returns_expected_predictions_and_shapes() -> None:
    """Centroid evaluator should classify a separable bundle correctly."""
    bundle = _make_separable_bundle()
    config = CentroidEvaluatorConfig(
        metric="cosine",
        batch_size=2,
        device="cpu",
        topk=(1,),
    )

    output = evaluate_centroid(bundle=bundle, config=config)

    expected_prototypes = torch.tensor(
        [
            [0.95, 0.05],
            [0.05, 0.95],
        ],
        dtype=torch.float32,
    )
    expected_predictions = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    torch.testing.assert_close(output.prototypes, expected_prototypes)
    torch.testing.assert_close(output.prototype_labels, torch.tensor([0, 1]))
    torch.testing.assert_close(output.predictions, expected_predictions)
    torch.testing.assert_close(output.labels, bundle.val.labels)

    assert tuple(output.logits.shape) == (4, 2)
    assert output.image_ids == bundle.val.image_ids
    assert _extract_accuracy(output.metrics) == pytest.approx(1.0)


def test_evaluate_centroid_l2_returns_expected_predictions() -> None:
    """Centroid evaluator should support l2 metric."""
    bundle = _make_separable_bundle()
    config = CentroidEvaluatorConfig(
        metric="l2",
        batch_size=4,
        device="cpu",
        topk=(1,),
    )

    output = evaluate_centroid(bundle=bundle, config=config)

    expected_predictions = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    torch.testing.assert_close(output.predictions, expected_predictions)
    assert _extract_accuracy(output.metrics) == pytest.approx(1.0)


def test_evaluate_centroid_is_batch_size_invariant() -> None:
    """Changing query batch size should not change final outputs."""
    bundle = _make_separable_bundle()

    output_small_batch = evaluate_centroid(
        bundle=bundle,
        config=CentroidEvaluatorConfig(
            metric="cosine",
            batch_size=1,
            device="cpu",
            topk=(1,),
        ),
    )
    output_large_batch = evaluate_centroid(
        bundle=bundle,
        config=CentroidEvaluatorConfig(
            metric="cosine",
            batch_size=16,
            device="cpu",
            topk=(1,),
        ),
    )

    torch.testing.assert_close(
        output_small_batch.predictions,
        output_large_batch.predictions,
    )
    torch.testing.assert_close(
        output_small_batch.labels,
        output_large_batch.labels,
    )
    torch.testing.assert_close(
        output_small_batch.logits,
        output_large_batch.logits,
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        output_small_batch.prototypes,
        output_large_batch.prototypes,
    )
    assert output_small_batch.image_ids == output_large_batch.image_ids
    assert output_small_batch.metrics == output_large_batch.metrics


def test_evaluate_centroid_raises_when_train_split_is_missing_a_class() -> None:
    """Evaluator should fail if a required prototype cannot be built."""
    train = _make_split(
        embeddings=[
            [1.0, 0.0],
            [0.8, 0.2],
        ],
        labels=[0, 0],
        prefix="train",
    )
    val = _make_split(
        embeddings=[
            [0.9, 0.1],
            [0.1, 0.9],
        ],
        labels=[0, 1],
        prefix="val",
    )
    bundle = _FakeBundle(train=train, val=val, num_classes=2)

    with pytest.raises(
        ValueError,
        match="train split must contain at least one sample for every class",
    ):
        evaluate_centroid(
            bundle=bundle,
            config=CentroidEvaluatorConfig(metric="cosine", topk=(1,)),
        )


def test_evaluate_centroid_raises_when_topk_exceeds_num_classes() -> None:
    """Evaluator should validate top-k against class count."""
    bundle = _make_separable_bundle()

    with pytest.raises(ValueError, match="topk must be <= num_classes"):
        evaluate_centroid(
            bundle=bundle,
            config=CentroidEvaluatorConfig(
                metric="cosine",
                batch_size=2,
                device="cpu",
                topk=(1, 3),
            ),
        )