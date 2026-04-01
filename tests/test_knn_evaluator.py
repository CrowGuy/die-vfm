"""Tests for die_vfm.evaluator.knn_evaluator."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from die_vfm.evaluator.knn_evaluator import (
    KnnEvaluatorConfig,
    KnnEvaluationOutput,
    _build_vote_logits,
    _compute_similarity,
    _resolve_topk,
    evaluate_knn,
)


def _make_split(
    embeddings: list[list[float]],
    labels: list[int],
    image_id_prefix: str,
) -> SimpleNamespace:
    """Builds a minimal split object compatible with evaluate_knn()."""
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return SimpleNamespace(
        embeddings=embeddings_tensor,
        labels=labels_tensor,
        image_ids=[
            f"{image_id_prefix}_{index}" for index in range(len(labels))
        ],
        num_samples=len(labels),
    )


def _make_bundle(
    train_embeddings: list[list[float]],
    train_labels: list[int],
    val_embeddings: list[list[float]],
    val_labels: list[int],
) -> SimpleNamespace:
    """Builds a minimal bundle object compatible with evaluate_knn()."""
    train_split = _make_split(
        embeddings=train_embeddings,
        labels=train_labels,
        image_id_prefix="train",
    )
    val_split = _make_split(
        embeddings=val_embeddings,
        labels=val_labels,
        image_id_prefix="val",
    )
    class_ids = sorted(set(train_labels))
    return SimpleNamespace(
        train=train_split,
        val=val_split,
        num_classes=len(class_ids),
        class_ids=class_ids,
    )


def test_knn_evaluator_returns_perfect_accuracy_for_separable_case() -> None:
    bundle = _make_bundle(
        train_embeddings=[
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        train_labels=[0, 0, 1, 1],
        val_embeddings=[
            [0.95, 0.05],
            [0.05, 0.95],
        ],
        val_labels=[0, 1],
    )
    config = KnnEvaluatorConfig(
        k=1,
        metric="cosine",
        weighting="uniform",
        batch_size=8,
        topk=(1,),
    )

    output = evaluate_knn(bundle=bundle, config=config)

    assert isinstance(output, KnnEvaluationOutput)
    assert output.predictions.tolist() == [0, 1]
    assert output.labels.tolist() == [0, 1]
    assert output.metrics["accuracy"] == pytest.approx(1.0)
    assert output.metrics["top1_accuracy"] == pytest.approx(1.0)
    assert output.logits.shape == (2, 2)
    assert output.neighbor_indices.shape == (2, 1)
    assert output.neighbor_labels.shape == (2, 1)
    assert output.neighbor_scores.shape == (2, 1)
    assert output.image_ids == ["val_0", "val_1"]


def test_knn_evaluator_uses_majority_vote_when_k_is_greater_than_one() -> None:
    bundle = _make_bundle(
        train_embeddings=[
            [1.00, 0.00],   # class 0
            [0.99, 0.01],   # class 0
            [0.98, 0.02],   # class 0
            [0.97, 0.03],   # class 0
            [0.96, 0.04],   # class 0
            [1.01, 0.00],   # class 1, closest single neighbor
        ],
        train_labels=[0, 0, 0, 0, 0, 1],
        val_embeddings=[
            [1.005, 0.0],
        ],
        val_labels=[0],
    )
    config = KnnEvaluatorConfig(
        k=5,
        metric="l2",
        weighting="uniform",
        batch_size=8,
        topk=(1,),
    )

    output = evaluate_knn(bundle=bundle, config=config)

    assert output.predictions.tolist() == [0]
    assert output.metrics["accuracy"] == pytest.approx(1.0)
    assert output.neighbor_labels.shape == (1, 5)
    assert output.neighbor_labels[0].tolist().count(0) >= 3


def test_knn_evaluator_distance_weighting_changes_prediction() -> None:
    bundle = _make_bundle(
        train_embeddings=[
            [1.00, 0.00],   # class 0, very close
            [0.00, 1.00],   # class 1
            [0.00, 0.99],   # class 1
        ],
        train_labels=[0, 1, 1],
        val_embeddings=[
            [0.90, 0.10],
        ],
        val_labels=[0],
    )

    uniform_output = evaluate_knn(
        bundle=bundle,
        config=KnnEvaluatorConfig(
            k=3,
            metric="cosine",
            weighting="uniform",
            batch_size=8,
            topk=(1,),
        ),
    )
    distance_output = evaluate_knn(
        bundle=bundle,
        config=KnnEvaluatorConfig(
            k=3,
            metric="cosine",
            weighting="distance",
            temperature=0.01,
            batch_size=8,
            topk=(1,),
        ),
    )

    assert uniform_output.predictions.tolist() == [1]
    assert distance_output.predictions.tolist() == [0]


def test_knn_evaluator_cosine_and_l2_can_produce_different_predictions() -> None:
    bundle = _make_bundle(
        train_embeddings=[
            [10.0, 0.0],   # class 0, same direction but far in l2
            [2.0, 2.0],    # class 1, closer in l2
        ],
        train_labels=[0, 1],
        val_embeddings=[
            [1.0, 0.0],
        ],
        val_labels=[0],
    )

    cosine_output = evaluate_knn(
        bundle=bundle,
        config=KnnEvaluatorConfig(
            k=1,
            metric="cosine",
            weighting="uniform",
            batch_size=8,
            topk=(1,),
        ),
    )
    l2_output = evaluate_knn(
        bundle=bundle,
        config=KnnEvaluatorConfig(
            k=1,
            metric="l2",
            weighting="uniform",
            batch_size=8,
            topk=(1,),
        ),
    )

    assert cosine_output.predictions.tolist() == [0]
    assert l2_output.predictions.tolist() == [1]


def test_knn_evaluator_batching_does_not_change_results() -> None:
    bundle = _make_bundle(
        train_embeddings=[
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        train_labels=[0, 0, 1, 1],
        val_embeddings=[
            [0.95, 0.05],
            [0.90, 0.10],
            [0.10, 0.90],
            [0.05, 0.95],
        ],
        val_labels=[0, 0, 1, 1],
    )

    output_small_batch = evaluate_knn(
        bundle=bundle,
        config=KnnEvaluatorConfig(
            k=1,
            metric="cosine",
            weighting="uniform",
            batch_size=1,
            topk=(1,),
        ),
    )
    output_large_batch = evaluate_knn(
        bundle=bundle,
        config=KnnEvaluatorConfig(
            k=1,
            metric="cosine",
            weighting="uniform",
            batch_size=128,
            topk=(1,),
        ),
    )

    assert torch.equal(
        output_small_batch.predictions,
        output_large_batch.predictions,
    )
    assert torch.equal(
        output_small_batch.logits,
        output_large_batch.logits,
    )
    assert torch.equal(
        output_small_batch.neighbor_indices,
        output_large_batch.neighbor_indices,
    )
    assert output_small_batch.metrics == output_large_batch.metrics


def test_knn_evaluator_reports_topk_metrics() -> None:
    bundle = _make_bundle(
        train_embeddings=[
            [1.0, 0.0],   # class 0
            [0.8, 0.2],   # class 1
            [0.6, 0.4],   # class 2
            [0.4, 0.6],   # class 3
            [0.2, 0.8],   # class 4
        ],
        train_labels=[0, 1, 2, 3, 4],
        val_embeddings=[
            [0.95, 0.05],
        ],
        val_labels=[1],
    )
    config = KnnEvaluatorConfig(
        k=5,
        metric="cosine",
        weighting="distance",
        temperature=0.5,
        batch_size=8,
        topk=(1, 5),
    )

    output = evaluate_knn(bundle=bundle, config=config)

    assert "accuracy" in output.metrics
    assert "top1_accuracy" in output.metrics
    assert "top5_accuracy" in output.metrics
    assert output.metrics["top5_accuracy"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"k": 0}, "k must be positive"),
        ({"metric": "dot"}, "metric must be 'cosine' or 'l2'"),
        ({"weighting": "rank"}, "weighting must be 'uniform' or 'distance'"),
        ({"temperature": 0.0}, "temperature must be positive"),
        ({"batch_size": 0}, "batch_size must be positive"),
        ({"topk": ()}, "topk must be non-empty"),
        ({"topk": (0,)}, "topk must contain only positive values"),
    ],
)
def test_knn_evaluator_config_validation_raises_value_error(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        KnnEvaluatorConfig(**kwargs)


def test_knn_evaluator_raises_when_k_exceeds_train_samples() -> None:
    bundle = _make_bundle(
        train_embeddings=[[1.0, 0.0], [0.0, 1.0]],
        train_labels=[0, 1],
        val_embeddings=[[0.9, 0.1]],
        val_labels=[0],
    )
    config = KnnEvaluatorConfig(
        k=3,
        metric="cosine",
        weighting="uniform",
        batch_size=8,
        topk=(1,),
    )

    with pytest.raises(
        ValueError,
        match="k must be <= number of train reference samples",
    ):
        evaluate_knn(bundle=bundle, config=config)


def test_knn_evaluator_raises_when_requested_topk_exceeds_num_classes() -> None:
    bundle = _make_bundle(
        train_embeddings=[[1.0, 0.0], [0.0, 1.0]],
        train_labels=[0, 1],
        val_embeddings=[[0.9, 0.1]],
        val_labels=[0],
    )
    config = KnnEvaluatorConfig(
        k=1,
        metric="cosine",
        weighting="uniform",
        batch_size=8,
        topk=(1, 3),
    )

    with pytest.raises(
        ValueError,
        match="Requested topk exceeds the number of classes",
    ):
        evaluate_knn(bundle=bundle, config=config)


def test_compute_similarity_for_cosine_returns_expected_ranking() -> None:
    query_embeddings = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    reference_embeddings = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )

    similarity = _compute_similarity(
        query_embeddings=query_embeddings,
        reference_embeddings=reference_embeddings,
        metric="cosine",
    )

    assert similarity.shape == (1, 2)
    assert similarity[0, 0] > similarity[0, 1]


def test_compute_similarity_for_l2_returns_negative_distances() -> None:
    query_embeddings = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    reference_embeddings = torch.tensor(
        [[1.0, 0.0], [3.0, 0.0]],
        dtype=torch.float32,
    )

    similarity = _compute_similarity(
        query_embeddings=query_embeddings,
        reference_embeddings=reference_embeddings,
        metric="l2",
    )

    assert similarity.shape == (1, 2)
    assert similarity[0, 0] > similarity[0, 1]


def test_build_vote_logits_uniform_accumulates_class_counts() -> None:
    neighbor_labels = torch.tensor([[0, 1, 1]], dtype=torch.long)
    neighbor_scores = torch.tensor([[0.9, 0.8, 0.7]], dtype=torch.float32)

    logits = _build_vote_logits(
        neighbor_labels=neighbor_labels,
        neighbor_scores=neighbor_scores,
        num_classes=3,
        weighting="uniform",
        temperature=0.1,
    )

    assert logits.shape == (1, 3)
    assert logits[0, 0].item() == pytest.approx(1.0)
    assert logits[0, 1].item() == pytest.approx(2.0)
    assert logits[0, 2].item() == pytest.approx(0.0)


def test_build_vote_logits_distance_favors_higher_scoring_neighbor() -> None:
    neighbor_labels = torch.tensor([[0, 1, 1]], dtype=torch.long)
    neighbor_scores = torch.tensor([[10.0, 1.0, 1.0]], dtype=torch.float32)

    logits = _build_vote_logits(
        neighbor_labels=neighbor_labels,
        neighbor_scores=neighbor_scores,
        num_classes=2,
        weighting="distance",
        temperature=0.1,
    )

    assert logits.shape == (1, 2)
    assert logits[0, 0] > logits[0, 1]


def test_build_vote_logits_raises_for_shape_mismatch() -> None:
    neighbor_labels = torch.tensor([[0, 1]], dtype=torch.long)
    neighbor_scores = torch.tensor([[0.9]], dtype=torch.float32)

    with pytest.raises(
        ValueError,
        match="neighbor_labels and neighbor_scores must have the same shape",
    ):
        _build_vote_logits(
            neighbor_labels=neighbor_labels,
            neighbor_scores=neighbor_scores,
            num_classes=2,
            weighting="uniform",
            temperature=0.1,
        )


def test_resolve_topk_returns_sorted_unique_values() -> None:
    resolved = _resolve_topk((5, 1, 5, 3), num_classes=5)

    assert resolved == (1, 3, 5)


def test_resolve_topk_raises_when_k_exceeds_num_classes() -> None:
    with pytest.raises(ValueError, match="topk must be <= num_classes"):
        _resolve_topk((1, 6), num_classes=5)


def test_knn_evaluation_output_validates_image_id_length() -> None:
    with pytest.raises(ValueError, match="image_ids length must match labels"):
        KnnEvaluationOutput(
            predictions=torch.tensor([0], dtype=torch.long),
            labels=torch.tensor([0], dtype=torch.long),
            logits=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            neighbor_indices=torch.tensor([[0]], dtype=torch.long),
            neighbor_labels=torch.tensor([[0]], dtype=torch.long),
            neighbor_scores=torch.tensor([[1.0]], dtype=torch.float32),
            image_ids=[],
            metrics={"accuracy": 1.0},
        )