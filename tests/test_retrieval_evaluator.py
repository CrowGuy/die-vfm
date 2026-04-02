"""Tests for retrieval evaluator."""

from __future__ import annotations

import pytest
import torch

from die_vfm.evaluator.io import LinearProbeDataBundle
from die_vfm.evaluator.io import LinearProbeSplitData
from die_vfm.evaluator.retrieval_evaluator import RetrievalEvaluatorConfig
from die_vfm.evaluator.retrieval_evaluator import evaluate_retrieval


def _make_split(
    split_name: str,
    embeddings: list[list[float]],
    labels: list[int],
    image_ids: list[str] | None = None,
) -> LinearProbeSplitData:
    """Builds a minimal validated split for evaluator tests."""
    num_samples = len(embeddings)
    if image_ids is None:
        image_ids = [f"{split_name}_{index}" for index in range(num_samples)]

    return LinearProbeSplitData(
        split_name=split_name,
        embeddings=torch.tensor(embeddings, dtype=torch.float32),
        labels=torch.tensor(labels, dtype=torch.long),
        original_labels=torch.tensor(labels, dtype=torch.long),
        image_ids=image_ids,
        metadata=[{} for _ in range(num_samples)],
        manifest=_DummyManifest(),
    )


def _make_bundle(
    train_embeddings: list[list[float]],
    train_labels: list[int],
    val_embeddings: list[list[float]],
    val_labels: list[int],
    train_image_ids: list[str] | None = None,
    val_image_ids: list[str] | None = None,
) -> LinearProbeDataBundle:
    """Builds a minimal train/val bundle for retrieval evaluation."""
    train_split = _make_split(
        split_name="train",
        embeddings=train_embeddings,
        labels=train_labels,
        image_ids=train_image_ids,
    )
    val_split = _make_split(
        split_name="val",
        embeddings=val_embeddings,
        labels=val_labels,
        image_ids=val_image_ids,
    )
    class_ids = sorted(set(train_labels))
    class_to_index = {
        class_id: index for index, class_id in enumerate(class_ids)
    }
    return LinearProbeDataBundle(
        train=train_split,
        val=val_split,
        class_ids=class_ids,
        class_to_index=class_to_index,
    )


class _DummyManifest:
    """Minimal manifest stub for split construction."""

    has_labels = True


def test_evaluate_retrieval_perfect_cosine_returns_one_for_recall_and_map() -> None:
    """Perfectly separated data should achieve perfect retrieval metrics."""
    bundle = _make_bundle(
        train_embeddings=[
            [10.0, 0.0],
            [9.0, 0.0],
            [0.0, 10.0],
            [0.0, 9.0],
        ],
        train_labels=[0, 0, 1, 1],
        val_embeddings=[
            [8.0, 0.0],
            [0.0, 8.0],
        ],
        val_labels=[0, 1],
    )
    config = RetrievalEvaluatorConfig(
        metric="cosine",
        batch_size=2,
        device="cpu",
        topk=(1, 2),
        save_predictions_topk=2,
    )

    output = evaluate_retrieval(bundle=bundle, config=config)

    assert output.query_labels.tolist() == [0, 1]
    assert output.topk_indices.shape == (2, 2)
    assert output.topk_labels.shape == (2, 2)
    assert output.topk_scores.shape == (2, 2)
    assert output.topk_matches.shape == (2, 2)

    assert output.metrics["recall_at_1"] == pytest.approx(1.0)
    assert output.metrics["recall_at_2"] == pytest.approx(1.0)
    assert output.metrics["map_at_1"] == pytest.approx(1.0)
    assert output.metrics["map_at_2"] == pytest.approx(1.0)
    assert output.metrics["num_valid_queries_for_map"] == pytest.approx(2.0)
    assert output.metrics["num_queries_without_positive"] == pytest.approx(0.0)


def test_evaluate_retrieval_recall_at_five_is_not_less_than_recall_at_one() -> None:
    """Recall@K should be monotonic non-decreasing as K increases."""
    bundle = _make_bundle(
        train_embeddings=[
            [0.95, 0.05],
            [0.80, 0.20],
            [0.10, 0.90],
            [0.20, 0.80],
            [0.75, 0.25],
        ],
        train_labels=[1, 0, 1, 1, 1],
        val_embeddings=[
            [1.0, 0.0],
        ],
        val_labels=[0],
    )
    config = RetrievalEvaluatorConfig(
        metric="cosine",
        batch_size=1,
        device="cpu",
        topk=(1, 5),
        save_predictions_topk=5,
    )

    output = evaluate_retrieval(bundle=bundle, config=config)

    assert output.metrics["recall_at_1"] == pytest.approx(0.0)
    assert output.metrics["recall_at_5"] == pytest.approx(1.0)
    assert (
        output.metrics["recall_at_5"] >= output.metrics["recall_at_1"]
    )


def test_evaluate_retrieval_cosine_and_l2_match_on_simple_ordering() -> None:
    """Cosine and L2 should agree on an easy ranking case."""
    bundle = _make_bundle(
        train_embeddings=[
            [1.0, 0.0],
            [0.9, 0.0],
            [0.0, 1.0],
            [0.0, 0.9],
        ],
        train_labels=[0, 0, 1, 1],
        val_embeddings=[
            [0.95, 0.0],
            [0.0, 0.95],
        ],
        val_labels=[0, 1],
    )

    cosine_output = evaluate_retrieval(
        bundle=bundle,
        config=RetrievalEvaluatorConfig(
            metric="cosine",
            batch_size=2,
            device="cpu",
            topk=(1, 2),
            save_predictions_topk=2,
        ),
    )
    l2_output = evaluate_retrieval(
        bundle=bundle,
        config=RetrievalEvaluatorConfig(
            metric="l2",
            batch_size=2,
            device="cpu",
            topk=(1, 2),
            save_predictions_topk=2,
        ),
    )

    assert cosine_output.topk_labels.tolist() == l2_output.topk_labels.tolist()
    assert cosine_output.topk_matches.tolist() == l2_output.topk_matches.tolist()
    assert cosine_output.metrics["recall_at_1"] == pytest.approx(
        l2_output.metrics["recall_at_1"]
    )
    assert cosine_output.metrics["recall_at_2"] == pytest.approx(
        l2_output.metrics["recall_at_2"]
    )
    assert cosine_output.metrics["map_at_1"] == pytest.approx(
        l2_output.metrics["map_at_1"]
    )
    assert cosine_output.metrics["map_at_2"] == pytest.approx(
        l2_output.metrics["map_at_2"]
    )


def test_evaluate_retrieval_ignores_queries_without_positive_for_map() -> None:
    """Queries with no positive in gallery should not contribute to mAP."""
    bundle = _make_bundle(
        train_embeddings=[
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        train_labels=[0, 1],
        val_embeddings=[
            [0.9, 0.1],
            [0.8, 0.2],
        ],
        val_labels=[0, 0],
    )
    config = RetrievalEvaluatorConfig(
        metric="cosine",
        batch_size=2,
        device="cpu",
        topk=(1, 2),
        save_predictions_topk=2,
        exclude_same_image_id=True,
    )

    output = evaluate_retrieval(bundle=bundle, config=config)

    assert output.metrics["num_valid_queries_for_map"] == pytest.approx(2.0)
    assert output.metrics["num_queries_without_positive"] == pytest.approx(0.0)

    bundle_same_id = _make_bundle(
        train_embeddings=[
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        train_labels=[0, 1],
        val_embeddings=[
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        val_labels=[0, 1],
        train_image_ids=["shared_a", "shared_b"],
        val_image_ids=["shared_a", "shared_b"],
    )

    output_same_id = evaluate_retrieval(
        bundle=bundle_same_id,
        config=RetrievalEvaluatorConfig(
            metric="cosine",
            batch_size=2,
            device="cpu",
            topk=(1,),
            save_predictions_topk=1,
            exclude_same_image_id=True,
        ),
    )

    assert output_same_id.metrics["num_valid_queries_for_map"] == pytest.approx(
        0.0
    )
    assert output_same_id.metrics[
        "num_queries_without_positive"
    ] == pytest.approx(2.0)
    assert output_same_id.metrics["map_at_1"] == pytest.approx(0.0)


def test_retrieval_config_raises_when_topk_exceeds_gallery_size() -> None:
    """Requested top-k larger than gallery size should fail."""
    bundle = _make_bundle(
        train_embeddings=[
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        train_labels=[0, 1],
        val_embeddings=[
            [1.0, 0.0],
        ],
        val_labels=[0],
    )
    config = RetrievalEvaluatorConfig(
        metric="cosine",
        batch_size=1,
        device="cpu",
        topk=(1, 3),
        save_predictions_topk=1,
    )

    with pytest.raises(ValueError, match="topk must be <= number of gallery"):
        evaluate_retrieval(bundle=bundle, config=config)


def test_retrieval_config_raises_when_save_predictions_topk_exceeds_gallery() -> None:
    """Saved prediction K must not exceed gallery size."""
    bundle = _make_bundle(
        train_embeddings=[
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        train_labels=[0, 1],
        val_embeddings=[
            [1.0, 0.0],
        ],
        val_labels=[0],
    )
    config = RetrievalEvaluatorConfig(
        metric="cosine",
        batch_size=1,
        device="cpu",
        topk=(1,),
        save_predictions_topk=3,
    )

    with pytest.raises(
        ValueError,
        match="save_predictions_topk must be <= number of train gallery samples",
    ):
        evaluate_retrieval(bundle=bundle, config=config)


def test_retrieval_bundle_raises_on_embedding_dim_mismatch() -> None:
    """Query/gallery embedding dimensions must match."""
    train_split = _make_split(
        split_name="train",
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        labels=[0, 1],
    )
    val_split = _make_split(
        split_name="val",
        embeddings=[[1.0, 0.0, 0.0]],
        labels=[0],
    )

    with pytest.raises(ValueError, match="embedding_dim must match"):
        LinearProbeDataBundle(
            train=train_split,
            val=val_split,
            class_ids=[0, 1],
            class_to_index={0: 0, 1: 1},
        )