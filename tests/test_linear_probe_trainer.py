"""Tests for die_vfm.evaluator.linear_probe_trainer."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from die_vfm.evaluator.io import LinearProbeDataBundle, LinearProbeSplitData
from die_vfm.evaluator.linear_probe import LinearProbeClassifier
from die_vfm.evaluator.linear_probe_trainer import (
    LinearProbeTrainerConfig,
    SplitEvaluationOutput,
    _build_optimizer,
    _iter_batch_indices,
    evaluate_linear_probe,
    train_linear_probe,
)


def _make_split(
    split_name: str,
    embeddings: torch.Tensor,
    original_labels: torch.Tensor,
    class_to_index: dict[int, int],
) -> LinearProbeSplitData:
    """Builds a LinearProbeSplitData object for tests."""
    remapped_labels = torch.tensor(
        [class_to_index[int(label)] for label in original_labels.tolist()],
        dtype=torch.long,
    )
    num_samples = int(embeddings.shape[0])

    return LinearProbeSplitData(
        split_name=split_name,
        embeddings=embeddings.clone(),
        labels=remapped_labels,
        original_labels=original_labels.clone().to(dtype=torch.long),
        image_ids=[f"{split_name}_{index:04d}" for index in range(num_samples)],
        metadata=[{"split": split_name, "index": index} for index in range(num_samples)],
        manifest=_make_manifest(
            split_name=split_name,
            num_samples=num_samples,
            embedding_dim=int(embeddings.shape[1]),
            has_labels=True,
        ),
    )


def _make_bundle() -> LinearProbeDataBundle:
    """Builds an easy linearly separable train/val bundle."""
    class_ids = [10, 20]
    class_to_index = {10: 0, 20: 1}

    train_embeddings = torch.tensor(
        [
            [-3.0, -2.5],
            [-2.5, -2.0],
            [-2.0, -3.0],
            [-1.5, -2.2],
            [2.0, 2.5],
            [2.5, 2.0],
            [3.0, 2.2],
            [1.8, 2.8],
        ],
        dtype=torch.float32,
    )
    train_original_labels = torch.tensor(
        [10, 10, 10, 10, 20, 20, 20, 20],
        dtype=torch.long,
    )

    val_embeddings = torch.tensor(
        [
            [-2.7, -2.1],
            [-1.8, -2.6],
            [2.2, 2.1],
            [2.9, 2.4],
        ],
        dtype=torch.float32,
    )
    val_original_labels = torch.tensor(
        [10, 10, 20, 20],
        dtype=torch.long,
    )

    train_split = _make_split(
        split_name="train",
        embeddings=train_embeddings,
        original_labels=train_original_labels,
        class_to_index=class_to_index,
    )
    val_split = _make_split(
        split_name="val",
        embeddings=val_embeddings,
        original_labels=val_original_labels,
        class_to_index=class_to_index,
    )

    return LinearProbeDataBundle(
        train=train_split,
        val=val_split,
        class_ids=class_ids,
        class_to_index=class_to_index,
    )


def _make_manifest(
    split_name: str,
    num_samples: int,
    embedding_dim: int,
    has_labels: bool,
) -> Any:
    """Builds a lightweight manifest stub for tests."""
    from die_vfm.artifacts.embedding_artifact import EmbeddingManifest, EmbeddingShardInfo

    return EmbeddingManifest(
        split=split_name,
        num_samples=num_samples,
        embedding_dim=embedding_dim,
        dtype="float32",
        has_labels=has_labels,
        num_shards=1,
        shards=[
            EmbeddingShardInfo(
                file_name="part-00000.pt",
                num_samples=num_samples,
            )
        ],
    )


def test_linear_probe_trainer_config_accepts_valid_arguments() -> None:
    """Builds a valid trainer config."""
    config = LinearProbeTrainerConfig(
        batch_size=8,
        num_epochs=10,
        learning_rate=0.05,
        weight_decay=0.01,
        optimizer_name="sgd",
        momentum=0.9,
        device="cpu",
        seed=7,
        selection_metric="val_accuracy",
    )

    assert config.batch_size == 8
    assert config.num_epochs == 10
    assert config.learning_rate == pytest.approx(0.05)
    assert config.weight_decay == pytest.approx(0.01)
    assert config.optimizer_name == "sgd"
    assert config.device == "cpu"
    assert config.seed == 7
    assert config.selection_metric == "val_accuracy"


@pytest.mark.parametrize(
    ("field_name", "kwargs", "match"),
    [
        ("batch_size", {"batch_size": 0}, "batch_size must be positive"),
        ("num_epochs", {"num_epochs": 0}, "num_epochs must be positive"),
        ("learning_rate", {"learning_rate": 0.0}, "learning_rate must be positive"),
        ("weight_decay", {"weight_decay": -1.0}, "weight_decay must be non-negative"),
        ("optimizer_name", {"optimizer_name": "rmsprop"}, "optimizer_name must be one of"),
        ("selection_metric", {"selection_metric": "accuracy"}, "selection_metric must be one of"),
    ],
)
def test_linear_probe_trainer_config_raises_for_invalid_arguments(
    field_name: str,
    kwargs: dict[str, Any],
    match: str,
) -> None:
    """Rejects invalid trainer config arguments."""
    del field_name  # Used for param readability.

    with pytest.raises(ValueError, match=match):
        LinearProbeTrainerConfig(**kwargs)


def test_iter_batch_indices_without_shuffle_covers_all_samples() -> None:
    """Creates sequential mini-batches that cover all indices exactly once."""
    batches = _iter_batch_indices(
        num_samples=10,
        batch_size=4,
        shuffle=False,
        seed=123,
    )

    assert len(batches) == 3
    assert torch.equal(batches[0], torch.tensor([0, 1, 2, 3], dtype=torch.long))
    assert torch.equal(batches[1], torch.tensor([4, 5, 6, 7], dtype=torch.long))
    assert torch.equal(batches[2], torch.tensor([8, 9], dtype=torch.long))

    concatenated = torch.cat(batches, dim=0)
    assert torch.equal(concatenated, torch.arange(10, dtype=torch.long))


def test_iter_batch_indices_with_shuffle_is_deterministic_for_same_seed() -> None:
    """Produces the same shuffled batches for the same seed."""
    batches_a = _iter_batch_indices(
        num_samples=9,
        batch_size=4,
        shuffle=True,
        seed=11,
    )
    batches_b = _iter_batch_indices(
        num_samples=9,
        batch_size=4,
        shuffle=True,
        seed=11,
    )

    assert len(batches_a) == len(batches_b)
    for batch_a, batch_b in zip(batches_a, batches_b):
        assert torch.equal(batch_a, batch_b)

    concatenated = torch.cat(batches_a, dim=0)
    assert sorted(concatenated.tolist()) == list(range(9))


def test_iter_batch_indices_raises_for_invalid_arguments() -> None:
    """Rejects invalid num_samples and batch_size values."""
    with pytest.raises(ValueError, match="num_samples must be positive"):
        _iter_batch_indices(
            num_samples=0,
            batch_size=4,
            shuffle=False,
            seed=0,
        )

    with pytest.raises(ValueError, match="batch_size must be positive"):
        _iter_batch_indices(
            num_samples=4,
            batch_size=0,
            shuffle=False,
            seed=0,
        )


def test_build_optimizer_returns_sgd() -> None:
    """Builds an SGD optimizer when requested."""
    model = LinearProbeClassifier(input_dim=2, num_classes=2)
    config = LinearProbeTrainerConfig(
        optimizer_name="sgd",
        learning_rate=0.1,
        momentum=0.8,
        weight_decay=0.01,
    )

    optimizer = _build_optimizer(model=model, config=config)

    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults["lr"] == pytest.approx(0.1)
    assert optimizer.defaults["momentum"] == pytest.approx(0.8)
    assert optimizer.defaults["weight_decay"] == pytest.approx(0.01)


def test_build_optimizer_returns_adamw() -> None:
    """Builds an AdamW optimizer when requested."""
    model = LinearProbeClassifier(input_dim=2, num_classes=2)
    config = LinearProbeTrainerConfig(
        optimizer_name="adamw",
        learning_rate=0.05,
        weight_decay=0.0,
    )

    optimizer = _build_optimizer(model=model, config=config)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == pytest.approx(0.05)
    assert optimizer.defaults["weight_decay"] == pytest.approx(0.0)


def test_evaluate_linear_probe_returns_expected_output_shapes() -> None:
    """Evaluates one split and returns aligned tensors."""
    bundle = _make_bundle()
    model = LinearProbeClassifier(
        input_dim=bundle.embedding_dim,
        num_classes=bundle.num_classes,
    )
    criterion = nn.CrossEntropyLoss()

    output = evaluate_linear_probe(
        model=model,
        split=bundle.val,
        criterion=criterion,
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert isinstance(output, SplitEvaluationOutput)
    assert output.logits.shape == (bundle.val.num_samples, bundle.num_classes)
    assert output.predictions.shape == (bundle.val.num_samples,)
    assert output.labels.shape == (bundle.val.num_samples,)
    assert len(output.image_ids) == bundle.val.num_samples
    assert output.loss >= 0.0
    assert 0.0 <= output.accuracy <= 1.0
    assert torch.equal(output.labels, bundle.val.labels)
    assert output.image_ids == bundle.val.image_ids


def test_train_linear_probe_learns_easy_separable_data() -> None:
    """Learns a simple linearly separable classification problem."""
    torch.manual_seed(0)
    bundle = _make_bundle()
    model = LinearProbeClassifier(
        input_dim=bundle.embedding_dim,
        num_classes=bundle.num_classes,
    )
    config = LinearProbeTrainerConfig(
        batch_size=4,
        num_epochs=50,
        learning_rate=0.05,
        weight_decay=0.0,
        optimizer_name="adamw",
        device="cpu",
        seed=123,
        selection_metric="val_accuracy",
    )

    result = train_linear_probe(
        model=model,
        bundle=bundle,
        config=config,
    )

    assert result.best_epoch >= 1
    assert len(result.history) == config.num_epochs
    assert set(result.train_metrics.keys()) == {"loss", "accuracy"}
    assert set(result.val_metrics.keys()) == {"loss", "accuracy"}

    assert result.train_metrics["loss"] >= 0.0
    assert result.val_metrics["loss"] >= 0.0
    assert result.train_metrics["accuracy"] >= 0.6
    assert result.val_metrics["accuracy"] >= 0.95

    assert result.val_output.logits.shape == (
        bundle.val.num_samples,
        bundle.num_classes,
    )
    assert result.val_output.predictions.shape == (bundle.val.num_samples,)
    assert torch.equal(result.val_output.labels, bundle.val.labels)
    assert result.val_output.image_ids == bundle.val.image_ids


def test_train_linear_probe_is_deterministic_for_same_seed() -> None:
    """Produces the same result for the same seed and inputs."""
    bundle = _make_bundle()

    config = LinearProbeTrainerConfig(
        batch_size=4,
        num_epochs=30,
        learning_rate=0.1,
        optimizer_name="sgd",
        momentum=0.0,
        device="cpu",
        seed=999,
        selection_metric="val_accuracy",
    )

    torch.manual_seed(7)
    model_a = LinearProbeClassifier(
        input_dim=bundle.embedding_dim,
        num_classes=bundle.num_classes,
    )
    result_a = train_linear_probe(
        model=model_a,
        bundle=bundle,
        config=config,
    )

    torch.manual_seed(7)
    model_b = LinearProbeClassifier(
        input_dim=bundle.embedding_dim,
        num_classes=bundle.num_classes,
    )
    result_b = train_linear_probe(
        model=model_b,
        bundle=bundle,
        config=config,
    )

    assert result_a.best_epoch == result_b.best_epoch
    assert result_a.train_metrics == pytest.approx(result_b.train_metrics)
    assert result_a.val_metrics == pytest.approx(result_b.val_metrics)
    assert result_a.history == pytest.approx(result_b.history)
    assert torch.equal(result_a.val_output.logits, result_b.val_output.logits)
    assert torch.equal(
        result_a.val_output.predictions,
        result_b.val_output.predictions,
    )
    assert torch.equal(result_a.val_output.labels, result_b.val_output.labels)
    assert result_a.val_output.image_ids == result_b.val_output.image_ids

    assert result_a.best_state_dict.keys() == result_b.best_state_dict.keys()
    for key in result_a.best_state_dict:
        assert torch.equal(result_a.best_state_dict[key], result_b.best_state_dict[key])


def test_train_linear_probe_supports_val_loss_selection() -> None:
    """Supports selecting the best checkpoint by validation loss."""
    torch.manual_seed(0)
    bundle = _make_bundle()
    model = LinearProbeClassifier(
        input_dim=bundle.embedding_dim,
        num_classes=bundle.num_classes,
    )
    config = LinearProbeTrainerConfig(
        batch_size=4,
        num_epochs=20,
        learning_rate=0.05,
        optimizer_name="adamw",
        weight_decay=0.0,
        device="cpu",
        seed=77,
        selection_metric="val_loss",
    )

    result = train_linear_probe(
        model=model,
        bundle=bundle,
        config=config,
    )

    assert result.best_epoch >= 1
    assert len(result.history) == config.num_epochs
    assert result.val_metrics["loss"] >= 0.0
    assert 0.0 <= result.val_metrics["accuracy"] <= 1.0


def test_train_linear_probe_history_contains_expected_fields() -> None:
    """Stores per-epoch train/val metrics in history."""
    bundle = _make_bundle()
    model = LinearProbeClassifier(
        input_dim=bundle.embedding_dim,
        num_classes=bundle.num_classes,
    )
    config = LinearProbeTrainerConfig(
        batch_size=4,
        num_epochs=5,
        learning_rate=0.1,
        optimizer_name="sgd",
        momentum=0.0,
        device="cpu",
        seed=1,
    )

    result = train_linear_probe(
        model=model,
        bundle=bundle,
        config=config,
    )

    assert len(result.history) == 5
    for index, epoch_record in enumerate(result.history, start=1):
        assert set(epoch_record.keys()) == {
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
        }
        assert epoch_record["epoch"] == pytest.approx(float(index))
        assert epoch_record["train_loss"] >= 0.0
        assert 0.0 <= epoch_record["train_accuracy"] <= 1.0
        assert epoch_record["val_loss"] >= 0.0
        assert 0.0 <= epoch_record["val_accuracy"] <= 1.0