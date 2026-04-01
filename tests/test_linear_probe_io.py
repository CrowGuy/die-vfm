"""Tests for die_vfm.evaluator.io."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from die_vfm.artifacts.embedding_artifact import (
    EmbeddingManifest,
    EmbeddingShardInfo,
)
from die_vfm.evaluator.io import load_linear_probe_bundle


def _write_embedding_split(
    split_dir: Path,
    split_name: str,
    embeddings: torch.Tensor,
    labels: torch.Tensor | None,
    image_ids: list[str] | None = None,
    metadata: list[dict[str, Any]] | None = None,
) -> None:
    """Writes a minimal single-shard embedding artifact split for tests."""
    split_dir.mkdir(parents=True, exist_ok=True)

    num_samples = int(embeddings.shape[0])

    if image_ids is None:
        image_ids = [f"{split_name}_{index:04d}" for index in range(num_samples)]
    if metadata is None:
        metadata = [{"index": index, "split": split_name} for index in range(num_samples)]

    payload = {
        "embeddings": embeddings,
        "labels": labels,
        "image_ids": image_ids,
        "metadata": metadata,
    }
    torch.save(payload, split_dir / "part-00000.pt")

    manifest = EmbeddingManifest(
        split=split_name,
        num_samples=num_samples,
        embedding_dim=int(embeddings.shape[1]),
        dtype=str(embeddings.dtype).replace("torch.", ""),
        has_labels=labels is not None,
        num_shards=1,
        shards=[EmbeddingShardInfo(file_name="part-00000.pt", num_samples=num_samples)],
    )
    manifest.save_yaml(split_dir / "manifest.yaml")


def test_load_linear_probe_bundle_success(tmp_path: Path) -> None:
    """Loads train/val artifacts and builds a valid bundle."""
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"

    train_embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.5],
            [0.9, 0.1, 0.4],
            [0.0, 1.0, 0.2],
            [0.1, 0.8, 0.3],
        ],
        dtype=torch.float32,
    )
    train_labels = torch.tensor([10, 10, 20, 20], dtype=torch.long)

    val_embeddings = torch.tensor(
        [
            [0.8, 0.2, 0.4],
            [0.2, 0.7, 0.3],
        ],
        dtype=torch.float32,
    )
    val_labels = torch.tensor([10, 20], dtype=torch.long)

    _write_embedding_split(train_dir, "train", train_embeddings, train_labels)
    _write_embedding_split(val_dir, "val", val_embeddings, val_labels)

    bundle = load_linear_probe_bundle(
        train_split_dir=train_dir,
        val_split_dir=val_dir,
        normalize_embeddings=False,
    )

    assert bundle.embedding_dim == 3
    assert bundle.num_classes == 2
    assert bundle.class_ids == [10, 20]
    assert bundle.class_to_index == {10: 0, 20: 1}

    assert torch.equal(bundle.train.embeddings, train_embeddings)
    assert torch.equal(bundle.val.embeddings, val_embeddings)

    assert torch.equal(bundle.train.original_labels, train_labels)
    assert torch.equal(bundle.val.original_labels, val_labels)

    assert torch.equal(bundle.train.labels, torch.tensor([0, 0, 1, 1], dtype=torch.long))
    assert torch.equal(bundle.val.labels, torch.tensor([0, 1], dtype=torch.long))

    assert bundle.train.num_samples == 4
    assert bundle.val.num_samples == 2
    assert len(bundle.train.image_ids) == 4
    assert len(bundle.val.metadata) == 2


def test_load_linear_probe_bundle_normalizes_embeddings(tmp_path: Path) -> None:
    """Applies row-wise L2 normalization when requested."""
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"

    train_embeddings = torch.tensor(
        [
            [3.0, 4.0],
            [5.0, 12.0],
        ],
        dtype=torch.float32,
    )
    train_labels = torch.tensor([0, 1], dtype=torch.long)

    val_embeddings = torch.tensor(
        [
            [8.0, 15.0],
            [7.0, 24.0],
        ],
        dtype=torch.float32,
    )
    val_labels = torch.tensor([0, 1], dtype=torch.long)

    _write_embedding_split(train_dir, "train", train_embeddings, train_labels)
    _write_embedding_split(val_dir, "val", val_embeddings, val_labels)

    bundle = load_linear_probe_bundle(
        train_split_dir=train_dir,
        val_split_dir=val_dir,
        normalize_embeddings=True,
    )

    train_norms = torch.linalg.norm(bundle.train.embeddings, dim=1)
    val_norms = torch.linalg.norm(bundle.val.embeddings, dim=1)

    assert torch.allclose(train_norms, torch.ones_like(train_norms), atol=1e-6)
    assert torch.allclose(val_norms, torch.ones_like(val_norms), atol=1e-6)


def test_load_linear_probe_bundle_raises_when_train_labels_missing(
    tmp_path: Path,
) -> None:
    """Raises if the train artifact has no labels."""
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"

    _write_embedding_split(
        train_dir,
        "train",
        embeddings=torch.randn(3, 4),
        labels=None,
    )
    _write_embedding_split(
        val_dir,
        "val",
        embeddings=torch.randn(2, 4),
        labels=torch.tensor([0, 1], dtype=torch.long),
    )

    with pytest.raises(ValueError, match="train split has no labels|requires labels"):
        load_linear_probe_bundle(train_dir, val_dir)


def test_load_linear_probe_bundle_raises_when_val_labels_missing(
    tmp_path: Path,
) -> None:
    """Raises if the val artifact has no labels."""
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"

    _write_embedding_split(
        train_dir,
        "train",
        embeddings=torch.randn(3, 4),
        labels=torch.tensor([0, 1, 0], dtype=torch.long),
    )
    _write_embedding_split(
        val_dir,
        "val",
        embeddings=torch.randn(2, 4),
        labels=None,
    )

    with pytest.raises(ValueError, match="val split has no labels|requires labels"):
        load_linear_probe_bundle(train_dir, val_dir)


def test_load_linear_probe_bundle_raises_when_embedding_dim_mismatch(
    tmp_path: Path,
) -> None:
    """Raises if train/val embedding dimensions differ."""
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"

    _write_embedding_split(
        train_dir,
        "train",
        embeddings=torch.randn(3, 4),
        labels=torch.tensor([0, 1, 0], dtype=torch.long),
    )
    _write_embedding_split(
        val_dir,
        "val",
        embeddings=torch.randn(2, 5),
        labels=torch.tensor([0, 1], dtype=torch.long),
    )

    with pytest.raises(ValueError, match="embedding_dim must match"):
        load_linear_probe_bundle(train_dir, val_dir)


def test_load_linear_probe_bundle_raises_when_val_contains_unseen_class(
    tmp_path: Path,
) -> None:
    """Raises if val includes a class not present in train."""
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"

    _write_embedding_split(
        train_dir,
        "train",
        embeddings=torch.randn(4, 3),
        labels=torch.tensor([0, 0, 1, 1], dtype=torch.long),
    )
    _write_embedding_split(
        val_dir,
        "val",
        embeddings=torch.randn(3, 3),
        labels=torch.tensor([0, 1, 2], dtype=torch.long),
    )

    with pytest.raises(ValueError, match="Unseen class ids: \\[2\\]"):
        load_linear_probe_bundle(train_dir, val_dir)


def test_load_linear_probe_bundle_remaps_non_contiguous_labels(
    tmp_path: Path,
) -> None:
    """Remaps arbitrary original label ids into contiguous class indices."""
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"

    _write_embedding_split(
        train_dir,
        "train",
        embeddings=torch.randn(4, 6),
        labels=torch.tensor([7, 42, 7, 99], dtype=torch.long),
    )
    _write_embedding_split(
        val_dir,
        "val",
        embeddings=torch.randn(3, 6),
        labels=torch.tensor([99, 42, 7], dtype=torch.long),
    )

    bundle = load_linear_probe_bundle(train_dir, val_dir)

    assert bundle.class_ids == [7, 42, 99]
    assert bundle.class_to_index == {7: 0, 42: 1, 99: 2}
    assert torch.equal(bundle.train.labels, torch.tensor([0, 1, 0, 2], dtype=torch.long))
    assert torch.equal(bundle.val.labels, torch.tensor([2, 1, 0], dtype=torch.long))


def test_load_linear_probe_bundle_raises_when_train_has_only_one_class(
    tmp_path: Path,
) -> None:
    """Raises if the train split does not contain at least two classes."""
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"

    _write_embedding_split(
        train_dir,
        "train",
        embeddings=torch.randn(3, 4),
        labels=torch.tensor([5, 5, 5], dtype=torch.long),
    )
    _write_embedding_split(
        val_dir,
        "val",
        embeddings=torch.randn(2, 4),
        labels=torch.tensor([5, 5], dtype=torch.long),
    )

    with pytest.raises(ValueError, match="at least 2 classes"):
        load_linear_probe_bundle(train_dir, val_dir)