"""Shared helpers for artifact-driven evaluator tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from die_vfm.artifacts.embedding_artifact import (
    EmbeddingManifest,
    EmbeddingShardInfo,
)


def write_embedding_split(
    split_dir: Path,
    split_name: str,
    embeddings: torch.Tensor,
    labels: torch.Tensor | None,
    image_ids: list[str] | None = None,
    metadata: list[dict[str, Any]] | None = None,
) -> None:
    """Writes a minimal single-shard embedding artifact split."""
    split_dir.mkdir(parents=True, exist_ok=True)

    num_samples = int(embeddings.shape[0])

    if image_ids is None:
        image_ids = [f"{split_name}_{index:04d}" for index in range(num_samples)]
    if metadata is None:
        metadata = [{"split": split_name, "index": index} for index in range(num_samples)]

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
        shards=[
            EmbeddingShardInfo(
                file_name="part-00000.pt",
                num_samples=num_samples,
            )
        ],
    )
    manifest.save_yaml(split_dir / "manifest.yaml")


def make_artifacts(root_dir: Path) -> tuple[Path, Path]:
    """Creates easy train/val embedding artifacts for evaluator script tests."""
    train_dir = root_dir / "train"
    val_dir = root_dir / "val"

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
    train_labels = torch.tensor([10, 10, 10, 10, 20, 20, 20, 20], dtype=torch.long)

    val_embeddings = torch.tensor(
        [
            [-2.7, -2.1],
            [-1.8, -2.6],
            [2.2, 2.1],
            [2.9, 2.4],
        ],
        dtype=torch.float32,
    )
    val_labels = torch.tensor([10, 10, 20, 20], dtype=torch.long)

    write_embedding_split(train_dir, "train", train_embeddings, train_labels)
    write_embedding_split(val_dir, "val", val_embeddings, val_labels)

    return train_dir, val_dir
