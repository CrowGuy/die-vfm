"""Embedding artifact loader for die_vfm."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from die_vfm.artifacts.embedding_artifact import (
    EmbeddingManifest,
    LoadedEmbeddingSplit,
    default_manifest_path,
    validate_embedding_shard_payload,
)


def load_embedding_split(
    split_dir: str | Path,
    map_location: str | torch.device = "cpu",
) -> LoadedEmbeddingSplit:
    """Loads one embedding artifact split from disk.

    Args:
        split_dir: Directory containing manifest.yaml and shard file(s).
        map_location: torch.load map_location, defaults to CPU.

    Returns:
        LoadedEmbeddingSplit with fully materialized tensors and metadata.
    """
    split_dir = Path(split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    if not split_dir.is_dir():
        raise NotADirectoryError(f"Split path is not a directory: {split_dir}")

    manifest = EmbeddingManifest.load_yaml(default_manifest_path(split_dir))

    shard_payloads = []
    for shard_info in manifest.shards:
        shard_path = split_dir / shard_info.file_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard file not found: {shard_path}")

        payload = torch.load(shard_path, map_location=map_location)
        if not isinstance(payload, dict):
            raise TypeError(
                f"Shard payload must be a dict, got {type(payload)} from {shard_path}."
            )

        validate_embedding_shard_payload(payload)

        actual_num_samples = int(payload["embeddings"].shape[0])
        if actual_num_samples != shard_info.num_samples:
            raise ValueError(
                "Shard num_samples does not match manifest shard info. "
                f"Got manifest={shard_info.num_samples}, loaded={actual_num_samples}, "
                f"file={shard_info.file_name}."
            )

        shard_payloads.append(payload)

    merged = _merge_shard_payloads(shard_payloads)

    artifact = LoadedEmbeddingSplit(
        manifest=manifest,
        embeddings=merged["embeddings"],
        labels=merged["labels"],
        image_ids=merged["image_ids"],
        metadata=merged["metadata"],
    )
    return artifact


def load_embedding_artifacts(
    embeddings_root: str | Path,
    required_splits: list[str] | tuple[str, ...] | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, LoadedEmbeddingSplit]:
    """Loads multiple embedding splits under one embeddings root directory.

    Example layout:
        runs/<run_name>/embeddings/
            train/
            val/
            test/

    Args:
        embeddings_root: Root directory containing per-split subdirectories.
        required_splits: Optional explicit split names to load. If omitted,
            all immediate child directories are considered.
        map_location: torch.load map_location, defaults to CPU.

    Returns:
        Mapping from split name to loaded artifact.
    """
    embeddings_root = Path(embeddings_root)
    if not embeddings_root.exists():
        raise FileNotFoundError(f"Embeddings root not found: {embeddings_root}")
    if not embeddings_root.is_dir():
        raise NotADirectoryError(
            f"Embeddings root path is not a directory: {embeddings_root}"
        )

    if required_splits is None:
        split_names = sorted(
            path.name for path in embeddings_root.iterdir() if path.is_dir()
        )
    else:
        split_names = list(required_splits)

    artifacts: dict[str, LoadedEmbeddingSplit] = {}
    for split_name in split_names:
        split_dir = embeddings_root / split_name
        artifacts[split_name] = load_embedding_split(
            split_dir=split_dir,
            map_location=map_location,
        )

    return artifacts


def _merge_shard_payloads(
    shard_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merges one or more shard payloads into a single split payload."""
    if not shard_payloads:
        raise ValueError("shard_payloads must be non-empty.")

    embeddings_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    image_ids: list[str] = []
    metadata: list[dict[str, Any]] = []

    saw_labels: bool | None = None
    embedding_dim: int | None = None
    embedding_dtype: torch.dtype | None = None

    for payload in shard_payloads:
        embeddings = payload["embeddings"]
        labels = payload["labels"]
        shard_image_ids = payload["image_ids"]
        shard_metadata = payload["metadata"]

        if embedding_dim is None:
            embedding_dim = int(embeddings.shape[1])
        elif int(embeddings.shape[1]) != embedding_dim:
            raise ValueError(
                "All shard embeddings must share the same embedding_dim. "
                f"Expected {embedding_dim}, got {int(embeddings.shape[1])}."
            )

        if embedding_dtype is None:
            embedding_dtype = embeddings.dtype
        elif embeddings.dtype != embedding_dtype:
            raise ValueError(
                "All shard embeddings must share the same dtype. "
                f"Expected {embedding_dtype}, got {embeddings.dtype}."
            )

        if saw_labels is None:
            saw_labels = labels is not None
        elif saw_labels != (labels is not None):
            raise ValueError(
                "Inconsistent label availability across shards: "
                "some shards have labels but others do not."
            )

        embeddings_list.append(embeddings)
        image_ids.extend(shard_image_ids)
        metadata.extend(shard_metadata)

        if labels is not None:
            labels_list.append(labels)

    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Duplicate image_ids detected across shard payloads.")

    merged_embeddings = torch.cat(embeddings_list, dim=0)

    merged_labels: torch.Tensor | None
    if saw_labels:
        merged_labels = torch.cat(labels_list, dim=0)
    else:
        merged_labels = None

    return {
        "embeddings": merged_embeddings,
        "labels": merged_labels,
        "image_ids": image_ids,
        "metadata": metadata,
    }