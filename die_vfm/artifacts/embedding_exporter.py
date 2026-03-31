"""Embedding artifact exporter for die_vfm."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from die_vfm.artifacts.embedding_artifact import (
    EMBEDDING_ARTIFACT_FORMAT,
    EMBEDDING_ARTIFACT_TYPE,
    EMBEDDING_ARTIFACT_VERSION,
    EmbeddingManifest,
    EmbeddingShardInfo,
    default_manifest_path,
    validate_embedding_shard_payload,
)


def export_split_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    output_dir: str | Path,
    split: str,
    device: str | torch.device,
) -> EmbeddingManifest:
    """Exports one dataset split into an embedding artifact.

    The exporter runs model inference over the dataloader, collects
    per-sample outputs on CPU, writes a single shard file for M1, and
    saves a manifest.yaml next to it.

    Args:
        model: The embedding model. Expected to return an object with
            `.embedding` of shape [B, D].
        dataloader: Iterable dataloader yielding batch dictionaries.
        output_dir: Target directory for the split artifact.
        split: Split name, e.g. "train", "val", or "test".
        device: Torch device used for inference.

    Returns:
        EmbeddingManifest describing the written artifact.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device)
    shard_path = output_dir / "part-00000.pt"
    manifest_path = default_manifest_path(output_dir)

    embeddings_chunks: list[torch.Tensor] = []
    labels_chunks: list[torch.Tensor] = []
    image_ids: list[str] = []
    metadata: list[dict[str, Any]] = []

    saw_any_batch = False
    saw_labels: bool | None = None

    was_training = model.training
    model.eval()

    try:
        with torch.inference_mode():
            for batch in dataloader:
                saw_any_batch = True

                batch_images, batch_labels, batch_image_ids, batch_metadata = (
                    _extract_batch_fields(batch)
                )

                batch_embeddings = _forward_embeddings(
                    model=model,
                    images=batch_images,
                    device=device,
                )

                batch_embeddings = batch_embeddings.detach().cpu()

                if batch_embeddings.ndim != 2:
                    raise ValueError(
                        "Model output embedding must have shape [B, D], "
                        f"got shape={tuple(batch_embeddings.shape)}."
                    )

                batch_size = int(batch_embeddings.shape[0])

                if batch_labels is None:
                    if saw_labels is None:
                        saw_labels = False
                    elif saw_labels:
                        raise ValueError(
                            "Inconsistent label availability across batches: "
                            "some batches have labels but others do not."
                        )
                else:
                    if saw_labels is None:
                        saw_labels = True
                    elif not saw_labels:
                        raise ValueError(
                            "Inconsistent label availability across batches: "
                            "some batches have labels but others do not."
                        )

                    batch_labels = batch_labels.detach().cpu()
                    if batch_labels.ndim != 1:
                        raise ValueError(
                            "Batch labels must have shape [B], "
                            f"got shape={tuple(batch_labels.shape)}."
                        )
                    if int(batch_labels.shape[0]) != batch_size:
                        raise ValueError(
                            "Batch label length must match embedding batch size. "
                            f"Got labels={int(batch_labels.shape[0])}, batch={batch_size}."
                        )
                    labels_chunks.append(batch_labels)

                if len(batch_image_ids) != batch_size:
                    raise ValueError(
                        "Batch image_ids length must match embedding batch size. "
                        f"Got image_ids={len(batch_image_ids)}, batch={batch_size}."
                    )

                if len(batch_metadata) != batch_size:
                    raise ValueError(
                        "Batch metadata length must match embedding batch size. "
                        f"Got metadata={len(batch_metadata)}, batch={batch_size}."
                    )

                embeddings_chunks.append(batch_embeddings)
                image_ids.extend(batch_image_ids)
                metadata.extend(batch_metadata)

        if not saw_any_batch:
            raise ValueError(
                f"Dataloader for split={split!r} produced no batches; "
                "cannot export empty embedding artifact in M1."
            )

        embeddings = torch.cat(embeddings_chunks, dim=0)

        labels: torch.Tensor | None
        if saw_labels:
            labels = torch.cat(labels_chunks, dim=0)
        else:
            labels = None

        payload = _build_shard_payload(
            embeddings=embeddings,
            labels=labels,
            image_ids=image_ids,
            metadata=metadata,
        )
        validate_embedding_shard_payload(payload)

        torch.save(payload, shard_path)

        manifest = EmbeddingManifest(
            artifact_type=EMBEDDING_ARTIFACT_TYPE,
            artifact_version=EMBEDDING_ARTIFACT_VERSION,
            format=EMBEDDING_ARTIFACT_FORMAT,
            split=split,
            num_samples=int(embeddings.shape[0]),
            embedding_dim=int(embeddings.shape[1]),
            dtype=_infer_embedding_dtype_str(embeddings),
            has_labels=labels is not None,
            num_shards=1,
            shards=[
                EmbeddingShardInfo(
                    file_name=shard_path.name,
                    num_samples=int(embeddings.shape[0]),
                )
            ],
        )
        manifest.save_yaml(manifest_path)
        return manifest

    finally:
        model.train(was_training)


def _extract_batch_fields(
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor | None, list[str], list[dict[str, Any]]]:
    """Extracts and validates required batch fields.

    Expected batch keys:
        - image: Tensor[B, ...]
        - label: Tensor[B] or None
        - image_id: list[str]
        - meta: list[dict]
    """
    if not isinstance(batch, dict):
        raise TypeError(f"Batch must be a dict, got {type(batch)}.")

    required_keys = {"image", "image_id", "meta"}
    missing_keys = required_keys - set(batch.keys())
    if missing_keys:
        raise ValueError(f"Missing required batch keys: {sorted(missing_keys)}")

    images = batch["image"]
    labels = batch.get("label")
    image_ids = batch["image_id"]
    metadata = batch["meta"]

    if not isinstance(images, torch.Tensor):
        raise TypeError(f"batch['image'] must be torch.Tensor, got {type(images)}.")

    if labels is not None and not isinstance(labels, torch.Tensor):
        raise TypeError(f"batch['label'] must be torch.Tensor or None, got {type(labels)}.")

    if not isinstance(image_ids, list):
        raise TypeError(f"batch['image_id'] must be list[str], got {type(image_ids)}.")
    if any(not isinstance(item, str) or not item for item in image_ids):
        raise ValueError("All batch['image_id'] entries must be non-empty strings.")

    if not isinstance(metadata, list):
        raise TypeError(f"batch['meta'] must be list[dict], got {type(metadata)}.")
    if any(not isinstance(item, dict) for item in metadata):
        raise ValueError("All batch['meta'] entries must be dictionaries.")

    return images, labels, image_ids, metadata


def _forward_embeddings(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Runs model forward and extracts the embedding tensor."""
    images = images.to(device, non_blocking=True)
    output = model(images)

    if not hasattr(output, "embedding"):
        raise AttributeError(
            "Model output must expose an `.embedding` attribute for artifact export."
        )

    embeddings = output.embedding
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError(
            f"Model output `.embedding` must be torch.Tensor, got {type(embeddings)}."
        )

    return embeddings


def _build_shard_payload(
    embeddings: torch.Tensor,
    labels: torch.Tensor | None,
    image_ids: list[str],
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    """Builds the serialized shard payload."""
    return {
        "embeddings": embeddings,
        "labels": labels,
        "image_ids": image_ids,
        "metadata": metadata,
    }


def _infer_embedding_dtype_str(embeddings: torch.Tensor) -> str:
    """Converts tensor dtype to manifest-friendly string form."""
    return str(embeddings.dtype).replace("torch.", "")