"""Embedding artifact schemas and validation helpers for die_vfm."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml


EMBEDDING_ARTIFACT_TYPE = "embedding_split"
EMBEDDING_ARTIFACT_VERSION = "v1"
EMBEDDING_ARTIFACT_FORMAT = "torch_pt"
EMBEDDING_MANIFEST_FILENAME = "manifest.yaml"


@dataclass(frozen=True)
class EmbeddingShardInfo:
    """Metadata for a single embedding shard file."""

    file_name: str
    num_samples: int

    def __post_init__(self) -> None:
        if not self.file_name:
            raise ValueError("EmbeddingShardInfo.file_name must be non-empty.")
        if self.num_samples < 0:
            raise ValueError(
                f"EmbeddingShardInfo.num_samples must be >= 0, got {self.num_samples}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Converts the shard info into a serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddingShardInfo":
        """Builds shard info from a manifest dictionary."""
        return cls(
            file_name=str(data["file_name"]),
            num_samples=int(data["num_samples"]),
        )


@dataclass(frozen=True)
class EmbeddingManifest:
    """Top-level manifest describing one embedding artifact split."""

    artifact_type: str = EMBEDDING_ARTIFACT_TYPE
    artifact_version: str = EMBEDDING_ARTIFACT_VERSION
    format: str = EMBEDDING_ARTIFACT_FORMAT

    split: str = ""
    num_samples: int = 0
    embedding_dim: int = 0
    dtype: str = "float32"
    has_labels: bool = False
    num_shards: int = 0
    shards: list[EmbeddingShardInfo] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.artifact_type != EMBEDDING_ARTIFACT_TYPE:
            raise ValueError(
                f"Unsupported artifact_type={self.artifact_type!r}. "
                f"Expected {EMBEDDING_ARTIFACT_TYPE!r}."
            )
        if self.artifact_version != EMBEDDING_ARTIFACT_VERSION:
            raise ValueError(
                f"Unsupported artifact_version={self.artifact_version!r}. "
                f"Expected {EMBEDDING_ARTIFACT_VERSION!r}."
            )
        if self.format != EMBEDDING_ARTIFACT_FORMAT:
            raise ValueError(
                f"Unsupported format={self.format!r}. "
                f"Expected {EMBEDDING_ARTIFACT_FORMAT!r}."
            )
        if not self.split:
            raise ValueError("EmbeddingManifest.split must be non-empty.")
        if self.num_samples < 0:
            raise ValueError(
                f"EmbeddingManifest.num_samples must be >= 0, got {self.num_samples}."
            )
        if self.embedding_dim <= 0:
            raise ValueError(
                "EmbeddingManifest.embedding_dim must be > 0, "
                f"got {self.embedding_dim}."
            )
        if self.num_shards <= 0:
            raise ValueError(
                f"EmbeddingManifest.num_shards must be > 0, got {self.num_shards}."
            )
        if len(self.shards) != self.num_shards:
            raise ValueError(
                "EmbeddingManifest.shards length must equal num_shards. "
                f"Got len(shards)={len(self.shards)} vs num_shards={self.num_shards}."
            )

        shard_sample_total = sum(shard.num_samples for shard in self.shards)
        if shard_sample_total != self.num_samples:
            raise ValueError(
                "Sum of shard num_samples must equal manifest num_samples. "
                f"Got shard total={shard_sample_total} vs manifest={self.num_samples}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Converts the manifest into a serializable dictionary."""
        return {
            "artifact_type": self.artifact_type,
            "artifact_version": self.artifact_version,
            "format": self.format,
            "split": self.split,
            "num_samples": self.num_samples,
            "embedding_dim": self.embedding_dim,
            "dtype": self.dtype,
            "has_labels": self.has_labels,
            "num_shards": self.num_shards,
            "shards": [shard.to_dict() for shard in self.shards],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddingManifest":
        """Builds a manifest from a deserialized dictionary."""
        return cls(
            artifact_type=str(data.get("artifact_type", EMBEDDING_ARTIFACT_TYPE)),
            artifact_version=str(
                data.get("artifact_version", EMBEDDING_ARTIFACT_VERSION)
            ),
            format=str(data.get("format", EMBEDDING_ARTIFACT_FORMAT)),
            split=str(data["split"]),
            num_samples=int(data["num_samples"]),
            embedding_dim=int(data["embedding_dim"]),
            dtype=str(data["dtype"]),
            has_labels=bool(data["has_labels"]),
            num_shards=int(data["num_shards"]),
            shards=[
                EmbeddingShardInfo.from_dict(item)
                for item in data.get("shards", [])
            ],
        )

    def save_yaml(self, path: str | Path) -> None:
        """Saves the manifest to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def load_yaml(cls, path: str | Path) -> "EmbeddingManifest":
        """Loads the manifest from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Manifest YAML must decode to dict, got {type(data)}.")
        return cls.from_dict(data)


@dataclass
class LoadedEmbeddingSplit:
    """Fully materialized embedding artifact for one dataset split."""

    manifest: EmbeddingManifest
    embeddings: torch.Tensor
    labels: torch.Tensor | None
    image_ids: list[str]
    metadata: list[dict[str, Any]]

    def __post_init__(self) -> None:
        validate_loaded_embedding_split(self)

    @property
    def num_samples(self) -> int:
        """Returns the number of rows in the loaded embedding tensor."""
        return int(self.embeddings.shape[0])

    @property
    def embedding_dim(self) -> int:
        """Returns the embedding feature dimension."""
        return int(self.embeddings.shape[1])

    @property
    def has_labels(self) -> bool:
        """Returns whether labels are available."""
        return self.labels is not None


def validate_embedding_shard_payload(payload: dict[str, Any]) -> None:
    """Validates the contents of one saved shard payload.

    Expected payload keys:
        - embeddings: Tensor[N, D]
        - labels: Tensor[N] | None
        - image_ids: list[str]
        - metadata: list[dict]
    """
    required_keys = {"embeddings", "labels", "image_ids", "metadata"}
    actual_keys = set(payload.keys())
    missing_keys = required_keys - actual_keys
    if missing_keys:
        raise ValueError(f"Missing shard payload keys: {sorted(missing_keys)}")

    embeddings = payload["embeddings"]
    labels = payload["labels"]
    image_ids = payload["image_ids"]
    metadata = payload["metadata"]

    if not isinstance(embeddings, torch.Tensor):
        raise TypeError(
            f"payload['embeddings'] must be torch.Tensor, got {type(embeddings)}."
        )
    if embeddings.ndim != 2:
        raise ValueError(
            "payload['embeddings'] must have shape [N, D], "
            f"got ndim={embeddings.ndim}."
        )

    num_samples = int(embeddings.shape[0])

    if labels is not None:
        if not isinstance(labels, torch.Tensor):
            raise TypeError(f"payload['labels'] must be torch.Tensor, got {type(labels)}.")
        if labels.ndim != 1:
            raise ValueError(
                f"payload['labels'] must have shape [N], got ndim={labels.ndim}."
            )
        if int(labels.shape[0]) != num_samples:
            raise ValueError(
                "payload['labels'] length must match payload['embeddings']. "
                f"Got labels={int(labels.shape[0])}, embeddings={num_samples}."
            )

    if not isinstance(image_ids, list):
        raise TypeError(f"payload['image_ids'] must be list[str], got {type(image_ids)}.")
    if len(image_ids) != num_samples:
        raise ValueError(
            "payload['image_ids'] length must match payload['embeddings']. "
            f"Got image_ids={len(image_ids)}, embeddings={num_samples}."
        )
    if any(not isinstance(image_id, str) or not image_id for image_id in image_ids):
        raise ValueError("All payload['image_ids'] entries must be non-empty strings.")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Duplicate image_ids detected in shard payload.")

    if not isinstance(metadata, list):
        raise TypeError(f"payload['metadata'] must be list[dict], got {type(metadata)}.")
    if len(metadata) != num_samples:
        raise ValueError(
            "payload['metadata'] length must match payload['embeddings']. "
            f"Got metadata={len(metadata)}, embeddings={num_samples}."
        )
    if any(not isinstance(item, dict) for item in metadata):
        raise ValueError("All payload['metadata'] entries must be dictionaries.")


def validate_loaded_embedding_split(artifact: LoadedEmbeddingSplit) -> None:
    """Validates a fully loaded embedding split object."""
    if not isinstance(artifact.embeddings, torch.Tensor):
        raise TypeError(
            f"artifact.embeddings must be torch.Tensor, got {type(artifact.embeddings)}."
        )
    if artifact.embeddings.ndim != 2:
        raise ValueError(
            "artifact.embeddings must have shape [N, D], "
            f"got ndim={artifact.embeddings.ndim}."
        )

    num_samples = int(artifact.embeddings.shape[0])
    embedding_dim = int(artifact.embeddings.shape[1])

    if artifact.labels is not None:
        if not isinstance(artifact.labels, torch.Tensor):
            raise TypeError(
                f"artifact.labels must be torch.Tensor, got {type(artifact.labels)}."
            )
        if artifact.labels.ndim != 1:
            raise ValueError(
                f"artifact.labels must have shape [N], got ndim={artifact.labels.ndim}."
            )
        if int(artifact.labels.shape[0]) != num_samples:
            raise ValueError(
                "artifact.labels length must match artifact.embeddings. "
                f"Got labels={int(artifact.labels.shape[0])}, embeddings={num_samples}."
            )

    if len(artifact.image_ids) != num_samples:
        raise ValueError(
            "artifact.image_ids length must match artifact.embeddings. "
            f"Got image_ids={len(artifact.image_ids)}, embeddings={num_samples}."
        )
    if len(artifact.metadata) != num_samples:
        raise ValueError(
            "artifact.metadata length must match artifact.embeddings. "
            f"Got metadata={len(artifact.metadata)}, embeddings={num_samples}."
        )

    if len(set(artifact.image_ids)) != len(artifact.image_ids):
        raise ValueError("Duplicate image_ids detected in loaded embedding split.")

    if artifact.manifest.num_samples != num_samples:
        raise ValueError(
            "manifest.num_samples does not match loaded embeddings. "
            f"Got manifest={artifact.manifest.num_samples}, loaded={num_samples}."
        )
    if artifact.manifest.embedding_dim != embedding_dim:
        raise ValueError(
            "manifest.embedding_dim does not match loaded embeddings. "
            f"Got manifest={artifact.manifest.embedding_dim}, loaded={embedding_dim}."
        )
    if artifact.manifest.has_labels != (artifact.labels is not None):
        raise ValueError(
            "manifest.has_labels does not match loaded labels availability. "
            f"Got manifest={artifact.manifest.has_labels}, "
            f"loaded={artifact.labels is not None}."
        )
    if artifact.manifest.dtype != str(artifact.embeddings.dtype).replace("torch.", ""):
        raise ValueError(
            "manifest.dtype does not match loaded embedding dtype. "
            f"Got manifest={artifact.manifest.dtype}, "
            f"loaded={str(artifact.embeddings.dtype).replace('torch.', '')}."
        )


def default_manifest_path(split_dir: str | Path) -> Path:
    """Returns the default manifest path under a split directory."""
    return Path(split_dir) / EMBEDDING_MANIFEST_FILENAME