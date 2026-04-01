"""I/O helpers for artifact-driven linear probe evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from die_vfm.artifacts.embedding_artifact import (
    EmbeddingManifest,
    LoadedEmbeddingSplit,
)
from die_vfm.artifacts.embedding_loader import load_embedding_split


@dataclass(frozen=True)
class LinearProbeSplitData:
    """One labeled split prepared for linear probe evaluation.

    Attributes:
        split_name: Dataset split name, e.g. "train" or "val".
        embeddings: Feature matrix with shape [N, D].
        labels: Remapped class indices in [0, num_classes - 1], shape [N].
        original_labels: Original label ids from embedding artifacts, shape [N].
        image_ids: Image ids aligned with embeddings and labels.
        metadata: Per-sample metadata aligned with embeddings and labels.
        manifest: Original embedding manifest for this split.
    """

    split_name: str
    embeddings: torch.Tensor
    labels: torch.Tensor
    original_labels: torch.Tensor
    image_ids: list[str]
    metadata: list[dict[str, Any]]
    manifest: EmbeddingManifest

    def __post_init__(self) -> None:
        """Validates split-level invariants."""
        if self.embeddings.ndim != 2:
            raise ValueError(
                f"{self.split_name} embeddings must have shape [N, D], "
                f"got ndim={self.embeddings.ndim}."
            )

        if self.labels.ndim != 1:
            raise ValueError(
                f"{self.split_name} labels must have shape [N], "
                f"got ndim={self.labels.ndim}."
            )

        if self.original_labels.ndim != 1:
            raise ValueError(
                f"{self.split_name} original_labels must have shape [N], "
                f"got ndim={self.original_labels.ndim}."
            )

        num_samples = int(self.embeddings.shape[0])
        if int(self.labels.shape[0]) != num_samples:
            raise ValueError(
                f"{self.split_name} labels length must match embeddings. "
                f"Got labels={int(self.labels.shape[0])}, "
                f"embeddings={num_samples}."
            )

        if int(self.original_labels.shape[0]) != num_samples:
            raise ValueError(
                f"{self.split_name} original_labels length must match embeddings. "
                f"Got original_labels={int(self.original_labels.shape[0])}, "
                f"embeddings={num_samples}."
            )

        if len(self.image_ids) != num_samples:
            raise ValueError(
                f"{self.split_name} image_ids length must match embeddings. "
                f"Got image_ids={len(self.image_ids)}, embeddings={num_samples}."
            )

        if len(self.metadata) != num_samples:
            raise ValueError(
                f"{self.split_name} metadata length must match embeddings. "
                f"Got metadata={len(self.metadata)}, embeddings={num_samples}."
            )

        if self.labels.dtype != torch.long:
            raise TypeError(
                f"{self.split_name} labels must be torch.long, "
                f"got {self.labels.dtype}."
            )

        if self.original_labels.dtype != torch.long:
            raise TypeError(
                f"{self.split_name} original_labels must be torch.long, "
                f"got {self.original_labels.dtype}."
            )

    @property
    def num_samples(self) -> int:
        """Returns the number of samples in this split."""
        return int(self.embeddings.shape[0])

    @property
    def embedding_dim(self) -> int:
        """Returns the embedding feature dimension."""
        return int(self.embeddings.shape[1])


@dataclass(frozen=True)
class LinearProbeDataBundle:
    """Train/val data bundle for linear probe evaluation.

    Attributes:
        train: Prepared train split.
        val: Prepared validation split.
        class_ids: Sorted original class ids from the train split.
        class_to_index: Mapping from original class id to contiguous index.
    """

    train: LinearProbeSplitData
    val: LinearProbeSplitData
    class_ids: list[int]
    class_to_index: dict[int, int]

    def __post_init__(self) -> None:
        """Validates bundle-level invariants."""
        if not self.class_ids:
            raise ValueError("class_ids must be non-empty.")

        if len(self.class_ids) != len(self.class_to_index):
            raise ValueError(
                "class_ids and class_to_index size mismatch. "
                f"Got len(class_ids)={len(self.class_ids)} and "
                f"len(class_to_index)={len(self.class_to_index)}."
            )

        if self.train.embedding_dim != self.val.embedding_dim:
            raise ValueError(
                "Train and val embedding_dim must match. "
                f"Got train={self.train.embedding_dim}, "
                f"val={self.val.embedding_dim}."
            )

        if self.num_classes < 2:
            raise ValueError(
                "Linear probe requires at least 2 classes, "
                f"got {self.num_classes}."
            )

    @property
    def embedding_dim(self) -> int:
        """Returns the shared embedding dimension."""
        return self.train.embedding_dim

    @property
    def num_classes(self) -> int:
        """Returns the number of classes."""
        return len(self.class_ids)


def load_linear_probe_bundle(
    train_split_dir: str,
    val_split_dir: str,
    normalize_embeddings: bool = False,
    map_location: str | torch.device = "cpu",
) -> LinearProbeDataBundle:
    """Loads and prepares train/val embedding artifacts for linear probe.

    This function is artifact-driven only. It does not touch dataloaders
    or model objects.

    Args:
        train_split_dir: Path to the train embedding split directory.
        val_split_dir: Path to the val embedding split directory.
        normalize_embeddings: Whether to L2-normalize embeddings row-wise.
        map_location: torch.load map_location.

    Returns:
        A fully validated LinearProbeDataBundle.

    Raises:
        FileNotFoundError: If artifact files are missing.
        ValueError: If labels are missing, dimensions mismatch, class space is
            invalid, or validation labels contain unseen classes.
        TypeError: If loaded labels have unsupported types.
    """
    train_artifact = load_embedding_split(
        split_dir=train_split_dir,
        map_location=map_location,
    )
    val_artifact = load_embedding_split(
        split_dir=val_split_dir,
        map_location=map_location,
    )

    _validate_required_labels(split_name="train", artifact=train_artifact)
    _validate_required_labels(split_name="val", artifact=val_artifact)
    _validate_embedding_compatibility(
        train_artifact=train_artifact,
        val_artifact=val_artifact,
    )

    train_original_labels = _canonicalize_labels(
        split_name="train",
        labels=train_artifact.labels,
    )
    val_original_labels = _canonicalize_labels(
        split_name="val",
        labels=val_artifact.labels,
    )

    class_ids = _build_class_ids(train_original_labels)
    class_to_index = {class_id: index for index, class_id in enumerate(class_ids)}

    _validate_val_class_coverage(
        train_class_ids=class_ids,
        val_original_labels=val_original_labels,
    )

    train_embeddings = _prepare_embeddings(
        embeddings=train_artifact.embeddings,
        normalize_embeddings=normalize_embeddings,
    )
    val_embeddings = _prepare_embeddings(
        embeddings=val_artifact.embeddings,
        normalize_embeddings=normalize_embeddings,
    )

    train_labels = _remap_labels(
        labels=train_original_labels,
        class_to_index=class_to_index,
    )
    val_labels = _remap_labels(
        labels=val_original_labels,
        class_to_index=class_to_index,
    )

    train_split = LinearProbeSplitData(
        split_name="train",
        embeddings=train_embeddings,
        labels=train_labels,
        original_labels=train_original_labels,
        image_ids=list(train_artifact.image_ids),
        metadata=list(train_artifact.metadata),
        manifest=train_artifact.manifest,
    )
    val_split = LinearProbeSplitData(
        split_name="val",
        embeddings=val_embeddings,
        labels=val_labels,
        original_labels=val_original_labels,
        image_ids=list(val_artifact.image_ids),
        metadata=list(val_artifact.metadata),
        manifest=val_artifact.manifest,
    )

    return LinearProbeDataBundle(
        train=train_split,
        val=val_split,
        class_ids=class_ids,
        class_to_index=class_to_index,
    )


def _validate_required_labels(
    split_name: str,
    artifact: LoadedEmbeddingSplit,
) -> None:
    """Ensures a split contains labels."""
    if artifact.labels is None:
        raise ValueError(
            f"Linear probe requires labels, but {split_name} split has no labels."
        )

    if not artifact.manifest.has_labels:
        raise ValueError(
            f"Linear probe requires labels, but {split_name} manifest reports "
            "has_labels=False."
        )


def _validate_embedding_compatibility(
    train_artifact: LoadedEmbeddingSplit,
    val_artifact: LoadedEmbeddingSplit,
) -> None:
    """Ensures train/val embeddings are compatible."""
    train_dim = int(train_artifact.embeddings.shape[1])
    val_dim = int(val_artifact.embeddings.shape[1])
    if train_dim != val_dim:
        raise ValueError(
            "Train and val embedding_dim must match for linear probe. "
            f"Got train={train_dim}, val={val_dim}."
        )


def _canonicalize_labels(
    split_name: str,
    labels: torch.Tensor | None,
) -> torch.Tensor:
    """Validates labels and converts them to torch.long."""
    if labels is None:
        raise ValueError(f"{split_name} labels must not be None.")

    if labels.ndim != 1:
        raise ValueError(
            f"{split_name} labels must have shape [N], got ndim={labels.ndim}."
        )

    if labels.dtype.is_floating_point or labels.dtype.is_complex:
        raise TypeError(
            f"{split_name} labels must be integer-valued, got dtype={labels.dtype}."
        )

    return labels.to(dtype=torch.long)


def _build_class_ids(labels: torch.Tensor) -> list[int]:
    """Builds the sorted train class id list."""
    unique_labels = torch.unique(labels, sorted=True)
    class_ids = [int(value) for value in unique_labels.tolist()]

    if not class_ids:
        raise ValueError("Train split contains no classes.")

    return class_ids


def _validate_val_class_coverage(
    train_class_ids: list[int],
    val_original_labels: torch.Tensor,
) -> None:
    """Ensures val labels are a subset of train labels."""
    train_class_id_set = set(train_class_ids)
    val_class_ids = {int(value) for value in torch.unique(val_original_labels).tolist()}
    unseen_class_ids = sorted(val_class_ids - train_class_id_set)

    if unseen_class_ids:
        raise ValueError(
            "Validation split contains labels not seen in the train split. "
            f"Unseen class ids: {unseen_class_ids}."
        )


def _prepare_embeddings(
    embeddings: torch.Tensor,
    normalize_embeddings: bool,
) -> torch.Tensor:
    """Returns embeddings ready for evaluator consumption."""
    prepared = embeddings.detach()

    if normalize_embeddings:
        prepared = F.normalize(prepared, p=2, dim=1)

    return prepared


def _remap_labels(
    labels: torch.Tensor,
    class_to_index: dict[int, int],
) -> torch.Tensor:
    """Remaps original label ids to contiguous class indices."""
    remapped = torch.empty_like(labels, dtype=torch.long)
    for index, label_value in enumerate(labels.tolist()):
        remapped[index] = class_to_index[int(label_value)]
    return remapped