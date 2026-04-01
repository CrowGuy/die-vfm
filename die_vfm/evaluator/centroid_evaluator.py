"""Centroid evaluator for artifact-driven embedding classification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from die_vfm.evaluator.io import LinearProbeDataBundle
from die_vfm.evaluator.metrics import summarize_classification_metrics


@dataclass(frozen=True)
class CentroidEvaluatorConfig:
    """Configuration for centroid-based evaluation.

    Attributes:
      metric: Prototype matching metric. Supported values: "cosine", "l2".
      batch_size: Number of query embeddings processed per batch.
      device: Execution device, e.g. "cpu" or "cuda".
      topk: Top-k values included in the metric summary.
    """

    metric: str = "cosine"
    batch_size: int = 1024
    device: str = "cpu"
    topk: tuple[int, ...] = (1, 5)

    def __post_init__(self) -> None:
        """Validates evaluator configuration."""
        if self.metric not in ("cosine", "l2"):
            raise ValueError(
                f"metric must be 'cosine' or 'l2', got {self.metric!r}."
            )
        if self.batch_size <= 0:
            raise ValueError(
                f"batch_size must be positive, got {self.batch_size}."
            )
        if not self.topk:
            raise ValueError("topk must be non-empty.")
        if any(k <= 0 for k in self.topk):
            raise ValueError(
                f"topk must contain only positive values, got {self.topk}."
            )


@dataclass(frozen=True)
class CentroidEvaluationOutput:
    """Final centroid evaluation outputs for one query split.

    Attributes:
      predictions: Predicted class indices with shape [N].
      labels: Ground-truth class indices with shape [N].
      logits: Class similarity scores with shape [N, C].
      prototype_labels: Prototype class indices with shape [C].
      prototypes: Class prototypes with shape [C, D].
      image_ids: Query image ids aligned with predictions.
      metrics: Flat metric dictionary for the query split.
    """

    predictions: torch.Tensor
    labels: torch.Tensor
    logits: torch.Tensor
    prototype_labels: torch.Tensor
    prototypes: torch.Tensor
    image_ids: list[str]
    metrics: dict[str, float]

    def __post_init__(self) -> None:
        """Validates result tensor shapes."""
        if self.predictions.ndim != 1:
            raise ValueError(
                "predictions must have shape [N], "
                f"got ndim={self.predictions.ndim}."
            )
        if self.labels.ndim != 1:
            raise ValueError(
                f"labels must have shape [N], got ndim={self.labels.ndim}."
            )
        if self.logits.ndim != 2:
            raise ValueError(
                f"logits must have shape [N, C], got ndim={self.logits.ndim}."
            )
        if self.prototype_labels.ndim != 1:
            raise ValueError(
                "prototype_labels must have shape [C], "
                f"got ndim={self.prototype_labels.ndim}."
            )
        if self.prototypes.ndim != 2:
            raise ValueError(
                f"prototypes must have shape [C, D], got ndim={self.prototypes.ndim}."
            )

        num_samples = int(self.labels.shape[0])
        num_classes = int(self.prototype_labels.shape[0])

        if int(self.predictions.shape[0]) != num_samples:
            raise ValueError(
                "predictions and labels batch size must match. "
                f"Got predictions={int(self.predictions.shape[0])}, "
                f"labels={num_samples}."
            )
        if int(self.logits.shape[0]) != num_samples:
            raise ValueError(
                "logits and labels batch size must match. "
                f"Got logits={int(self.logits.shape[0])}, labels={num_samples}."
            )
        if int(self.logits.shape[1]) != num_classes:
            raise ValueError(
                "logits class dimension must match prototype_labels. "
                f"Got logits.shape[1]={int(self.logits.shape[1])}, "
                f"prototype_labels={num_classes}."
            )
        if int(self.prototypes.shape[0]) != num_classes:
            raise ValueError(
                "prototypes class dimension must match prototype_labels. "
                f"Got prototypes.shape[0]={int(self.prototypes.shape[0])}, "
                f"prototype_labels={num_classes}."
            )
        if len(self.image_ids) != num_samples:
            raise ValueError(
                "image_ids length must match labels. "
                f"Got image_ids={len(self.image_ids)}, labels={num_samples}."
            )


def evaluate_centroid(
    bundle: LinearProbeDataBundle,
    config: CentroidEvaluatorConfig,
) -> CentroidEvaluationOutput:
    """Runs artifact-driven centroid evaluation on val embeddings.

    The train split is used to build one prototype per class.
    The val split is used as the query set.

    Args:
      bundle: Prepared labeled embedding bundle.
      config: Centroid evaluator configuration.

    Returns:
      Centroid predictions, prototypes, and aggregated metrics.

    Raises:
      ValueError: If bundle dimensions or class coverage are inconsistent.
    """
    _validate_bundle(bundle=bundle, config=config)

    device = torch.device(config.device)
    train_embeddings = bundle.train.embeddings.to(device=device)
    train_labels = bundle.train.labels.to(device=device)
    query_embeddings = bundle.val.embeddings.to(device=device)
    query_labels = bundle.val.labels.to(device=device)

    prototypes = _build_class_prototypes(
        embeddings=train_embeddings,
        labels=train_labels,
        num_classes=bundle.num_classes,
    )
    prototype_labels = torch.arange(
        bundle.num_classes, dtype=torch.long, device=device
    )

    all_logits = []
    num_queries = bundle.val.num_samples
    for start in range(0, num_queries, config.batch_size):
        end = min(start + config.batch_size, num_queries)
        query_batch = query_embeddings[start:end]
        logits = _compute_similarity(
            query_embeddings=query_batch,
            prototype_embeddings=prototypes,
            metric=config.metric,
        )
        all_logits.append(logits)

    logits = torch.cat(all_logits, dim=0)
    predictions = torch.argmax(logits, dim=1)
    metrics = summarize_classification_metrics(
        logits=logits,
        labels=query_labels,
        topk=_resolve_topk(config.topk, bundle.num_classes),
    )

    return CentroidEvaluationOutput(
        predictions=predictions.cpu(),
        labels=query_labels.cpu(),
        logits=logits.cpu(),
        prototype_labels=prototype_labels.cpu(),
        prototypes=prototypes.cpu(),
        image_ids=list(bundle.val.image_ids),
        metrics=metrics,
    )


def _validate_bundle(
    bundle: LinearProbeDataBundle,
    config: CentroidEvaluatorConfig,
) -> None:
    """Validates bundle-level constraints for centroid evaluation."""
    del config  # Reserved for future evaluator-specific validation.

    if bundle.train.num_samples == 0:
        raise ValueError("train split must be non-empty.")
    if bundle.val.num_samples == 0:
        raise ValueError("val split must be non-empty.")
    if bundle.num_classes <= 1:
        raise ValueError(
            f"num_classes must be greater than 1, got {bundle.num_classes}."
        )
    if max(bundle.train.labels.tolist()) >= bundle.num_classes:
        raise ValueError(
            "train labels must be in [0, num_classes - 1]. "
            f"Got max_train_label={int(bundle.train.labels.max().item())}, "
            f"num_classes={bundle.num_classes}."
        )
    if max(bundle.val.labels.tolist()) >= bundle.num_classes:
        raise ValueError(
            "val labels must be in [0, num_classes - 1]. "
            f"Got max_val_label={int(bundle.val.labels.max().item())}, "
            f"num_classes={bundle.num_classes}."
        )


def _build_class_prototypes(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Builds one mean prototype per class.

    Args:
      embeddings: Train embeddings with shape [N, D].
      labels: Remapped train class indices with shape [N].
      num_classes: Total number of classes.

    Returns:
      Prototype matrix with shape [C, D].

    Raises:
      ValueError: If tensor shapes are invalid or a class is missing.
    """
    if embeddings.ndim != 2:
        raise ValueError(
            "embeddings must have shape [N, D], "
            f"got ndim={embeddings.ndim}."
        )
    if labels.ndim != 1:
        raise ValueError(
            f"labels must have shape [N], got ndim={labels.ndim}."
        )
    if int(embeddings.shape[0]) != int(labels.shape[0]):
        raise ValueError(
            "embeddings and labels batch size must match. "
            f"Got embeddings={int(embeddings.shape[0])}, "
            f"labels={int(labels.shape[0])}."
        )
    if labels.dtype != torch.long:
        raise TypeError(f"labels must be torch.long, got {labels.dtype}.")
    if num_classes <= 1:
        raise ValueError(
            f"num_classes must be greater than 1, got {num_classes}."
        )

    embedding_dim = int(embeddings.shape[1])
    prototypes = torch.zeros(
        (num_classes, embedding_dim),
        dtype=embeddings.dtype,
        device=embeddings.device,
    )
    counts = torch.zeros(
        num_classes,
        dtype=embeddings.dtype,
        device=embeddings.device,
    )

    prototypes.scatter_add_(
        dim=0,
        index=labels.unsqueeze(1).expand(-1, embedding_dim),
        src=embeddings,
    )
    counts.scatter_add_(
        dim=0,
        index=labels,
        src=torch.ones_like(labels, dtype=embeddings.dtype),
    )

    missing_classes = torch.nonzero(counts == 0, as_tuple=False).flatten()
    if int(missing_classes.numel()) > 0:
        raise ValueError(
            "train split must contain at least one sample for every class. "
            f"Missing classes: {missing_classes.tolist()}."
        )

    prototypes = prototypes / counts.unsqueeze(1)
    return prototypes


def _compute_similarity(
    query_embeddings: torch.Tensor,
    prototype_embeddings: torch.Tensor,
    metric: str,
) -> torch.Tensor:
    """Computes query-to-prototype similarity matrix.

    Args:
      query_embeddings: Query matrix with shape [B, D].
      prototype_embeddings: Prototype matrix with shape [C, D].
      metric: Supported values: "cosine", "l2".

    Returns:
      Similarity matrix with shape [B, C]. Larger is better.
    """
    _validate_embeddings(
        query_embeddings=query_embeddings,
        prototype_embeddings=prototype_embeddings,
    )

    if metric == "cosine":
        normalized_queries = F.normalize(query_embeddings, p=2, dim=1)
        normalized_prototypes = F.normalize(prototype_embeddings, p=2, dim=1)
        return normalized_queries @ normalized_prototypes.transpose(0, 1)
    if metric == "l2":
        distances = torch.cdist(query_embeddings, prototype_embeddings, p=2)
        return -distances
    raise ValueError(f"Unsupported metric: {metric!r}.")


def _resolve_topk(
    requested_topk: tuple[int, ...],
    num_classes: int,
) -> tuple[int, ...]:
    """Returns sorted, unique top-k values validated against class count."""
    unique_topk = sorted(set(requested_topk))
    if unique_topk[-1] > num_classes:
        raise ValueError(
            "topk must be <= num_classes. "
            f"Got topk={tuple(unique_topk)}, num_classes={num_classes}."
        )
    return tuple(unique_topk)


def _validate_embeddings(
    query_embeddings: torch.Tensor,
    prototype_embeddings: torch.Tensor,
) -> None:
    """Validates centroid embedding tensors."""
    if query_embeddings.ndim != 2:
        raise ValueError(
            "query_embeddings must have shape [B, D], "
            f"got ndim={query_embeddings.ndim}."
        )
    if prototype_embeddings.ndim != 2:
        raise ValueError(
            "prototype_embeddings must have shape [C, D], "
            f"got ndim={prototype_embeddings.ndim}."
        )

    query_dim = int(query_embeddings.shape[1])
    prototype_dim = int(prototype_embeddings.shape[1])
    if query_dim != prototype_dim:
        raise ValueError(
            "Embedding dimension mismatch between query and prototypes. "
            f"Got query_dim={query_dim}, prototype_dim={prototype_dim}."
        )