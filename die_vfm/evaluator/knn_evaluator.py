"""kNN evaluator for artifact-driven embedding classification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from die_vfm.evaluator.io import LinearProbeDataBundle
from die_vfm.evaluator.metrics import summarize_classification_metrics


@dataclass(frozen=True)
class KnnEvaluatorConfig:
    """Configuration for k-nearest neighbors evaluation.

    Attributes:
      k: Number of nearest neighbors used for voting.
      metric: Neighbor search metric. Supported values: "cosine", "l2".
      weighting: Voting scheme. Supported values: "uniform", "distance".
      temperature: Positive temperature used for distance-weighted voting.
      batch_size: Number of query embeddings processed per batch.
      device: Execution device, e.g. "cpu" or "cuda".
      topk: Top-k values included in the metric summary.
    """

    k: int = 20
    metric: str = "cosine"
    weighting: str = "uniform"
    temperature: float = 0.07
    batch_size: int = 1024
    device: str = "cpu"
    topk: tuple[int, ...] = (1, 5)

    def __post_init__(self) -> None:
        """Validates evaluator configuration."""
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}.")
        if self.metric not in ("cosine", "l2"):
            raise ValueError(
                f"metric must be 'cosine' or 'l2', got {self.metric!r}."
            )
        if self.weighting not in ("uniform", "distance"):
            raise ValueError(
                "weighting must be 'uniform' or 'distance', "
                f"got {self.weighting!r}."
            )
        if self.temperature <= 0.0:
            raise ValueError(
                f"temperature must be positive, got {self.temperature}."
            )
        if self.batch_size <= 0:
            raise ValueError(
                f"batch_size must be positive, got {self.batch_size}."
            )
        if not self.topk:
            raise ValueError("topk must be non-empty.")
        if any(k <= 0 for k in self.topk):
            raise ValueError(f"topk must contain only positive values, got {self.topk}.")


@dataclass(frozen=True)
class KnnEvaluationOutput:
    """Final kNN evaluation outputs for one query split.

    Attributes:
      predictions: Predicted class indices with shape [N].
      labels: Ground-truth class indices with shape [N].
      logits: Class vote scores with shape [N, C].
      neighbor_indices: Reference-set indices of nearest neighbors, shape [N, K].
      neighbor_labels: Neighbor class indices, shape [N, K].
      neighbor_scores: Neighbor similarity scores used for ranking, shape [N, K].
      image_ids: Query image ids aligned with predictions.
      metrics: Flat metric dictionary for the query split.
    """

    predictions: torch.Tensor
    labels: torch.Tensor
    logits: torch.Tensor
    neighbor_indices: torch.Tensor
    neighbor_labels: torch.Tensor
    neighbor_scores: torch.Tensor
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
        if self.neighbor_indices.ndim != 2:
            raise ValueError(
                "neighbor_indices must have shape [N, K], "
                f"got ndim={self.neighbor_indices.ndim}."
            )
        if self.neighbor_labels.ndim != 2:
            raise ValueError(
                "neighbor_labels must have shape [N, K], "
                f"got ndim={self.neighbor_labels.ndim}."
            )
        if self.neighbor_scores.ndim != 2:
            raise ValueError(
                "neighbor_scores must have shape [N, K], "
                f"got ndim={self.neighbor_scores.ndim}."
            )

        num_samples = int(self.labels.shape[0])
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
        if int(self.neighbor_indices.shape[0]) != num_samples:
            raise ValueError(
                "neighbor_indices and labels batch size must match. "
                f"Got neighbor_indices={int(self.neighbor_indices.shape[0])}, "
                f"labels={num_samples}."
            )
        if int(self.neighbor_labels.shape[0]) != num_samples:
            raise ValueError(
                "neighbor_labels and labels batch size must match. "
                f"Got neighbor_labels={int(self.neighbor_labels.shape[0])}, "
                f"labels={num_samples}."
            )
        if int(self.neighbor_scores.shape[0]) != num_samples:
            raise ValueError(
                "neighbor_scores and labels batch size must match. "
                f"Got neighbor_scores={int(self.neighbor_scores.shape[0])}, "
                f"labels={num_samples}."
            )
        if len(self.image_ids) != num_samples:
            raise ValueError(
                "image_ids length must match labels. "
                f"Got image_ids={len(self.image_ids)}, labels={num_samples}."
            )


def evaluate_knn(
    bundle: LinearProbeDataBundle,
    config: KnnEvaluatorConfig,
) -> KnnEvaluationOutput:
    """Runs artifact-driven kNN evaluation on val embeddings.

    The train split is used as the reference set.
    The val split is used as the query set.

    Args:
      bundle: Prepared labeled embedding bundle.
      config: kNN evaluator configuration.

    Returns:
      kNN predictions, neighbor metadata, and aggregated metrics.

    Raises:
      ValueError: If k exceeds the number of reference samples or if bundle
        dimensions are inconsistent with the config.
    """
    _validate_bundle(bundle=bundle, config=config)

    device = torch.device(config.device)
    reference_embeddings = bundle.train.embeddings.to(device=device)
    reference_labels = bundle.train.labels.to(device=device)
    query_embeddings = bundle.val.embeddings.to(device=device)
    query_labels = bundle.val.labels.to(device=device)

    all_logits = []
    all_neighbor_indices = []
    all_neighbor_labels = []
    all_neighbor_scores = []

    num_queries = bundle.val.num_samples
    for start in range(0, num_queries, config.batch_size):
        end = min(start + config.batch_size, num_queries)
        query_batch = query_embeddings[start:end]

        similarity = _compute_similarity(
            query_embeddings=query_batch,
            reference_embeddings=reference_embeddings,
            metric=config.metric,
        )
        neighbor_scores, neighbor_indices = torch.topk(
            similarity,
            k=config.k,
            dim=1,
            largest=True,
            sorted=True,
        )
        neighbor_labels = reference_labels[neighbor_indices]
        logits = _build_vote_logits(
            neighbor_labels=neighbor_labels,
            neighbor_scores=neighbor_scores,
            num_classes=bundle.num_classes,
            weighting=config.weighting,
            temperature=config.temperature,
        )

        all_logits.append(logits)
        all_neighbor_indices.append(neighbor_indices)
        all_neighbor_labels.append(neighbor_labels)
        all_neighbor_scores.append(neighbor_scores)

    logits = torch.cat(all_logits, dim=0)
    neighbor_indices = torch.cat(all_neighbor_indices, dim=0)
    neighbor_labels = torch.cat(all_neighbor_labels, dim=0)
    neighbor_scores = torch.cat(all_neighbor_scores, dim=0)
    predictions = torch.argmax(logits, dim=1)

    metrics = summarize_classification_metrics(
        logits=logits,
        labels=query_labels,
        topk=_resolve_topk(config.topk, bundle.num_classes),
    )

    return KnnEvaluationOutput(
        predictions=predictions.cpu(),
        labels=query_labels.cpu(),
        logits=logits.cpu(),
        neighbor_indices=neighbor_indices.cpu(),
        neighbor_labels=neighbor_labels.cpu(),
        neighbor_scores=neighbor_scores.cpu(),
        image_ids=list(bundle.val.image_ids),
        metrics=metrics,
    )


def _validate_bundle(
    bundle: LinearProbeDataBundle,
    config: KnnEvaluatorConfig,
) -> None:
    """Validates bundle-level constraints for kNN evaluation."""
    if bundle.train.num_samples == 0:
        raise ValueError("train split must be non-empty.")
    if bundle.val.num_samples == 0:
        raise ValueError("val split must be non-empty.")
    if config.k > bundle.train.num_samples:
        raise ValueError(
            "k must be <= number of train reference samples. "
            f"Got k={config.k}, train_num_samples={bundle.train.num_samples}."
        )
    if max(config.topk) > bundle.num_classes:
        raise ValueError(
            "Requested topk exceeds the number of classes. "
            f"Got topk={config.topk}, num_classes={bundle.num_classes}."
        )


def _compute_similarity(
    query_embeddings: torch.Tensor,
    reference_embeddings: torch.Tensor,
    metric: str,
) -> torch.Tensor:
    """Computes query-to-reference similarity matrix.

    Args:
      query_embeddings: Query matrix with shape [B, D].
      reference_embeddings: Reference matrix with shape [N, D].
      metric: Supported values: "cosine", "l2".

    Returns:
      Similarity matrix with shape [B, N]. Larger is better.
    """
    _validate_embeddings(
        query_embeddings=query_embeddings,
        reference_embeddings=reference_embeddings,
    )

    if metric == "cosine":
        normalized_queries = F.normalize(query_embeddings, p=2, dim=1)
        normalized_references = F.normalize(reference_embeddings, p=2, dim=1)
        return normalized_queries @ normalized_references.transpose(0, 1)

    if metric == "l2":
        distances = torch.cdist(query_embeddings, reference_embeddings, p=2)
        return -distances

    raise ValueError(f"Unsupported metric: {metric!r}.")


def _build_vote_logits(
    neighbor_labels: torch.Tensor,
    neighbor_scores: torch.Tensor,
    num_classes: int,
    weighting: str,
    temperature: float,
) -> torch.Tensor:
    """Aggregates neighbor votes into class-score logits.

    Args:
      neighbor_labels: Neighbor class indices with shape [B, K].
      neighbor_scores: Ranking scores with shape [B, K].
      num_classes: Number of classes.
      weighting: Supported values: "uniform", "distance".
      temperature: Positive temperature for weighted voting.

    Returns:
      Vote logits with shape [B, C].
    """
    if neighbor_labels.ndim != 2:
        raise ValueError(
            "neighbor_labels must have shape [B, K], "
            f"got ndim={neighbor_labels.ndim}."
        )
    if neighbor_scores.ndim != 2:
        raise ValueError(
            "neighbor_scores must have shape [B, K], "
            f"got ndim={neighbor_scores.ndim}."
        )
    if neighbor_labels.shape != neighbor_scores.shape:
        raise ValueError(
            "neighbor_labels and neighbor_scores must have the same shape. "
            f"Got labels={tuple(neighbor_labels.shape)}, "
            f"scores={tuple(neighbor_scores.shape)}."
        )
    if num_classes <= 1:
        raise ValueError(
            f"num_classes must be greater than 1, got {num_classes}."
        )

    if weighting == "uniform":
        weights = torch.ones_like(neighbor_scores)
    elif weighting == "distance":
        weights = torch.softmax(neighbor_scores / temperature, dim=1)
    else:
        raise ValueError(f"Unsupported weighting: {weighting!r}.")

    batch_size = int(neighbor_labels.shape[0])
    logits = torch.zeros(
        (batch_size, num_classes),
        dtype=neighbor_scores.dtype,
        device=neighbor_scores.device,
    )
    logits.scatter_add_(dim=1, index=neighbor_labels, src=weights)
    return logits


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
    reference_embeddings: torch.Tensor,
) -> None:
    """Validates kNN embedding tensors."""
    if query_embeddings.ndim != 2:
        raise ValueError(
            "query_embeddings must have shape [B, D], "
            f"got ndim={query_embeddings.ndim}."
        )
    if reference_embeddings.ndim != 2:
        raise ValueError(
            "reference_embeddings must have shape [N, D], "
            f"got ndim={reference_embeddings.ndim}."
        )
    query_dim = int(query_embeddings.shape[1])
    reference_dim = int(reference_embeddings.shape[1])
    if query_dim != reference_dim:
        raise ValueError(
            "Embedding dimension mismatch between query and reference. "
            f"Got query_dim={query_dim}, reference_dim={reference_dim}."
        )