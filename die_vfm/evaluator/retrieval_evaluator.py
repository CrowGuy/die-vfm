"""Retrieval evaluator for artifact-driven embedding ranking."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from die_vfm.evaluator.io import LinearProbeDataBundle


@dataclass(frozen=True)
class RetrievalEvaluatorConfig:
    """Configuration for retrieval evaluation.

    Attributes:
      metric: Retrieval similarity metric. Supported values: "cosine", "l2".
      batch_size: Number of query embeddings processed per batch.
      device: Execution device, e.g. "cpu" or "cuda".
      topk: Top-k values used for Recall@K and mAP@K reporting.
      save_predictions_topk: Number of retrieved neighbors to keep in the
        output object for downstream writing/debugging.
      exclude_same_image_id: Whether to exclude gallery entries whose image_id
        exactly matches the query image_id.
    """

    metric: str = "cosine"
    batch_size: int = 1024
    device: str = "cpu"
    topk: tuple[int, ...] = (1, 5)
    save_predictions_topk: int = 10
    exclude_same_image_id: bool = False

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
        if self.save_predictions_topk <= 0:
            raise ValueError(
                "save_predictions_topk must be positive, "
                f"got {self.save_predictions_topk}."
            )


@dataclass(frozen=True)
class RetrievalEvaluationOutput:
    """Final retrieval evaluation outputs for one query split.

    Attributes:
      query_labels: Ground-truth query class indices with shape [Q].
      topk_indices: Retrieved gallery indices with shape [Q, Kp].
      topk_labels: Retrieved gallery class indices with shape [Q, Kp].
      topk_scores: Retrieved gallery similarity scores with shape [Q, Kp].
      topk_matches: Whether each retrieved gallery item is relevant, shape
        [Q, Kp].
      image_ids: Query image ids aligned with query_labels.
      topk_image_ids: Retrieved gallery image ids aligned with topk tensors.
      metrics: Flat metric dictionary for the query split.
    """

    query_labels: torch.Tensor
    topk_indices: torch.Tensor
    topk_labels: torch.Tensor
    topk_scores: torch.Tensor
    topk_matches: torch.Tensor
    image_ids: list[str]
    topk_image_ids: list[list[str]]
    metrics: dict[str, float]

    def __post_init__(self) -> None:
        """Validates result tensor shapes."""
        if self.query_labels.ndim != 1:
            raise ValueError(
                "query_labels must have shape [Q], "
                f"got ndim={self.query_labels.ndim}."
            )
        if self.topk_indices.ndim != 2:
            raise ValueError(
                "topk_indices must have shape [Q, K], "
                f"got ndim={self.topk_indices.ndim}."
            )
        if self.topk_labels.ndim != 2:
            raise ValueError(
                "topk_labels must have shape [Q, K], "
                f"got ndim={self.topk_labels.ndim}."
            )
        if self.topk_scores.ndim != 2:
            raise ValueError(
                "topk_scores must have shape [Q, K], "
                f"got ndim={self.topk_scores.ndim}."
            )
        if self.topk_matches.ndim != 2:
            raise ValueError(
                "topk_matches must have shape [Q, K], "
                f"got ndim={self.topk_matches.ndim}."
            )

        num_queries = int(self.query_labels.shape[0])
        if int(self.topk_indices.shape[0]) != num_queries:
            raise ValueError(
                "topk_indices and query_labels batch size must match. "
                f"Got topk_indices={int(self.topk_indices.shape[0])}, "
                f"query_labels={num_queries}."
            )
        if int(self.topk_labels.shape[0]) != num_queries:
            raise ValueError(
                "topk_labels and query_labels batch size must match. "
                f"Got topk_labels={int(self.topk_labels.shape[0])}, "
                f"query_labels={num_queries}."
            )
        if int(self.topk_scores.shape[0]) != num_queries:
            raise ValueError(
                "topk_scores and query_labels batch size must match. "
                f"Got topk_scores={int(self.topk_scores.shape[0])}, "
                f"query_labels={num_queries}."
            )
        if int(self.topk_matches.shape[0]) != num_queries:
            raise ValueError(
                "topk_matches and query_labels batch size must match. "
                f"Got topk_matches={int(self.topk_matches.shape[0])}, "
                f"query_labels={num_queries}."
            )

        num_retrieved = int(self.topk_indices.shape[1])
        if int(self.topk_labels.shape[1]) != num_retrieved:
            raise ValueError(
                "topk_labels retrieval dimension must match topk_indices. "
                f"Got topk_labels.shape[1]={int(self.topk_labels.shape[1])}, "
                f"topk_indices.shape[1]={num_retrieved}."
            )
        if int(self.topk_scores.shape[1]) != num_retrieved:
            raise ValueError(
                "topk_scores retrieval dimension must match topk_indices. "
                f"Got topk_scores.shape[1]={int(self.topk_scores.shape[1])}, "
                f"topk_indices.shape[1]={num_retrieved}."
            )
        if int(self.topk_matches.shape[1]) != num_retrieved:
            raise ValueError(
                "topk_matches retrieval dimension must match topk_indices. "
                f"Got topk_matches.shape[1]={int(self.topk_matches.shape[1])}, "
                f"topk_indices.shape[1]={num_retrieved}."
            )
        if len(self.image_ids) != num_queries:
            raise ValueError(
                "image_ids length must match query_labels. "
                f"Got image_ids={len(self.image_ids)}, "
                f"query_labels={num_queries}."
            )
        if len(self.topk_image_ids) != num_queries:
            raise ValueError(
                "topk_image_ids outer length must match query_labels. "
                f"Got topk_image_ids={len(self.topk_image_ids)}, "
                f"query_labels={num_queries}."
            )
        for row_index, row in enumerate(self.topk_image_ids):
            if len(row) != num_retrieved:
                raise ValueError(
                    "Each topk_image_ids row must match retrieval dimension. "
                    f"Got len(topk_image_ids[{row_index}])={len(row)}, "
                    f"topk_indices.shape[1]={num_retrieved}."
                )


def evaluate_retrieval(
    bundle: LinearProbeDataBundle,
    config: RetrievalEvaluatorConfig,
) -> RetrievalEvaluationOutput:
    """Runs artifact-driven retrieval evaluation on val embeddings.

    The train split is used as the gallery/reference set.
    The val split is used as the query set.

    Args:
      bundle: Prepared labeled embedding bundle.
      config: Retrieval evaluator configuration.

    Returns:
      Retrieval neighbors and aggregated metrics.

    Raises:
      ValueError: If bundle dimensions or retrieval settings are invalid.
    """
    _validate_bundle(bundle=bundle, config=config)

    device = torch.device(config.device)
    gallery_embeddings = bundle.train.embeddings.to(device=device)
    gallery_labels = bundle.train.labels.to(device=device)
    query_embeddings = bundle.val.embeddings.to(device=device)
    query_labels = bundle.val.labels.to(device=device)

    resolved_topk = _resolve_topk(
        requested_topk=config.topk,
        num_gallery=bundle.train.num_samples,
    )
    prediction_topk = min(config.save_predictions_topk, bundle.train.num_samples)
    compute_topk = max(max(resolved_topk), prediction_topk)

    gallery_label_counts = torch.bincount(
        gallery_labels, minlength=bundle.num_classes
    )
    gallery_image_id_to_index = {
        image_id: index for index, image_id in enumerate(bundle.train.image_ids)
    }

    all_topk_indices = []
    all_topk_labels = []
    all_topk_scores = []
    all_topk_matches = []

    recall_hits_by_k = {k: [] for k in resolved_topk}
    ap_numerators_by_k = {k: [] for k in resolved_topk}
    ap_denominators_by_k = {k: [] for k in resolved_topk}
    valid_query_masks = []

    num_queries = bundle.val.num_samples
    for start in range(0, num_queries, config.batch_size):
        end = min(start + config.batch_size, num_queries)
        query_batch = query_embeddings[start:end]
        query_labels_batch = query_labels[start:end]
        query_image_ids_batch = bundle.val.image_ids[start:end]

        similarity = _compute_similarity(
            query_embeddings=query_batch,
            gallery_embeddings=gallery_embeddings,
            metric=config.metric,
        )

        excluded_gallery_indices = _resolve_excluded_gallery_indices(
            query_image_ids=query_image_ids_batch,
            gallery_image_id_to_index=gallery_image_id_to_index,
        )
        if config.exclude_same_image_id:
            similarity = _mask_excluded_gallery_entries(
                similarity=similarity,
                excluded_gallery_indices=excluded_gallery_indices,
            )

        topk_scores, topk_indices = torch.topk(
            similarity,
            k=compute_topk,
            dim=1,
            largest=True,
            sorted=True,
        )
        topk_labels = gallery_labels[topk_indices]
        topk_matches = topk_labels.eq(query_labels_batch.unsqueeze(1))

        positive_counts = gallery_label_counts[query_labels_batch].to(torch.long)
        if config.exclude_same_image_id:
            excluded_positive_counts = _count_excluded_positives(
                query_labels=query_labels_batch,
                gallery_labels=gallery_labels,
                excluded_gallery_indices=excluded_gallery_indices,
            )
            positive_counts = positive_counts - excluded_positive_counts

        valid_for_map = positive_counts > 0
        valid_query_masks.append(valid_for_map)

        precision_at_rank = _compute_precision_at_rank(matches=topk_matches)
        cumulative_ap_numerator = torch.cumsum(
            precision_at_rank * topk_matches.to(dtype=topk_scores.dtype),
            dim=1,
        )

        for k in resolved_topk:
            match_prefix = topk_matches[:, :k]
            recall_hits = torch.any(match_prefix, dim=1).to(dtype=torch.float32)
            recall_hits_by_k[k].append(recall_hits)

            ap_numerator = cumulative_ap_numerator[:, k - 1]
            ap_denominator = torch.minimum(
                positive_counts,
                torch.full_like(positive_counts, fill_value=k),
            )
            ap_numerators_by_k[k].append(ap_numerator)
            ap_denominators_by_k[k].append(ap_denominator)

        all_topk_indices.append(topk_indices[:, :prediction_topk])
        all_topk_labels.append(topk_labels[:, :prediction_topk])
        all_topk_scores.append(topk_scores[:, :prediction_topk])
        all_topk_matches.append(topk_matches[:, :prediction_topk])

    prediction_indices = torch.cat(all_topk_indices, dim=0)
    prediction_labels = torch.cat(all_topk_labels, dim=0)
    prediction_scores = torch.cat(all_topk_scores, dim=0)
    prediction_matches = torch.cat(all_topk_matches, dim=0)

    metrics = _build_retrieval_metrics(
        recall_hits_by_k=recall_hits_by_k,
        ap_numerators_by_k=ap_numerators_by_k,
        ap_denominators_by_k=ap_denominators_by_k,
        valid_query_masks=valid_query_masks,
        metric_name=config.metric,
        num_queries=bundle.val.num_samples,
        num_gallery=bundle.train.num_samples,
        topk=resolved_topk,
    )

    return RetrievalEvaluationOutput(
        query_labels=query_labels.cpu(),
        topk_indices=prediction_indices.cpu(),
        topk_labels=prediction_labels.cpu(),
        topk_scores=prediction_scores.cpu(),
        topk_matches=prediction_matches.cpu(),
        image_ids=list(bundle.val.image_ids),
        topk_image_ids=_gather_topk_image_ids(
            topk_indices=prediction_indices.cpu(),
            gallery_image_ids=bundle.train.image_ids,
        ),
        metrics=metrics,
    )


def _validate_bundle(
    bundle: LinearProbeDataBundle,
    config: RetrievalEvaluatorConfig,
) -> None:
    """Validates bundle-level constraints for retrieval evaluation."""
    if bundle.train.num_samples == 0:
        raise ValueError("train split must be non-empty.")
    if bundle.val.num_samples == 0:
        raise ValueError("val split must be non-empty.")
    if bundle.num_classes <= 1:
        raise ValueError(
            f"num_classes must be greater than 1, got {bundle.num_classes}."
        )

    resolved_topk = _resolve_topk(
        requested_topk=config.topk,
        num_gallery=bundle.train.num_samples,
    )
    del resolved_topk

    if config.save_predictions_topk > bundle.train.num_samples:
        raise ValueError(
            "save_predictions_topk must be <= number of train gallery samples. "
            f"Got save_predictions_topk={config.save_predictions_topk}, "
            f"train_num_samples={bundle.train.num_samples}."
        )


def _compute_similarity(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    metric: str,
) -> torch.Tensor:
    """Computes query-to-gallery similarity matrix.

    Args:
      query_embeddings: Query matrix with shape [B, D].
      gallery_embeddings: Gallery matrix with shape [N, D].
      metric: Supported values: "cosine", "l2".

    Returns:
      Similarity matrix with shape [B, N]. Larger is better.
    """
    _validate_embeddings(
        query_embeddings=query_embeddings,
        gallery_embeddings=gallery_embeddings,
    )

    if metric == "cosine":
        normalized_queries = F.normalize(query_embeddings, p=2, dim=1)
        normalized_gallery = F.normalize(gallery_embeddings, p=2, dim=1)
        return normalized_queries @ normalized_gallery.transpose(0, 1)
    if metric == "l2":
        distances = torch.cdist(query_embeddings, gallery_embeddings, p=2)
        return -distances

    raise ValueError(f"Unsupported metric: {metric!r}.")


def _resolve_topk(
    requested_topk: tuple[int, ...],
    num_gallery: int,
) -> tuple[int, ...]:
    """Returns sorted, unique top-k values validated against gallery size."""
    unique_topk = sorted(set(requested_topk))
    if unique_topk[-1] > num_gallery:
        raise ValueError(
            "topk must be <= number of gallery samples. "
            f"Got topk={tuple(unique_topk)}, num_gallery={num_gallery}."
        )
    return tuple(unique_topk)


def _validate_embeddings(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
) -> None:
    """Validates retrieval embedding tensors."""
    if query_embeddings.ndim != 2:
        raise ValueError(
            "query_embeddings must have shape [B, D], "
            f"got ndim={query_embeddings.ndim}."
        )
    if gallery_embeddings.ndim != 2:
        raise ValueError(
            "gallery_embeddings must have shape [N, D], "
            f"got ndim={gallery_embeddings.ndim}."
        )
    query_dim = int(query_embeddings.shape[1])
    gallery_dim = int(gallery_embeddings.shape[1])
    if query_dim != gallery_dim:
        raise ValueError(
            "Embedding dimension mismatch between query and gallery. "
            f"Got query_dim={query_dim}, gallery_dim={gallery_dim}."
        )


def _resolve_excluded_gallery_indices(
    query_image_ids: list[str],
    gallery_image_id_to_index: dict[str, int],
) -> list[int | None]:
    """Resolves excluded gallery index for each query image id."""
    return [
        gallery_image_id_to_index.get(image_id) for image_id in query_image_ids
    ]


def _mask_excluded_gallery_entries(
    similarity: torch.Tensor,
    excluded_gallery_indices: list[int | None],
) -> torch.Tensor:
    """Masks excluded gallery entries so they cannot be retrieved."""
    if similarity.ndim != 2:
        raise ValueError(
            f"similarity must have shape [B, N], got ndim={similarity.ndim}."
        )

    masked = similarity.clone()
    min_value = torch.finfo(masked.dtype).min
    for row_index, gallery_index in enumerate(excluded_gallery_indices):
        if gallery_index is None:
            continue
        masked[row_index, gallery_index] = min_value
    return masked


def _count_excluded_positives(
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
    excluded_gallery_indices: list[int | None],
) -> torch.Tensor:
    """Counts how many positive gallery entries were excluded per query."""
    excluded_positive_counts = torch.zeros_like(query_labels, dtype=torch.long)
    for row_index, gallery_index in enumerate(excluded_gallery_indices):
        if gallery_index is None:
            continue
        if gallery_labels[gallery_index] == query_labels[row_index]:
            excluded_positive_counts[row_index] = 1
    return excluded_positive_counts


def _compute_precision_at_rank(matches: torch.Tensor) -> torch.Tensor:
    """Computes precision@rank for each query/rank position."""
    if matches.ndim != 2:
        raise ValueError(
            f"matches must have shape [B, K], got ndim={matches.ndim}."
        )

    cumulative_hits = torch.cumsum(matches.to(dtype=torch.float32), dim=1)
    rank_positions = torch.arange(
        1,
        int(matches.shape[1]) + 1,
        device=matches.device,
        dtype=torch.float32,
    ).unsqueeze(0)
    return cumulative_hits / rank_positions


def _build_retrieval_metrics(
    recall_hits_by_k: dict[int, list[torch.Tensor]],
    ap_numerators_by_k: dict[int, list[torch.Tensor]],
    ap_denominators_by_k: dict[int, list[torch.Tensor]],
    valid_query_masks: list[torch.Tensor],
    metric_name: str,
    num_queries: int,
    num_gallery: int,
    topk: tuple[int, ...],
) -> dict[str, float]:
    """Builds flat retrieval metric outputs."""
    valid_mask = torch.cat(valid_query_masks, dim=0)
    num_valid_queries = int(valid_mask.sum().item())
    num_invalid_queries = int((~valid_mask).sum().item())

    metrics: dict[str, float] = {
        "num_queries": float(num_queries),
        "num_gallery": float(num_gallery),
        "num_valid_queries_for_map": float(num_valid_queries),
        "num_queries_without_positive": float(num_invalid_queries),
        "metric_cosine": 1.0 if metric_name == "cosine" else 0.0,
        "metric_l2": 1.0 if metric_name == "l2" else 0.0,
    }

    for k in topk:
        recall_hits = torch.cat(recall_hits_by_k[k], dim=0)
        metrics[f"recall_at_{k}"] = float(recall_hits.mean().item())

        ap_numerator = torch.cat(ap_numerators_by_k[k], dim=0)
        ap_denominator = torch.cat(ap_denominators_by_k[k], dim=0)

        if num_valid_queries == 0:
            metrics[f"map_at_{k}"] = 0.0
            continue

        valid_ap = ap_numerator[valid_mask] / ap_denominator[valid_mask].to(
            dtype=ap_numerator.dtype
        )
        metrics[f"map_at_{k}"] = float(valid_ap.mean().item())

    return metrics


def _gather_topk_image_ids(
    topk_indices: torch.Tensor,
    gallery_image_ids: list[str],
) -> list[list[str]]:
    """Collects retrieved gallery image ids from top-k indices."""
    if topk_indices.ndim != 2:
        raise ValueError(
            f"topk_indices must have shape [Q, K], got ndim={topk_indices.ndim}."
        )

    gathered: list[list[str]] = []
    for row in topk_indices.tolist():
        gathered.append([gallery_image_ids[index] for index in row])
    return gathered